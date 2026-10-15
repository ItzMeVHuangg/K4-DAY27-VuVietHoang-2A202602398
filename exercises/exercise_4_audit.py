"""Exercise 4 — Structured SQLite audit trail + durable checkpointer.

Zero setup — SQLite stores everything in a single file (`./hitl_audit.db`).
The audit_events schema is created automatically on first connection.

Goals:

1. Use AsyncSqliteSaver so the graph can resume after a crash.
2. Define and emit an `AuditEntry` (common/schemas.py) for every meaningful step,
   so the full session can be replayed.
3. Verify with `uv run python -m audit.replay --thread <id>`.

Approach:
    - Read `node_fetch_pr` below — it is the one fully-worked example. Pay attention
      to *why* each AuditEntry field has the value it does at that step.
    - For every other node, you decide what to log. Field reference:
      `common/schemas.py:AuditEntry`. Helper: `risk_level_for(confidence)`.
    - Implement the one-line body of `audit()`. Everything else (graph wiring,
      checkpointer setup, interrupt/resume loop) is already done for you.

TODOs to complete (10 total): the audit() body, ONE AuditEntry per node for
analyze/route/commit/auto_approve/synthesize, TWO each for human_approval and
escalate (before + after the interrupt).

Extras on top of the TODOs:
    - The "before interrupt" rows are de-duplicated: a node that calls interrupt()
      re-runs from the top on resume, so without the check every HITL step would
      log its "pending" row twice.
    - Bonus 4 (auto-edit): when the reviewer picks `edit`, node_commit asks the LLM
      to rewrite the review using the reviewer's feedback, then posts it.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import time
import uuid

import openai
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from rich.console import Console
from rich.panel import Panel

from common.db import db_conn, db_path, write_audit_event
from common.github import fetch_pr, post_review_comment
from common.llm import get_llm
from common.schemas import (
    AUTO_APPROVE_THRESHOLD,
    ESCALATE_THRESHOLD,
    AuditEntry,
    PRAnalysis,
    ReviewState,
    risk_level_for,
)


console = Console()
# Gemini/OpenRouter occasionally answer 503 "high demand" / 429 — retry those with backoff.
LLM_RETRY = dict(
    stop_after_attempt=5,
    exponential_jitter_params={"initial": 10, "max": 60},  # free-tier quota resets per minute
    retry_if_exception_type=(openai.InternalServerError, openai.RateLimitError, openai.APIConnectionError),
)
# Let the checkpointer (de)serialize our Pydantic state types without warnings.
SERDE = JsonPlusSerializer(allowed_msgpack_modules=[
    ("common.schemas", "PRAnalysis"), ("common.schemas", "ReviewComment"),
])
AGENT_ID = "pr-review-agent@v0.1"

ANALYZE_PROMPT = f"""You are a senior software engineer reviewing a GitHub pull request.
Read the diff carefully and return a structured review:
- summary: one paragraph on what the PR does.
- risk_factors: concrete risks (security, data loss, migrations, missing tests, ...).
- comments: specific review comments tied to a file (and line when you can tell).
- confidence: how sure you are that your review is complete and correct AND that
  the PR is safe to merge with only your comments. Calibrate honestly:
    {AUTO_APPROVE_THRESHOLD:.2f}-1.00  trivial/mechanical change (typo, docs, dependency bump)
    {ESCALATE_THRESHOLD:.2f}-{AUTO_APPROVE_THRESHOLD - 0.01:.2f}  normal feature change, looks fine but a human should
               double-check an open question (e.g. schema migration, compatibility)
    0.00-{ESCALATE_THRESHOLD - 0.01:.2f}  security-sensitive code (auth, passwords, tokens, crypto, SQL
               built from strings, network sync), multiple red flags, or intent you
               cannot verify from the diff alone
- confidence_reasoning: why you picked that value.
- escalation_questions: if confidence < {ESCALATE_THRESHOLD:.2f}, 2-4 specific questions for the
  human reviewer, each referencing the file/section of the diff it is about and asking
  about intent, threat model or production context. Otherwise [].
"""


def _ms(t0: float) -> int:
    return int((time.monotonic() - t0) * 1000)


def _reviewer_id() -> str:
    """GitHub username of the human reviewer (set GITHUB_USER in .env)."""
    return os.environ.get("GITHUB_USER") or "unknown-reviewer"


async def audit(state, entry: AuditEntry) -> None:
    """Write one structured AuditEntry row to the `audit_events` table.

    `thread_id` and `pr_url` are taken from `state` so callers only build
    the entry itself.
    """
    await write_audit_event(thread_id=state["thread_id"], pr_url=state["pr_url"], entry=entry)


async def _already_logged(state, action: str, decision: str) -> bool:
    """True if this thread already has an (action, decision) row.

    Used for the pre-interrupt rows: on resume LangGraph re-executes the node
    from the top, which would otherwise write the "pending" row a second time.
    """
    async with db_conn() as conn:
        async with conn.execute(
            "SELECT 1 FROM audit_events WHERE thread_id = ? AND action = ? AND decision = ? LIMIT 1",
            (state["thread_id"], action, decision),
        ) as cur:
            return await cur.fetchone() is not None


# ─── Reference example — read this carefully ───────────────────────────────
async def node_fetch_pr(state):
    console.print("[cyan]→ fetch_pr[/cyan]")
    t0 = time.monotonic()
    with console.status("[dim]Fetching PR from GitHub...[/dim]"):
        pr = fetch_pr(state["pr_url"])
    console.print(f"  [green]✓[/green] {len(pr.files_changed)} files, head {pr.head_sha[:7]}")
    # We've only fetched the diff, not analyzed it. So:
    #   - confidence is unknown → 0.0
    #   - risk_level can't be derived from confidence yet → "med" as neutral default
    #   - decision is "pending" — nothing has been decided
    #   - reviewer_id is None — no human is involved at this stage
    #   - reason is a short human-readable summary of what happened
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="fetch_pr",
        confidence=0.0,
        risk_level="med",
        decision="pending",
        reason=f"Fetched {len(pr.files_changed)} files, head={pr.head_sha[:7]}",
        execution_time_ms=int((time.monotonic() - t0) * 1000),
    ))
    return {
        "pr_title": pr.title,
        "pr_diff": pr.diff,
        "pr_files": pr.files_changed,
        "pr_head_sha": pr.head_sha,
    }
# ───────────────────────────────────────────────────────────────────────────


async def node_analyze(state):
    console.print("[cyan]→ analyze[/cyan]")
    t0 = time.monotonic()
    llm = get_llm().with_structured_output(PRAnalysis).with_retry(**LLM_RETRY)
    with console.status("[dim]LLM reviewing the diff...[/dim]"):
        a: PRAnalysis = await llm.ainvoke([
            {"role": "system", "content": ANALYZE_PROMPT},
            {"role": "user", "content": (
                f"Title: {state['pr_title']}\n"
                f"Files: {', '.join(state['pr_files'])}\n"
                f"Diff:\n{state['pr_diff']}"
            )},
        ])
    console.print(f"  [green]✓[/green] confidence={a.confidence:.0%}, {len(a.comments)} comment(s)")
    # The LLM has produced a confidence, but no routing decision exists yet.
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="analyze",
        confidence=a.confidence,
        risk_level=risk_level_for(a.confidence),
        decision="pending",
        reason=a.confidence_reasoning,
        execution_time_ms=_ms(t0),
    ))
    return {"analysis": a}


async def node_route(state):
    console.print("[cyan]→ route[/cyan]")
    t0 = time.monotonic()
    c = state["analysis"].confidence
    if c >= AUTO_APPROVE_THRESHOLD:
        decision = "auto_approve"
    elif c < ESCALATE_THRESHOLD:
        decision = "escalate"
    else:
        decision = "human_approval"
    console.print(f"  [green]✓[/green] decision=[bold]{decision}[/bold] (confidence={c:.0%})")
    # Map the graph branch onto the AuditEntry decision vocabulary.
    audit_decision = {"auto_approve": "auto", "human_approval": "pending", "escalate": "escalate"}[decision]
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="route",
        confidence=c,
        risk_level=risk_level_for(c),
        decision=audit_decision,
        reason=f"confidence={c:.2f} → {decision} "
               f"(auto ≥ {AUTO_APPROVE_THRESHOLD}, escalate < {ESCALATE_THRESHOLD})",
        execution_time_ms=_ms(t0),
    ))
    return {"decision": decision}


async def node_human_approval(state):
    t0 = time.monotonic()
    a = state["analysis"]

    # Before the interrupt: the request is shown to a human, nobody answered yet.
    if not await _already_logged(state, "human_approval", "pending"):
        await audit(state, AuditEntry(
            agent_id=AGENT_ID,
            action="human_approval",
            confidence=a.confidence,
            risk_level=risk_level_for(a.confidence),
            reviewer_id=None,
            decision="pending",
            reason="Approval requested from human reviewer",
            execution_time_ms=_ms(t0),
        ))

    resp = interrupt({
        "kind": "approval_request",
        "pr_url": state["pr_url"],
        "refined": bool(state.get("escalation_answers")),
        "confidence": a.confidence,
        "confidence_reasoning": a.confidence_reasoning,
        "summary": a.summary,
        "risk_factors": a.risk_factors,
        "comments": [c.model_dump() for c in a.comments],
        "diff_preview": state["pr_diff"][:2000],
    })

    # After resume: we know who answered and what they chose.
    choice = resp.get("choice")
    feedback = resp.get("feedback") or None
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="human_approval",
        confidence=a.confidence,
        risk_level=risk_level_for(a.confidence),
        reviewer_id=_reviewer_id(),
        decision=choice,
        reason=feedback,
        execution_time_ms=_ms(t0),
    ))
    return {"human_choice": choice, "human_feedback": feedback}


def _render_comment_body(state) -> str:
    a = state["analysis"]
    lines = [f"### Automated review (confidence {a.confidence:.0%})", "", a.summary, ""]
    for c in a.comments:
        lines.append(f"- **[{c.severity}]** `{c.file}:{c.line or '?'}` — {c.body}")
    if state.get("human_feedback"):
        lines.append(f"\n_Reviewer note: {state['human_feedback']}_")
    if state.get("escalation_answers"):
        lines.append("\n_Reviewer answered escalation questions:_")
        for q, ans in state["escalation_answers"].items():
            lines.append(f"> **{q}** {ans}")
    return "\n".join(lines)


def _post(state) -> str:
    try:
        post_review_comment(state["pr_url"], _render_comment_body(state))
        console.print(f"  [green]✓[/green] posted comment to {state['pr_url']}")
        return "committed"
    except Exception as e:
        console.print(f"  [red]✗[/red] post failed: {e}")
        return "commit_failed"


async def _apply_edit(state) -> PRAnalysis:
    """Bonus 4 — rewrite the review following the reviewer's edit instructions."""
    llm = get_llm().with_structured_output(PRAnalysis).with_retry(**LLM_RETRY)
    with console.status("[dim]LLM rewriting review from reviewer feedback...[/dim]"):
        return await llm.ainvoke([
            {"role": "system", "content": (
                "Senior reviewer. Rewrite the review below following the human reviewer's "
                "edit instructions exactly. Keep the same structured format and leave "
                "escalation_questions empty."
            )},
            {"role": "user", "content": (
                f"Diff:\n{state['pr_diff']}\n\n"
                f"Current review:\n{state['analysis'].model_dump_json(indent=2)}\n\n"
                f"Reviewer edit instructions:\n{state.get('human_feedback') or '(none)'}"
            )},
        ])


async def node_commit(state):
    console.print("[cyan]→ commit[/cyan]")
    t0 = time.monotonic()
    update: dict = {}
    choice = state.get("human_choice")
    # Paths that converge here:
    #   1. human_approval → commit (post if approved; rewrite + post if edit)
    #   2. escalate → synthesize → commit (always post the refined review)
    #   3. escalate → synthesize → human_approval → commit (UI: final confirm)
    if choice == "edit":
        edited = await _apply_edit(state)
        update["analysis"] = edited
        state = {**state, "analysis": edited}
        action = _post(state)
        decision = "edit"
    elif choice == "approve" or (state.get("escalation_answers") and choice is None):
        action = _post(state)
        decision = "approve"
    else:
        console.print(f"  [yellow]·[/yellow] skipping comment (choice={choice})")
        action = "rejected"
        decision = "reject"
    a = state["analysis"]
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="commit",
        confidence=a.confidence,
        risk_level=risk_level_for(a.confidence),
        reviewer_id=_reviewer_id(),
        decision=decision,
        reason=f"{action}: {len(a.comments)} comment(s)"
               + (" after escalation Q&A" if state.get("escalation_answers") else ""),
        execution_time_ms=_ms(t0),
    ))
    return {**update, "final_action": action}


async def node_auto_approve(state):
    console.print("[cyan]→ auto_approve[/cyan]  [dim]high confidence — posting directly[/dim]")
    t0 = time.monotonic()
    a = state["analysis"]
    action = _post(state)
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="auto_approve",
        confidence=a.confidence,
        risk_level=risk_level_for(a.confidence),
        reviewer_id=None,
        decision="auto",
        reason=f"{action}: {len(a.comments)} comment(s), no human in the loop",
        execution_time_ms=_ms(t0),
    ))
    return {"final_action": f"auto_{action}"}


async def node_escalate(state):
    t0 = time.monotonic()
    a = state["analysis"]
    questions = a.escalation_questions or ["What is the intent of this PR?"]

    # Before the interrupt: questions are out, reviewer hasn't answered yet.
    if not await _already_logged(state, "escalate", "escalate"):
        await audit(state, AuditEntry(
            agent_id=AGENT_ID,
            action="escalate",
            confidence=a.confidence,
            risk_level=risk_level_for(a.confidence),
            reviewer_id=None,
            decision="escalate",
            reason=f"Asked reviewer {len(questions)} question(s): " + " | ".join(questions),
            execution_time_ms=_ms(t0),
        ))

    answers = interrupt({
        "kind": "escalation",
        "pr_url": state["pr_url"],
        "confidence": a.confidence,
        "confidence_reasoning": a.confidence_reasoning,
        "summary": a.summary,
        "risk_factors": a.risk_factors,
        "questions": questions,
        "diff_preview": state["pr_diff"][:2000],
    })

    # After resume: the reviewer's answers are in.
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="escalate",
        confidence=a.confidence,
        risk_level=risk_level_for(a.confidence),
        reviewer_id=_reviewer_id(),
        decision="answered",
        reason=" | ".join(f"{q} → {ans}" for q, ans in answers.items()),
        execution_time_ms=_ms(t0),
    ))
    return {"escalation_answers": answers}


async def node_synthesize(state):
    console.print("[cyan]→ synthesize[/cyan]")
    t0 = time.monotonic()
    initial: PRAnalysis = state["analysis"]
    qa = "\n".join(f"Q: {q}\nA: {a}" for q, a in (state.get("escalation_answers") or {}).items())
    llm = get_llm().with_structured_output(PRAnalysis).with_retry(**LLM_RETRY)
    with console.status("[dim]LLM refining review with reviewer answers...[/dim]"):
        refined: PRAnalysis = await llm.ainvoke([
            {"role": "system", "content": (
                "Senior reviewer. You previously reviewed this PR with low confidence and asked "
                "a human reviewer clarifying questions. Using their answers, produce a refined, "
                "final review: keep comments that still apply, drop concerns the answers resolved, "
                "and add comments for problems the answers revealed. Re-estimate confidence given "
                "the new context. Leave escalation_questions empty."
            )},
            {"role": "user", "content": (
                f"Title: {state['pr_title']}\nDiff:\n{state['pr_diff']}\n\n"
                f"Initial analysis:\n{initial.model_dump_json(indent=2)}\n\n"
                f"Reviewer Q&A:\n{qa}"
            )},
        ])
    console.print(f"  [green]✓[/green] refined confidence={refined.confidence:.0%}")
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="synthesize",
        confidence=refined.confidence,
        risk_level=risk_level_for(refined.confidence),
        reviewer_id=None,
        decision="pending",
        reason=f"confidence {initial.confidence:.2f} → {refined.confidence:.2f}. "
               f"{refined.confidence_reasoning}",
        execution_time_ms=_ms(t0),
    ))
    # node_commit will run next and post the refined review to the PR.
    return {"analysis": refined}


def build_graph(checkpointer, confirm_after_escalation: bool = False):
    """Compile the review graph.

    `confirm_after_escalation=True` (used by the Streamlit UI) sends the refined
    review from `synthesize` through `human_approval` for a final confirm instead
    of posting it straight away.
    """
    g = StateGraph(ReviewState)
    for name, fn in [
        ("fetch_pr", node_fetch_pr), ("analyze", node_analyze), ("route", node_route),
        ("auto_approve", node_auto_approve), ("human_approval", node_human_approval),
        ("commit", node_commit), ("escalate", node_escalate), ("synthesize", node_synthesize),
    ]:
        g.add_node(name, fn)
    g.add_edge(START, "fetch_pr")
    g.add_edge("fetch_pr", "analyze")
    g.add_edge("analyze", "route")
    g.add_conditional_edges(
        "route", lambda s: s["decision"],
        {"auto_approve": "auto_approve", "human_approval": "human_approval", "escalate": "escalate"},
    )
    g.add_edge("auto_approve", END)
    g.add_edge("human_approval", "commit")
    g.add_edge("commit", END)
    g.add_edge("escalate", "synthesize")
    g.add_edge("synthesize", "human_approval" if confirm_after_escalation else "commit")
    checkpointer.serde = SERDE
    return g.compile(checkpointer=checkpointer)


def handle_interrupt(payload):
    kind = payload["kind"]
    if kind == "approval_request":
        console.print(Panel.fit(
            f"{payload['summary']}\n\n[dim]{payload['confidence_reasoning']}[/dim]",
            title=f"conf={payload['confidence']:.0%}", border_style="green",
        ))
        for c in payload.get("comments", []):
            console.print(f"  [{c['severity']}] {c['file']}:{c.get('line') or '?'} — {c['body']}")
        choice = ""
        while choice not in {"approve", "reject", "edit"}:
            choice = console.input("approve/reject/edit? ").strip().lower()
        return {"choice": choice, "feedback": console.input("Feedback: ").strip()}
    console.print(Panel.fit(
        payload["summary"] + (
            "\n\n[red]Risks:[/red] " + "; ".join(payload["risk_factors"])
            if payload.get("risk_factors") else ""
        ),
        title=f"Escalation conf={payload['confidence']:.0%}", border_style="yellow",
    ))
    return {q: console.input(f"Q: {q}\nA: ").strip() for q in payload["questions"]}


async def run(pr_url: str, thread_id: str | None):
    thread_id = thread_id or str(uuid.uuid4())
    console.rule("[bold]Exercise 4 — SQLite audit trail[/bold]")
    console.print(f"[dim]PR: {pr_url}[/dim]")
    console.print(f"[dim]thread_id = {thread_id}[/dim]\n")

    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        app = build_graph(cp)
        cfg = {"configurable": {"thread_id": thread_id}}

        # Resuming an existing thread: don't restart it from fetch_pr.
        snapshot = await app.aget_state(cfg)
        pending = [i for t in snapshot.tasks for i in t.interrupts]
        if pending:  # paused at interrupt() — ask the human again
            console.print(f"[dim]Resuming thread paused at {snapshot.next}[/dim]")
            result = {"__interrupt__": pending}
        elif snapshot.next:  # crashed mid-run (e.g. LLM error) — continue from the last checkpoint
            console.print(f"[dim]Continuing crashed thread from {snapshot.next}[/dim]")
            result = await app.ainvoke(None, cfg)
        else:
            result = await app.ainvoke({"pr_url": pr_url, "thread_id": thread_id}, cfg)
        while "__interrupt__" in result:
            payload = result["__interrupt__"][0].value
            result = await app.ainvoke(Command(resume=handle_interrupt(payload)), cfg)

        console.rule("Final")
        console.print(f"final_action = {result.get('final_action')}")
        console.print(f"\n[dim]Replay:[/dim] uv run python -m audit.replay --thread {thread_id}")


def main():
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--pr", required=True)
    p.add_argument("--thread", help="Resume an existing thread")
    args = p.parse_args()
    asyncio.run(run(args.pr, args.thread))


if __name__ == "__main__":
    main()
