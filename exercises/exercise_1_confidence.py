"""Exercise 1 — Confidence scoring + routing.

Build a small LangGraph that fetches a PR, analyzes it, then routes to one of
three terminal nodes by confidence. Goal: see the three branches print
different messages on different PRs.

"""

from __future__ import annotations

import argparse

import openai
from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from rich.console import Console

from common.github import fetch_pr
from common.llm import get_llm
from common.schemas import (
    AUTO_APPROVE_THRESHOLD,
    ESCALATE_THRESHOLD,
    PRAnalysis,
    ReviewState,
)


console = Console()
# Gemini/OpenRouter occasionally answer 503 "high demand" / 429 — retry those with backoff.
LLM_RETRY = dict(
    stop_after_attempt=5,
    exponential_jitter_params={"initial": 10, "max": 60},  # free-tier quota resets per minute
    retry_if_exception_type=(openai.InternalServerError, openai.RateLimitError, openai.APIConnectionError),
)

SYSTEM_PROMPT = """You are a senior software engineer reviewing a GitHub pull request.
Read the diff carefully and return a structured review:
- summary: one paragraph on what the PR does.
- risk_factors: concrete risks (security, data loss, migrations, missing tests, ...).
- comments: specific review comments tied to a file (and line when you can tell).
- confidence: how sure you are that your review is complete and correct AND that
  the PR is safe to merge with only your comments. Calibrate honestly:
    0.85-1.00  trivial/mechanical change (typo, docs, dependency bump), nothing to ask
    0.58-0.72  normal feature change, looks fine but a human should double-check
               an open question (e.g. schema/data migration, backward compatibility)
    0.00-0.57  security-sensitive code (auth, passwords, tokens, crypto, SQL built
               from strings, network sync), multiple red flags, or intent you cannot
               verify from the diff alone
- confidence_reasoning: why you picked that value.
- escalation_questions: if confidence < 0.58, 2-4 specific questions for the human
  reviewer, each referencing the file/section of the diff it is about. Otherwise [].
"""


def node_fetch_pr(state: ReviewState) -> dict:
    console.print("[cyan]→ fetch_pr[/cyan]")
    with console.status("[dim]Fetching PR from GitHub...[/dim]"):
        pr = fetch_pr(state["pr_url"])
    console.print(f"  [green]✓[/green] {len(pr.files_changed)} files, head {pr.head_sha[:7]}")
    return {
        "pr_title": pr.title, "pr_diff": pr.diff,
        "pr_files": pr.files_changed, "pr_head_sha": pr.head_sha,
    }


def node_analyze(state: ReviewState) -> dict:
    console.print("[cyan]→ analyze[/cyan]")
    llm = get_llm().with_structured_output(PRAnalysis).with_retry(**LLM_RETRY)
    with console.status("[dim]LLM thinking...[/dim]"):
        analysis = llm.invoke([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Title: {state['pr_title']}\n"
                f"Files: {', '.join(state['pr_files'])}\n"
                f"Diff:\n{state['pr_diff']}"
            )},
        ])
    console.print(f"  [green]✓[/green] confidence={analysis.confidence:.0%}, {len(analysis.comments)} comment(s)")
    return {"analysis": analysis}


def node_route(state: ReviewState) -> dict:
    console.print("[cyan]→ route[/cyan]")
    c = state["analysis"].confidence
    if c >= AUTO_APPROVE_THRESHOLD:
        decision = "auto_approve"
    elif c < ESCALATE_THRESHOLD:
        decision = "escalate"
    else:
        decision = "human_approval"
    console.print(f"  [green]✓[/green] decision=[bold]{decision}[/bold] (confidence={c:.0%})")
    return {"decision": decision}


def node_auto_approve(state: ReviewState) -> dict:
    console.print("[green]✓ AUTO APPROVE[/green] — high confidence, no human needed")
    return {"final_action": "auto_approved"}


def node_human_approval(state: ReviewState) -> dict:
    console.print("[yellow]✓ HUMAN APPROVAL[/yellow] — placeholder, exercise 2 will pause here")
    return {"final_action": "pending_human_approval"}


def node_escalate(state: ReviewState) -> dict:
    console.print("[red]✓ ESCALATE[/red] — placeholder, exercise 3 will ask the reviewer questions")
    return {"final_action": "pending_escalation"}


def build_graph():
    g = StateGraph(ReviewState)
    for name, fn in [
        ("fetch_pr", node_fetch_pr), ("analyze", node_analyze), ("route", node_route),
        ("auto_approve", node_auto_approve), ("human_approval", node_human_approval),
        ("escalate", node_escalate),
    ]:
        g.add_node(name, fn)
    g.add_edge(START, "fetch_pr")
    g.add_edge("fetch_pr", "analyze")
    g.add_edge("analyze", "route")
    g.add_conditional_edges(
        "route", lambda s: s["decision"],
        {"auto_approve": "auto_approve", "human_approval": "human_approval", "escalate": "escalate"},
    )
    for terminal in ("auto_approve", "human_approval", "escalate"):
        g.add_edge(terminal, END)
    return g.compile()


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--pr", required=True)
    args = parser.parse_args()

    console.rule("[bold]Exercise 1 — confidence routing[/bold]")
    console.print(f"[dim]PR: {args.pr}[/dim]\n")

    app = build_graph()
    final = app.invoke({"pr_url": args.pr})

    console.rule("Final")
    console.print(f"confidence = {final['analysis'].confidence:.0%}")
    console.print(f"action     = {final.get('final_action')}")


if __name__ == "__main__":
    main()
