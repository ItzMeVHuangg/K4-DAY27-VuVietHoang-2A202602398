"""Exercise 5 — Streamlit approval UI for the HITL PR review agent.

Run with:
    uv run streamlit run app.py

Goal: wrap the LangGraph built in exercises 1–4 in a web UI that adapts to
the confidence bucket of each PR.

Routing thresholds (common/schemas.py):
    > 72%        auto_approve     UI shows a success card; reviewer does nothing
    58 – 72%     human_approval   UI shows Approve / Reject / Edit buttons
    <  58%       escalate         UI shows a question form for the reviewer,
                                  then the refined review for a final confirm
"""

from __future__ import annotations

import asyncio
import os
import uuid

import streamlit as st
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from common.db import db_conn, db_path, replay_events
from common.schemas import AUTO_APPROVE_THRESHOLD, ESCALATE_THRESHOLD
from exercises.exercise_4_audit import _render_comment_body, build_graph


load_dotenv()


# ─── Session state ─────────────────────────────────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "pr_url" not in st.session_state:
    st.session_state.pr_url = ""
if "interrupt_payload" not in st.session_state:
    st.session_state.interrupt_payload = None
if "final" not in st.session_state:
    st.session_state.final = None
if "error" not in st.session_state:
    st.session_state.error = None
if "stalled" not in st.session_state:  # run crashed mid-graph (e.g. LLM 429) — can be continued
    st.session_state.stalled = False
if "pr_url_input" not in st.session_state:  # backing value of the PR URL text box
    st.session_state.pr_url_input = ""


# ─── Graph helpers ─────────────────────────────────────────────────────────
def _cfg(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


async def run_graph(pr_url: str, thread_id: str, resume_value=None, continue_run: bool = False):
    """Invoke the graph once. Returns the final result or {'__interrupt__': ...}.

    `continue_run=True` re-enters a thread that crashed mid-run from its last checkpoint.
    """
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        app = build_graph(cp, confirm_after_escalation=True)
        cfg = _cfg(thread_id)
        if continue_run:
            return await app.ainvoke(None, cfg)
        if resume_value is None:
            return await app.ainvoke({"pr_url": pr_url, "thread_id": thread_id}, cfg)
        return await app.ainvoke(Command(resume=resume_value), cfg)


async def load_thread(thread_id: str) -> tuple[dict, dict | None, bool]:
    """Read a thread back from the checkpointer: (state values, pending interrupt payload, stalled)."""
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        app = build_graph(cp, confirm_after_escalation=True)
        snap = await app.aget_state(_cfg(thread_id))
    pending = None
    for task in snap.tasks:
        if task.interrupts:
            pending = task.interrupts[0].value
            break
    return dict(snap.values or {}), pending, bool(snap.next) and pending is None


async def recent_threads(limit: int = 15) -> list[dict]:
    """Same query as `audit.replay.list_threads`, returned as rows instead of printed."""
    async with db_conn() as conn:
        async with conn.execute(
            """
            SELECT thread_id,
                   pr_url,
                   MAX(timestamp) AS last_event,
                   CASE MAX(CASE risk_level WHEN 'high' THEN 3 WHEN 'med' THEN 2 ELSE 1 END)
                        WHEN 3 THEN 'high' WHEN 2 THEN 'med' ELSE 'low' END AS worst_risk,
                   COUNT(*)       AS events
              FROM audit_events
             GROUP BY thread_id, pr_url
             ORDER BY MAX(timestamp) DESC
             LIMIT ?
            """,
            (limit,),
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]


def apply_result(result: dict) -> None:
    """Store a graph result in session state (either a pending interrupt or the final state)."""
    st.session_state.stalled = False
    if "__interrupt__" in result:
        st.session_state.interrupt_payload = result["__interrupt__"][0].value
        st.session_state.final = None
    else:
        st.session_state.interrupt_payload = None
        st.session_state.final = result


def open_thread(thread_id: str) -> None:
    """Point the UI at an existing thread and restore it from the checkpointer."""
    values, pending, stalled = asyncio.run(load_thread(thread_id))
    st.session_state.thread_id = thread_id
    st.session_state.pr_url = values.get("pr_url", "")
    st.session_state.pr_url_input = st.session_state.pr_url  # runs before the form renders
    st.session_state.interrupt_payload = pending
    st.session_state.final = values if pending is None and values.get("final_action") else None
    st.session_state.stalled = stalled
    st.session_state.error = None
    st.query_params["thread"] = thread_id


# Restore the session after a full page reload: thread_id lives in the URL.
if st.session_state.thread_id is None and st.query_params.get("thread"):
    try:
        open_thread(st.query_params["thread"])
    except Exception as e:  # stale / unknown thread id in the URL
        st.session_state.error = f"Could not restore thread: {e}"


# ─── Page setup ────────────────────────────────────────────────────────────
st.set_page_config(page_title="HITL PR Review", layout="wide")
st.title("HITL PR Review Agent")
st.caption(
    f"auto-approve ≥ {AUTO_APPROVE_THRESHOLD:.0%} · human approval "
    f"{ESCALATE_THRESHOLD:.0%}–{AUTO_APPROVE_THRESHOLD:.0%} · escalate < {ESCALATE_THRESHOLD:.0%}"
)


# ─── Sidebar — reviewer + recent sessions ──────────────────────────────────
RISK_BADGE = {"low": "🟢", "med": "🟡", "high": "🔴"}

with st.sidebar:
    reviewer = st.text_input(
        "Reviewer (GitHub username)", value=os.environ.get("GITHUB_USER", ""),
        help="Written to audit_events.reviewer_id on every HITL step.",
    )
    if reviewer:
        os.environ["GITHUB_USER"] = reviewer

    st.header("Recent sessions")
    try:
        threads = asyncio.run(recent_threads())
    except Exception as e:
        threads = []
        st.caption(f"(audit DB unavailable: {e})")
    if not threads:
        st.caption("No sessions yet.")
    for t in threads:
        pr_short = t["pr_url"].split("github.com/")[-1]
        label = f"{RISK_BADGE.get(t['worst_risk'], '⚪')} {pr_short} · {t['thread_id'][:8]}"
        current = t["thread_id"] == st.session_state.thread_id
        if st.button(label, key=f"thread_{t['thread_id']}", width="stretch",
                     type="primary" if current else "secondary",
                     help=f"{t['events']} events · last {t['last_event']}"):
            open_thread(t["thread_id"])
            st.rerun()


# ─── Top form — start a new review ─────────────────────────────────────────
with st.form("start"):
    pr_url = st.text_input(
        "PR URL", key="pr_url_input",
        placeholder="https://github.com/VinUni-AI20k/PR-Demo/pull/1",
    )
    submitted = st.form_submit_button("Run review", type="primary")


# ─── Renderers per interrupt kind ──────────────────────────────────────────
def render_approval_card(payload: dict) -> dict | None:
    """58–72% bucket (and final confirm after escalation): review + 3 buttons."""
    conf = payload["confidence"]
    if payload.get("refined"):
        st.subheader(f"Refined review — final confirm · confidence {conf:.0%}")
        st.info("The agent re-analyzed the PR with your answers. Confirm before it posts.")
    else:
        st.subheader(f"Approval requested — confidence {conf:.0%}")
    st.progress(conf)
    st.caption(payload["confidence_reasoning"])
    st.markdown(payload["summary"])
    if payload.get("risk_factors"):
        st.warning("Risks: " + "; ".join(payload["risk_factors"]))

    st.markdown("**Proposed comments**")
    for c in payload.get("comments", []) or [{"severity": "-", "file": "-", "body": "(none)"}]:
        st.markdown(f"- **[{c['severity']}]** `{c['file']}:{c.get('line') or '?'}` — {c['body']}")

    with st.expander("Diff", expanded=not payload.get("refined")):
        st.code(payload.get("diff_preview", ""), language="diff")

    feedback = st.text_input(
        "Feedback (optional — required for Edit: tell the agent what to change)",
        key=f"approval_feedback_{st.session_state.thread_id}_{payload.get('refined')}",
    )
    col1, col2, col3 = st.columns(3)
    if col1.button("Approve", type="primary", width="stretch"):
        return {"choice": "approve", "feedback": feedback}
    if col2.button("Reject", width="stretch"):
        return {"choice": "reject", "feedback": feedback}
    if col3.button("Edit", width="stretch"):
        if not feedback.strip():
            st.error("Write what should change in the feedback box, then click Edit.")
            return None
        return {"choice": "edit", "feedback": feedback}
    return None


def render_escalation_card(payload: dict) -> dict | None:
    """< 58% bucket: show risk factors + question form. Return {question: answer} or None."""
    conf = payload["confidence"]
    st.subheader(f"Strong escalation — confidence {conf:.0%}")
    st.progress(conf)
    st.caption(payload["confidence_reasoning"])
    if payload.get("risk_factors"):
        st.error("**Risk factors**\n\n" + "\n".join(f"- {r}" for r in payload["risk_factors"]))
    st.markdown(payload["summary"])
    if payload.get("diff_preview"):
        with st.expander("Diff"):
            st.code(payload["diff_preview"], language="diff")

    with st.form(f"escalation_{st.session_state.thread_id}"):
        st.markdown("**The agent needs your input on:**")
        answers: dict[str, str] = {}
        for i, q in enumerate(payload["questions"]):
            answers[q] = st.text_area(f"{i + 1}. {q}", key=f"esc_{st.session_state.thread_id}_{i}")
        if st.form_submit_button("Submit answers", type="primary"):
            if any(not a.strip() for a in answers.values()):
                st.error("Please answer every question.")
                return None
            return {q: a.strip() for q, a in answers.items()}
    return None


def render_final(final: dict) -> None:
    action = final.get("final_action", "?")
    a = final.get("analysis")
    pr = st.session_state.pr_url
    if action.startswith("auto_committed"):
        st.success(f"✓ Auto-approved — confidence {a.confidence:.0%}, no human needed. "
                   "The agent already posted its review.")
        st.markdown(a.summary)
        st.link_button("View comment on GitHub", pr)
    elif action.startswith("committed"):
        st.success(f"✓ {action} — comment posted to {pr}")
        st.link_button("View comment on GitHub", pr)
    elif action == "rejected":
        st.warning("Rejected — no comment posted")
    elif "failed" in action:
        st.error(f"{action} — posting to GitHub failed (check GITHUB_TOKEN scope; see terminal log)")
    else:
        st.info(f"final_action = {action}")

    if a is not None and action != "rejected":
        with st.expander("Comment body", expanded=False):
            st.markdown(_render_comment_body(final))


def render_timeline(thread_id: str) -> None:
    events = asyncio.run(replay_events(thread_id))
    if not events:
        return
    with st.expander(f"Audit trail ({len(events)} events)"):
        st.dataframe(
            [{k: ev[k] for k in ("timestamp", "action", "confidence", "risk_level",
                                 "decision", "reviewer_id", "execution_time_ms", "reason")}
             for ev in events],
            width="stretch", hide_index=True,
        )
    st.caption(f"thread_id = {thread_id}  ·  replay: "
               f"`uv run python -m audit.replay --thread {thread_id}`")


# ─── Main flow ─────────────────────────────────────────────────────────────
if submitted and pr_url:
    st.session_state.pr_url = pr_url
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.interrupt_payload = None
    st.session_state.final = None
    st.session_state.error = None
    st.query_params["thread"] = st.session_state.thread_id

    with st.spinner("Fetching PR + asking the LLM..."):
        try:
            apply_result(asyncio.run(run_graph(pr_url, st.session_state.thread_id)))
        except Exception as e:
            st.session_state.error = f"{type(e).__name__}: {e}"
            st.session_state.stalled = True

if st.session_state.error:
    st.error(st.session_state.error)

# Render the current interrupt card, if any
payload = st.session_state.interrupt_payload
if payload is not None:
    kind = payload["kind"]
    answer = render_approval_card(payload) if kind == "approval_request" else render_escalation_card(payload)
    if answer is not None:
        with st.spinner("Resuming..."):
            try:
                apply_result(asyncio.run(run_graph(
                    st.session_state.pr_url, st.session_state.thread_id, resume_value=answer,
                )))
                st.session_state.error = None
            except Exception as e:
                st.session_state.error = f"{type(e).__name__}: {e}"
                st.session_state.interrupt_payload = None
                st.session_state.stalled = True
        st.rerun()

# A run that crashed mid-graph (LLM quota, network...) resumes from its last checkpoint
if st.session_state.stalled and st.session_state.interrupt_payload is None:
    st.warning("This run stopped before finishing. Its progress is checkpointed — "
               "continue from where it stopped (wait a minute first if the LLM hit a rate limit).")
    if st.button("Continue run", type="primary"):
        with st.spinner("Continuing from the last checkpoint..."):
            try:
                apply_result(asyncio.run(run_graph(
                    st.session_state.pr_url, st.session_state.thread_id, continue_run=True,
                )))
                st.session_state.error = None
            except Exception as e:
                st.session_state.error = f"{type(e).__name__}: {e}"
        st.rerun()

# Render final state, if reached
if st.session_state.final is not None:
    render_final(st.session_state.final)

if st.session_state.thread_id:
    render_timeline(st.session_state.thread_id)
