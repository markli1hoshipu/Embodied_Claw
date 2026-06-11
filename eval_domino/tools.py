"""Eval-pipeline plumbing on top of pipeline.tools (same runs root, same mailbox contract).
Adds: benchmark/model path constants and direct_ask (interrupt-based escalation from
deterministic nodes that have no Agent attached, e.g. resource_gate)."""
from __future__ import annotations

import json
from pathlib import Path

from langgraph.types import interrupt

from pipeline.tools import (find_unanswered, log_transition, new_escalation_id, notify,  # noqa: F401
                            now, read_reply, run_dir, runs_root, sh, write_question)

DOMINO = Path("/work/markhsp/DOMINO")
# Activation prelude for any DOMINO subprocess (conda env + Vulkan loader/ICD exports).
DOMINO_ENV = "source /work/markhsp/DOMINO/domino_env.sh"
MODELS_CACHE = Path("/work/markhsp/models_cache")


def direct_ask(run_id: str, node: str, question: str, context: str = "",
               options: list | None = None, recommendation: int | None = None) -> dict:
    """Agent._ask_user without the Agent: same question.json/reply mailbox + interrupt(),
    so the bridge renders it identically. Re-entry safe (re-attaches to unanswered question)."""
    rd = run_dir(run_id)
    esc_id = find_unanswered(rd, node) or new_escalation_id(node)
    q = write_question(rd, esc_id, node=node, agent="pipeline", question=question,
                       context=context, options=options, recommendation=recommendation)
    reply = read_reply(rd, esc_id)
    if reply is None:
        log_transition(run_id, node, "escalated", esc_id)
        notify(f"[{run_id}:{node}] {q['question']}")
        reply = interrupt({"escalation_id": esc_id, **q})
    return reply


def eval_config(run_id: str) -> dict:
    p = run_dir(run_id) / "config.json"
    return json.loads(p.read_text()) if p.exists() else {}


def cancel_requested(run_id: str) -> bool:
    return (run_dir(run_id) / "CANCEL").exists()
