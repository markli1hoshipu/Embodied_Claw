"""Shared node runner: skip-if-done guard, upstream gating, agent invocation, transitions log."""
from __future__ import annotations

import json

from pipeline import tools
from pipeline.state import StageStatus


def _status(status: str, started: str | None, agent: str | None = None, error: str | None = None,
            artifacts=(), escalation: dict | None = None) -> StageStatus:
    return {"status": status, "started_at": started, "finished_at": tools.now(),  # type: ignore
            "artifact_paths": [str(a) for a in artifacts], "agent_thread_id": agent,
            "escalation": escalation, "error": error}


def agent_node(stage: str, make_agent, prompt_fn, requires: tuple = (), extra=None):
    """Wrap one agent-driven node. Code before the agent loop is cheap/idempotent — on
    interrupt-resume the node re-runs from the top and the persisted conversation replays."""
    def node(state: dict) -> dict:
        if (state.get(stage) or {}).get("status") == "succeeded":
            return {}
        bad = [u for u in requires if state[u]["status"] != "succeeded"]
        if bad:
            return {stage: _status("skipped", tools.now(),
                                   error=f"upstream not succeeded: {', '.join(bad)}")}
        started = tools.now()
        tools.log_transition(state["run_id"], stage, "running")
        agent = make_agent(state)
        result = agent.run_node(stage, prompt_fn(state)) or {}
        status = result.get("status") if result.get("status") in ("succeeded", "failed") else "failed"
        out = {stage: _status(status, started, agent=agent.name, error=result.get("error"),
                              artifacts=result.get("artifact_paths") or [],
                              escalation=agent.last_escalation)}
        if result.get("notes"):
            out["shared_notes"] = {agent.name: result["notes"]}
        if status == "succeeded" and extra is not None:
            more = extra(state, result)
            if "__error__" in more:
                out[stage] = _status("failed", started, agent=agent.name, error=more["__error__"])
            else:
                out.update(more)
        tools.log_transition(state["run_id"], stage, out[stage]["status"],
                             result.get("summary") or out[stage].get("error"))
        return out
    return node


def cfg_excerpt(state: dict, *keys: str) -> str:
    cfg = state.get("config") or {}
    picked = {k: cfg.get(k) for k in keys} if keys else cfg
    return json.dumps(picked, indent=2)[:6000]
