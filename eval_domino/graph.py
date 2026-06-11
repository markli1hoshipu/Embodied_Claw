"""Eval FSM wiring: linear intake -> resource_gate -> model_prepare -> bench_preflight ->
run_matrix -> aggregate_report. Failure routing happens inside nodes (upstream gating in
agent_node skips downstream stages), GPU leases are released in finalize on every path."""
from __future__ import annotations

import json
import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from eval_domino import gpu, tools
from eval_domino.state import STAGES, EvalState


def build_graph(checkpointer=None):
    from eval_domino.nodes import (aggregate_report, bench_preflight, intake,
                                     model_prepare, resource_gate, run_matrix)
    g = StateGraph(EvalState)
    for name, mod in [("intake", intake), ("resource_gate", resource_gate),
                      ("model_prepare", model_prepare), ("bench_preflight", bench_preflight),
                      ("run_matrix", run_matrix), ("aggregate_report", aggregate_report)]:
        g.add_node(name, mod.node)
    g.add_edge(START, "intake")
    for a, b in zip(STAGES, STAGES[1:]):
        g.add_edge(a, b)
    g.add_edge(STAGES[-1], END)
    return g.compile(checkpointer=checkpointer)


def open_checkpointer(run_id: str) -> SqliteSaver:
    rd = tools.run_dir(run_id)
    rd.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(rd / "state.sqlite"), check_same_thread=False)
    return SqliteSaver(conn)


def pending_interrupts(graph, conf) -> list:
    snap = graph.get_state(conf)
    return [i for t in snap.tasks for i in getattr(t, "interrupts", ())]


def finalize_run(run_id: str, values: dict) -> None:
    """artifacts.json + summary.md + lease release. Runs on every terminal path; release is
    idempotent and stale leases are also reaped by pid-liveness, so no path leaks a GPU."""
    freed = gpu.release(run_id)
    if freed:
        tools.log_transition(run_id, "resource_gate", "running", f"released GPUs {freed}")
    rd = tools.run_dir(run_id)
    arts = {s: (values.get(s) or {}).get("artifact_paths", []) for s in STAGES}
    (rd / "artifacts.json").write_text(json.dumps(arts, indent=2))
    lines = [f"# Eval run `{run_id}`", ""]
    for s in STAGES:
        st = values.get(s) or {}
        extra = st.get("error") or "; ".join(st.get("artifact_paths") or [])
        lines.append(f"- **{s}** — {st.get('status', 'pending')} "
                     f"({st.get('started_at')} -> {st.get('finished_at')}) {extra}")
    (rd / "summary.md").write_text("\n".join(lines) + "\n")
    if all((values.get(s) or {}).get("status") == "succeeded" for s in STAGES):
        tf = rd / "transitions.jsonl"
        last = tf.read_text().splitlines()[-1] if tf.exists() and tf.stat().st_size else ""
        if '"node": "run"' not in last:
            tools.log_transition(run_id, "run", "done")
