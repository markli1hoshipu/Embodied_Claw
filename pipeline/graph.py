"""StateGraph wiring (langgraph 1.2.4, verified API): linear 0->1->2, fan-out 2->{3,4},
list-edge join {3,4}->5 (guarantees train waits for BOTH branches), 5->6. SqliteSaver per run."""
from __future__ import annotations

import json
import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from pipeline import tools
from pipeline.state import STAGES, PipelineState


def build_graph(checkpointer=None):
    from pipeline.nodes import (filter_build, ingest, intake, norm_stats, train,
                                upload_dataset, upload_model)
    g = StateGraph(PipelineState)
    for name, mod in [("intake", intake), ("ingest", ingest), ("filter_build", filter_build),
                      ("upload_dataset", upload_dataset), ("norm_stats", norm_stats),
                      ("train", train), ("upload_model", upload_model)]:
        g.add_node(name, mod.node)
    g.add_edge(START, "intake")
    g.add_edge("intake", "ingest")
    g.add_edge("ingest", "filter_build")
    g.add_edge("filter_build", "upload_dataset")        # fan-out: Node 3 + Node 4 concurrent
    g.add_edge("filter_build", "norm_stats")
    g.add_edge(["upload_dataset", "norm_stats"], "train")  # join: train waits for both
    g.add_edge("train", "upload_model")
    g.add_edge("upload_model", END)
    return g.compile(checkpointer=checkpointer)


def open_checkpointer(run_id: str) -> SqliteSaver:
    """runs/<run_id>/state.sqlite; check_same_thread=False is mandatory — langgraph executes
    parallel nodes in a thread pool (SqliteSaver serializes access internally)."""
    rd = tools.run_dir(run_id)
    rd.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(rd / "state.sqlite"), check_same_thread=False)
    return SqliteSaver(conn)


def pending_interrupts(graph, conf) -> list:
    snap = graph.get_state(conf)
    return [i for t in snap.tasks for i in getattr(t, "interrupts", ())]


def finalize_run(run_id: str, values: dict) -> None:
    """artifacts.json + summary.md at run end (spec section 2)."""
    rd = tools.run_dir(run_id)
    arts = {s: (values.get(s) or {}).get("artifact_paths", []) for s in STAGES}
    (rd / "artifacts.json").write_text(json.dumps(arts, indent=2))
    lines = [f"# Run `{run_id}`", ""]
    for s in STAGES:
        st = values.get(s) or {}
        extra = st.get("error") or "; ".join(st.get("artifact_paths") or [])
        lines.append(f"- **{s}** — {st.get('status', 'pending')} "
                     f"({st.get('started_at')} -> {st.get('finished_at')}) {extra}")
    (rd / "summary.md").write_text("\n".join(lines) + "\n")
    if all((values.get(s) or {}).get("status") == "succeeded" for s in STAGES):
        tf = rd / "transitions.jsonl"  # run-level 'done' line — the bridge marks the run done
        last = tf.read_text().splitlines()[-1] if tf.exists() and tf.stat().st_size else ""
        if '"node": "run"' not in last:
            tools.log_transition(run_id, "run", "done")
