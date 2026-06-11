"""StateGraph wiring, sqlite checkpointer, retry policies (langgraph 1.2.x API, verified)."""
from __future__ import annotations

import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from pipeline import nodes
from pipeline.state import PipelineState

PIPELINE_RUNS_DIR = Path("/work/markhsp/Embodied_Claw/pipeline_runs")
_ORDER = ["ingest", "filter_build", "norm_stats", "train", "upload"]

def _continue_unless_failed(stage: str, nxt: str):
    def route(state: PipelineState) -> str:
        return END if state[stage]["status"] == "failed" else nxt
    return route

def build_graph(checkpointer=None, ingest_retry: RetryPolicy | None = None):
    b = StateGraph(PipelineState)
    # 5 attempts, exponential backoff, only on HF 429s (spec ingest retry policy).
    # `ingest_retry` override exists so tests can exercise the SAME retry_on semantics fast.
    b.add_node("ingest", nodes.ingest_source,
               retry_policy=ingest_retry or RetryPolicy(max_attempts=5, initial_interval=5.0,
                                                        backoff_factor=2.0,
                                                        retry_on=nodes.TransientHFError))
    b.add_node("filter_build", nodes.filter_and_build)
    b.add_node("norm_stats", nodes.compute_norm_stats)
    b.add_node("train", nodes.train)
    b.add_node("upload", nodes.upload_to_hf)
    b.add_edge(START, "ingest")
    for cur, nxt in zip(_ORDER, _ORDER[1:]):  # conditional edges: halt as soon as a stage fails
        b.add_conditional_edges(cur, _continue_unless_failed(cur, nxt), {nxt: nxt, END: END})
    b.add_edge("upload", END)
    return b.compile(checkpointer=checkpointer)

def open_checkpointer(run_id: str) -> SqliteSaver:
    """One sqlite DB per run_id; thread_id == run_id. check_same_thread=False is mandatory —
    langgraph executes nodes in a thread pool (SqliteSaver serializes access internally)."""
    PIPELINE_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(PIPELINE_RUNS_DIR / f"{run_id}.sqlite"), check_same_thread=False)
    return SqliteSaver(conn)
