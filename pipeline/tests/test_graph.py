import helpers as H
import pytest

import pipeline.graph as G
from pipeline import nodes
from pipeline.state import STAGES, init_state


def _all_artifacts(cfg, monkeypatch):
    H.make_sources(cfg)
    H.make_drop_lists(nodes, cfg)
    H.make_built(nodes, cfg, episodes=13)
    H.make_norm(nodes, cfg)
    H.make_final_ckpt(nodes, cfg)
    monkeypatch.setattr(nodes, "_hf_api", lambda token=None: H.FakeApi(exists=True))
    monkeypatch.setattr(nodes, "_sh", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no shell")))


def test_full_graph_skips_everything_and_checkpoints(cfg, sandbox, monkeypatch, tmp_path):
    _all_artifacts(cfg, monkeypatch)
    monkeypatch.setattr(G, "PIPELINE_RUNS_DIR", tmp_path / "pipeline_runs")
    saver = G.open_checkpointer(cfg["run_id"])
    graph = G.build_graph(checkpointer=saver)
    final = graph.invoke(init_state(cfg),
                         config={"configurable": {"thread_id": cfg["run_id"]}}, durability="sync")
    assert {s: final[s]["status"] for s in STAGES} == {s: "skipped" for s in STAGES}
    tup = saver.get_tuple({"configurable": {"thread_id": cfg["run_id"]}})
    assert tup is not None
    assert tup.checkpoint["channel_values"]["train"]["status"] == "skipped"


def test_graph_halts_at_first_failed_stage(cfg, sandbox, monkeypatch):
    H.make_sources(cfg)       # ingest skips
    H.make_drop_lists(nodes, cfg)
    monkeypatch.setattr(nodes, "find_train_pids", lambda n: [])  # no live run; rebuild allowed
    monkeypatch.setattr(nodes, "_sh", lambda cmd, env=None, timeout=None: H.CP(1, "", "builder exploded"))
    final = G.build_graph().invoke(init_state(cfg))
    assert final["ingest"]["status"] == "skipped"
    assert final["filter_build"]["status"] == "failed"
    assert "builder exploded" in final["filter_build"]["error"]
    assert final["norm_stats"]["status"] == "pending"   # never ran: conditional edge halted
    assert final["train"]["status"] == "pending"
    assert final["upload"]["status"] == "pending"


def test_graph_retry_policy_retries_transient_429s(cfg, sandbox, monkeypatch):
    """The ingest RetryPolicy must actually catch TransientHFError and re-invoke the node —
    same retry_on semantics as production, tiny interval so the test stays fast."""
    from langgraph.types import RetryPolicy

    _all_artifacts(cfg, monkeypatch)  # everything downstream of ingest skips
    calls = {"n": 0}

    def flaky_ingest(state):
        calls["n"] += 1
        if calls["n"] < 3:
            raise nodes.TransientHFError("HF 429 rate limit on org/official")
        return {"ingest": {"status": "succeeded", "started_at": None, "finished_at": None,
                           "error": None, "artifact_paths": []}}

    monkeypatch.setattr(nodes, "ingest_source", flaky_ingest)
    graph = G.build_graph(ingest_retry=RetryPolicy(max_attempts=5, initial_interval=0.001,
                                                   backoff_factor=1.0, jitter=False,
                                                   retry_on=nodes.TransientHFError))
    final = graph.invoke(init_state(cfg))
    assert calls["n"] == 3                                # raised twice, retried, then succeeded
    assert final["ingest"]["status"] == "succeeded"
    assert final["upload"]["status"] == "skipped"


def test_graph_retry_policy_gives_up_after_max_attempts(cfg, sandbox, monkeypatch):
    from langgraph.types import RetryPolicy

    _all_artifacts(cfg, monkeypatch)
    calls = {"n": 0}

    def always_429(state):
        calls["n"] += 1
        raise nodes.TransientHFError("HF 429 rate limit")

    monkeypatch.setattr(nodes, "ingest_source", always_429)
    graph = G.build_graph(ingest_retry=RetryPolicy(max_attempts=5, initial_interval=0.001,
                                                   backoff_factor=1.0, jitter=False,
                                                   retry_on=nodes.TransientHFError))
    with pytest.raises(nodes.TransientHFError):
        graph.invoke(init_state(cfg))
    assert calls["n"] == 5


def test_rerun_same_thread_resumes_via_skip_checks(cfg, sandbox, monkeypatch, tmp_path):
    _all_artifacts(cfg, monkeypatch)
    monkeypatch.setattr(G, "PIPELINE_RUNS_DIR", tmp_path / "pipeline_runs")
    saver = G.open_checkpointer(cfg["run_id"])
    graph = G.build_graph(checkpointer=saver)
    kw = {"config": {"configurable": {"thread_id": cfg["run_id"]}}, "durability": "sync"}
    graph.invoke(init_state(cfg), **kw)
    final = graph.invoke(init_state(cfg), **kw)  # second run: restarts from START, all skip again
    assert all(final[s]["status"] == "skipped" for s in STAGES)
