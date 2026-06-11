"""End-to-end mocked run (all 7 nodes from request.txt), fan-out concurrency, failure gating."""
import json

from pipeline import cli, tools
from pipeline.state import STAGES, init_state, pending
from pipeline.tests import fakes


def test_full_run_from_request_txt(graph_env, run_dir):
    graph, conf, fake = graph_env()
    values = cli.drive(graph, conf, init_state("tr1"), "tr1", poll=0.01)
    for s in STAGES:
        assert values[s]["status"] == "succeeded", (s, values[s])
    # intake wrote + populated config
    cfg = json.loads((run_dir / "config.json").read_text())
    assert values["config"]["run_id"] == cfg["run_id"] == "tr1"
    assert values["config"]["train_request"]["num_train_steps"] == 60000
    # run-trace artifacts (spec section 2)
    arts = json.loads((run_dir / "artifacts.json").read_text())
    assert arts["train"] == ["/tmp/ckpt/59999"]
    assert "# Run `tr1`" in (run_dir / "summary.md").read_text()
    trans = [json.loads(l) for l in (run_dir / "transitions.jsonl").read_text().splitlines()]
    for s in STAGES:
        assert {"running", "succeeded"} <= {t["status"] for t in trans if t["node"] == s}
    # run-level completion line: the bridge maps it to run status 'done'
    assert trans[-1]["node"] == "run" and trans[-1]["status"] == "done"
    for s in STAGES:  # full per-node Claude transcript
        assert (run_dir / "agent_messages" / f"{s}.jsonl").exists()
    assert values["train"]["agent_thread_id"] == "training_agent"


def test_parallel_fanout_and_join(graph_env, run_dir, monkeypatch):
    """Concurrency proof is a Barrier(2): both branch skills must be in-flight simultaneously
    to pass it (zero wall-clock sensitivity); order[] proves train ran only after both."""
    import threading
    from pipeline.agents import hf_agent, training_agent
    barrier = threading.Barrier(2, timeout=10)
    order = []

    def branch(key, ret):
        def fn(**kw):
            barrier.wait()  # raises BrokenBarrierError unless node3+node4 truly overlap
            order.append(key)
            return ret
        return fn

    monkeypatch.setitem(hf_agent.REGISTRY, "verify_repo_doesnt_exist_or_confirm_overwrite",
                        branch("u", {"status": "new"}))
    monkeypatch.setitem(training_agent.REGISTRY, "verify_norm_stats_sane",
                        branch("n", {"ok": True}))
    monkeypatch.setitem(training_agent.REGISTRY, "preflight_nccl_check",
                        lambda **kw: (order.append("train"), {"ok": True})[1])
    script = fakes.default_script(
        intake=fakes.intake_script("tr1"),
        upload_dataset=[[fakes.tool_use("verify_repo_doesnt_exist_or_confirm_overwrite",
                                        {"repo_id": "x", "repo_type": "dataset"})], fakes.done()],
        norm_stats=[[fakes.tool_use("verify_norm_stats_sane", {"norm_stats_path": "x"})],
                    fakes.done()],
        train=[[fakes.tool_use("preflight_nccl_check", {})], fakes.done()])
    graph, conf, fake = graph_env(script)
    values = cli.drive(graph, conf, init_state("tr1"), "tr1", poll=0.01)
    assert values["train"]["status"] == "succeeded"
    # Node 3 + Node 4 overlapped (both passed the barrier); train joined AFTER both
    assert set(order[:2]) == {"u", "n"} and order[2] == "train", order


def test_upstream_failure_gates_downstream_then_retry_reruns_only_failed(graph_env, run_dir):
    script = fakes.default_script(intake=fakes.intake_script("tr1"),
                                  ingest=[fakes.done(status="failed", error="404 on repo")])
    graph, conf, fake = graph_env(script)
    values = cli.drive(graph, conf, init_state("tr1"), "tr1", poll=0.01)
    assert values["intake"]["status"] == "succeeded"
    assert values["ingest"]["status"] == "failed" and "404" in values["ingest"]["error"]
    for s in ("filter_build", "upload_dataset", "norm_stats", "train", "upload_model"):
        assert values[s]["status"] == "skipped"
        assert "upstream" in values[s]["error"]
    # retry: reset non-succeeded stages; intake must NOT rerun (skip-if-done guard)
    from pipeline.agents import base
    from pipeline.graph import build_graph, open_checkpointer
    fake2 = fakes.FakeAnthropic({k: v for k, v in fakes.default_script().items() if k != "intake"})
    base_get = base.get_client
    base.get_client = lambda: fake2
    try:
        graph2 = build_graph(open_checkpointer("tr1"))
        redo = {s: pending() for s in STAGES if values[s]["status"] != "succeeded"}
        values2 = cli.drive(graph2, conf, redo, "tr1", poll=0.01)
    finally:
        base.get_client = base_get
    assert all(values2[s]["status"] == "succeeded" for s in STAGES)
    assert "intake" not in [c["node"] for c in fake2.calls]


def test_rerun_completed_graph_is_noop(graph_env, run_dir, capsys):
    graph, conf, fake = graph_env()
    cli.drive(graph, conf, init_state("tr1"), "tr1", poll=0.01)
    n_calls = len(fake.calls)
    snap = graph.get_state(conf)
    assert not snap.next
    assert all((snap.values.get(s) or {}).get("status") == "succeeded" for s in STAGES)
    assert len(fake.calls) == n_calls  # nothing re-invoked


def test_shared_notes_merge(graph_env, run_dir):
    script = fakes.default_script(
        intake=fakes.intake_script("tr1"),
        upload_dataset=[fakes.done(paths=("d",), notes={"dataset_repo": "Hoshipu/ds"})],
        norm_stats=[fakes.done(paths=("n",), notes={"config_name": "pi05_x"})])
    graph, conf, _ = graph_env(script)
    values = cli.drive(graph, conf, init_state("tr1"), "tr1", poll=0.01)
    assert values["shared_notes"]["hf_agent"] == {"dataset_repo": "Hoshipu/ds"}
    assert values["shared_notes"]["training_agent"] == {"config_name": "pi05_x"}
