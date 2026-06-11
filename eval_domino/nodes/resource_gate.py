"""resource_gate — deterministic (no agent): lease the confirmed GPU count. Fail-fast when
zero are free (user decision 2026-06-10); one re-confirm when fewer are free than confirmed
(never a silent downgrade)."""
from __future__ import annotations

from eval_domino import gpu, tools
from pipeline.nodes import _status
from pipeline.tools import now


def node(state: dict) -> dict:
    stage = "resource_gate"
    if (state.get(stage) or {}).get("status") == "succeeded":
        # Resume after finalize released the leases: stage says done but the lease files are
        # gone — fall through and re-acquire instead of running downstream unleased.
        import json
        held = {int(p.stem[3:]) for p in gpu.LEASE_ROOT.glob("gpu*.lease")
                if json.loads(p.read_text()).get("run_id") == state["run_id"]} \
            if gpu.LEASE_ROOT.is_dir() else set()
        wanted = set((state["config"].get("resources") or {}).get("gpu_ids") or [])
        if wanted and wanted <= held:
            return {}
    if state["intake"]["status"] != "succeeded":
        return {stage: _status("skipped", now(), error="upstream not succeeded: intake")}
    started = now()
    tools.log_transition(state["run_id"], stage, "running")
    cfg = dict(state["config"])
    bench = __import__(f"eval_domino.benchmarks.{cfg['benchmark']['name']}", fromlist=["x"])
    requested = int(cfg["resources"]["gpus_requested"])
    free = gpu.free_gpus(bench.MIN_FREE_MB)
    if not free:
        msg = ("no free GPUs (need >= "
               f"{bench.MIN_FREE_MB // 1024}GB free, none qualify) — re-trigger the eval when "
               "capacity frees up")
        tools.log_transition(state["run_id"], stage, "failed", msg)
        return {stage: _status("failed", started, error=msg)}
    if len(free) < requested:
        reply = tools.direct_ask(
            state["run_id"], stage,
            f"Only {len(free)} GPU(s) free now (you confirmed {requested}). Proceed with "
            f"{len(free)}, or cancel?",
            options=[{"id": 1, "label": f"Proceed with {len(free)}"},
                     {"id": 99, "label": "Cancel"}], recommendation=1)
        if not (reply.get("type") == "option" and reply.get("option") == 1):
            return {stage: _status("failed", started, error="cancelled at resource gate")}
        requested = len(free)
    ids = gpu.acquire(state["run_id"], requested, bench.MIN_FREE_MB)
    if not ids:  # raced away between snapshot and acquire
        msg = "GPUs were taken between confirmation and lease — re-trigger the eval"
        tools.log_transition(state["run_id"], stage, "failed", msg)
        return {stage: _status("failed", started, error=msg)}
    cfg["resources"]["gpu_ids"] = ids
    p = tools.run_dir(state["run_id"]) / "config.json"
    p.write_text(__import__("json").dumps(cfg, indent=2))
    tools.log_transition(state["run_id"], stage, "succeeded", f"leased GPUs {ids}")
    return {"config": cfg,
            stage: _status("succeeded", started, artifacts=[str(p)])}
