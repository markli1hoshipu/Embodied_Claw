"""benchmark_agent skills: preflight -> task plan -> shard orchestration -> aggregate.

Shard truth lives in runs/<id>/task_plan.json (single writer: run_pending_shards), shard logs in
runs/<id>/shards/<shard_id>.log, digests in progress.json + transitions.jsonl (the bridge tails
transitions and posts to Slack — one digest per orchestrator window ~= the 30-min heartbeat).

Failure semantics (user decision 2026-06-10): low scores are DATA (record, continue); program
errors PAUSE the queue and return control to the agent for fix mode (<=2 fix attempts/shard),
retry_shard() puts the probe back."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path

from eval_domino import tools
from eval_domino.benchmarks import get as get_benchmark

MAX_ATTEMPTS = 3  # 1 original + 2 post-fix retries


def _plan_path(run_id: str) -> Path:
    return tools.run_dir(run_id) / "task_plan.json"


def _load_plan(run_id: str) -> dict:
    return json.loads(_plan_path(run_id).read_text())


def _save_plan(run_id: str, plan: dict) -> None:
    p = _plan_path(run_id)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(plan, indent=2))
    tmp.rename(p)


def _shard_log(run_id: str, shard_id: str) -> Path:
    d = tools.run_dir(run_id) / "shards"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{shard_id}.log"


def _pid_alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def _proc_done(pid) -> bool:
    """Shard wrappers are OUR children: kill-0 reports exited-but-unreaped zombies as alive
    forever, so reap with waitpid(WNOHANG). After a driver restart they are not our children —
    fall back to kill-0 (the old driver's zombies died with it)."""
    try:
        done_pid, _ = os.waitpid(int(pid), os.WNOHANG)
        return done_pid != 0
    except ChildProcessError:
        return not _pid_alive(pid)
    except (OSError, TypeError, ValueError):
        return True


def env_preflight(run_id: str, gpu_id: int) -> dict:
    """Benchmark runtime check on the leased GPU (DOMINO: SAPIEN RT render test) + assets."""
    cfg = tools.eval_config(run_id)
    adapter = get_benchmark(cfg["benchmark"]["name"])
    r = tools.sh(adapter.runtime_check_cmd(gpu_id), timeout=600)
    ok = r.returncode == 0 and "Render Well" in r.stdout
    return {"ok": ok, "rc": r.returncode,
            "log_tail": (r.stdout + r.stderr)[-1500:] if not ok else "Render Well"}


def build_task_plan(run_id: str) -> dict:
    """One shard per (task, seed) from config.json. Idempotent: an existing plan is returned
    untouched (resume must not reset shard states)."""
    if _plan_path(run_id).exists():
        plan = _load_plan(run_id)
        return {"ok": True, "existing": True, "n_shards": len(plan["shards"])}
    cfg = tools.eval_config(run_id)
    b = cfg["benchmark"]
    shards = [{"id": f"{t}-s{b.get('seed', 0)}", "task": t, "seed": b.get("seed", 0),
               "status": "pending", "attempts": 0, "pid": None, "gpu": None,
               "launched_at": None, "metrics": None} for t in b["tasks"]]
    plan = {"run_id": run_id, "benchmark": b["name"], "task_config": b["task_config"],
            "created_at": tools.now(), "paused": False, "shards": shards}
    _save_plan(run_id, plan)
    eta_min = len(shards) * 25 // max(1, len(cfg["resources"]["gpu_ids"]))
    return {"ok": True, "existing": False, "n_shards": len(shards), "eta_min_rough": eta_min}


def _launch(run_id: str, shard: dict, gpu_id: int, cfg: dict, adapter) -> None:
    m = cfg["model"]
    cmd = adapter.shard_cmd(shard["task"], cfg["benchmark"]["task_config"],
                           m["train_config_name"], m["model_name"],
                           m["checkpoint_step"] if m.get("checkpoint_step") is not None
                           else m.get("checkpoint_id"),
                           shard["seed"], gpu_id, policy=m["family"],
                           server_log=str(_shard_log(run_id, shard["id"]).with_suffix(".server.log")),
                           episodes=int(cfg["benchmark"].get("episodes_per_task") or 100))
    log = open(_shard_log(run_id, shard["id"]), "wb")  # truncate per attempt: triage must
    # classify THIS attempt's output, not stale tracebacks from earlier ones
    proc = subprocess.Popen(["bash", "-c", cmd], stdout=log, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, start_new_session=True)
    shard.update(status="running", pid=proc.pid, gpu=gpu_id, launched_at=time.time(),
                 attempts=shard["attempts"] + 1)


def _collect(run_id: str, shard: dict, cfg: dict, adapter) -> None:
    """Shard process exited: result file newer than launch == done; otherwise classify."""
    res = adapter.parse_shard_result(shard["task"], cfg["benchmark"]["task_config"],
                                     cfg["model"]["model_name"], policy=cfg["model"]["family"])
    fresh = res and Path(res["_result_dir"]).stat().st_mtime >= (shard["launched_at"] or 0)
    if fresh:
        shard.update(status="done", pid=None,
                     metrics={k: v for k, v in res.items() if not k.startswith("_")},
                     result_dir=res["_result_dir"])
        return
    tail = _shard_log(run_id, shard["id"]).read_text(errors="replace")[-3000:]
    kind = adapter.classify_log(tail)
    if shard["attempts"] >= MAX_ATTEMPTS:
        shard.update(status="failed", pid=None, error=f"{kind}; attempts exhausted")
    elif kind == "program_error":
        shard.update(status="error_paused", pid=None)
    else:  # exited without result and without a known signature — treat as program error too
        shard.update(status="error_paused", pid=None)


def run_pending_shards(run_id: str, max_minutes: int = 25) -> dict:
    """Bounded orchestrator window: keep every leased GPU busy with one shard, collect exits,
    write a heartbeat digest, return control. Statuses returned: 'running' (re-invoke),
    'program_error' (enter fix mode: read shard log, fix, retry_shard, re-invoke),
    'cancelled' (CANCEL file), 'done' (all shards terminal)."""
    cfg = tools.eval_config(run_id)
    adapter = get_benchmark(cfg["benchmark"]["name"])
    gpus = cfg["resources"]["gpu_ids"]
    deadline = time.time() + max_minutes * 60
    plan = _load_plan(run_id)

    def shards(*statuses):
        return [s for s in plan["shards"] if s["status"] in statuses]

    # re-attach after driver restart: running shards whose pid died get collected below
    while True:
        for s in shards("running"):
            if _proc_done(s["pid"]):
                _collect(run_id, s, cfg, adapter)
        if tools.cancel_requested(run_id):
            for s in shards("running"):
                try:
                    os.killpg(int(s["pid"]), signal.SIGTERM)
                except (OSError, TypeError, ValueError):
                    pass
                s.update(status="pending", pid=None, attempts=max(0, s["attempts"] - 1))
            plan["paused"] = True
            _save_plan(run_id, plan)
            _digest(run_id, plan, note="cancelled by user")
            return {"status": "cancelled", **_counts(plan)}
        if shards("error_paused"):
            plan["paused"] = True
            _save_plan(run_id, plan)
            bad = shards("error_paused")[0]
            _digest(run_id, plan, note=f"program error in {bad['id']} — fix mode")
            return {"status": "program_error", "shard_id": bad["id"],
                    "attempts": bad["attempts"],
                    "log_tail": _shard_log(run_id, bad["id"]).read_text(errors="replace")[-3000:],
                    **_counts(plan)}
        if not plan["paused"]:
            for g in gpus:
                if not any(s["gpu"] == g and s["status"] == "running" for s in plan["shards"]):
                    nxt = next(iter(shards("pending")), None)
                    if nxt:
                        _launch(run_id, nxt, g, cfg, adapter)
        _save_plan(run_id, plan)
        if not shards("pending", "running"):
            _digest(run_id, plan, note="all shards terminal")
            return {"status": "done", **_counts(plan)}
        if time.time() >= deadline:
            _digest(run_id, plan)
            return {"status": "running", **_counts(plan)}
        time.sleep(20)


def _counts(plan: dict) -> dict:
    c: dict = {}
    for s in plan["shards"]:
        c[s["status"]] = c.get(s["status"], 0) + 1
    done = [s for s in plan["shards"] if s["status"] == "done"]
    out = {"counts": c, "n_total": len(plan["shards"])}
    rates = [s["metrics"].get("success_rate") for s in done
             if isinstance(s.get("metrics"), dict) and
             isinstance(s["metrics"].get("success_rate"), (int, float))]
    if rates:
        out["mean_success_rate_so_far"] = round(sum(rates) / len(rates), 4)
    return out


def _digest(run_id: str, plan: dict, note: str = "") -> None:
    info = _counts(plan)
    msg = (f"eval progress: {info['counts'].get('done', 0)}/{info['n_total']} shards done, "
           f"{info['counts'].get('failed', 0)} failed"
           + (f", mean SR {info['mean_success_rate_so_far']}"
              if "mean_success_rate_so_far" in info else "")
           + (f" — {note}" if note else ""))
    (tools.run_dir(run_id) / "progress.json").write_text(
        json.dumps({"ts": tools.now(), **info}, indent=2))
    tools.log_transition(run_id, "run_matrix", "running", msg)


def shard_log_tail(run_id: str, shard_id: str, max_bytes: int = 6000) -> str:
    """Tail of one shard's log (fix-mode diagnosis)."""
    p = _shard_log(run_id, shard_id)
    return p.read_text(errors="replace")[-max_bytes:] if p.exists() else "(no log)"


def retry_shard(run_id: str, shard_id: str) -> dict:
    """After a fix: put an error_paused/failed shard back to pending and unpause the queue.
    The next run_pending_shards launches it first (probe) alongside normal dispatch."""
    plan = _load_plan(run_id)
    s = next((x for x in plan["shards"] if x["id"] == shard_id), None)
    if s is None:
        return {"ok": False, "error": f"no shard '{shard_id}'"}
    s.update(status="pending", pid=None)
    plan["paused"] = False
    # probe first: move it to the front
    plan["shards"].sort(key=lambda x: x["id"] != shard_id)
    _save_plan(run_id, plan)
    return {"ok": True, "attempts_so_far": s["attempts"]}


def skip_shard(run_id: str, shard_id: str, reason: str) -> dict:
    """Give up on one shard (marks failed) and unpause — the rest of the matrix continues."""
    plan = _load_plan(run_id)
    s = next((x for x in plan["shards"] if x["id"] == shard_id), None)
    if s is None:
        return {"ok": False, "error": f"no shard '{shard_id}'"}
    s.update(status="failed", pid=None, error=reason)
    plan["paused"] = False
    _save_plan(run_id, plan)
    return {"ok": True}


def aggregate_results(run_id: str) -> dict:
    """Per-task metrics table + overall means -> results.json / results.md."""
    plan = _load_plan(run_id)
    cfg = tools.eval_config(run_id)
    rows: list[dict] = []
    num: dict[str, list] = {}
    for s in plan["shards"]:
        row = {"task": s["task"], "seed": s["seed"], "status": s["status"]}
        if isinstance(s.get("metrics"), dict):
            for k, v in s["metrics"].items():
                if isinstance(v, (int, float)):
                    row[k] = v
                    num.setdefault(k, []).append(v)
        rows.append(row)
    overall = {k: round(sum(v) / len(v), 4) for k, v in num.items() if v}
    done = sum(1 for s in plan["shards"] if s["status"] == "done")
    out = {"run_id": run_id, "model": cfg.get("model"), "benchmark": cfg.get("benchmark"),
           "n_done": done, "n_total": len(plan["shards"]), "overall": overall, "per_task": rows}
    rd = tools.run_dir(run_id)
    (rd / "results.json").write_text(json.dumps(out, indent=2))
    keys = sorted(num)
    md = [f"# Eval results — {run_id}", "",
          f"model `{cfg['model'].get('model_name')}` step {cfg['model'].get('checkpoint_step')}"
          f" on **{cfg['benchmark'].get('name')}** / `{cfg['benchmark'].get('task_config')}`",
          f"shards: {done}/{len(plan['shards'])} done", "",
          "| task | status | " + " | ".join(keys) + " |",
          "|---|---|" + "---|" * len(keys)]
    for r in rows:
        md.append(f"| {r['task']} | {r['status']} | "
                  + " | ".join(str(r.get(k, "—")) for k in keys) + " |")
    md += ["", "**overall**: " + ", ".join(f"{k}={v}" for k, v in overall.items())]
    (rd / "results.md").write_text("\n".join(md) + "\n")
    return {"ok": True, "overall": overall, "n_done": done, "n_total": len(plan["shards"]),
            "paths": [str(rd / "results.json"), str(rd / "results.md")]}
