"""DoD: kill the process mid-train; rerun; only the dead node re-executes (real process kill,
real sqlite persistence — the Anthropic client and the launch skill are faked in the child)."""
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CHILD = str(REPO / "pipeline" / "tests" / "_child_driver.py")


def _spawn(mode: str, run_id: str, trace: Path, env_tmp: Path):
    env = {**os.environ,
           "EMBODIED_CLAW_RUNS": str(env_tmp / "runs"),
           "EMBODIED_CLAW_AGENTS": str(env_tmp / "agents"),
           "EMBODIED_CLAW_CONFIG": str(env_tmp / "config"),
           "EMBODIED_CLAW_CACHE": str(env_tmp / "cache")}
    env.pop("ANTHROPIC_API_KEY", None)
    return subprocess.run([sys.executable, CHILD, mode, run_id, str(trace)],
                          capture_output=True, text=True, env=env, cwd=str(REPO), timeout=120)


def test_kill_mid_train_then_resume_runs_only_dead_node(env, run_dir):
    t1, t2 = env / "trace1.txt", env / "trace2.txt"
    r1 = _spawn("crash", "tr1", t1, env)
    assert r1.returncode == 17, r1.stderr[-2000:]  # died mid-train-node, checkpoint persisted
    seen1 = t1.read_text().split()
    assert seen1[:3] == ["intake", "intake", "ingest"] and "train" in seen1
    assert "upload_model" not in seen1
    assert (run_dir / "state.sqlite").exists()

    r2 = _spawn("resume", "tr1", t2, env)
    assert r2.returncode == 0, r2.stderr[-2000:]
    statuses = json.loads(r2.stdout.split("FINISHED ", 1)[1])
    assert statuses["train"] == "succeeded" and statuses["upload_model"] == "succeeded"
    assert all(v == "succeeded" for v in statuses.values()), statuses
    # only the dead node (and its downstream) hit the model again — nodes 0-4 never re-ran
    seen2 = set(t2.read_text().split())
    assert seen2 <= {"train", "upload_model"}, seen2
