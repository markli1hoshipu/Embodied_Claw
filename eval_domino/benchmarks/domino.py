"""DOMINO adapter (RoboTwin 2.0-based dynamic manipulation, /work/markhsp/DOMINO).
Result contract: eval_policy.py writes eval_result/<task>/<policy>/<task_config>/<ckpt>/<ts>/
{_metrics.json,_episodes_detail.json,_result.txt}. Known env gotchas live in domino_env.sh +
memory project_domino.md (curobo v0.7.8 + nvcc 12.6, warp 1.14 patch, Vulkan exports)."""
from __future__ import annotations

import json
from pathlib import Path

from eval_domino.tools import DOMINO, DOMINO_ENV

NAME = "domino"
MIN_FREE_MB = 100_000  # a GPU already hosting a ~57GB model server must NEVER be re-leased
SHARD_TIMEOUT_S = 3 * 3600
_NOT_TASKS = {"robot", "camera", "curobo"}

# Program-error signatures (shard infra broke) vs eval-outcome failures (policy just did badly).
ERROR_PATTERNS = ("Traceback", "CUDA error", "illegal instruction", "out of memory",
                  "ModuleNotFoundError", "ImportError", "cudaError", "Segmentation fault",
                  "vk::", "RuntimeError", "SERVER_FAILED_TO_START", "Failed to connect to server")


def list_tasks() -> list[str]:
    return sorted(p.stem for p in (DOMINO / "envs").glob("*.py")
                  if not p.stem.startswith("_") and p.stem not in _NOT_TASKS)


def task_configs() -> list[str]:
    return sorted(p.stem for p in (DOMINO / "task_config").glob("*.yml")
                  if not p.stem.startswith("_"))


def runtime_check_cmd(gpu_id: int) -> str:
    return (f"{DOMINO_ENV} && cd {DOMINO} && CUDA_VISIBLE_DEVICES={gpu_id} "
            f"python script/test_render.py")


def shard_cmd(task: str, task_config: str, train_config_name: str, model_name: str,
              checkpoint_id: int, seed: int, gpu_id: int, policy: str = "pi05",
              server_log: str = "/tmp/pi05_server.log", episodes: int = 100) -> str:
    """One shard = model server (py3.11 venv, jax) + sim client (py3.10 domino env) on one GPU,
    talking over localhost:921<gpu>. The server dies with the shard (same process group + trap),
    so cancel/cleanup needs no extra bookkeeping. Model load can take minutes — the client only
    starts once the server log says it is listening.
    Port base is 9210, NOT 9100: the system node_exporter listens on *:9100 (foreign process,
    never kill it), so 9100+gpu_id deterministically EADDRINUSEs every GPU-0 shard."""
    port = 9210 + int(gpu_id)
    server_py = DOMINO / "policy" / policy / ".venv" / "bin" / "python"
    overrides = (f"--task_name {task} --task_config {task_config} "
                 f"--train_config_name {train_config_name} --model_name {model_name} "
                 f"--checkpoint_id {checkpoint_id} --ckpt_setting {model_name} "
                 f"--seed {seed} --policy_name {policy} --test_num {int(episodes)}")
    return (
        # cd must NOT be chained into the backgrounded list with '&&': 'cd X && srv &'
        # backgrounds the whole list, so the main shell (which runs the client) never
        # changes directory (client died with ENOENT on script/eval_policy_client.py),
        # and $SRV becomes the wrapping subshell instead of the python pid (the EXIT
        # trap then kills the subshell and leaks the server on the GPU). Keep cd as its
        # own statement and the server launch a simple command so $! is the python pid.
        f"cd {DOMINO} || exit 16; "
        # Server imports envs -> open3d/sapien, which need conda-env libGL/vulkan libs.
        # Do not depend on the parent shell having them (domino_env.sh exports the same path).
        # PYTHONUNBUFFERED=1: 'Model server started' must reach the log promptly or the
        # readiness grep below times out at 15 min against block-buffered stdout and
        # kills a healthy server.
        f"LD_LIBRARY_PATH=/work/markhsp/miniforge3/envs/domino/lib:$LD_LIBRARY_PATH "
        f"CUDA_VISIBLE_DEVICES={gpu_id} XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 PYTHONUNBUFFERED=1 "
        f"{server_py} script/policy_model_server.py --port {port} "
        f"--config policy/{policy}/deploy_policy.yml --overrides {overrides} "
        f"> {server_log} 2>&1 & SRV=$!; "
        f"trap 'kill $SRV 2>/dev/null' EXIT; up=0; "
        f"for i in $(seq 1 180); do "
        f"grep -q 'Model server started' {server_log} && up=1 && break; "
        f"kill -0 $SRV 2>/dev/null || break; sleep 5; done; "
        f"if [ $up -ne 1 ]; then echo SERVER_FAILED_TO_START; tail -60 {server_log}; exit 17; fi; "
        f"{DOMINO_ENV} && CUDA_VISIBLE_DEVICES={gpu_id} PYTHONWARNINGS=ignore::UserWarning "
        f"python script/eval_policy_client.py --port {port} "
        f"--config policy/{policy}/deploy_policy.yml --overrides {overrides}")


def parse_shard_result(task: str, task_config: str, model_name: str,
                       policy: str = "pi05") -> dict | None:
    """Newest _metrics.json for this shard, or None if the run never wrote one (program error)."""
    d = DOMINO / "eval_result" / task / policy / task_config / model_name
    if not d.is_dir():
        return None
    metrics = sorted(d.glob("*/_metrics.json"), key=lambda p: p.stat().st_mtime)
    if not metrics:
        return None
    out = json.loads(metrics[-1].read_text())
    out["_result_dir"] = str(metrics[-1].parent)
    return out


def classify_log(tail: str) -> str:
    """'program_error' if an infra signature appears; otherwise 'ok'."""
    return "program_error" if any(p in tail for p in ERROR_PATTERNS) else "ok"
