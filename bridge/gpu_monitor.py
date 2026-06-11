"""Fixed-cadence GPU usage reporter — no agent, no interactivity.

Every GPU_MONITOR_INTERVAL_S (default 600s) posts one message to GPU_MONITOR_CHANNEL_ID:
per-GPU memory/utilization plus which job runs there (the process's working-directory folder
name, e.g. DOMINO / openpi / Embodied_Claw). Reads SLACK_BOT_TOKEN from the repo .env.

    nohup .venv/bin/python -m bridge.gpu_monitor > logs/gpu_monitor.log 2>&1 &
"""
from __future__ import annotations

import datetime as dt
import os
import subprocess
import time
from pathlib import Path

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from bridge.config import load_env_file

REPO = Path(__file__).resolve().parent.parent
CHANNEL = os.environ.get("GPU_MONITOR_CHANNEL_ID", "C0B9VG5BULE")
INTERVAL = int(os.environ.get("GPU_MONITOR_INTERVAL_S", "600"))


def _sh(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout


def _job_folders(pids: list[str]) -> str:
    names = []
    for pid in pids:
        try:
            cwd = os.readlink(f"/proc/{pid}/cwd")
            name = Path(cwd).name or cwd
        except OSError:
            name = "?"
        if name not in names:
            names.append(name)
    return ", ".join(names) if names else "idle"


def snapshot_message() -> str:
    gpus = _sh(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits"]).strip().splitlines()
    apps = _sh(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                "--format=csv,noheader"]).strip().splitlines()
    uuid_of = dict(line.split(", ") for line in _sh(
        ["nvidia-smi", "--query-gpu=index,gpu_uuid", "--format=csv,noheader"]
    ).strip().splitlines())
    pids_by_gpu: dict[str, list[str]] = {}
    uuid_to_idx = {v: k for k, v in uuid_of.items()}
    for line in apps:
        if not line.strip():
            continue
        uuid, pid = [x.strip() for x in line.split(",")]
        pids_by_gpu.setdefault(uuid_to_idx.get(uuid, "?"), []).append(pid)
    lines = [f"*GPU status* — {dt.datetime.now():%m-%d %H:%M}"]
    for row in gpus:
        idx, util, used, total = [x.strip() for x in row.split(",")]
        jobs = _job_folders(pids_by_gpu.get(idx, []))
        bar = "🟩" if int(used) > 10_000 else "⬜"
        lines.append(f"{bar} GPU {idx}: {int(used) // 1024}/{int(total) // 1024} GB, "
                     f"{util}% util — {jobs}")
    return "\n".join(lines)


def main() -> None:
    load_env_file(REPO / ".env")
    client = WebClient(token=os.environ["SLACK_BOT_TOKEN"])
    while True:
        try:
            client.chat_postMessage(channel=CHANNEL, text=snapshot_message())
        except SlackApiError as e:
            print(f"{dt.datetime.now():%H:%M:%S} slack error: "
                  f"{e.response.get('error')}", flush=True)
        except Exception as e:  # noqa: BLE001 — monitor must survive transient failures
            print(f"{dt.datetime.now():%H:%M:%S} error: {e}", flush=True)
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
