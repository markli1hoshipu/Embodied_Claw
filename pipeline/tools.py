"""Shared plumbing: paths, bash/hf wrappers, transitions log, notifications, and the
file-mailbox half of the ask_user escalation protocol (spec section 7 — byte-level contract)."""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import shlex
import subprocess
import time
import uuid
from pathlib import Path

REPO = Path("/work/markhsp/Embodied_Claw")
OPENPI = Path("/work/markhsp/openpi")
LEROBOT_ROOT = Path("/work/markhsp/datasets/lerobot")
DATASETS_ROOT = Path("/work/markhsp/datasets")
PCA_DIR = Path("/work/markhsp/behavior1k-xvla/behavior1k_training/pca_filter")
STAGING_ROOT = Path("/work/markhsp/hf_staging")
FFPROBE = "/work/markhsp/miniforge3/envs/ffmpeg7/bin/ffprobe"
FFMPEG_LIB = "/work/markhsp/miniforge3/envs/ffmpeg7/lib"
CONDA = "source /work/markhsp/miniforge3/etc/profile.d/conda.sh && conda activate xvla-stable"
UPLOAD_CACHE = Path.home() / ".cache/huggingface/upload"

HF_ENV = {"HF_HUB_DISABLE_XET": "1", "HF_HUB_ENABLE_HF_TRANSFER": "0"}
TRAIN_ENV = {**HF_ENV, "HF_LEROBOT_HOME": "/home/mark-li/.cache/huggingface/lerobot",
             "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.95", "JAX_PLATFORMS": "cuda",
             "NCCL_NVLS_ENABLE": "0", "NCCL_P2P_DISABLE": "0", "NCCL_IB_DISABLE": "0"}


def runs_root() -> Path:
    return Path(os.environ.get("EMBODIED_CLAW_RUNS", str(REPO / "runs")))


def agents_root() -> Path:
    return Path(os.environ.get("EMBODIED_CLAW_AGENTS", str(REPO / "agents")))


def config_dir() -> Path:
    return Path(os.environ.get("EMBODIED_CLAW_CONFIG", str(REPO / "config")))


def run_dir(run_id: str) -> Path:
    return runs_root() / run_id


def now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def sh(cmd: str, env: dict | None = None, timeout: int | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "-c", cmd], capture_output=True, text=True,
                          env={**os.environ, **(env or {})}, timeout=timeout)


def hf_api(token: str | None = None):
    from huggingface_hub import HfApi
    return HfApi(token=token or os.environ.get("HF_TOKEN"))


def require_hf_token(stage: str) -> str:
    tok = os.environ.get("HF_TOKEN")
    if not tok:
        raise RuntimeError(f"HF_TOKEN is not set in the environment but '{stage}' needs it. "
                           "Run `export HF_TOKEN=...` and retry; the token is never hardcoded.")
    return tok


def find_train_pids(config_name: str) -> list[int]:
    """pgrep the live training python. Matches `python.*scripts/train\\.py.*<cfg>( |$)` — NOT a bare
    substring (bash watchers/tee carry 'scripts/train.py'; the trailing boundary stops prefix
    collisions like recovery vs recovery2/3). rc>=2 (pgrep itself broke) raises — it must never
    read as 'no process running', a false [] could double-launch an --overwrite train script."""
    if not config_name:
        raise ValueError("config_name required — an empty pattern matches every train process")
    pat = rf"python.*scripts/train\.py.*{re.escape(config_name)}( |$)"
    r = subprocess.run(["pgrep", "-f", pat], capture_output=True, text=True)
    if r.returncode >= 2:
        raise RuntimeError(f"pgrep failed (rc={r.returncode}): {(r.stderr or r.stdout).strip()}")
    return [int(x) for x in r.stdout.split()]


def live_pids(config_name: str) -> list[int]:
    """find_train_pids, fail-safe direction: pgrep error reads as 'assume alive' (sentinel [-1])
    so nothing destructive (launch/rebuild) can fire on an unhandled error path."""
    try:
        return find_train_pids(config_name)
    except RuntimeError as e:
        print(f"[train] WARNING: {e}; assuming the run is alive", flush=True)
        return [-1]


def log_transition(run_id: str, node: str, status: str, detail: str | None = None) -> None:
    rd = run_dir(run_id)
    rd.mkdir(parents=True, exist_ok=True)
    line = {"ts": now(), "run_id": run_id, "node": node, "status": status}
    if detail:
        line["detail"] = str(detail)[:500]
    with open(rd / "transitions.jsonl", "a") as f:
        f.write(json.dumps(line) + "\n")


def notify(summary: str) -> None:
    """Fire the configured notification (config/notifications.toml). Defaults per spec 7.1:
    notify-send + an append to ~/.cache/embodied_claw/inbox.log. Never raises."""
    cmd, log = "notify-send 'Embodied Claw' {summary}", None
    try:
        toml_p = config_dir() / "notifications.toml"
        if toml_p.exists():
            import tomllib
            conf = tomllib.loads(toml_p.read_text()).get("notify", {})
            cmd, log = conf.get("command", cmd), conf.get("inbox_log")
        log = Path(os.path.expanduser(log or os.environ.get(
            "EMBODIED_CLAW_CACHE", "~/.cache/embodied_claw") + "/inbox.log"))
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "a") as f:
            f.write(f"{now()} {summary}\n")
        subprocess.run(["bash", "-c", cmd.format(summary=shlex.quote(summary))],
                       capture_output=True, timeout=10)
    except Exception:
        pass  # notification is best-effort; the mailbox + CLI inbox stay authoritative


# ------------------------------------------------------------------ escalation mailbox (spec 7)

def esc_dir(rd: Path) -> Path:
    return rd / "escalations"


def new_escalation_id(node: str) -> str:
    return f"{node}_{int(time.time())}_{uuid.uuid4().hex[:4]}"


def write_question(rd: Path, esc_id: str, node: str, agent: str, question: str,
                   context: str = "", options: list | None = None,
                   recommendation: int | None = None) -> dict:
    """Idempotent: if <esc_id>.question.json already exists, return its parsed content unchanged
    (re-entry must re-attach, never double-post). Atomic write (tmp+rename) for the FS watcher."""
    d = esc_dir(rd)
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{esc_id}.question.json"
    if p.exists():
        return json.loads(p.read_text())
    payload: dict = {"node": node, "agent": agent, "question": question, "context": context}
    if options:
        payload["options"] = options
    if recommendation is not None:
        payload["recommendation"] = recommendation
    payload["created_at"] = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.rename(p)
    return payload


def find_unanswered(rd: Path, node: str | None = None) -> str | None:
    """Newest unanswered escalation_id (question.json with no sibling reply), optionally for one node."""
    best: tuple[float, str] | None = None
    for q in esc_dir(rd).glob("*.question.json") if esc_dir(rd).is_dir() else ():
        eid = q.name[:-len(".question.json")]
        if list(esc_dir(rd).glob(f"{eid}.reply.*")):
            continue
        if node is not None and json.loads(q.read_text()).get("node") != node:
            continue
        if best is None or q.stat().st_mtime > best[0]:
            best = (q.stat().st_mtime, eid)
    return best[1] if best else None


def parse_reply(text: str) -> dict:
    """Spec 7.2(B): a single integer is an option id; anything else is a free-form message."""
    t = text.strip()
    if re.fullmatch(r"-?\d+", t):
        return {"type": "option", "option": int(t)}
    return {"type": "message", "message": t}


def read_reply(rd: Path, esc_id: str) -> dict | None:
    """Reply files match by escalation_id prefix, ANY extension: <esc_id>.reply.*
    An empty/blank file reads as 'no reply yet' (torn non-atomic write tolerance)."""
    for p in sorted(esc_dir(rd).glob(f"{esc_id}.reply.*")) if esc_dir(rd).is_dir() else ():
        text = p.read_text()
        if not text.strip():
            return None
        return parse_reply(text)
    return None
