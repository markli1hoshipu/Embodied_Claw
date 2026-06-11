"""pi05 adapter — DOMINO's vendored-openpi baseline (DOMINO/policy/pi05). Checkpoints MUST sit
at policy/pi05/checkpoints/<train_config_name>/<model_name>/<step>/ with params/ and
assets/<robotwin_repo_id>/ (norm stats) — that exact layout is what pi_model.PI0 loads.
B1K-embodiment checkpoints are NOT DOMINO-compatible (aloha-agilex dual-arm expected);
verify_layout flags this via the train-config registry check."""
from __future__ import annotations

import re
from pathlib import Path

from eval_domino.tools import DOMINO, DOMINO_ENV

FAMILY = "pi05"
POLICY_DIR = DOMINO / "policy" / "pi05"
CKPT_ROOT = POLICY_DIR / "checkpoints"
_CONFIG_REGISTRY = POLICY_DIR / "src" / "openpi" / "training" / "config.py"
# The vendored openpi needs py>=3.11 while the sim env is py3.10 — model runs in its own venv
# (uv sync in POLICY_DIR) and serves actions over DOMINO's policy_model_server socket protocol.
SERVER_PY = POLICY_DIR / ".venv" / "bin" / "python"


def checkpoint_dir(train_config_name: str, model_name: str, step: int) -> Path:
    return CKPT_ROOT / train_config_name / model_name / str(step)


def list_local() -> list[dict]:
    out = []
    for cfg in CKPT_ROOT.iterdir() if CKPT_ROOT.is_dir() else ():
        for model in cfg.iterdir() if cfg.is_dir() else ():
            steps = sorted(int(s.name) for s in model.iterdir()
                           if s.is_dir() and s.name.isdigit())
            if steps:
                out.append({"train_config_name": cfg.name, "model_name": model.name,
                            "steps": steps})
    return out


def config_in_registry(train_config_name: str) -> bool:
    """The vendored openpi resolves configs by name from its config.py — a checkpoint whose
    train config is missing there cannot be loaded, however valid its files are."""
    if not _CONFIG_REGISTRY.exists():
        return False
    return bool(re.search(rf'name="{re.escape(train_config_name)}"',
                          _CONFIG_REGISTRY.read_text()))


def verify_layout(train_config_name: str, model_name: str, step: int) -> dict:
    d = checkpoint_dir(train_config_name, model_name, step)
    assets = sorted(p.name for p in (d / "assets").iterdir()) if (d / "assets").is_dir() else []
    checks = {
        "checkpoint_dir_exists": d.is_dir(),
        "params_present": (d / "params").is_dir(),
        "assets_repo_ids": assets,                      # PI0 takes assets/<first entry> as repo_id
        "config_in_registry": config_in_registry(train_config_name),
    }
    checks["ok"] = bool(checks["checkpoint_dir_exists"] and checks["params_present"]
                        and assets and checks["config_in_registry"])
    return {"path": str(d), **checks}


def smoke_cmd(train_config_name: str, model_name: str, step: int, gpu_id: int) -> str:
    """Load the policy the way the model server does (in the py3.11 server venv — no sim deps)
    and push one dummy observation through get_action. Must run from DOMINO root (PI0 uses
    relative checkpoint paths)."""
    smoke = POLICY_DIR / "scripts" / "_eval_smoke.py"
    return (f"cd {DOMINO} && "
            f"XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 CUDA_VISIBLE_DEVICES={gpu_id} "
            f"{SERVER_PY} {smoke} {train_config_name} {model_name} {step}")
