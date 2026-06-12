"""Artifact reconciliation — enter the FSM at any point.

v2 nodes skip on graph state (state.<stage>.status == "succeeded"), which a fresh run_id
knows nothing about. This module grounds stage status in ON-DISK EVIDENCE (v1's needs_*
philosophy) and turns it into pre-seeded graph state, so the graph naturally starts at the
first stage the filesystem cannot vouch for:

    seeds, report, conflicts = reconcile.seed(cfg)             # auto (cli run default)
    seeds, report, conflicts = reconcile.seed(cfg, from_node="upload_model", force=...)

Upload stages are never auto-seeded (proving them needs the HF API; re-upload is cheap and
dedups by SHA). Everything here is offline: filesystem + json only.
"""
from __future__ import annotations

import json
from pathlib import Path

from pipeline import tools
from pipeline.state import STAGES

AUTO_SEEDABLE = ("ingest", "filter_build", "norm_stats", "train")


def derive_names(cfg: dict) -> dict:
    """dataset/train-config names. Explicit config keys win; else derived from
    outputs.hf_dataset_repo basename + the pi05_<b1k_...> convention."""
    ds = cfg.get("dataset_name")
    if not ds:
        repo = (cfg.get("outputs") or {}).get("hf_dataset_repo") or ""
        ds = repo.rsplit("/", 1)[-1] or None
    tc = cfg.get("train_config_name")
    if not tc and ds:
        tc = f"pi05_{ds}" if ds.startswith("b1k_") else f"pi05_b1k_{ds}"
    return {"dataset_name": ds, "train_config_name": tc,
            "dataset_dir": (tools.LEROBOT_ROOT / ds) if ds else None}


def train_done(cfg: dict) -> tuple[bool, str]:
    """Final checkpoint exists and is complete (params/ + _CHECKPOINT_METADATA at step
    num_train_steps-1, any exp dir). Shared by the train-node guard and the reconciler."""
    names = derive_names(cfg)
    tc = names["train_config_name"]
    steps = (cfg.get("train_request") or {}).get("num_train_steps")
    if not tc or not steps:
        return False, "train_config_name/num_train_steps underivable from config"
    root = tools.OPENPI / "checkpoints" / tc
    for meta in sorted(root.glob(f"*/{steps - 1}/_CHECKPOINT_METADATA")):
        if (meta.parent / "params").is_dir():
            return True, str(meta.parent)
    return False, f"no complete {steps - 1} checkpoint under {root}"


def evidence(cfg: dict) -> dict:
    """stage -> {'done': bool, 'why': str}; filesystem evidence only."""
    names = derive_names(cfg)
    ev: dict = {}

    srcs = (cfg.get("data_request") or {}).get("sources") or []
    dirs = [Path(s["local_dir"]) for s in srcs if isinstance(s, dict) and s.get("local_dir")]
    if srcs and len(dirs) == len(srcs):
        present = [d for d in dirs if d.is_dir() and any(d.iterdir())]
        ev["ingest"] = {"done": len(present) == len(dirs),
                        "why": f"{len(present)}/{len(dirs)} source dirs present and non-empty"}
    else:
        ev["ingest"] = {"done": False,
                        "why": "sources missing local_dir — nothing checkable on disk"}

    dd = names["dataset_dir"]
    info = (dd / "meta" / "info.json") if dd else None
    if info is not None and info.exists():
        try:
            n = json.loads(info.read_text()).get("total_episodes") or 0
        except (OSError, json.JSONDecodeError):
            n = 0
        ev["filter_build"] = {"done": n > 0, "why": f"{dd.name}: total_episodes={n}"}
    else:
        ev["filter_build"] = {"done": False, "why": f"no lerobot dataset at {dd}"}

    ns = (dd / "norm_stats.json") if dd else None
    if ns is not None and ns.exists() and ns.stat().st_size > 1024:
        ev["norm_stats"] = {"done": True, "why": f"{ns} ({ns.stat().st_size} bytes)"}
    else:
        ev["norm_stats"] = {"done": False, "why": f"no norm_stats.json at {dd}"}

    done, why = train_done(cfg)
    ev["train"] = {"done": done, "why": why}
    return ev


def _seeded(why: str) -> dict:
    return {"status": "succeeded", "started_at": tools.now(), "finished_at": tools.now(),
            "artifact_paths": [why], "agent_thread_id": None, "escalation": None,
            "error": None}


def seed(cfg: dict, from_node: str | None = None,
         force: bool = False) -> tuple[dict, list, list]:
    """Returns (seeds, report, conflicts).
    seeds:     {stage: StageStatus(succeeded)} to merge into the initial graph state
    report:    [(stage, 'seeded'|'live', why)] for the human
    conflicts: [(stage, why)] — only in --from mode: upstream stages with no disk evidence
               (caller refuses unless force; forced stages are seeded with a 'forced' note).
    Auto mode seeds only AUTO_SEEDABLE stages with positive evidence. --from mode seeds every
    stage before from_node in STAGES order (intake's evidence is config.json itself —
    callers ensure it exists before asking for --from)."""
    ev = evidence(cfg)
    seeds: dict = {}
    report: list = []
    conflicts: list = []
    if from_node is None:
        for s in AUTO_SEEDABLE:
            if ev[s]["done"]:
                seeds[s] = _seeded(f"reconciled from disk: {ev[s]['why']}")
                report.append((s, "seeded", ev[s]["why"]))
            else:
                report.append((s, "live", ev[s]["why"]))
        return seeds, report, conflicts
    if from_node not in STAGES:
        raise ValueError(f"unknown node '{from_node}' (choose from {', '.join(STAGES)})")
    for s in STAGES[:STAGES.index(from_node)]:
        e = ev.get(s)
        if s == "intake" or (e and e["done"]):
            why = "config.json present" if s == "intake" else e["why"]
            seeds[s] = _seeded(f"reconciled from disk: {why}")
            report.append((s, "seeded", why))
        elif force:
            why = (e or {}).get("why", "no offline evidence (uploads are only provable via HF)")
            seeds[s] = _seeded(f"FORCED by --from {from_node} --force despite: {why}")
            report.append((s, "seeded(FORCED)", why))
        else:
            conflicts.append((s, (e or {}).get("why",
                              "no offline evidence (uploads are only provable via HF)")))
    return seeds, report, conflicts


def format_report(report: list, conflicts: list = ()) -> str:
    lines = ["reconcile (disk evidence -> stage seeding):"]
    for s, verdict, why in report:
        lines.append(f"  {s:<15} {verdict:<14} {why}")
    for s, why in conflicts:
        lines.append(f"  {s:<15} {'CONFLICT':<14} {why}")
    return "\n".join(lines)
