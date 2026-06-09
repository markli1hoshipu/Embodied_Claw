"""Pipeline driver: python -m pipeline.cli [--dry-run] [--status RUN_ID] runs/<cfg>.yaml ...
Run from /work/markhsp/Embodied_Claw using pipeline/.venv/bin/python (YAMLs execute sequentially)."""
from __future__ import annotations

import argparse
import fcntl
import json
import sys
from pathlib import Path

import yaml

from pipeline import nodes
from pipeline.graph import PIPELINE_RUNS_DIR, build_graph, open_checkpointer
from pipeline.state import STAGES, init_state

REQUIRED_KEYS = ["run_id", "dataset_name", "train_config_name", "hf_model_repo", "hf_dataset_repo",
                 "sources", "builder_script", "pca_thresh", "dup_factor", "train_script",
                 "num_train_steps", "save_interval", "batch_size", "peak_lr"]

def load_config(path: str) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        sys.exit(f"{path}: missing config keys: {', '.join(missing)}")
    return cfg

def dry_run(cfg: dict) -> None:
    """Evaluate every needs_* predicate against the real filesystem; print, execute nothing.
    Completely side-effect free AND offline (read-only fs/pgrep/log-tail checks only — the
    upload row never touches the HF API)."""
    rows = []
    if nodes.needs_ingest(cfg):
        missing = [s["hf_repo"] for s in cfg["sources"] if not nodes.source_ready(s)]
        rows.append(("ingest", f"RUN (missing sources: {', '.join(missing)})"))
    else:
        n = len(cfg["sources"])
        rows.append(("ingest", f"SKIP (artifacts present: {n}/{n} sources ready)"))
    exp = nodes.expected_episodes(cfg)
    if nodes.needs_build(cfg):
        rows.append(("filter_build", f"RUN (meta/info.json missing or stale; expected episodes={exp})"))
    else:
        got = nodes.built_info(cfg)["total_episodes"]
        rows.append(("filter_build", f"SKIP (info.json present, total_episodes={got} == expected "
                                     f"{exp if exp is not None else '(unknown: sources/drop lists unreadable)'})"))
    p = nodes.norm_stats_path(cfg)
    rows.append(("norm_stats", "RUN (norm_stats.json missing)" if nodes.needs_norm(cfg)
                 else f"SKIP (norm_stats.json present, {p.stat().st_size} bytes)"))
    train_pending = nodes.needs_train(cfg)
    if not train_pending:
        rows.append(("train", f"SKIP (final checkpoint present: {nodes.final_ckpt(cfg)})"))
    else:
        pids = nodes._live_pids(cfg["train_config_name"])
        if pids:
            rows.append(("train", f"ATTACH (already running, pids={pids}, last progress: "
                                  f"{nodes.last_progress(cfg) or 'n/a'})"))
        else:
            rows.append(("train", "RUN (no live process; would pre-flight psum then setsid-launch)"))
    if train_pending:
        rows.append(("upload", "PENDING (blocked on train)"))
    else:
        # Deliberately no nodes.needs_upload() here: that issues real HF API GETs and dry-run
        # promises to stay offline. A real run re-evaluates the skip check itself.
        rows.append(("upload", "PENDING (train done; would check HF repo contents — not queried in dry-run)"))
    print(f"=== {cfg['run_id']} (dry run) ===")
    for name, desc in rows:
        print(f"  {name:<13} {desc}")

def show_status(run_id: str) -> None:
    """Print the checkpointed PipelineState for a run_id from its sqlite DB.
    Exit codes: 0 for a readable state AND for a run that has never been executed (an expected,
    legitimate condition); 1 only for genuinely broken states (DB present but no checkpoint)."""
    db = PIPELINE_RUNS_DIR / f"{run_id}.sqlite"
    if not db.exists():
        print(f"run '{run_id}' has never been executed (no checkpoint DB at {db})")
        return
    saver = open_checkpointer(run_id)
    tup = saver.get_tuple({"configurable": {"thread_id": run_id}})
    if tup is None:
        sys.exit(f"no checkpoint for thread_id={run_id} in {db}")
    vals = tup.checkpoint.get("channel_values", {})
    print(json.dumps({k: vals[k] for k in ("config", *STAGES) if k in vals}, indent=2, default=str))

def run(cfg: dict) -> dict | None:
    # Single-instance lock per run_id: two concurrent CLI invocations of the same run could
    # double-launch the --overwrite train script (the sqlite checkpointer does NOT lock the thread).
    PIPELINE_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = PIPELINE_RUNS_DIR / f"{cfg['run_id']}.lock"
    lock = open(lock_path, "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        lock.close()
        sys.exit(f"another pipeline instance holds {lock_path}; refusing to run "
                 f"'{cfg['run_id']}' concurrently")
    try:
        saver = open_checkpointer(cfg["run_id"])
        graph = build_graph(checkpointer=saver)
        conf = {"configurable": {"thread_id": cfg["run_id"]}}
        try:
            # durability="sync": commit each superstep's checkpoint before moving on (kill -9 safe).
            final = graph.invoke(init_state(cfg), config=conf, durability="sync")
        except nodes.TransientHFError as e:
            # Exhausted graph RetryPolicy: record a failed ingest status (instead of leaving it
            # 'pending' forever) and let the caller continue with the next YAML.
            graph.update_state(conf, {"ingest": nodes._stage(
                "failed", None, error=f"HF 429 retries exhausted: {e}")}, as_node="ingest")
            print(f"=== {cfg['run_id']} ===\n  ingest        failed     HF 429 retries exhausted: {e}")
            return None
        print(f"=== {cfg['run_id']} ===")
        for s in STAGES:
            st = final[s]
            extra = st.get("error") or "; ".join(st.get("artifact_paths") or [])
            print(f"  {s:<13} {st['status']:<10} {extra}")
        return final
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()

def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="pipeline.cli", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("yamls", nargs="*", help="run config YAML(s), executed sequentially")
    ap.add_argument("--dry-run", action="store_true",
                    help="print per-stage done/skip vs would-run table; execute nothing")
    ap.add_argument("--status", metavar="RUN_ID", help="print checkpointed state for a run and exit")
    args = ap.parse_args(argv)
    if args.status:
        show_status(args.status)
        return
    if not args.yamls:
        ap.error("provide at least one YAML config (or --status RUN_ID)")
    for cfg in [load_config(p) for p in args.yamls]:
        dry_run(cfg) if args.dry_run else run(cfg)

if __name__ == "__main__":
    main()
