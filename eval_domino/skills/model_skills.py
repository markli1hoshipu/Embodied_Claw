"""model_agent skills: resolve -> fetch -> adapt-check -> smoke -> launch_spec.
All paths flow through the model-family adapter (eval_domino.models.<family>); v1 family
is pi05 on DOMINO's vendored openpi."""
from __future__ import annotations

import difflib
import json

from eval_domino import gpu as gpu_mod
from eval_domino import tools
from eval_domino.models import get as get_family

HF_ENV = {"HF_HUB_DISABLE_XET": "1", "HF_HUB_ENABLE_HF_TRANSFER": "0",
          "HF_HUB_DOWNLOAD_TIMEOUT": "120"}


def gpu_snapshot_summary() -> dict:
    """Free/leased GPU snapshot for the confirm card (free = no lease + enough headroom)."""
    snap = gpu_mod.snapshot()
    free = [g["id"] for g in snap if g["leased_by"] is None
            and g["mem_free_mb"] >= gpu_mod.DEFAULT_MIN_FREE_MB]
    return {"free_gpu_ids": free, "n_free": len(free),
            "detail": [{**g, "mem_free_gb": g.pop("mem_free_mb") // 1024} for g in snap]}


def save_eval_config(run_id: str, config_json: str) -> dict:
    """Validate + persist runs/<id>/config.json. Required: model{family,train_config_name,
    model_name}, benchmark{name,tasks,task_config}, resources{gpus_requested}. Unknown
    benchmark/family/task_config/tasks are rejected here, never at GPU time."""
    cfg = json.loads(config_json)
    errs = []
    model, bench, res = cfg.get("model") or {}, cfg.get("benchmark") or {}, cfg.get("resources") or {}
    for path, key in (("model", "family"), ("model", "train_config_name"), ("model", "model_name"),
                      ("benchmark", "name"), ("benchmark", "task_config")):
        if not (cfg.get(path) or {}).get(key):
            errs.append(f"missing {path}.{key}")
    if not isinstance(res.get("gpus_requested"), int) or res.get("gpus_requested", 0) < 1:
        errs.append("resources.gpus_requested must be an int >= 1 (from the confirm card)")
    if not errs:
        try:
            adapter = __import__(f"eval_domino.benchmarks.{bench['name']}",
                                 fromlist=["x"])
        except ImportError:
            return {"ok": False, "errors": [f"unknown benchmark '{bench.get('name')}'"]}
        known = adapter.list_tasks()
        tasks = bench.get("tasks")
        if tasks in (None, "all", []):
            cfg["benchmark"]["tasks"] = known
        else:
            unknown = [t for t in tasks if t not in known]
            if unknown:
                sugg = {t: difflib.get_close_matches(t, known, n=2) for t in unknown}
                errs.append(f"unknown tasks {unknown}; close matches: {sugg}")
        if bench.get("task_config") not in adapter.task_configs():
            errs.append(f"unknown task_config '{bench.get('task_config')}'; "
                        f"available: {adapter.task_configs()}")
        try:
            get_family(model["family"])
        except ImportError:
            errs.append(f"unknown model family '{model.get('family')}'")
    if errs:
        return {"ok": False, "errors": errs}
    cfg["run_id"] = run_id
    cfg["benchmark"].setdefault("seed", 0)
    p = tools.run_dir(run_id) / "config.json"
    p.write_text(json.dumps(cfg, indent=2))
    return {"ok": True, "path": str(p), "n_tasks": len(cfg["benchmark"]["tasks"])}


def list_local_models(family: str = "pi05") -> list[dict]:
    """Checkpoints already in the benchmark's expected layout."""
    return get_family(family).list_local()


def resolve_model_ref(ref: str, family: str = "pi05", checkpoint_step: int = None) -> dict:
    """Map a user ref (local '<train_config>/<model_name>' or HF 'org/repo') to a concrete
    checkpoint. Returns found=False with fuzzy candidates instead of guessing."""
    fam = get_family(family)
    local = fam.list_local()
    # local form first: <train_config_name>/<model_name> or bare <model_name>
    for m in local:
        full = f"{m['train_config_name']}/{m['model_name']}"
        if ref in (full, m["model_name"]):
            step = int(checkpoint_step) if checkpoint_step is not None else m["steps"][-1]
            return {"found": True, "kind": "local", "train_config_name": m["train_config_name"],
                    "model_name": m["model_name"], "checkpoint_step": step,
                    "available_steps": m["steps"],
                    "step_exists": step in m["steps"]}
    # HF form
    try:
        from huggingface_hub import HfApi
        info = HfApi().repo_info(ref, repo_type="model")
        return {"found": True, "kind": "hf", "hf_repo": ref, "revision": info.sha,
                "note": "call fetch_model to place it into the benchmark checkpoint layout"}
    except Exception as e:  # noqa: BLE001 — not-found and network errors both mean 'no'
        not_found = type(e).__name__ in ("RepositoryNotFoundError", "HFValidationError")
        names = [f"{m['train_config_name']}/{m['model_name']}" for m in local]
        return {"found": False,
                "error": f"'{ref}' not local and HF lookup failed: {type(e).__name__}",
                "hf_definitely_missing": not_found,
                "local_candidates": difflib.get_close_matches(ref.split("/")[-1],
                                                              [n.split("/")[-1] for n in names],
                                                              n=5, cutoff=0.3),
                "all_local": names}


def fetch_model(hf_repo: str, family: str, train_config_name: str, model_name: str,
                checkpoint_step: int) -> dict:
    """snapshot_download '<step>/*' into the adapter's checkpoint layout. Idempotent: an
    existing valid layout short-circuits (verify with check_benchmark_compat after)."""
    fam = get_family(family)
    dest = fam.checkpoint_dir(train_config_name, model_name, checkpoint_step)
    if (dest / "params").is_dir():
        return {"ok": True, "cached": True, "path": str(dest)}
    from huggingface_hub import snapshot_download
    import os
    os.environ.update(HF_ENV)
    local = snapshot_download(repo_id=hf_repo, repo_type="model",
                              allow_patterns=[f"{checkpoint_step}/*"],
                              local_dir=str(dest.parent.parent / "_hf" / model_name))
    src = f"{local}/{checkpoint_step}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = tools.sh(f"cp -al '{src}' '{dest}' 2>/dev/null || cp -r '{src}' '{dest}'")
    if r.returncode != 0:
        return {"ok": False, "error": r.stderr[-1000:]}
    return {"ok": True, "cached": False, "path": str(dest)}


def check_benchmark_compat(family: str, train_config_name: str, model_name: str,
                           checkpoint_step: int) -> dict:
    """Hard adapt-check gate. Failure modes carry a diagnosis the user can act on."""
    res = get_family(family).verify_layout(train_config_name, model_name, checkpoint_step)
    if not res["ok"]:
        why = []
        if not res["checkpoint_dir_exists"]:
            why.append(f"checkpoint dir missing: {res['path']}")
        elif not res["params_present"]:
            why.append("params/ missing — incomplete download or wrong step")
        if not res["assets_repo_ids"]:
            why.append("assets/<repo_id>/ (norm stats) missing — checkpoint was not trained "
                       "against this benchmark's data")
        if not res["config_in_registry"]:
            why.append(f"train config '{train_config_name}' is not in the vendored openpi "
                       "registry — likely a different-embodiment checkpoint (e.g. B1K); DOMINO "
                       "needs a pi05 finetuned on RoboTwin/DOMINO data "
                       "(see DOMINO/policy/pi05/finetune.sh)")
        res["diagnosis"] = "; ".join(why)
    return res


def smoke_test_inference(family: str, train_config_name: str, model_name: str,
                         checkpoint_step: int, gpu_id: int) -> dict:
    """Run the adapter's smoke script on the leased GPU (load + one dummy get_action)."""
    cmd = get_family(family).smoke_cmd(train_config_name, model_name, checkpoint_step, gpu_id)
    r = tools.sh(cmd, timeout=1800)
    ok = r.returncode == 0 and "SMOKE_OK" in r.stdout
    return {"ok": ok, "rc": r.returncode,
            "result_line": next((l for l in r.stdout.splitlines() if "SMOKE_OK" in l), None),
            "log_tail": (r.stdout + r.stderr)[-2500:] if not ok else ""}


def write_launch_spec(run_id: str, spec_json: str) -> dict:
    """Persist launch_spec.json (machine contract for the benchmark agent) + launch_doc.md
    (human doc). Required keys enforced — prose is never the interface."""
    spec = json.loads(spec_json)
    required = ("model", "benchmark", "runtime", "smoke")
    missing = [k for k in required if k not in spec]
    if missing:
        return {"ok": False, "error": f"launch_spec missing keys: {missing}"}
    rd = tools.run_dir(run_id)
    (rd / "launch_spec.json").write_text(json.dumps(spec, indent=2))
    m, b, rt = spec["model"], spec["benchmark"], spec["runtime"]
    doc = (f"# Launch spec — {run_id}\n\n"
           f"**Model**: {m.get('family')} `{m.get('train_config_name')}/{m.get('model_name')}` "
           f"step {m.get('checkpoint_step')}\n\n"
           f"**Benchmark**: {b.get('name')} / task_config `{b.get('task_config')}` / "
           f"{len(b.get('tasks', []))} tasks\n\n"
           f"**Runtime**: `{rt.get('activation', '')}`, "
           f"XLA mem fraction {rt.get('xla_mem_fraction')}\n\n"
           f"**Smoke**: {spec['smoke'].get('result_line')}\n\n"
           f"Launch one shard:\n```bash\n{spec.get('example_cmd', '(see benchmark adapter)')}\n```\n")
    (rd / "launch_doc.md").write_text(doc)
    return {"ok": True, "paths": [str(rd / "launch_spec.json"), str(rd / "launch_doc.md")]}
