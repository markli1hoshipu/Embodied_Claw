"""training_agent skills (spec section 5). v1 launch safety ported verbatim: pgrep-attach with
re.escape'd config + trailing boundary, liveness re-check immediately before any launch
(TOCTOU; train .sh passes --overwrite), stale-log byte marks gating crash classification."""
import json
import math
import pathlib
import re
import shlex
import time

from pipeline import tools

PREFLIGHT_PY = """import jax, jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental.shard_map import shard_map
mesh = Mesh(jax.devices(), ('x',))
y = jax.jit(lambda a: shard_map(lambda a: jax.lax.psum(a,'x'), mesh=mesh, in_specs=P('x'), out_specs=P())(a))(
    jax.device_put(jnp.ones((jax.device_count(),)), NamedSharding(mesh, P('x'))))
assert float(y) == float(jax.device_count())
print('NCCL OK')"""

_EXPORTS = ('export PATH="$HOME/.local/bin:$PATH" && '
            f"export LD_LIBRARY_PATH={tools.FFMPEG_LIB}:${{LD_LIBRARY_PATH:-}}")

# Verbatim insert template (openpi config.py:983-1003); num_workers=48 is a learned gotcha
# (num_workers=2 starves 8 GPUs). wandb/fsdp/save_interval/base_model come from train_request —
# the spec contract is to honor the user's exact values, never silently override them.
CONFIG_TEMPLATE = '''    TrainConfig(
        name="{name}",
        wandb_enabled={wandb_enabled},
        num_workers=48,
        model=pi0_config.Pi0Config(pi05=True, action_horizon=50, discrete_state_input=False),
        data=LeRobotB1KDataConfig(
            repo_id="{repo_id}",
            base_config=DataConfig(prompt_from_task=True),
        ),
        batch_size={batch_size},
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2_000,
            peak_lr={peak_lr},
            decay_steps={num_train_steps},
            decay_lr={decay_lr},
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/{base_model}/params"),
        num_train_steps={num_train_steps},
        fsdp_devices={fsdp_devices},
        save_interval={save_interval},
    ),
'''


def ensure_train_config_present(config_name: str, repo_id_path: str, batch_size: int = 32,
                                peak_lr: float = 2.5e-5, num_train_steps: int = 60000,
                                wandb_enabled: bool = False, fsdp_devices: int = 1,
                                save_interval: int = 1000, base_model: str = "pi05_base",
                                config_path: str = None) -> dict:
    """Insert a TrainConfig entry into openpi src/openpi/training/config.py if absent
    (decay_lr = peak_lr/10, warmup 2000, cosine decay to num_train_steps). Pass the
    train_request values (wandb_enabled, fsdp_devices, save_interval, base_model) verbatim —
    they are structured because openpi consumes them as exact values."""
    cp = pathlib.Path(config_path or tools.OPENPI / "src/openpi/training/config.py")
    text = cp.read_text()
    if f'name="{config_name}"' in text:
        return {"added": False, "config_block": ""}
    block = CONFIG_TEMPLATE.format(name=config_name, repo_id=repo_id_path, batch_size=batch_size,
                                   peak_lr=peak_lr, num_train_steps=num_train_steps,
                                   decay_lr=peak_lr / 10, wandb_enabled=bool(wandb_enabled),
                                   fsdp_devices=int(fsdp_devices),
                                   save_interval=int(save_interval), base_model=base_model)
    lines = text.splitlines(keepends=True)
    anchor = next((i for i, ln in enumerate(lines) if ln.startswith("_CONFIGS = [")), None)
    if anchor is None:
        return {"error": f"_CONFIGS list not found in {cp}"}
    close = next((i for i in range(anchor + 1, len(lines)) if lines[i].rstrip() == "]"), None)
    if close is None:
        return {"error": f"closing ] of _CONFIGS not found in {cp}"}
    cp.write_text("".join(lines[:close]) + block + "".join(lines[close:]))
    return {"added": True, "config_block": block}


def compute_norm_stats_fast(config_name: str, max_frames: int = 50000,
                            dataset_dir: str = None) -> dict:
    """`uv run python scripts/compute_norm_stats.py --config-name <cfg> --max-frames N` from
    openpi (sampled fast path — a full pass takes 60+ min; spec section 9 row 8)."""
    env = {"HF_LEROBOT_HOME": tools.TRAIN_ENV["HF_LEROBOT_HOME"],
           "LD_LIBRARY_PATH": f"{tools.FFMPEG_LIB}:"}
    t0 = time.time()
    r = tools.sh(f"cd {tools.OPENPI} && {_EXPORTS} && uv run python scripts/compute_norm_stats.py "
                 f"--config-name {shlex.quote(config_name)} --max-frames {int(max_frames)}",
                 env=env, timeout=2 * 3600)
    if r.returncode != 0:
        return {"error": ((r.stderr or "") + (r.stdout or ""))[-2000:]}
    p = pathlib.Path(dataset_dir or ".") / "norm_stats.json"
    if not p.is_file() or p.stat().st_size <= 1024:
        return {"error": f"{p} missing or <= 1 KB after compute"}
    return {"norm_stats_path": str(p), "duration_s": round(time.time() - t0, 1),
            "sample_size": int(max_frames)}


def _scan_nans(obj, prefix: str, hits: list) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            _scan_nans(v, f"{prefix}.{k}" if prefix else str(k), hits)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _scan_nans(v, f"{prefix}[{i}]", hits)
    elif isinstance(obj, float) and math.isnan(obj):
        hits.append(prefix)


def verify_norm_stats_sane(norm_stats_path: str, expected_keys: list = ("actions", "state")) -> dict:
    """Parse norm_stats.json; require the expected top-level stat keys and zero NaNs anywhere."""
    try:
        data = json.loads(pathlib.Path(norm_stats_path).read_text())
    except (OSError, json.JSONDecodeError) as e:
        return {"ok": False, "nan_keys": [], "missing_keys": list(expected_keys),
                "error": f"unreadable: {e}"}
    keys = set(data)
    for v in data.values():
        if isinstance(v, dict):
            keys |= set(v)
    missing = [k for k in expected_keys if k not in keys]
    nans: list = []
    _scan_nans(data, "", nans)
    return {"ok": not missing and not nans, "nan_keys": nans[:20], "missing_keys": missing}


def preflight_nccl_check(config_name: str = "") -> dict:
    """8-GPU shard_map psum probe before burning hours (spec section 9 row 4). ALWAYS pass
    config_name: when a matching train run is already live, the probe is suppressed — it would
    try to preallocate 95% of every GPU under the live run and OOM/interfere."""
    if config_name and tools.live_pids(config_name):
        return {"ok": True, "error": None, "suggested_env": {},
                "skipped": "matching train run is live — GPU probe suppressed (attach instead)"}
    r = tools.sh(f"cd {tools.OPENPI} && {_EXPORTS} && uv run python -c {shlex.quote(PREFLIGHT_PY)}",
                 env=tools.TRAIN_ENV, timeout=600)
    if r.returncode != 0 or "NCCL OK" not in r.stdout:
        return {"ok": False, "error": "multi-GPU psum pre-flight failed: "
                                      + ((r.stderr or r.stdout) or "")[-800:],
                "suggested_env": {"NCCL_NVLS_ENABLE": "0"}}
    return {"ok": True, "error": None, "suggested_env": {}}


def verify_train_script_patched(script_path: str = None) -> dict:
    """Pre-flight for spec section 9 row 5: the first-batch wandb.Image gather in openpi
    scripts/train.py must sit under an `if config.wandb_enabled:` guard (unguarded it runs the
    gather — and crashes — on wandb-disabled runs). Read-only grep; never edits openpi."""
    p = pathlib.Path(script_path or tools.OPENPI / "scripts/train.py")
    try:
        lines = p.read_text().splitlines()
    except OSError as e:
        return {"ok": False, "evidence": f"cannot read {p}: {e}"}
    sites = [i for i, ln in enumerate(lines) if "wandb.Image(" in ln]
    for i in sites:
        if not any("if config.wandb_enabled" in lines[j] for j in range(max(0, i - 5), i)):
            return {"ok": False, "evidence": f"unguarded wandb.Image at {p}:{i + 1}: "
                                             f"{lines[i].strip()[:200]} — needs the "
                                             "`if config.wandb_enabled:` patch"}
    return {"ok": True, "evidence": (f"{len(sites)} wandb.Image site(s) guarded by "
                                     "config.wandb_enabled" if sites
                                     else "no wandb.Image call present")}


def clone_train_script_if_needed(template: str, new_name: str, config_name: str,
                                 scripts_dir: str = None) -> dict:
    """Clone a train_*.sh template, retargeting its --config-name to config_name."""
    sd = pathlib.Path(scripts_dir or tools.OPENPI / "scripts")
    dst = sd / new_name
    if dst.exists():
        return {"path": str(dst), "created": False}
    text = pathlib.Path(template).read_text()
    text, n = re.subn(r"(--config[-_]name[= ])\S+", rf"\g<1>{config_name}", text)
    if n == 0:
        return {"error": f"no --config-name flag found in template {template}"}
    dst.write_text(text)
    dst.chmod(0o755)
    return {"path": str(dst), "created": True}


def launch_detached_train(script_path: str, log_path: str, config_name: str,
                          startup_timeout_s: int = 120) -> dict:
    """Detached `setsid bash <script>` (load-bearing: the run outlives this process). Attaches
    instead of launching when a matching train process is alive, and re-checks liveness right
    before the setsid (TOCTOU guard — a duplicate launch would --overwrite the in-flight run).
    After the setsid it polls until pgrep matches the python (uv resolution + imports can take
    tens of seconds): success NEVER returns pid=None — a None pid invites a retry-launch while
    the first launch is still warming up."""
    pids = tools.live_pids(config_name)
    if pids:
        return {"attached": True, "pid": pids[0], "pids": pids, "log_path": log_path}
    log = pathlib.Path(log_path)
    pids = tools.live_pids(config_name)  # immediate pre-launch re-check
    if pids:
        return {"attached": True, "pid": pids[0], "pids": pids, "log_path": log_path}
    cmd = (f"cd {tools.OPENPI} && {_EXPORTS} && setsid bash {shlex.quote(script_path)} "
           f"</dev/null >{shlex.quote(str(log))} 2>&1 & disown")
    r = tools.sh(cmd, env=tools.TRAIN_ENV)
    if r.returncode != 0:
        return {"error": f"failed to launch {script_path}: {(r.stderr or r.stdout)[-500:]}"}
    deadline = time.time() + startup_timeout_s
    while True:  # setsid ... & disown exits 0 even for a bad script path — wait for the pid
        pids = tools.live_pids(config_name)
        if pids:
            return {"attached": False, "pid": pids[0], "log_path": str(log)}
        if time.time() >= deadline:
            break
        time.sleep(2)
    tail = _log_tail(str(log)) if log.exists() else "(no log written)"
    return {"error": f"launched {script_path} but no matching train process appeared within "
                     f"{startup_timeout_s}s — startup failure? log tail: {tail[-800:]}"}


def _log_tail(log_path: str, since_byte: int = 0) -> str:
    p = pathlib.Path(log_path)
    try:
        size = p.stat().st_size
    except OSError:
        return ""
    if size <= since_byte:
        return ""  # nothing new since attach/launch — stale content is not evidence
    with open(p, "rb") as f:
        f.seek(max(since_byte, size - 8000))
        return f.read().decode("utf-8", "replace").replace("\r", "\n")


def _last_loss(text: str):
    m = re.findall(r"loss[=:\s]+(nan|[0-9][0-9.eE+-]*)", text, re.IGNORECASE)
    try:
        return float(m[-1]) if m else None
    except ValueError:
        return None


def monitor_train_progress(log_path: str, ckpt_dir: str, num_steps: int, config_name: str,
                           poll_interval: int = 60, max_polls: int = 30,
                           loss_ceiling: float = None, deadline_s: int = 24 * 3600) -> dict:
    """Bounded poll loop — returns every max_polls polls with {"status": "running", "last_step",
    "recent_loss"} so the agent observes mid-training progress and re-invokes (Node-5 divergence
    trigger). Short-circuits with {"status": "diverged", "loss"} on a NaN loss or recent loss >
    loss_ceiling (pass ~10x the typical end-of-run loss, e.g. 0.08, once past warmup). 'done'
    when <ckpt_dir>/<num_steps-1>/ holds params/ + _CHECKPOINT_METADATA (final save lands at
    N-1). 'crashed' only after 3 consecutive empty-pid polls (the pgrep blind window between
    `setsid bash` and the python exec must not read as a crash) + a 5s checkpoint-flush grace;
    deadline expiry with a live pid is 'running', never 'crashed'. Crash evidence = log bytes
    written AFTER attach."""
    cd = pathlib.Path(ckpt_dir)
    final = cd / str(int(num_steps) - 1)
    mark = pathlib.Path(log_path).stat().st_size if pathlib.Path(log_path).exists() else 0

    def _done():
        return (final / "_CHECKPOINT_METADATA").exists() and (final / "params").is_dir()

    def _step():
        steps = [int(p.name) for p in cd.glob("[0-9]*") if p.is_dir() and p.name.isdigit()]
        return max(steps, default=None)

    deadline = time.time() + deadline_s
    empty_polls, loss = 0, None
    for _ in range(max(1, int(max_polls))):
        if _done():
            return {"status": "done", "last_step": int(num_steps) - 1,
                    "final_loss": _last_loss(_log_tail(log_path, mark)), "since_byte": mark}
        loss = _last_loss(_log_tail(log_path, mark))
        if loss is not None and (math.isnan(loss) or (loss_ceiling and loss > loss_ceiling)):
            return {"status": "diverged", "loss": loss, "last_step": _step(),
                    "since_byte": mark, "log_tail": _log_tail(log_path, mark)[-1500:]}
        if not tools.live_pids(config_name):
            empty_polls += 1
            if empty_polls >= 3:
                time.sleep(5)  # grace for the final checkpoint flush
                if _done():
                    return {"status": "done", "last_step": int(num_steps) - 1,
                            "final_loss": _last_loss(_log_tail(log_path, mark)),
                            "since_byte": mark}
                return {"status": "crashed", "last_step": _step(), "final_loss": loss,
                        "since_byte": mark, "log_tail": _log_tail(log_path, mark)[-1500:]}
        else:
            empty_polls = 0
        if time.time() >= deadline:
            break
        time.sleep(poll_interval)
    return {"status": "running", "last_step": _step(), "recent_loss": loss,
            "since_byte": mark, "note": "still training — re-invoke to keep monitoring"}


def classify_train_crash(log_path: str, since_byte: int = 0) -> dict:
    """Crash triage on bytes written after attach/launch (stale-log gate). Categories:
    nccl_nvls (CUDA failure 401 / nvls.cc) | oom | data_loader | unknown."""
    tail = _log_tail(log_path, since_byte)
    low = tail.lower()
    cats = [("nccl_nvls", ("cuda failure 401", "nvls.cc")),
            ("oom", ("resource_exhausted", "out of memory", "cuda out of memory")),
            ("data_loader", ("dataloader", "torchcodec", "decode error", "video_reader"))]
    for cat, needles in cats:
        for n in needles:
            if n in low:
                line = next((ln for ln in tail.splitlines() if n in ln.lower()), n)
                return {"category": cat, "evidence": line[-300:]}
    return {"category": "unknown", "evidence": tail[-500:]}


def restart_with_workaround(category: str, script_path: str, log_path: str, config_name: str,
                            rechecks: int = 3, recheck_interval_s: float = 5.0) -> dict:
    """Relaunch after a classified crash. nccl_nvls: NCCL_NVLS_ENABLE=0 is already exported in
    TRAIN_ENV — relaunch, but only after liveness stays empty across several spaced re-checks
    (cooldown: a 'crash' seen during the pgrep blind window must attach, never double-launch).
    oom / data_loader / unknown: no known automatic workaround — escalate to the user."""
    if category != "nccl_nvls":
        return {"error": f"no known automatic workaround for '{category}' — escalate via ask_user"}
    for i in range(max(1, int(rechecks))):
        pids = tools.live_pids(config_name)
        if pids:
            return {"attached": True, "pid": pids[0], "pids": pids, "log_path": log_path,
                    "applied_env": {"NCCL_NVLS_ENABLE": "0"}}
        if i + 1 < rechecks:
            time.sleep(recheck_interval_s)
    out = launch_detached_train(script_path, log_path, config_name)
    return {**out, "applied_env": {"NCCL_NVLS_ENABLE": "0"}}
