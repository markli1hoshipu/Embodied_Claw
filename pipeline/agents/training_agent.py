"""training_agent — owns Node 4 (norm-stats), Node 5 (train)."""
from pipeline.agents import base
from pipeline.skills import training_skills as t

REGISTRY = {
    "ensure_train_config_present": t.ensure_train_config_present,
    "compute_norm_stats_fast": t.compute_norm_stats_fast,
    "verify_norm_stats_sane": t.verify_norm_stats_sane,
    "preflight_nccl_check": t.preflight_nccl_check,
    "verify_train_script_patched": t.verify_train_script_patched,
    "clone_train_script_if_needed": t.clone_train_script_if_needed,
    "launch_detached_train": t.launch_detached_train,
    "monitor_train_progress": t.monitor_train_progress,
    "classify_train_crash": t.classify_train_crash,
    "restart_with_workaround": t.restart_with_workaround,
}

SYSTEM = ("You own openpi training: TrainConfig insertion, norm stats, NCCL preflight, detached "
          "launches and monitoring. Launch safety is non-negotiable: launch_detached_train "
          "attaches to a live matching process instead of double-launching (train .sh passes "
          "--overwrite). The final checkpoint lands at step N-1 (e.g. 59999 for a 60k run). "
          "Prior healthy runs converged to loss ~0.0035-0.008.")


def make_agent(state: dict, skill_names: tuple, builtins: tuple = ()) -> base.Agent:
    return base.Agent("training_agent", state["run_id"],
                      {n: REGISTRY[n] for n in skill_names}, SYSTEM, builtins=builtins)
