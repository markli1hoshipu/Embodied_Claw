"""hf_agent — owns Node 3 (dataset upload), Node 6 (model upload)."""
from pipeline.agents import base
from pipeline.skills import hf_skills as h

REGISTRY = {
    "propose_repo_name": h.propose_repo_name,
    "lookup_past_naming_conventions": h.lookup_past_naming_conventions,
    "verify_repo_doesnt_exist_or_confirm_overwrite": h.verify_repo_doesnt_exist_or_confirm_overwrite,
    "hardlink_stage_checkpoints": h.hardlink_stage_checkpoints,
    "upload_large_folder_resilient": h.upload_large_folder_resilient,
    "upload_norm_stats_per_checkpoint": h.upload_norm_stats_per_checkpoint,
    "confirm_upload_complete": h.confirm_upload_complete,
}

SYSTEM = ("You own HuggingFace publishing. Conventions: datasets Hoshipu/b1k_<task>_<variant>, "
          "models Hoshipu/pi05-b1kt0-<variant>-lr<lr>. train_state/ is always excluded from "
          "model uploads; per-step assets/norm_stats.json is injected after the bulk upload. "
          "If outputs.* repo names are null, propose names and confirm with the user before "
          "creating anything.")


def make_agent(state: dict, skill_names: tuple, builtins: tuple = ()) -> base.Agent:
    return base.Agent("hf_agent", state["run_id"],
                      {n: REGISTRY[n] for n in skill_names}, SYSTEM, builtins=builtins)
