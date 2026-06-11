"""data_agent — owns Node 0 (intake), Node 1 (ingest), Node 2 (filter+build)."""
from pipeline.agents import base
from pipeline.skills import data_skills as d
from pipeline.skills.hf_skills import lookup_past_naming_conventions, propose_repo_name

REGISTRY = {
    "parse_user_request": d.parse_user_request, "write_run_config": d.write_run_config,
    "propose_repo_name": propose_repo_name,
    "lookup_past_naming_conventions": lookup_past_naming_conventions,
    "download_hf_snapshot": d.download_hf_snapshot, "download_hf_zip": d.download_hf_zip,
    "verify_dataset_integrity": d.verify_dataset_integrity,
    "inventory_dataset": d.inventory_dataset, "run_pca": d.run_pca,
    "apply_drop_list": d.apply_drop_list, "trim_by_skills": d.trim_by_skills,
    "extract_keyframes": d.extract_keyframes, "cap_at_video_length": d.cap_at_video_length,
    "slice_and_renumber": d.slice_and_renumber_to_file,  # tool-facing wrapper (writes dst_path)
    "symlink_videos": d.symlink_videos, "emit_lerobot_meta": d.emit_lerobot_meta,
    "build_curated_dataset": d.build_curated_dataset,
}

SYSTEM = ("You own data: intake (natural language -> RunConfig), source ingest, and the curated "
          "LeRobot build. The data_request fields are free-form natural language — you interpret "
          "them at the relevant node. Known layout: official b1k raw trees have data/ "
          "annotations/ videos/ per task; skill ids for task-0 are 1=move-to 2=pick-up 67=press "
          "(3=place-on is the dropped tail).")


def make_agent(state: dict, skill_names: tuple, builtins: tuple = ()) -> base.Agent:
    return base.Agent("data_agent", state["run_id"],
                      {n: REGISTRY[n] for n in skill_names}, SYSTEM, builtins=builtins)
