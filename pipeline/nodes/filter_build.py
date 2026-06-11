"""Node 2 — filter_and_build (data_agent, inherits Node 1 context)."""
from pipeline.agents.data_agent import make_agent
from pipeline.nodes import agent_node, cfg_excerpt

SKILLS = ("run_pca", "apply_drop_list", "trim_by_skills", "extract_keyframes",
          "cap_at_video_length", "slice_and_renumber", "symlink_videos", "emit_lerobot_meta",
          "build_curated_dataset", "verify_dataset_integrity", "inventory_dataset")


def prompt(state: dict) -> str:
    return f"""TASK: execute data_request.filter_description and produce ONE contiguous LeRobot \
v2.1 dataset under /work/markhsp/datasets/lerobot/<dataset_name>/ — task_index=0 forced, the \
task prompt applied. Typical recipe: run_pca for outlier drop lists (two-pass: cuts first, then \
chosen threshold), then build_curated_dataset (trims official episodes by skills, caps perturb \
episodes at video length, duplicates them, symlinks videos to realpaths, emits meta/*). ALWAYS \
pass train_config_name to build_curated_dataset (convention: pi05_b1k_<dataset_name>) — it \
gates the destructive rebuild against a live training process streaming the same dataset.

ESCALATE: requested skill ids absent from the source annotations; outlier filter would drop \
>20% of episodes; video/parquet length mismatch on >30% of episodes; duplication pushing \
frame count past 1M.

SUCCESS: verify_dataset_integrity(root, "lerobot") ok — meta/info.json present, episode count \
matches your prediction, zero broken video symlinks -> complete_node(status="succeeded", \
artifact_paths=[<lerobot dir>]).

CONFIG:
{cfg_excerpt(state, 'data_request', 'outputs')}"""


node = agent_node("filter_build",
                  lambda s: make_agent(s, SKILLS, builtins=("bash", "read_file")),
                  prompt, requires=("ingest",))
