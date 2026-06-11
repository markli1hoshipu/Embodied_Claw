"""Node 4 — compute_norm_stats (training_agent). Runs in parallel with Node 3."""
from pipeline.agents.training_agent import make_agent
from pipeline.nodes import agent_node, cfg_excerpt

SKILLS = ("ensure_train_config_present", "compute_norm_stats_fast", "verify_norm_stats_sane")


def prompt(state: dict) -> str:
    arts = state["filter_build"]["artifact_paths"]
    return f"""TASK: derive the openpi train config name from the run (convention: \
pi05_b1k_<dataset_name>), ensure_train_config_present (inserts the TrainConfig entry into \
src/openpi/training/config.py if absent, repo_id = the local dataset path; pass the \
train_request values EXACTLY — wandb_enabled, fsdp_devices, save_interval, base_model — they \
are confirmed user intent, never silently override them; propose changes via ask_user), then \
compute_norm_stats_fast (always --max-frames 50000) and verify_norm_stats_sane (openpi norm \
stats carry actions/state; extend expected_keys with image keys from the dataset's \
meta/info.json features only if the produced file is expected to contain them).

ESCALATE: the compute throws (usually a dataloader/video-decode issue from Node 2) or any \
quantile is NaN.

SUCCESS: norm_stats.json exists, parses, expected keys present, no NaNs -> \
complete_node(status="succeeded", artifact_paths=[<norm_stats.json>]).

LOCAL DATASET: {arts}
CONFIG:
{cfg_excerpt(state, 'train_request', 'outputs')}"""


node = agent_node("norm_stats",
                  lambda s: make_agent(s, SKILLS, builtins=("bash", "read_file")),
                  prompt, requires=("filter_build",))
