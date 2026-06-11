"""Node 3 — upload_dataset (hf_agent). Runs in parallel with Node 4."""
from pipeline.agents.hf_agent import make_agent
from pipeline.nodes import agent_node, cfg_excerpt

SKILLS = ("propose_repo_name", "lookup_past_naming_conventions",
          "verify_repo_doesnt_exist_or_confirm_overwrite", "upload_large_folder_resilient",
          "confirm_upload_complete")


def prompt(state: dict) -> str:
    arts = state["filter_build"]["artifact_paths"]
    return f"""TASK: upload the built LeRobot dataset (local path below) to an HF dataset repo \
(outputs.hf_dataset_repo; the symlinked videos resolve to real bytes on upload).

ESCALATE: hf_dataset_repo is null (propose names from past conventions, ask approval); the \
repo exists with conflicting content (verify_... returns exists_conflict — confirm overwrite \
or pick a new name); upload stalled after the skill's internal restarts.

SUCCESS: confirm_upload_complete ok with expected_files=["meta/info.json", ...] and SHA spot \
check -> complete_node(status="succeeded", artifact_paths=["https://huggingface.co/datasets/<repo>"]).

LOCAL DATASET: {arts}
CONFIG:
{cfg_excerpt(state, 'outputs')}"""


node = agent_node("upload_dataset", lambda s: make_agent(s, SKILLS), prompt,
                  requires=("filter_build",))
