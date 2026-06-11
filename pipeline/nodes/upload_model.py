"""Node 6 — upload_model (hf_agent, inherits Node 3 context)."""
from pipeline.agents.hf_agent import make_agent
from pipeline.nodes import agent_node, cfg_excerpt

SKILLS = ("propose_repo_name", "lookup_past_naming_conventions",
          "verify_repo_doesnt_exist_or_confirm_overwrite", "hardlink_stage_checkpoints",
          "upload_large_folder_resilient", "upload_norm_stats_per_checkpoint",
          "confirm_upload_complete")


def prompt(state: dict) -> str:
    return f"""TASK: publish checkpoints to outputs.hf_model_repo. Enumerate steps from \
train_request.save_interval up to num_train_steps; hardlink_stage_checkpoints into \
/work/markhsp/hf_staging/{state['run_id']}/ (final save sits at N-1 on disk, staged as ckpt-N); \
upload_large_folder_resilient with ignore_patterns \
["ckpt-*/train_state/*","ckpt-*/train_state/**","ckpt-*/assets/*"]; then \
upload_norm_stats_per_checkpoint and confirm_upload_complete (params + per-step \
assets/norm_stats.json, SHA spot check).

ESCALATE: same as the dataset upload — null repo name, exists_conflict, or a stall the skill \
could not recover.

SUCCESS: complete_node(status="succeeded", artifact_paths=["https://huggingface.co/<repo>"]).

FINAL CHECKPOINT: {state['train']['artifact_paths']}
NORM_STATS: {state['norm_stats']['artifact_paths']}
CONFIG:
{cfg_excerpt(state, 'train_request', 'outputs')}"""


node = agent_node("upload_model", lambda s: make_agent(s, SKILLS, builtins=("bash",)),
                  prompt, requires=("train",))
