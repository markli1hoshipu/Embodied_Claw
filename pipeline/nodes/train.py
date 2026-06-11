"""Node 5 — train (training_agent, inherits Node 4 context). Joins Nodes 3+4."""
from pipeline.agents.training_agent import make_agent
from pipeline.nodes import agent_node, cfg_excerpt

SKILLS = ("preflight_nccl_check", "verify_train_script_patched", "clone_train_script_if_needed",
          "launch_detached_train", "monitor_train_progress", "classify_train_crash",
          "restart_with_workaround")


def prompt(state: dict) -> str:
    return f"""TASK: train to train_request.num_train_steps. Sequence: \
verify_train_script_patched (wandb.Image gather must be guarded — escalate on needs-patch, \
never edit openpi yourself); preflight_nccl_check (ALWAYS pass config_name — it suppresses the \
GPU probe when a matching run is already live); if train_request.wandb_enabled is true, check \
WANDB_API_KEY is set (bash: test -n "$WANDB_API_KEY") and escalate if missing; \
clone_train_script_if_needed; launch_detached_train (it ATTACHES if a matching run is live — \
never double-launch); monitor_train_progress (ckpt_dir = \
/work/markhsp/openpi/checkpoints/<config_name>/<exp>/; it returns "running" periodically — \
re-invoke until done, and pass loss_ceiling ~10x the typical end-of-run loss once past warmup \
so divergence surfaces); on "diverged", escalate; on "crashed", classify_train_crash with the \
monitor's since_byte, then restart_with_workaround (max 2 relaunches) or escalate. Finish \
with a short table of all current models under /work/markhsp/openpi/checkpoints/.

ESCALATE: preflight fails with NCCL_NVLS_ENABLE=0 already set; verify_train_script_patched \
returns needs-patch; >2 deaths with different errors; loss NaN or >10x the typical \
end-of-run value (monitor returns "diverged"); wandb_enabled=true but WANDB_API_KEY missing.

SUCCESS: <ckpt_dir>/<num_train_steps-1>/ holds params/ + _CHECKPOINT_METADATA and final loss \
is plausible (memory: ~0.0035-0.008) -> complete_node(status="succeeded", \
artifact_paths=[<final ckpt dir>]).

NORM_STATS: {state['norm_stats']['artifact_paths']}
CONFIG:
{cfg_excerpt(state, 'train_request')}"""


node = agent_node("train", lambda s: make_agent(s, SKILLS, builtins=("bash", "read_file")),
                  prompt, requires=("norm_stats",))
