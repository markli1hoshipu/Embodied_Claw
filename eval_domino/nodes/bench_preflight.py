"""bench_preflight (benchmark_agent): runtime check on the leased GPU + task plan + ETA post."""
from eval_domino.nodes import agent_node, cfg_excerpt
from eval_domino.agents.benchmark_agent import make_agent

SKILLS = ("env_preflight", "build_task_plan")


def prompt(state: dict) -> str:
    rid = state["run_id"]
    gpus = (state["config"].get("resources") or {}).get("gpu_ids") or []
    return f"""TASK: preflight the benchmark runtime, then plan the matrix. Sequence: \
env_preflight(run_id="{rid}", gpu_id={gpus[0] if gpus else 'MISSING'}) — on failure read the \
log_tail; environment fixes from your memory may apply (one bounded attempt), else escalate; \
build_task_plan(run_id="{rid}") and finish with a one-line plan summary (n shards, GPUs \
{gpus}, rough ETA) in complete_node's summary so it lands in the Slack thread.

SUCCESS: both ok -> complete_node(status="succeeded", artifact_paths=[<task_plan.json path>]).

CONFIG:
{cfg_excerpt(state, 'benchmark', 'resources')}"""


node = agent_node("bench_preflight",
                  lambda s: make_agent(s, SKILLS, builtins=("bash", "read_file")),
                  prompt, requires=("model_prepare",))
