"""run_matrix (benchmark_agent): drive run_pending_shards to completion with two-tier triage.
Heartbeat digests are written by the skill itself every orchestrator window (~25 min) into
transitions.jsonl, which the bridge broadcasts — the ~30-min Slack cadence costs no agent turn."""
from eval_domino.nodes import agent_node, cfg_excerpt
from eval_domino.agents.benchmark_agent import make_agent

SKILLS = ("run_pending_shards", "shard_log_tail", "retry_shard", "skip_shard")


def prompt(state: dict) -> str:
    rid = state["run_id"]
    return f"""TASK: run the eval matrix to completion. Loop run_pending_shards(run_id="{rid}") \
and act on its status: "running" -> re-invoke; "done" -> finish; "cancelled" -> \
complete_node(status="failed", error="cancelled by user (partial results kept)"); \
"program_error" -> FIX MODE on the returned shard_id: diagnose from log_tail (+ shard_log_tail \
/ read_file / bash for deeper digging), apply a bounded fix, retry_shard, re-invoke. Max 2 fix \
attempts per shard — then either skip_shard(reason=...) if it is shard-local, or escalate if \
systemic (every shard would hit it) or the fix needs the user. append_memory every fix that \
works. LOW SCORES ARE DATA: never retry or "fix" a shard because its success_rate is low.

ESCALATE: systemic program errors you cannot fix in 2 attempts; anything needing credentials \
or changes that could disturb the live training (never touch other processes' GPUs).

SUCCESS: status "done" -> complete_node(status="succeeded", \
artifact_paths=[<task_plan.json>, <progress.json>]). Failed shards are acceptable — they are \
reported, not hidden.

CONFIG:
{cfg_excerpt(state, 'benchmark', 'resources')}"""


node = agent_node("run_matrix",
                  lambda s: make_agent(s, SKILLS, builtins=("bash", "read_file"),
                                       max_iterations=150),
                  prompt, requires=("bench_preflight",))
