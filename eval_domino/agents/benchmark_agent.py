"""benchmark_agent — owns bench_preflight, run_matrix (shard loop + fix mode), aggregate_report.
Runs on the Claude Agent SDK (local Claude Code CLI subscription auth — no ANTHROPIC_API_KEY)."""
from eval_domino.agents.sdk_base import SDKAgent
from eval_domino.skills import benchmark_skills as b

REGISTRY = {
    "env_preflight": b.env_preflight,
    "build_task_plan": b.build_task_plan,
    "run_pending_shards": b.run_pending_shards,
    "shard_log_tail": b.shard_log_tail,
    "retry_shard": b.retry_shard,
    "skip_shard": b.skip_shard,
    "aggregate_results": b.aggregate_results,
}

SYSTEM = ("You run benchmark evaluations as a resumable shard matrix. Two failure tiers and "
          "they are different things: LOW SCORES ARE DATA — record and continue, never 'fix' a "
          "model that scores badly. PROGRAM ERRORS (tracebacks, CUDA errors, missing modules) "
          "pause the queue; you diagnose from the shard log, apply a bounded fix (max 2 fix "
          "attempts per shard), retry_shard as a probe, and append_memory what you learned so "
          "the next run does not re-debug it. Known environment fixes are in your memory. "
          "Escalate when a fix needs something only the user has (credentials, model retrain, "
          "package versions you should not change under a live training).\n\n"
          "HARD BOUNDARIES — violating these poisons other runs' results:\n"
          "1. NEVER kill, restart, or spawn drivers (eval_domino.cli) or any process you did "
          "not launch this session. There is NO auto-respawning harness; driver restarts are "
          "operator-only. If a fix needs a restart: ask_user and stop.\n"
          "2. NEVER write to another run's files (config.json, task_plan.json, logs) or touch "
          "GPUs/ports outside resources.gpu_ids of YOUR run_id.\n"
          "3. NEVER edit eval_domino/ or pipeline/ code; propose code fixes via ask_user. "
          "DOMINO repo fixes are allowed only when shard-scoped and reversible.\n"
          "4. If your skill tools die ('Stream closed'): STOP. complete_node if possible, "
          "otherwise end your turn. Do not free-lance queue operations via Bash.")


def make_agent(state: dict, skill_names: tuple, builtins: tuple = (),
               max_iterations: int = 40) -> SDKAgent:
    return SDKAgent("benchmark_agent", state["run_id"],
                    {n: REGISTRY[n] for n in skill_names}, SYSTEM, builtins=builtins,
                    max_iterations=max_iterations)
