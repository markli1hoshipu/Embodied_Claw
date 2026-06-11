"""aggregate_report (benchmark_agent): metrics table + final Slack summary. Runs after
run_matrix even on partial/cancelled matrices (requires run_matrix to have at least run)."""
from eval_domino.nodes import agent_node, cfg_excerpt
from eval_domino.agents.benchmark_agent import make_agent

SKILLS = ("aggregate_results",)


def prompt(state: dict) -> str:
    rid = state["run_id"]
    return f"""TASK: aggregate_results(run_id="{rid}"), then complete_node with a compact \
human-readable summary: model + benchmark + overall metrics (success rate / manipulation \
score) + done/failed counts + the results.md path. That summary line is what the requester \
sees in Slack — make it the line you would want to read.

SUCCESS: complete_node(status="succeeded", artifact_paths=[results.json, results.md paths]).

CONFIG:
{cfg_excerpt(state, 'model', 'benchmark')}"""


node = agent_node("aggregate_report", lambda s: make_agent(s, SKILLS, builtins=("read_file",)),
                  prompt, requires=("run_matrix",))
