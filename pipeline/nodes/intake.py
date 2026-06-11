"""Node 0 — intake (data_agent): natural language -> RunConfig (spec section 4 / section 1)."""
import json

from pipeline import tools
from pipeline.agents.data_agent import make_agent
from pipeline.nodes import agent_node

SKILLS = ("parse_user_request", "write_run_config", "propose_repo_name",
          "lookup_past_naming_conventions")


def prompt(state: dict) -> str:
    rid = state["run_id"]
    req = (tools.run_dir(rid) / "request.txt").read_text()
    return f"""TASK: parse the user's request into a structured RunConfig and write it with \
write_run_config (run_id="{rid}"). Call parse_user_request first for the schema. Fill every \
field by explicit user statement or high-confidence inference; keep data_request fields \
free-form natural language; keep description = the raw request verbatim. Pass write_run_config \
a confidence map ({{field: "explicit"|"inferred"|"confirmed"}}) marking how each field was \
filled — it is persisted as the config's _confidence audit trail.

ESCALATE (ask_user — never guess): ambiguous source repo identity, unspecified filter method, \
output repo naming the user left open ("like the others" => propose 1-3 names from \
lookup_past_naming_conventions and confirm), or unstated training hyperparameters. Whatever \
survives intake is confirmed user intent for all downstream nodes.

SUCCESS: write_run_config returned ok, then complete_node(status="succeeded", \
artifact_paths=[<config.json path>]).

USER REQUEST (runs/{rid}/request.txt):
{req}"""


def _extra(state: dict, result: dict) -> dict:
    p = tools.run_dir(state["run_id"]) / "config.json"
    if not p.exists():
        return {"__error__": "agent reported success but config.json was not written"}
    return {"config": json.loads(p.read_text())}


_agent_node = agent_node("intake", lambda s: make_agent(s, SKILLS), prompt, extra=_extra)


def node(state: dict) -> dict:
    """Operator fast path: a config.json with _confirmed=true (written by pipeline.operator
    after the user confirmed the full config in chat) IS the intake outcome — no agent call,
    no API key needed for this node. Anything else falls through to the intake agent."""
    if (state.get("intake") or {}).get("status") != "succeeded":
        p = tools.run_dir(state["run_id"]) / "config.json"
        if p.exists():
            try:
                cfg = json.loads(p.read_text())
            except (OSError, json.JSONDecodeError):
                cfg = {}
            if cfg.get("_confirmed"):
                tools.log_transition(state["run_id"], "intake", "succeeded",
                                     "operator pre-confirmed config")
                return {"intake": {"status": "succeeded", "started_at": tools.now(),
                                   "finished_at": tools.now(), "artifact_paths": [str(p)],
                                   "agent_thread_id": None, "escalation": None, "error": None},
                        "config": cfg}
    return _agent_node(state)
