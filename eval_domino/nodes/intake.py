"""intake (model_agent): request -> EvalConfig + the ALWAYS-shown confirm card (config summary
+ GPU-count options from a live snapshot — user decision 2026-06-10: always confirm, GPU count
chosen by the user on the card)."""
import json

from eval_domino import tools
from eval_domino.nodes import agent_node
from eval_domino.agents.model_agent import make_agent

SKILLS = ("gpu_snapshot_summary", "save_eval_config", "list_local_models", "resolve_model_ref")


def prompt(state: dict) -> str:
    rid = state["run_id"]
    req = (tools.run_dir(rid) / "request.txt").read_text()
    return f"""TASK: parse the user's eval request into an EvalConfig and confirm it. Steps:
1. Parse: model ref (resolve_model_ref / list_local_models to ground it), benchmark name \
(default "domino"), tasks (list or "all"), task_config, checkpoint step (latest if unsaid).
2. ESCALATE (ask_user) for anything ambiguous — model not resolvable (offer the candidates), \
unknown tasks, missing task_config. Never guess.
3. ALWAYS confirm before any GPU work: call gpu_snapshot_summary, then ask_user with a compact \
summary of the parsed config in `context` and options = one per usable GPU count \
(id 1..n_free, label "Run with N GPU(s)") plus id 99 "Cancel". recommendation=1. If n_free is \
0, still ask with only Cancel + a context note that GPUs are busy (the run will fail-fast at \
the gate if they confirm nothing).
4. On a count: save_eval_config(run_id="{rid}", config_json=...) including \
resources.gpus_requested. On Cancel or an Edit-style reply: incorporate and re-confirm, or \
complete_node(status="failed", error="cancelled by user").

SUCCESS: save_eval_config ok -> complete_node(status="succeeded", \
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
    """Operator fast-path: a config.json with _confirmed=true was already validated and
    confirmed conversationally in the Slack thread — no agent, no confirm card."""
    import json as _json

    from pipeline.nodes import _status
    from pipeline.tools import now

    p = tools.run_dir(state["run_id"]) / "config.json"
    if (state.get("intake") or {}).get("status") != "succeeded" and p.exists():
        cfg = _json.loads(p.read_text())
        if cfg.get("_confirmed"):
            tools.log_transition(state["run_id"], "intake", "succeeded",
                                 "pre-confirmed by operator thread")
            return {"config": cfg,
                    "intake": _status("succeeded", now(), artifacts=[str(p)])}
    return _agent_node(state)
