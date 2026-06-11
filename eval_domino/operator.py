"""Operator layer — "Claude in the Slack thread".

One conversational operator session per Slack thread (claude-agent-sdk, subscription auth,
session continuity via --resume). The bridge drops inbound messages into
runs/_operator/<thread_ts>/inbox/*.txt and posts anything the operator writes to
outbox/*.md back to the thread. Each message is handled by a short-lived process:

    python -m eval_domino.operator handle <thread_ts>

The operator converses freely and drives the eval FSM as a tool: it launches runs
(pre-confirmed config skips the intake card), reads progress/results, answers escalations,
cancels runs, and debugs within governance bounds. It is the *head*; eval_domino is the hands.
"""
from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
import time

import anyio
from claude_agent_sdk import ClaudeAgentOptions, create_sdk_mcp_server, query, tool

from eval_domino import gpu as gpu_mod
from eval_domino import tools
from pipeline import inbox as pipeline_inbox

OPERATOR_ROOT_NAME = "_operator"


def op_dir(thread_ts: str):
    d = tools.runs_root() / OPERATOR_ROOT_NAME / thread_ts
    (d / "inbox").mkdir(parents=True, exist_ok=True)
    (d / "outbox").mkdir(parents=True, exist_ok=True)
    return d


# ------------------------------------------------------------------------- operator tools
def _mk_tools(thread_ts: str) -> list:
    @tool("launch_eval_run", "Launch a benchmark eval run (the FSM handles GPU leasing, model "
          "prep, shard matrix, reporting). Pass the full structured config you confirmed with "
          "the user — it skips the FSM's own confirmation card. Returns the run_id.",
          {"type": "object", "properties": {
              "run_id": {"type": "string", "description": "short slug, e.g. eval_pi05_clean"},
              "request": {"type": "string", "description": "the user's request, verbatim"},
              "config_json": {"type": "string", "description":
                  "EvalConfig JSON: model{family,train_config_name,model_name,checkpoint_step},"
                  " benchmark{name,tasks|'all',task_config,episodes_per_task,seed},"
                  " resources{gpus_requested}"}},
           "required": ["run_id", "request", "config_json"]})
    async def launch_eval_run(args):
        rid = args["run_id"]
        rd = tools.run_dir(rid)
        if (rd / "request.txt").exists():
            return _t({"error": f"run '{rid}' already exists — pick a new run_id "
                                f"or use run_status"})
        rd.mkdir(parents=True, exist_ok=True)
        cfg = json.loads(args["config_json"])
        cfg["run_id"], cfg["_confirmed"] = rid, True
        # validate through the same gate intake uses
        from eval_domino.skills.model_skills import save_eval_config
        cfg.setdefault("resources", {}).setdefault("gpus_requested", 1)
        res = save_eval_config(rid, json.dumps(cfg))
        if not res.get("ok"):
            return _t({"error": "config invalid", "details": res})
        saved = json.loads((rd / "config.json").read_text())  # validator expanded tasks/defaults
        saved["_confirmed"] = True
        json.dump(saved, open(rd / "config.json", "w"), indent=2)
        (rd / "request.txt").write_text(args["request"] + "\n")
        (rd / ".operator_thread").write_text(thread_ts)
        subprocess.Popen([sys.executable, "-m", "eval_domino.cli", "run", "--detach", rid],
                         cwd=str(tools.runs_root().parent), stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, start_new_session=True)
        return _t({"ok": True, "run_id": rid, "note": "driver launched detached; transitions "
                   "will stream to this thread via the bridge"})

    @tool("run_status", "Progress of an eval run: stage statuses, shard counts, latest digest.",
          {"type": "object", "properties": {"run_id": {"type": "string"}},
           "required": ["run_id"]})
    async def run_status(args):
        rid = args["run_id"]
        rd = tools.run_dir(rid)
        out: dict = {"run_id": rid}
        tp = rd / "task_plan.json"
        if tp.exists():
            plan = json.loads(tp.read_text())
            counts: dict = {}
            for s in plan["shards"]:
                counts[s["status"]] = counts.get(s["status"], 0) + 1
            out["shards"] = counts
            out["running_tasks"] = [s["task"] for s in plan["shards"] if s["status"] == "running"]
        for name in ("progress.json", "config.json"):
            p = rd / name
            if p.exists():
                out[name] = json.loads(p.read_text())
        tf = rd / "transitions.jsonl"
        if tf.exists():
            out["last_transitions"] = tf.read_text().splitlines()[-5:]
        return _t(out)

    @tool("list_runs", "List eval runs and their latest status.", {"type": "object",
                                                                   "properties": {}})
    async def list_runs(args):
        out = []
        for d in sorted(tools.runs_root().iterdir()):
            if not d.is_dir() or d.name.startswith("_"):
                continue
            tf = d / "transitions.jsonl"
            last = tf.read_text().splitlines()[-1] if tf.exists() and tf.stat().st_size else ""
            out.append({"run_id": d.name, "last": last[-200:]})
        return _t(out[-20:])

    @tool("eval_results", "Final/partial results of a run (results.json if aggregated, else "
          "per-task metrics from the task plan).",
          {"type": "object", "properties": {"run_id": {"type": "string"}},
           "required": ["run_id"]})
    async def eval_results(args):
        rd = tools.run_dir(args["run_id"])
        if (rd / "results.json").exists():
            return _t(json.loads((rd / "results.json").read_text()))
        tp = rd / "task_plan.json"
        if not tp.exists():
            return _t({"error": "no task plan yet"})
        plan = json.loads(tp.read_text())
        rows = [{"task": s["task"], "status": s["status"], **(s.get("metrics") or {})}
                for s in plan["shards"] if s["status"] == "done"]
        return _t({"done": rows, "note": "run not aggregated yet"})

    @tool("answer_escalation", "Answer a pending FSM escalation for a run (use after the user "
          "told you their choice in chat — option id or free-text).",
          {"type": "object", "properties": {
              "run_id": {"type": "string"}, "node": {"type": "string"},
              "option": {"type": "integer"}, "message": {"type": "string"}},
           "required": ["run_id"]})
    async def answer_escalation(args):
        target = pipeline_inbox.resolve_target(run_id=args["run_id"], node=args.get("node"),
                                               latest=False)
        if target is None:
            return _t({"error": "no pending escalation for that run"})
        text = str(args["option"]) if args.get("option") is not None else args.get("message", "")
        if not text:
            return _t({"error": "provide option or message"})
        p = pipeline_inbox.write_reply(target["run_id"], target["escalation_id"], text)
        return _t({"ok": True, "answered": target["escalation_id"], "reply_file": str(p)})

    @tool("cancel_run", "Cooperative cancel: the run stops at the next shard boundary and "
          "reports partial results.", {"type": "object",
                                       "properties": {"run_id": {"type": "string"}},
                                       "required": ["run_id"]})
    async def cancel_run(args):
        rd = tools.run_dir(args["run_id"])
        if not rd.is_dir():
            return _t({"error": "unknown run"})
        (rd / "CANCEL").touch()
        return _t({"ok": True})

    @tool("gpu_status", "Live GPU snapshot: memory and which run leases each GPU.",
          {"type": "object", "properties": {}})
    async def gpu_status(args):
        snap = gpu_mod.snapshot()
        return _t([{**g, "mem_free_gb": g.pop("mem_free_mb") // 1024} for g in snap])

    return [launch_eval_run, run_status, list_runs, eval_results, answer_escalation,
            cancel_run, gpu_status]


def _t(payload) -> dict:
    return {"content": [{"type": "text",
                         "text": payload if isinstance(payload, str)
                         else json.dumps(payload, default=str)}]}


SYSTEM = """You are the evaluation operator for this lab's GPU server, living in a Slack \
thread. Teammates @ you to run robot-policy benchmark evaluations (currently: pi05-family \
models on the DOMINO benchmark) and ask about progress and results.

How to work:
- Converse naturally; your final message each turn is posted to the Slack thread (markdown).
- Confirm the parsed eval config (model, tasks, task_config, episodes, GPU count vs the live \
gpu_status) with the user BEFORE launch_eval_run — your confirmation replaces the FSM's card.
- Launched runs are autonomous (the FSM leases GPUs, prepares the model, runs the shard \
matrix, posts heartbeats to this thread). Don't babysit; answer questions from run_status / \
eval_results when asked, and summarize results.md when a run finishes.
- If a run escalates a question, relay it conversationally and use answer_escalation once the \
user decides.
- Debugging: you may inspect anything (Bash/Read). Destructive actions (killing processes, \
restarting drivers, deleting files) require explicit user confirmation in this thread first. \
Never touch the training pipeline's processes or other users' jobs.
- Be concise: thread messages, not essays. Lead with the answer.

Reporting results:
- ALWAYS report results as a table: one row per task (or per model when comparing runs), \
columns at minimum success_rate and manipulation_score_mean, plus total_episodes when episode \
counts differ between runs.
- Tasks/models that have not run (pending, running, failed, cancelled) still get a row — fill \
their metric cells with N/A, never omit them or leave blanks.
- Slack renders no markdown tables: format the table as a fixed-width code block (```) with \
aligned columns.
- After the table, add at most 2-3 sentences of takeaways.

When something is unsure or not built yet:
- If the user asks for a capability, benchmark, model family, or behavior that does not exist \
in this system yet (or you cannot verify it exists), do NOT improvise or half-build it — \
describe in 1-2 sentences what implementing it would take, then ASK the user whether to \
implement it or not, and wait for their answer.
- If you are unsure about any fact (paths, model names, run state), check with your tools \
first; if still unsure, say so plainly and ask — never present a guess as fact."""


# ------------------------------------------------------------------------- session handling
async def _run_session(thread_ts: str, prompt_text: str) -> str:
    d = op_dir(thread_ts)
    sess_p = d / "session.json"
    sess = json.loads(sess_p.read_text()) if sess_p.exists() else {}
    options = ClaudeAgentOptions(
        system_prompt=SYSTEM,
        mcp_servers={"ops": create_sdk_mcp_server(name="ops", version="1.0.0",
                                                  tools=_mk_tools(thread_ts))},
        allowed_tools=[f"mcp__ops__{n}" for n in
                       ("launch_eval_run", "run_status", "list_runs", "eval_results",
                        "answer_escalation", "cancel_run", "gpu_status")] + ["Bash", "Read",
                                                                             "Grep", "Glob"],
        permission_mode="bypassPermissions",
        max_turns=40,
        # operator work is status queries + table formatting + config confirmation — a fast
        # model keeps Slack replies snappy; override with OPERATOR_MODEL=opus for hard debugging
        model=os.environ.get("OPERATOR_MODEL", "sonnet"),
        cwd=str(tools.runs_root().parent),
        resume=sess.get("session_id"),
        env={**os.environ, "MCP_TOOL_TIMEOUT": "600000"},
    )
    final_text, session_id = "", sess.get("session_id")
    async for message in query(prompt=prompt_text, options=options):
        kind = type(message).__name__
        if kind == "AssistantMessage":
            texts = [getattr(b, "text", "") for b in (getattr(message, "content", []) or [])]
            texts = [t for t in texts if t]
            if texts:
                final_text = "\n".join(texts)
        elif kind == "ResultMessage":
            session_id = getattr(message, "session_id", None) or session_id
    if session_id:
        sess_p.write_text(json.dumps({"session_id": session_id, "updated_at": tools.now()}))
    return final_text or "(operator produced no reply)"


def handle(thread_ts: str) -> None:
    """Drain this thread's inbox through the operator session; write replies to the outbox.
    flock serializes concurrent messages for one thread; distinct threads run in parallel."""
    d = op_dir(thread_ts)
    with open(d / ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        while True:
            pending = sorted((d / "inbox").glob("*.txt"))
            if not pending:
                return
            texts = []
            for p in pending:
                texts.append(p.read_text().strip())
                p.rename(p.with_suffix(".done"))
            reply = anyio.run(_run_session, thread_ts, "\n\n".join(texts))
            out = d / "outbox" / f"{int(time.time() * 1000)}.md"
            tmp = out.with_suffix(".tmp")
            tmp.write_text(reply)
            tmp.rename(out)


def main(argv=None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) >= 2 and argv[0] == "handle":
        handle(argv[1])
    else:
        sys.exit("usage: python -m eval_domino.operator handle <thread_ts>")


if __name__ == "__main__":
    main()
