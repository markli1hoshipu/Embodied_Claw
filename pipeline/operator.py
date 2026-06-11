"""Training operator — "Claude in the Slack thread" for the pi0.5 training pipeline.

Same architecture as eval_domino.operator (the bridge is shared): one conversational
operator session per Slack thread (claude-agent-sdk, subscription auth, session continuity
via --resume). The bridge drops inbound messages into runs/_operator/<thread_ts>/inbox/*.txt
and posts anything the operator writes to outbox/*.md back to the thread. Each message is
handled by a short-lived process:

    python -m pipeline.operator handle <thread_ts>

The operator answers status questions (checkpoints, datasets, runs, GPUs), launches
pre-confirmed training runs through the FSM, and relays/answers pipeline escalations.
It is the *head*; the pipeline FSM is the hands.
"""
from __future__ import annotations

import fcntl
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import anyio
from claude_agent_sdk import ClaudeAgentOptions, create_sdk_mcp_server, query, tool

from pipeline import inbox as pipeline_inbox
from pipeline import tools

OPERATOR_ROOT_NAME = "_operator"


def op_dir(thread_ts: str) -> Path:
    d = tools.runs_root() / OPERATOR_ROOT_NAME / thread_ts
    (d / "inbox").mkdir(parents=True, exist_ok=True)
    (d / "outbox").mkdir(parents=True, exist_ok=True)
    return d


# --------------------------------------------------------------- pure impls (unit-tested)
def last_progress(config_name: str, logs_dir: Path | None = None) -> str | None:
    """Latest tqdm Progress line from the newest train log for this config (tr-\\r aware)."""
    logs_dir = logs_dir or (tools.OPENPI / "logs")
    logs = sorted(logs_dir.glob(f"{config_name}_*.log"))
    if not logs:
        return None
    r = tools.sh(f"tr '\\r' '\\n' < {shlex.quote(str(logs[-1]))} | grep -a 'Progress on' | tail -1")
    line = r.stdout.strip()
    return line[-220:] or None


def checkpoint_status_impl(ckpt_root: Path | None = None, logs_dir: Path | None = None,
                           live=None) -> list[dict]:
    """One row per (train_config, exp): saved steps, final-step completeness, live process."""
    ckpt_root = ckpt_root or (tools.OPENPI / "checkpoints")
    live = live or tools.live_pids
    out: list[dict] = []
    for cfg_dir in sorted(p for p in ckpt_root.iterdir() if p.is_dir()) if ckpt_root.is_dir() else []:
        try:
            pids = live(cfg_dir.name)
        except Exception:
            pids = []
        for exp_dir in sorted(p for p in cfg_dir.iterdir() if p.is_dir()):
            steps = sorted(int(p.name) for p in exp_dir.iterdir()
                           if p.is_dir() and p.name.isdigit())
            row: dict = {"train_config": cfg_dir.name, "exp": exp_dir.name, "steps": steps,
                         "final_step": steps[-1] if steps else None,
                         "final_complete": bool(steps) and
                         (exp_dir / str(steps[-1]) / "_CHECKPOINT_METADATA").exists(),
                         "live_pids": pids}
            if pids:
                row["last_progress"] = last_progress(cfg_dir.name, logs_dir)
            out.append(row)
    return out


def dataset_status_impl(root: Path | None = None) -> list[dict]:
    """One row per LeRobot dataset: episodes, frames, norm-stats presence."""
    root = root or tools.LEROBOT_ROOT
    out = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []:
        info = d / "meta" / "info.json"
        if not info.exists():
            continue
        try:
            meta = json.loads(info.read_text())
        except (OSError, json.JSONDecodeError):
            meta = {}
        out.append({"dataset": d.name, "episodes": meta.get("total_episodes"),
                    "frames": meta.get("total_frames"),
                    "norm_stats": (d / "norm_stats.json").exists()})
    return out


def run_status_impl(run_id: str) -> dict:
    rd = tools.run_dir(run_id)
    if not rd.is_dir():
        return {"error": f"unknown run '{run_id}'"}
    out: dict = {"run_id": run_id}
    for name in ("config.json", "artifacts.json"):
        p = rd / name
        if p.exists():
            try:
                out[name] = json.loads(p.read_text())
            except (OSError, json.JSONDecodeError):
                pass
    tf = rd / "transitions.jsonl"
    if tf.exists():
        out["last_transitions"] = tf.read_text().splitlines()[-8:]
    out["pending_escalations"] = [
        {"node": e["node"], "escalation_id": e["escalation_id"], "question": e["question"]}
        for e in pipeline_inbox.pending_escalations() if e["run_id"] == run_id]
    if (rd / "summary.md").exists():
        out["summary_md"] = (rd / "summary.md").read_text()[-2000:]
    return out


def list_runs_impl() -> list[dict]:
    out = []
    for d in sorted(tools.runs_root().iterdir()) if tools.runs_root().is_dir() else []:
        if not d.is_dir() or d.name.startswith("_"):
            continue
        cfg = {}
        if (d / "config.json").exists():
            try:
                cfg = json.loads((d / "config.json").read_text())
            except (OSError, json.JSONDecodeError):
                pass
        kind = ("eval" if "benchmark" in cfg or (d / "task_plan.json").exists()
                else "train" if "train_request" in cfg or (d / "request.txt").exists()
                else "?")
        tf = d / "transitions.jsonl"
        last = tf.read_text().splitlines()[-1] if tf.exists() and tf.stat().st_size else ""
        out.append({"run_id": d.name, "kind": kind, "last": last[-200:]})
    return out[-25:]


def launch_train_run_impl(thread_ts: str, run_id: str, request: str,
                          config_json: str, spawn=None) -> dict:
    rd = tools.run_dir(run_id)
    if (rd / "request.txt").exists():
        return {"error": f"run '{run_id}' already exists — pick a new run_id or use run_status"}
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return {"error": "ANTHROPIC_API_KEY is not set in the bridge/operator environment — "
                         "the FSM's agent nodes (ingest/filter_build/...) need it. Add it to "
                         "the bridge service env (.env + systemd unit) before launching runs. "
                         "Status/escalation tools keep working without it."}
    try:
        cfg = json.loads(config_json)
    except json.JSONDecodeError as e:
        return {"error": f"config_json is not valid JSON: {e}"}
    cfg["run_id"], cfg["_confirmed"] = run_id, True
    rd.mkdir(parents=True, exist_ok=True)
    from pipeline.skills.data_skills import write_run_config  # same validation gate as intake
    res = write_run_config(run_id, cfg)
    if not res.get("ok"):
        return {"error": "config invalid", "details": res}
    (rd / "request.txt").write_text(request + "\n")
    (rd / ".operator_thread").write_text(thread_ts)
    spawn = spawn or (lambda rid: subprocess.Popen(
        [sys.executable, "-m", "pipeline.cli", "run", "--detach", rid],
        cwd=str(tools.REPO), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True))
    spawn(run_id)
    return {"ok": True, "run_id": run_id,
            "note": "driver launched detached; the pre-confirmed config skips the intake "
                    "escalations; transitions will stream to this thread via the bridge"}


def gpu_status_impl() -> list[dict]:
    try:  # prefer the eval stack's lease-aware snapshot when present
        from eval_domino import gpu as gpu_mod
        return [{**g, "mem_free_gb": g.pop("mem_free_mb") // 1024} for g in gpu_mod.snapshot()]
    except Exception:
        r = tools.sh("nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu "
                     "--format=csv,noheader,nounits")
        rows = []
        for line in r.stdout.strip().splitlines():
            try:
                i, used, total, util = [x.strip() for x in line.split(",")]
                rows.append({"gpu": int(i), "mem_used_mb": int(used),
                             "mem_total_mb": int(total), "util_pct": int(util)})
            except ValueError:
                continue
        return rows


# ------------------------------------------------------------------------- operator tools
def _mk_tools(thread_ts: str) -> list:
    @tool("checkpoint_status", "Status of all training checkpoints on this server: per "
          "train-config/exp the saved steps, whether the final step is complete "
          "(_CHECKPOINT_METADATA), live training pids and the latest Progress line.",
          {"type": "object", "properties": {}})
    async def checkpoint_status(args):
        return _t(checkpoint_status_impl())

    @tool("dataset_status", "All built LeRobot datasets: episodes, frames, norm-stats presence.",
          {"type": "object", "properties": {}})
    async def dataset_status(args):
        return _t(dataset_status_impl())

    @tool("list_runs", "List pipeline + eval runs and their latest transition.",
          {"type": "object", "properties": {}})
    async def list_runs(args):
        return _t(list_runs_impl())

    @tool("run_status", "Progress of one run: stage transitions, config, artifacts, pending "
          "escalations, final summary if done.",
          {"type": "object", "properties": {"run_id": {"type": "string"}},
           "required": ["run_id"]})
    async def run_status(args):
        return _t(run_status_impl(args["run_id"]))

    @tool("launch_train_run", "Launch a training-pipeline run (FSM: ingest -> filter_build -> "
          "norm_stats||upload_dataset -> train -> upload_model). Pass the full RunConfig JSON "
          "you confirmed with the user — it skips the FSM intake's own escalations. Schema: "
          '{description, data_request:{task_description, sources:[{description, hf_repo, '
          'repo_type, kind(snapshot|single_file_zip), allow_patterns, filename, local_dir}], '
          'filter_description(free-form)}, train_request:{base_model, num_train_steps, '
          'batch_size, peak_lr, fsdp_devices, save_interval, wandb_enabled}, '
          'outputs:{hf_dataset_repo, hf_model_repo}}',
          {"type": "object", "properties": {
              "run_id": {"type": "string", "description": "short slug, e.g. perturb_recovery4"},
              "request": {"type": "string", "description": "the user's request, verbatim"},
              "config_json": {"type": "string", "description": "RunConfig JSON per the schema"}},
           "required": ["run_id", "request", "config_json"]})
    async def launch_train_run(args):
        return _t(launch_train_run_impl(thread_ts, args["run_id"], args["request"],
                                        args["config_json"]))

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

    @tool("gpu_status", "Live GPU snapshot: memory/util (and eval-lease owners when available).",
          {"type": "object", "properties": {}})
    async def gpu_status(args):
        return _t(gpu_status_impl())

    return [checkpoint_status, dataset_status, list_runs, run_status, launch_train_run,
            answer_escalation, gpu_status]


def _t(payload) -> dict:
    return {"content": [{"type": "text",
                         "text": payload if isinstance(payload, str)
                         else json.dumps(payload, default=str)}]}


SYSTEM = """You are the training operator for this lab's GPU server, living in a Slack \
thread. Teammates @ you to manage the pi0.5 BEHAVIOR-1K training pipeline: check checkpoint \
and dataset status, launch data+training runs, monitor progress, and answer pipeline \
escalations.

How to work:
- Converse naturally; your final message each turn is posted to the Slack thread (markdown).
- Status questions (checkpoints, datasets, runs, GPUs) -> answer directly from your tools. \
Lead with the answer. Slack renders no markdown tables: format tables as fixed-width code \
blocks (```) with aligned columns; after the table add at most 2-3 sentences of takeaways.
- Launching a run: draft the RunConfig from the request (keep data_request fields free-form \
natural language; propose HF repo names following the Hoshipu conventions when unstated), \
confirm the FULL config with the user in this thread, then launch_train_run — your chat \
confirmation replaces the FSM intake's own escalation round. Launched runs are autonomous; \
don't babysit. If a run escalates a question, relay it conversationally and use \
answer_escalation once the user decides.
- TRAINING SAFETY (non-negotiable): never start, restart, or kill a training process via \
Bash — training launches ONLY through launch_train_run (the FSM attach-guards against \
double launches; train scripts use --overwrite, a manual duplicate launch destroys a live \
run). Killing processes, deleting datasets/checkpoints, or freeing GPUs requires the user's \
explicit confirmation in this thread first. Never touch eval runs or other users' jobs.
- Debugging: you may inspect anything (Bash/Read/Grep/Glob), read-only by default.
- Be concise: thread messages, not essays.

When something is unsure or not built yet:
- If the user asks for a capability or behavior that does not exist in this system yet (or \
you cannot verify it exists), do NOT improvise or half-build it — describe in 1-2 sentences \
what implementing it would take, then ASK the user whether to implement it, and wait.
- If you are unsure about any fact (paths, config names, run state), check with your tools \
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
                       ("checkpoint_status", "dataset_status", "list_runs", "run_status",
                        "launch_train_run", "answer_escalation", "gpu_status")] + [
                       "Bash", "Read", "Grep", "Glob"],
        permission_mode="bypassPermissions",
        max_turns=40,
        # status queries + table formatting + config confirmation — a fast model keeps Slack
        # replies snappy; override with OPERATOR_MODEL=opus for hard debugging
        model=os.environ.get("OPERATOR_MODEL", "sonnet"),
        cwd=str(tools.REPO),
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
        sys.exit("usage: python -m pipeline.operator handle <thread_ts>")


if __name__ == "__main__":
    main()
