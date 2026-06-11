"""AnthropicAgent: manual messages.create tool loop (verified SDK 0.109.0 surface). Run-scoped
context persists per agent per run in runs/<id>/agent_conv/<agent>.json, so node re-entry
(crash or interrupt-resume) replays from the exact pending tool call instead of starting blind.
ask_user pauses the graph via interrupt(); the file mailbox (pipeline.tools) is the contract."""
from __future__ import annotations

import inspect
import json
import os
import types
import typing
from pathlib import Path

from langgraph.errors import GraphInterrupt
from langgraph.types import interrupt

from pipeline import tools

DEFAULT_MODEL = "claude-sonnet-4-6"

SYSTEM = """You are {name}, one of three long-lived agents driving the Embodied Claw pipeline \
(pi0.5 / BEHAVIOR-1K data curation, training, and HuggingFace publishing).

Rules:
- Work the node's TASK until its SUCCESS criteria hold, then call complete_node with \
status="succeeded" and the artifact paths. If unrecoverable, complete_node with status="failed" \
and a precise error.
- NEVER guess on an ESCALATE trigger: call ask_user (with options and a recommendation when you \
have one) and act on the reply. Anything that survived intake is confirmed user intent.
- Skills return JSON; verify their outputs before declaring success.
- Append durable lessons (user preferences, gotchas) with append_memory — append-only, never ask \
to delete.

## Your long-term memory (agents/{name}/memory.md)
{memory}"""

def _obj(req: list, **props) -> dict:
    return {"type": "object", "properties": props, "required": req}


_BUILTIN_SCHEMAS = {
    "ask_user": ("Escalate a question to the user and block until they reply. Use for every "
                 "ESCALATE trigger. Provide numbered options plus a recommendation when possible.",
                 _obj(["question"], question={"type": "string"}, context={"type": "string"},
                      options={"type": "array", "items": _obj([], id={"type": "integer"},
                                                              label={"type": "string"})},
                      recommendation={"type": "integer"})),
    "complete_node": ("Finish this node. status must be 'succeeded' or 'failed'.",
                      _obj(["status"], status={"type": "string", "enum": ["succeeded", "failed"]},
                           artifact_paths={"type": "array", "items": {"type": "string"}},
                           summary={"type": "string"}, error={"type": "string"},
                           notes={"type": "object"})),
    "append_memory": ("Append one durable lesson to your long-term memory.md.",
                      _obj(["text"], text={"type": "string"})),
    "read_file": ("Read a text file (truncated).",
                  _obj(["path"], path={"type": "string"}, max_bytes={"type": "integer"})),
    "bash": ("Run a shell command for inspection (ls/ffprobe/grep). Heavy work belongs in skills.",
             _obj(["command"], command={"type": "string"}, timeout={"type": "integer"})),
}


def get_client():
    """Pre-check the env: with no key the SDK only fails at request time with a raw TypeError."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is not set; export it before running the pipeline.")
    import anthropic
    return anthropic.Anthropic(max_retries=5)


def _json_type(ann) -> str:
    if ann is inspect.Parameter.empty:
        return "string"
    origin = typing.get_origin(ann)
    if origin in (list, tuple):
        return "array"
    if origin is dict:
        return "object"
    if isinstance(ann, types.UnionType) or origin is typing.Union:
        for a in typing.get_args(ann):
            if a is not type(None):
                return _json_type(a)
    return {str: "string", int: "integer", float: "number", bool: "boolean",
            list: "array", tuple: "array", dict: "object"}.get(ann, "string")


def tool_schema(fn) -> dict:
    """input_schema mirroring the skill's python signature (spec section 5 signatures).
    get_type_hints resolves `from __future__ import annotations` strings — raw
    p.annotation would be the string 'int' and silently map every param to "string"."""
    try:
        hints = typing.get_type_hints(fn)
    except Exception:  # unresolvable forward refs: fall back to raw annotations
        hints = {}
    props, req = {}, []
    for p in inspect.signature(fn).parameters.values():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        props[p.name] = {"type": _json_type(hints.get(p.name, p.annotation))}
        if p.default is inspect.Parameter.empty:
            req.append(p.name)
    return {"type": "object", "properties": props, "required": req}


class Agent:
    def __init__(self, name: str, run_id: str, skills: dict, system_intro: str = "",
                 builtins: tuple = (), client=None, model: str | None = None,
                 max_iterations: int | None = None):
        self.name, self.run_id, self.skills = name, run_id, dict(skills)
        self.system_intro = system_intro
        self.client = client or get_client()
        self.model = (model or os.environ.get(f"EMBODIED_CLAW_MODEL_{name.upper()}")
                      or os.environ.get("EMBODIED_CLAW_MODEL") or DEFAULT_MODEL)
        self.max_iterations = max_iterations or int(os.environ.get("EMBODIED_CLAW_MAX_ITER", "40"))
        self.last_escalation: dict | None = None
        self._node = ""
        if "read_file" in builtins:
            self.skills["read_file"] = lambda path, max_bytes=20000: \
                Path(path).read_text(errors="replace")[:max_bytes]
        if "bash" in builtins:
            self.skills["bash"] = self._bash
        self.conv_path = tools.run_dir(run_id) / "agent_conv" / f"{name}.json"
        try:
            self.conv = json.loads(self.conv_path.read_text())
        except (OSError, json.JSONDecodeError):
            self.conv = {"messages": [], "started_nodes": [], "pending": {}}

    @staticmethod
    def _bash(command: str, timeout: int = 120) -> dict:
        r = tools.sh(command, timeout=timeout)
        return {"rc": r.returncode, "stdout": r.stdout[-8000:], "stderr": r.stderr[-4000:]}

    def memory_path(self) -> Path:
        return tools.agents_root() / self.name / "memory.md"

    def _save(self) -> None:
        self.conv_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.conv_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.conv))
        tmp.rename(self.conv_path)

    def _log(self, obj: dict) -> None:
        p = tools.run_dir(self.run_id) / "agent_messages" / f"{self._node}.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a") as f:
            f.write(json.dumps({"ts": tools.now(), "agent": self.name, **obj}, default=str) + "\n")

    def _tools_param(self) -> list[dict]:
        out = []
        for n, fn in self.skills.items():
            if n in _BUILTIN_SCHEMAS:
                desc, schema = _BUILTIN_SCHEMAS[n]
            else:
                desc = inspect.getdoc(fn).split("\n\n")[0] if inspect.getdoc(fn) else n
                schema = tool_schema(fn)
            out.append({"name": n, "description": desc, "input_schema": schema})
        for n in ("ask_user", "complete_node", "append_memory"):
            desc, schema = _BUILTIN_SCHEMAS[n]
            out.append({"name": n, "description": desc, "input_schema": schema})
        out[-1]["cache_control"] = {"type": "ephemeral"}  # 2nd breakpoint: tools block
        return out

    def _create(self):
        mem = self.memory_path()
        memory = mem.read_text() if mem.exists() else "(empty)"
        system = [{"type": "text",
                   "text": SYSTEM.format(name=self.name, memory=memory) + "\n\n" + self.system_intro,
                   "cache_control": {"type": "ephemeral"}}]
        kw = dict(model=self.model, max_tokens=8192, system=system, tools=self._tools_param(),
                  tool_choice={"type": "auto", "disable_parallel_tool_use": True},
                  messages=self.conv["messages"])
        if os.environ.get("EMBODIED_CLAW_THINKING", "adaptive") != "off":
            kw["thinking"] = {"type": "adaptive"}
        return self.client.messages.create(**kw)

    def _append_user(self, content) -> None:
        blocks = [{"type": "text", "text": content}] if isinstance(content, str) else content
        msgs = self.conv["messages"]
        if msgs and msgs[-1]["role"] == "user":  # API wants alternating roles: merge
            prev = msgs[-1]["content"]
            prev = [{"type": "text", "text": prev}] if isinstance(prev, str) else prev
            msgs[-1]["content"] = prev + blocks
        else:
            msgs.append({"role": "user", "content": blocks})
        self._save()

    def _pending_tool_uses(self) -> list[dict]:
        msgs = self.conv["messages"]
        if msgs and msgs[-1]["role"] == "assistant":
            return [b for b in msgs[-1]["content"]
                    if isinstance(b, dict) and b.get("type") == "tool_use"]
        return []

    def _ask_user(self, tool_use_id: str, args: dict) -> str:
        """Spec section 7 protocol. Re-entry safe: re-attaches to an existing unanswered
        question.json (or consumes an existing reply) instead of double-posting. Pauses the
        graph via interrupt() — the resume value is the parsed reply dict."""
        rd = tools.run_dir(self.run_id)
        esc_id = self.conv["pending"].get(tool_use_id) or tools.find_unanswered(rd, self._node) \
            or tools.new_escalation_id(self._node)
        self.conv["pending"][tool_use_id] = esc_id
        self._save()
        q = tools.write_question(rd, esc_id, node=self._node, agent=self.name,
                                 question=args.get("question", ""), context=args.get("context", ""),
                                 options=args.get("options"),
                                 recommendation=args.get("recommendation"))
        reply = tools.read_reply(rd, esc_id)
        if reply is None:
            tools.log_transition(self.run_id, self._node, "escalated", esc_id)
            tools.notify(f"[{self.run_id}:{self._node}] {q['question']}")
            reply = interrupt({"escalation_id": esc_id, **q})  # GraphInterrupt until user replies
        self.conv["pending"].pop(tool_use_id, None)
        self.last_escalation = {"question": q["question"], "context": q.get("context", ""),
                                "user_reply": reply}
        self._save()
        tools.log_transition(self.run_id, self._node, "running", f"reply consumed for {esc_id}")
        if reply.get("type") == "option":
            label = next((o["label"] for o in q.get("options", [])
                          if o.get("id") == reply["option"]), "")
            return f"User chose option {reply['option']}: {label}"
        return f"User replied: {reply.get('message', '')}"

    def _execute_tools(self, pending: list[dict]) -> dict | None:
        results, final = [], None
        for b in pending:
            name, args, tid = b["name"], b.get("input") or {}, b["id"]
            if name == "complete_node":
                final = {k: args.get(k) for k in ("status", "artifact_paths", "summary",
                                                  "error", "notes")}
                results.append({"type": "tool_result", "tool_use_id": tid,
                                "content": "node completion recorded"})
            elif name == "ask_user":
                results.append({"type": "tool_result", "tool_use_id": tid,
                                "content": self._ask_user(tid, args)})  # may raise GraphInterrupt
            elif name == "append_memory":
                p = self.memory_path()
                p.parent.mkdir(parents=True, exist_ok=True)
                with open(p, "a") as f:
                    f.write(args.get("text", "").rstrip() + "\n")
                results.append({"type": "tool_result", "tool_use_id": tid, "content": "appended"})
            else:
                try:
                    out = self.skills[name](**args)
                    results.append({"type": "tool_result", "tool_use_id": tid,
                                    "content": json.dumps(out, default=str)[:30000]})
                except GraphInterrupt:
                    raise
                except Exception as e:  # noqa: BLE001 — surfaced to the agent to reason about
                    results.append({"type": "tool_result", "tool_use_id": tid,
                                    "content": f"Error: {type(e).__name__}: {e}", "is_error": True})
        self._append_user(results)
        self._log({"event": "tool_results",
                   "results": [{**r, "content": str(r["content"])[:2000]} for r in results]})
        return final

    def run_node(self, node: str, prompt: str) -> dict:
        """Drive the tool loop until complete_node. Returns its payload
        {"status","artifact_paths","summary","error","notes"}."""
        self._node = node
        if node not in self.conv["started_nodes"]:
            self._append_user(f"## NODE: {node}\n\n{prompt}")
            self.conv["started_nodes"].append(node)
            self._save()
            self._log({"event": "node_start", "prompt": prompt})
        nudged = False
        for _ in range(self.max_iterations):
            pending = self._pending_tool_uses()
            if pending:
                final = self._execute_tools(pending)
                if final is not None:
                    return final
                continue
            resp = self._create()
            blocks = [blk.model_dump(exclude_none=True) for blk in resp.content]
            self.conv["messages"].append({"role": "assistant", "content": blocks})
            self._save()
            self._log({"event": "assistant", "stop_reason": resp.stop_reason, "content": blocks})
            if resp.stop_reason != "tool_use":
                if nudged:
                    return {"status": "failed", "error": f"agent ended turn twice without "
                                                         f"complete_node in node {node}"}
                nudged = True
                self._append_user("You ended your turn without calling complete_node. Either "
                                  "continue with tools or call complete_node with the outcome.")
        return {"status": "failed",
                "error": f"max iterations ({self.max_iterations}) exceeded in node {node}"}
