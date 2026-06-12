"""Claude Agent SDK backend for FSM agents (pipeline + eval_domino).

Auth comes from the locally logged-in Claude Code CLI (subscription) — no ANTHROPIC_API_KEY.
Mirrors pipeline.agents.base.Agent's surface (name / last_escalation / run_node) so
pipeline.nodes.agent_node drives it unchanged. Skills become in-process MCP tools; bash and
file reads use Claude Code's own Bash/Read tools (permission_mode=bypassPermissions).

Escalations: ask_user BLOCKS inside the tool, polling the same question.json/reply.* mailbox
the bridge renders — no GraphInterrupt. A driver crash mid-wait resumes by re-running the node;
ask_user re-attaches to the unanswered question (or consumes an existing reply) idempotently.

Long tools (multi-hour train monitoring, 25-min eval shards) — MCP_TOOL_TIMEOUT is raised
accordingly. Backend selection lives in pipeline.agents.base.make_backend.
"""
from __future__ import annotations

import inspect
import json
import os
import time

import anyio
from claude_agent_sdk import ClaudeAgentOptions, create_sdk_mcp_server, query, tool

from pipeline import tools
from pipeline.agents.base import SYSTEM, tool_schema

_ASK_USER_SCHEMA = {
    "type": "object",
    "properties": {
        "question": {"type": "string"},
        "context": {"type": "string"},
        "options": {"type": "array", "items": {
            "type": "object",
            "properties": {"id": {"type": "integer"}, "label": {"type": "string"}},
            "required": ["id", "label"]}},
        "recommendation": {"type": "integer"},
    },
    "required": ["question"],
}
_COMPLETE_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["succeeded", "failed"]},
        "artifact_paths": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string"},
        "error": {"type": "string"},
        "notes": {"type": "string"},
    },
    "required": ["status"],
}


def _text(payload) -> dict:
    return {"content": [{"type": "text",
                         "text": payload if isinstance(payload, str)
                         else json.dumps(payload, default=str)}]}


class SDKAgent:
    def __init__(self, name: str, run_id: str, skills: dict, system_intro: str = "",
                 builtins: tuple = (), model: str | None = None,
                 max_iterations: int | None = None):
        self.name, self.run_id, self.skills = name, run_id, dict(skills)
        self.system_intro = system_intro
        self.model = (model or os.environ.get(f"EMBODIED_CLAW_MODEL_{name.upper()}")
                      or os.environ.get("EMBODIED_CLAW_MODEL"))
        self.max_iterations = max_iterations or int(os.environ.get("EMBODIED_CLAW_MAX_ITER", "60"))
        self.builtins = builtins
        self.last_escalation: dict | None = None
        self._node = ""
        self._result: dict | None = None

    # ------------------------------------------------------------------ tool implementations
    def memory_path(self):
        return tools.agents_root() / self.name / "memory.md"

    def _log(self, obj: dict) -> None:
        p = tools.run_dir(self.run_id) / "agent_messages" / f"{self._node}.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a") as f:
            f.write(json.dumps({"ts": tools.now(), "agent": self.name, **obj}, default=str) + "\n")

    def _ask_user_blocking(self, args: dict) -> str:
        """Same mailbox contract as pipeline spec section 7, minus the interrupt: write the
        question, wait for the bridge/CLI to drop a reply file, re-notify daily."""
        rd = tools.run_dir(self.run_id)
        esc_id = (tools.find_unanswered(rd, self._node)
                  or tools.new_escalation_id(self._node))
        q = tools.write_question(rd, esc_id, node=self._node, agent=self.name,
                                 question=args.get("question", ""),
                                 context=args.get("context", ""),
                                 options=args.get("options"),
                                 recommendation=args.get("recommendation"))
        reply = tools.read_reply(rd, esc_id)
        if reply is None:
            tools.log_transition(self.run_id, self._node, "escalated", esc_id)
            tools.notify(f"[{self.run_id}:{self._node}] {q['question']}")
            waited = 0.0
            while reply is None:
                time.sleep(15)
                waited += 15
                if waited >= 24 * 3600:
                    waited = 0.0
                    tools.notify(f"[{self.run_id}:{self._node}] still awaiting reply: "
                                 f"{q['question']}")
                reply = tools.read_reply(rd, esc_id)
        self.last_escalation = {"question": q["question"], "context": q.get("context", ""),
                                "user_reply": reply}
        tools.log_transition(self.run_id, self._node, "running", f"reply consumed for {esc_id}")
        if reply.get("type") == "option":
            label = next((o["label"] for o in q.get("options", [])
                          if o.get("id") == reply["option"]), "")
            return f"User chose option {reply['option']}: {label}"
        return f"User replied: {reply.get('message', '')}"

    # ------------------------------------------------------------------ MCP server assembly
    def _mcp_tools(self) -> list:
        out = []

        def make_skill_tool(skill_name: str, fn):
            desc = (inspect.getdoc(fn) or skill_name).split("\n\n")[0]

            @tool(skill_name, desc, tool_schema(fn))
            async def handler(args, _fn=fn, _name=skill_name):
                self._log({"tool": _name, "input": args})
                try:
                    result = await anyio.to_thread.run_sync(lambda: _fn(**args))
                except Exception as e:  # noqa: BLE001 — agent sees the failure and adapts
                    result = {"error": f"{type(e).__name__}: {e}"}
                self._log({"tool": _name, "result": result})
                return _text(result)
            return handler

        for n, fn in self.skills.items():
            out.append(make_skill_tool(n, fn))

        @tool("ask_user", "Escalate a question to the user and wait for their reply "
              "(options show as Slack buttons; always include a recommendation).",
              _ASK_USER_SCHEMA)
        async def ask_user(args):
            self._log({"tool": "ask_user", "input": args})
            reply = await anyio.to_thread.run_sync(lambda: self._ask_user_blocking(args))
            self._log({"tool": "ask_user", "result": reply})
            return _text(reply)

        @tool("complete_node", "Finish this node. REQUIRED as your final action.",
              _COMPLETE_SCHEMA)
        async def complete_node(args):
            self._result = {"status": args.get("status", "failed"),
                            "artifact_paths": args.get("artifact_paths") or [],
                            "summary": args.get("summary", ""),
                            "error": args.get("error"), "notes": args.get("notes")}
            self._log({"tool": "complete_node", "input": args})
            return _text("node recorded — you are done; do not call more tools")

        @tool("append_memory", "Append a durable lesson to your agent memory "
              "(survives across runs).", {"type": "object",
                                          "properties": {"lesson": {"type": "string"}},
                                          "required": ["lesson"]})
        async def append_memory(args):
            p = self.memory_path()
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "a") as f:
                f.write(f"- {args.get('lesson', '').strip()}\n")
            return _text("recorded")

        out += [ask_user, complete_node, append_memory]
        return out

    def _options(self) -> ClaudeAgentOptions:
        mem = self.memory_path()
        memory = mem.read_text() if mem.exists() else "(empty)"
        allowed = [f"mcp__skills__{n}" for n in self.skills] + [
            "mcp__skills__ask_user", "mcp__skills__complete_node", "mcp__skills__append_memory"]
        if "bash" in self.builtins:
            allowed.append("Bash")
        if "read_file" in self.builtins:
            allowed.append("Read")
        return ClaudeAgentOptions(
            system_prompt=SYSTEM.format(name=self.name, memory=memory)
            + "\n\n" + self.system_intro
            + "\n\nFinish by calling complete_node — a run without it counts as failed.",
            mcp_servers={"skills": create_sdk_mcp_server(
                name="skills", version="1.0.0", tools=self._mcp_tools())},
            allowed_tools=allowed,
            permission_mode="bypassPermissions",
            max_turns=self.max_iterations,
            model=self.model,
            cwd=str(tools.REPO),
            env={**os.environ, "MCP_TOOL_TIMEOUT": "172800000", "MCP_TIMEOUT": "120000"},
        )

    # ------------------------------------------------------------------ node runner
    async def _run(self, prompt_str: str) -> None:
        self._last_assistant_text = ""
        async for message in query(prompt=prompt_str, options=self._options()):
            kind = type(message).__name__
            if kind == "AssistantMessage":
                for block in getattr(message, "content", []) or []:
                    txt = getattr(block, "text", None)
                    if txt:
                        self._last_assistant_text = txt
                        self._log({"assistant": txt[:2000]})
            elif kind == "ResultMessage":
                self._log({"result_meta": {
                    "is_error": getattr(message, "is_error", None),
                    "num_turns": getattr(message, "num_turns", None),
                    "total_cost_usd": getattr(message, "total_cost_usd", None)}})

    def run_node(self, node_name: str, prompt_str: str) -> dict:
        """Subscription limit hits ('hit your session limit ... resets Xpm') wait and retry —
        an overnight eval must survive the quota window, not fail its node."""
        self._node, self._result = node_name, None
        deadline = time.time() + float(os.environ.get("EMBODIED_CLAW_SDK_LIMIT_WAIT_H", os.environ.get("EVAL_SDK_LIMIT_WAIT_H", "8"))) * 3600
        while True:
            try:
                anyio.run(self._run, prompt_str)
            except Exception as e:  # noqa: BLE001 — surface as node failure, never crash the graph
                hint = self._last_assistant_text[:300]
                limited = "session limit" in hint.lower() or "usage limit" in hint.lower()
                if limited and time.time() < deadline:
                    tools.log_transition(self.run_id, self._node, "running",
                                         f"claude usage limit hit ({hint}); retrying in 15 min")
                    tools.notify(f"[{self.run_id}:{self._node}] usage limit; waiting")
                    time.sleep(900)
                    continue
                return {"status": "failed", "artifact_paths": [],
                        "error": f"sdk agent error: {type(e).__name__}: {e}"
                                 + (f" | last assistant text: {hint}" if hint else "")}
            return self._result or {"status": "failed", "artifact_paths": [],
                                    "error": "agent finished without calling complete_node"}
