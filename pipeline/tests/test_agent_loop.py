"""Agent tool-use loop: dispatch, complete_node, errors, guards, transcripts, caching, memory."""
import json

from pipeline import tools
from pipeline.agents.base import Agent, get_client, tool_schema
from pipeline.tests.fakes import FakeAnthropic, done, tool_use


def make(run_dir, script, skills=None, **kw):
    fake = FakeAnthropic(script)
    a = Agent("data_agent", "tr1", skills or {}, "intro", client=fake, **kw)
    return a, fake


def test_no_api_key_clear_error(env, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    try:
        get_client()
        raise AssertionError("expected RuntimeError")
    except RuntimeError as e:
        assert "ANTHROPIC_API_KEY" in str(e)


def test_dispatch_and_complete(run_dir):
    a, fake = make(run_dir, {"x": [[tool_use("echo", {"v": 7})],
                                   done(paths=("/p",), summary="all good")]},
                   skills={"echo": lambda v: {"got": v}})
    out = a.run_node("x", "do the thing")
    assert out["status"] == "succeeded" and out["artifact_paths"] == ["/p"]
    # tool_result round-tripped into the conversation
    blob = json.dumps(a.conv["messages"])
    assert '"got": 7' in blob.replace('\\"', '"') or "'got': 7" in blob or "got" in blob
    # transcript jsonl per node
    tr = (run_dir / "agent_messages" / "x.jsonl").read_text().splitlines()
    events = [json.loads(l)["event"] for l in tr]
    assert "node_start" in events and "assistant" in events and "tool_results" in events


def test_skill_exception_becomes_is_error_result(run_dir):
    def boom(**kw):
        raise ValueError("bad input")
    a, _ = make(run_dir, {"x": [[tool_use("boom", {})], done()]}, skills={"boom": boom})
    assert a.run_node("x", "p")["status"] == "succeeded"
    results = [b for m in a.conv["messages"] if m["role"] == "user" and isinstance(m["content"], list)
               for b in m["content"] if isinstance(b, dict) and b.get("type") == "tool_result"]
    assert any(r.get("is_error") and "bad input" in r["content"] for r in results)


def test_max_iterations_guard(run_dir):
    script = {"x": [[tool_use("noop", {})] for _ in range(5)]}
    a, _ = make(run_dir, script, skills={"noop": lambda: "ok"}, max_iterations=4)
    out = a.run_node("x", "p")
    assert out["status"] == "failed" and "max iterations" in out["error"]


def test_nudge_then_complete_and_double_endturn_fails(run_dir):
    a, _ = make(run_dir, {"x": [[{"type": "text", "text": "musing"}], done()]})
    assert a.run_node("x", "p")["status"] == "succeeded"
    assert any("without calling complete_node" in json.dumps(m) for m in a.conv["messages"])
    b, _ = make(run_dir, {"y": [[{"type": "text", "text": "a"}], [{"type": "text", "text": "b"}]]})
    out = b.run_node("y", "p")
    assert out["status"] == "failed" and "twice" in out["error"]


def test_system_caching_memory_and_request_shape(run_dir):
    a, fake = make(run_dir, {"x": [done()]})
    a.run_node("x", "p")
    kw = fake.calls[0]["kwargs"]
    assert kw["model"] == "claude-sonnet-4-6"
    assert kw["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "Hoshipu/example-data_agent" in kw["system"][0]["text"]  # memory.md inlined
    assert kw["tools"][-1]["cache_control"] == {"type": "ephemeral"}
    assert kw["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}
    assert kw["thinking"] == {"type": "adaptive"}
    names = [t["name"] for t in kw["tools"]]
    assert {"ask_user", "complete_node", "append_memory"} <= set(names)


def test_thinking_blocks_tolerated(run_dir):
    script = {"x": [[{"type": "thinking", "thinking": "hmm", "signature": "s"},
                     tool_use("echo", {"v": 1})], done()]}
    a, _ = make(run_dir, script, skills={"echo": lambda v: v})
    assert a.run_node("x", "p")["status"] == "succeeded"


def test_append_memory_tool(run_dir):
    a, _ = make(run_dir, {"x": [[tool_use("append_memory", {"text": "- learned: prefer p98"})],
                                done()]})
    a.run_node("x", "p")
    assert "- learned: prefer p98" in a.memory_path().read_text()


def test_run_scoped_context_spans_nodes(run_dir):
    a, fake = make(run_dir, {"n1": [done()], "n2": [done()]})
    a.run_node("n1", "first")
    a2 = Agent("data_agent", "tr1", {}, "intro", client=fake)  # fresh instance, same disk conv
    a2.run_node("n2", "second")
    text = json.dumps(a2.conv["messages"])
    assert "## NODE: n1" in text and "## NODE: n2" in text
    assert a2.conv["started_nodes"] == ["n1", "n2"]


def test_tool_schema_from_signature():
    def f(repo_id: str, allow_patterns: list = None, max_workers: int = 8,
          thresh: float = 0.5, flag: bool = False):
        pass
    s = tool_schema(f)
    assert s["required"] == ["repo_id"]
    assert s["properties"]["repo_id"]["type"] == "string"
    assert s["properties"]["allow_patterns"]["type"] == "array"
    assert s["properties"]["max_workers"]["type"] == "integer"
    assert s["properties"]["thresh"]["type"] == "number"
    assert s["properties"]["flag"]["type"] == "boolean"


def test_model_env_override(run_dir, monkeypatch):
    monkeypatch.setenv("EMBODIED_CLAW_MODEL_DATA_AGENT", "claude-opus-4-8")
    a, fake = make(run_dir, {"x": [done()]})
    a.run_node("x", "p")
    assert fake.calls[0]["kwargs"]["model"] == "claude-opus-4-8"


def test_escalated_transition_logged_on_ask_user(run_dir, monkeypatch, notify_calls):
    """ask_user writes question.json + escalated transition + notify before interrupting
    (spec 7.1 steps 3-4). Outside a langgraph runtime interrupt() raises a context error
    instead of GraphInterrupt — either way it must PROPAGATE (never swallowed); in-graph
    pause is covered in test_escalation. notify is mocked by the autouse conftest fixture."""
    import pytest
    a, _ = make(run_dir, {"x": [[tool_use("ask_user", {"question": "which repo?"})]]})
    with pytest.raises(Exception):
        a.run_node("x", "p")
    qs = list((run_dir / "escalations").glob("*.question.json"))
    assert len(qs) == 1 and json.loads(qs[0].read_text())["question"] == "which repo?"
    trans = [json.loads(l) for l in (run_dir / "transitions.jsonl").read_text().splitlines()]
    assert any(t["status"] == "escalated" and t["node"] == "x" for t in trans)
    assert any("which repo?" in c for c in notify_calls)  # 7.1 step 4: notification fired
    # reply pre-dropped -> replay consumes it without interrupting again
    eid = qs[0].name[: -len(".question.json")]
    (run_dir / "escalations" / f"{eid}.reply.txt").write_text("use the second one")
    a2 = Agent("data_agent", "tr1", {}, "intro",
               client=FakeAnthropic({"x": [done()]}))
    out = a2.run_node("x", "p")
    assert out["status"] == "succeeded"
    assert a2.last_escalation["user_reply"] == {"type": "message", "message": "use the second one"}
