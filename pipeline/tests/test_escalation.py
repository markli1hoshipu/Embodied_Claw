"""Escalation pause/resume across simulated process restarts (DoD: file-drop reply resume;
re-entry never double-posts; concurrent node 3+4 escalations resolve independently)."""
import json

from langgraph.types import Command

from pipeline import cli, inbox, tools
from pipeline.agents import base
from pipeline.graph import build_graph, open_checkpointer, pending_interrupts
from pipeline.state import init_state
from pipeline.tests import fakes

ASK = [fakes.tool_use("ask_user", {
    "question": "PCA at p98 would drop episodes 102, 103, 137. OK?",
    "context": "cuts: p95=4.42 (drop 7), p98=4.83 (drop 3)",
    "options": [{"id": 1, "label": "p98 — drop 3 (recommended)"},
                {"id": 2, "label": "skip PCA"}],
    "recommendation": 1})]


def _fresh_graph(monkeypatch, script):
    fake = fakes.FakeAnthropic(script)
    monkeypatch.setattr(base, "get_client", lambda: fake)
    return build_graph(open_checkpointer("tr1")), {"configurable": {"thread_id": "tr1"}}, fake


def test_escalate_kill_filedrop_restart_resume(run_dir, monkeypatch):
    # process 1: run until filter_build escalates, then "die" (drop all objects)
    script = fakes.default_script(intake=fakes.intake_script("tr1"), filter_build=[ASK])
    graph, conf, _ = _fresh_graph(monkeypatch, script)
    result = graph.invoke(init_state("tr1"), conf, durability="sync")
    assert "__interrupt__" in result
    qs = list((run_dir / "escalations").glob("*.question.json"))
    assert len(qs) == 1
    q = json.loads(qs[0].read_text())
    assert q["node"] == "filter_build" and q["agent"] == "data_agent"
    trans = (run_dir / "transitions.jsonl").read_text()
    assert '"escalated"' in trans
    # user replies via plain file drop (spec 7.2 B)
    eid = qs[0].name[: -len(".question.json")]
    (run_dir / "escalations" / f"{eid}.reply.txt").write_text("1")
    # process 2: fresh graph + checkpointer + client — driver picks the reply up and resumes
    rest = fakes.default_script(filter_build=[fakes.done(paths=("/tmp/lerobot/ds",))])
    rest.pop("intake"), rest.pop("ingest")
    graph2, conf2, fake2 = _fresh_graph(monkeypatch, rest)
    assert pending_interrupts(graph2, conf2), "interrupt must survive restart via sqlite"
    values = cli.drive(graph2, conf2, None, "tr1", poll=0.01)
    assert values["filter_build"]["status"] == "succeeded"
    assert values["filter_build"]["escalation"]["user_reply"] == {"type": "option", "option": 1}
    assert values["upload_model"]["status"] == "succeeded"
    assert "intake" not in [c["node"] for c in fake2.calls]  # completed nodes never re-ran
    assert len(list((run_dir / "escalations").glob("*.question.json"))) == 1


def test_reentry_never_double_posts(run_dir, monkeypatch):
    script = fakes.default_script(intake=fakes.intake_script("tr1"), filter_build=[ASK])
    graph, conf, _ = _fresh_graph(monkeypatch, script)
    graph.invoke(init_state("tr1"), conf, durability="sync")
    assert len(list((run_dir / "escalations").glob("*.question.json"))) == 1
    # naive post-crash re-run with NO reply: node re-enters, re-attaches, re-interrupts
    graph2, conf2, fake2 = _fresh_graph(monkeypatch, {})
    result = graph2.invoke(None, conf2, durability="sync")
    assert "__interrupt__" in result
    assert len(list((run_dir / "escalations").glob("*.question.json"))) == 1  # no double-post
    assert fake2.calls == []  # replayed the pending tool call; no new model turn needed


def test_concurrent_escalations_resolve_independently(run_dir, monkeypatch):
    ask_u = [fakes.tool_use("ask_user", {"question": "dataset repo name?",
                                         "options": [{"id": 1, "label": "Hoshipu/b1k_x"}]})]
    ask_n = [fakes.tool_use("ask_user", {"question": "norm stats look odd — proceed?"})]
    script = fakes.default_script(intake=fakes.intake_script("tr1"),
                                  upload_dataset=[ask_u, fakes.done(paths=("hf://d",))],
                                  norm_stats=[ask_n, fakes.done(paths=("/n.json",))])
    graph, conf, _ = _fresh_graph(monkeypatch, script)
    result = graph.invoke(init_state("tr1"), conf, durability="sync")
    ints = result["__interrupt__"]
    assert len(ints) == 2
    by_node = {i.value["node"]: i for i in ints}
    assert set(by_node) == {"upload_dataset", "norm_stats"}
    # reply ONLY to norm_stats; resume per-interrupt-id — upload_dataset stays escalated
    inbox.write_reply("tr1", by_node["norm_stats"].value["escalation_id"], "yes proceed")
    r2 = graph.invoke(Command(resume={by_node["norm_stats"].id:
                                      tools.read_reply(run_dir,
                                                       by_node["norm_stats"].value["escalation_id"])}),
                      conf, durability="sync")
    vals = graph.get_state(conf).values
    assert vals["norm_stats"]["status"] == "succeeded"
    assert [i.value["node"] for i in r2["__interrupt__"]] == ["upload_dataset"]
    assert vals.get("train", {}).get("status") != "succeeded"  # join still waiting
    inbox.write_reply("tr1", by_node["upload_dataset"].value["escalation_id"], "1")
    values = cli.drive(graph, conf, None, "tr1", poll=0.01)
    assert values["upload_dataset"]["status"] == "succeeded"
    assert values["train"]["status"] == "succeeded"  # ran only after both branches resolved


def test_cli_reply_latest_routes_to_newest(run_dir, monkeypatch, capsys):
    eid = tools.new_escalation_id("filter_build")
    tools.write_question(run_dir, eid, node="filter_build", agent="data_agent", question="q?")
    cli.main(["reply", "--latest", "--message", "go"])
    assert tools.read_reply(run_dir, eid) == {"type": "message", "message": "go"}
    assert "reply recorded" in capsys.readouterr().out
