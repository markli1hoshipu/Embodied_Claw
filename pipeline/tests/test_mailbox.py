"""Mailbox file-format conformance — spec section 7 is a byte-level contract."""
import json
import re

from pipeline import inbox, tools


def test_escalation_id_format(env):
    eid = tools.new_escalation_id("filter_build")
    assert re.fullmatch(r"filter_build_\d{10}_[0-9a-f]{4}", eid)


def test_question_json_fields_and_idempotence(run_dir):
    eid = tools.new_escalation_id("filter_build")
    q = tools.write_question(run_dir, eid, node="filter_build", agent="data_agent",
                             question="PCA at p98 would drop episodes 102, 103, 137. OK?",
                             context="132 clips; p98=4.83 (drop 3)",
                             options=[{"id": 1, "label": "p98 (recommended)"},
                                      {"id": 2, "label": "skip"}],
                             recommendation=1)
    p = run_dir / "escalations" / f"{eid}.question.json"
    assert p.exists()
    data = json.loads(p.read_text())
    assert list(data) == ["node", "agent", "question", "context", "options",
                          "recommendation", "created_at"]
    assert data["node"] == "filter_build" and data["agent"] == "data_agent"
    assert data["options"][0] == {"id": 1, "label": "p98 (recommended)"}
    assert data["recommendation"] == 1
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", data["created_at"])
    # idempotent re-write: returns existing content, never re-posts
    q2 = tools.write_question(run_dir, eid, node="filter_build", agent="data_agent",
                              question="DIFFERENT")
    assert q2["question"] == q["question"]
    assert len(list((run_dir / "escalations").glob("*.question.json"))) == 1


def test_question_without_options_omits_keys(run_dir):
    eid = tools.new_escalation_id("ingest")
    q = tools.write_question(run_dir, eid, node="ingest", agent="data_agent", question="404?")
    assert "options" not in q and "recommendation" not in q


def test_parse_reply_integer_vs_freeform():
    assert tools.parse_reply("1") == {"type": "option", "option": 1}
    assert tools.parse_reply("  2 \n") == {"type": "option", "option": 2}
    assert tools.parse_reply("Use p98 but force-drop 280") == \
        {"type": "message", "message": "Use p98 but force-drop 280"}


def test_reply_prefix_match_any_extension(run_dir):
    eid = tools.new_escalation_id("train")
    tools.write_question(run_dir, eid, node="train", agent="training_agent", question="q")
    assert tools.read_reply(run_dir, eid) is None
    (run_dir / "escalations" / f"{eid}.reply.json").write_text("3")
    assert tools.read_reply(run_dir, eid) == {"type": "option", "option": 3}


def test_find_unanswered_filters_by_node_and_skips_replied(run_dir):
    e1 = tools.new_escalation_id("ingest")
    e2 = tools.new_escalation_id("filter_build")
    tools.write_question(run_dir, e1, node="ingest", agent="data_agent", question="a")
    tools.write_question(run_dir, e2, node="filter_build", agent="data_agent", question="b")
    assert tools.find_unanswered(run_dir, "filter_build") == e2
    (run_dir / "escalations" / f"{e2}.reply.txt").write_text("ok")
    assert tools.find_unanswered(run_dir, "filter_build") is None
    assert tools.find_unanswered(run_dir, "ingest") == e1


def test_inbox_listing_and_format(run_dir):
    eid = tools.new_escalation_id("filter_build")
    tools.write_question(run_dir, eid, node="filter_build", agent="data_agent",
                         question="thresh?", options=[{"id": 1, "label": "p98"}],
                         recommendation=1)
    entries = inbox.pending_escalations()
    assert len(entries) == 1 and entries[0]["run_id"] == "tr1"
    text = inbox.format_inbox(entries)
    assert "[tr1:filter_build]" in text and "thresh?" in text and "(recommended)" in text
    assert inbox.format_inbox([]) == "inbox empty — no pending escalations."


def test_write_reply_and_resolve(run_dir):
    eid = tools.new_escalation_id("upload_dataset")
    tools.write_question(run_dir, eid, node="upload_dataset", agent="hf_agent", question="name?")
    target = inbox.resolve_target(run_id="tr1", node="upload_dataset")
    assert target["escalation_id"] == eid
    p = inbox.write_reply("tr1", eid, "Hoshipu/b1k_new")
    assert p.read_text() == "Hoshipu/b1k_new"
    assert inbox.resolve_target(latest=True) is None  # now answered
