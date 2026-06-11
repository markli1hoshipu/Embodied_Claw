"""render.py — Block Kit per spec 13.7, redaction per 13.8, transition lines per 13.3."""
from bridge.render import chunked, redact, render_escalation, render_transition

ESC = {
    "node": "filter_build", "agent": "data_agent",
    "question": "What threshold?", "context": "132 clips total",
    "options": [{"id": i, "label": f"option {i}"} for i in range(1, 8)],
    "recommendation": 2,
}


def test_blocks_structure():
    blocks = render_escalation(ESC, "filter_build_1_ab")
    assert blocks[0]["type"] == "header"
    assert "filter_build" in blocks[0]["text"]["text"]
    assert blocks[1]["type"] == "section"
    assert blocks[1]["text"]["text"] == "What threshold?"
    assert blocks[2]["type"] == "context"
    assert "custom message" in blocks[-1]["elements"][0]["text"]


def test_five_button_chunking():
    blocks = render_escalation(ESC, "x_1_a")
    actions = [b for b in blocks if b["type"] == "actions"]
    assert [len(a["elements"]) for a in actions] == [5, 2]  # 7 options -> 5 + 2


def test_recommendation_checkmark_and_action_ids():
    blocks = render_escalation(ESC, "x_1_a")
    buttons = [el for b in blocks if b["type"] == "actions" for el in b["elements"]]
    assert buttons[1]["text"]["text"].endswith("✓")
    assert not buttons[0]["text"]["text"].endswith("✓")
    assert buttons[0]["action_id"] == "esc:x_1_a:opt:1"
    assert buttons[0]["value"] == "1"


def test_label_75_char_limit():
    esc = dict(ESC, options=[{"id": 1, "label": "L" * 200}], recommendation=None)
    blocks = render_escalation(esc, "x_1_a")
    button = [b for b in blocks if b["type"] == "actions"][0]["elements"][0]
    assert len(button["text"]["text"]) == 75


def test_recommendation_cue_survives_long_labels():
    """Truncate-then-mark: a >=73-char recommended label must still end with the checkmark
    (and carry the truncation-proof primary style)."""
    esc = dict(ESC, options=[{"id": 1, "label": "L" * 200}, {"id": 2, "label": "short"}],
               recommendation=1)
    blocks = render_escalation(esc, "x_1_a")
    rec, other = [b for b in blocks if b["type"] == "actions"][0]["elements"]
    assert rec["text"]["text"].endswith("✓") and len(rec["text"]["text"]) <= 75
    assert rec.get("style") == "primary"
    assert not other["text"]["text"].endswith("✓") and "style" not in other


def test_no_options_no_actions_block():
    esc = {"node": "train", "agent": "training_agent", "question": "Loss diverged. Continue?"}
    blocks = render_escalation(esc, "train_1_a")
    assert not [b for b in blocks if b["type"] == "actions"]
    assert "custom message" in blocks[-1]["elements"][0]["text"]


def test_redaction_patterns():
    secret = "token hf_" + "A" * 24 + " and xoxb-1234-abcd and sk-ant-xyz123 and ghp_" + "b" * 12
    out = redact(secret)
    for marker in ("hf_A", "xoxb-1", "sk-ant-x", "ghp_b"):
        assert marker not in out
    assert out.count("[REDACTED]") == 4


def test_redaction_applied_inside_blocks():
    esc = dict(ESC, question="push with hf_" + "Z" * 30,
               options=[{"id": 1, "label": "use xapp-12345-secret"}], recommendation=None)
    blocks = render_escalation(esc, "x_1_a")
    flat = str(blocks)
    assert "hf_Z" not in flat and "xapp-12345" not in flat
    assert "[REDACTED]" in flat


def test_chunked():
    assert chunked([1, 2, 3], 2) == [[1, 2], [3]]
    assert chunked([], 5) == []


def test_render_transition_tolerates_unknown_fields():
    line = render_transition({"ts": "t", "run_id": "r1", "node": "ingest",
                              "status": "succeeded", "detail": "144 parquets",
                              "totally_new_field": {"x": 1}})
    assert "`r1`" in line and "ingest" in line and "succeeded" in line and "144 parquets" in line


def test_render_transition_minimal_and_redacted():
    line = render_transition({"status": "failed", "detail": "auth hf_" + "Q" * 25})
    assert "❌" in line and "hf_Q" not in line
