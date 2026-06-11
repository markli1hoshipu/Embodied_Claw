"""Escalation JSON -> Slack Block Kit (spec 13.7) + transition lines (13.3) + redaction (13.8)."""
from __future__ import annotations

import re

# Known token shapes (spec 13.8 + addenda): redacted from ANY text posted to Slack.
SECRET_PATTERNS = [
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"xoxb-[A-Za-z0-9-]{5,}"),
    re.compile(r"xapp-[A-Za-z0-9-]{5,}"),
    re.compile(r"ghp_[A-Za-z0-9]{10,}"),
    re.compile(r"sk-ant-[A-Za-z0-9_-]{5,}"),
]

STATUS_EMOJI = {
    "launched": "\U0001F680", "running": "▶️", "started": "▶️",
    "succeeded": "✅", "done": "\U0001F3C1", "escalated": "❓",
    "failed": "❌", "skipped": "⏭️",
}


def redact(text: str) -> str:
    for pat in SECRET_PATTERNS:
        text = pat.sub("[REDACTED]", text)
    return text


def redact_deep(obj):
    """Recursively redact every string inside a Block Kit structure."""
    if isinstance(obj, str):
        return redact(obj)
    if isinstance(obj, dict):
        return {k: redact_deep(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact_deep(v) for v in obj]
    return obj


def chunked(seq: list, n: int) -> list[list]:
    return [seq[i:i + n] for i in range(0, len(seq), n)]


def render_escalation(esc: dict, escalation_id: str) -> list[dict]:
    """Per spec 13.7: header, question, context, <=5-button action rows, free-form hint."""
    blocks: list[dict] = [
        {"type": "header", "text": {"type": "plain_text",
         "text": f"❓ {esc.get('node', '?')} — {esc.get('agent', '?')}"[:150]}},
        {"type": "section", "text": {"type": "mrkdwn", "text": esc.get("question", "(no question)")}},
    ]
    if esc.get("context"):
        blocks.append({"type": "context", "elements": [
            {"type": "mrkdwn", "text": f"_{esc['context']}_"}]})
    if esc.get("options"):
        buttons = []
        for opt in esc["options"]:
            label = str(opt.get("label", opt["id"]))[:75]  # Slack 75-char limit
            button = {
                "type": "button",
                "text": {"type": "plain_text", "text": label},
                "action_id": f"esc:{escalation_id}:opt:{opt['id']}",
                "value": str(opt["id"]),
            }
            if opt["id"] == esc.get("recommendation"):
                # truncate FIRST so the cue survives long labels; style is truncation-proof
                button["text"]["text"] = label[:71] + "  ✓"
                button["style"] = "primary"
            buttons.append(button)
        for chunk in chunked(buttons, 5):  # Slack actions block holds max 5 buttons
            blocks.append({"type": "actions", "elements": chunk})
    blocks.append({"type": "context", "elements": [
        {"type": "mrkdwn", "text": "_Or reply in this thread with a custom message._"}]})
    return redact_deep(blocks)


def render_transition(event: dict) -> str:
    """One status line per transitions.jsonl event (tolerates unknown/missing fields)."""
    status = str(event.get("status", "?"))
    emoji = STATUS_EMOJI.get(status.lower(), "ℹ️")
    line = f"{emoji} `{event.get('run_id', '?')}`"
    if event.get("node"):
        line += f" / {event['node']}"
    line += f" — {status}"
    if event.get("detail"):
        line += f" ({event['detail']})"
    return redact(line)
