"""CLI helpers for the escalation mailbox: list pending questions, write replies, poll."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from pipeline import tools


def pending_escalations(root: Path | None = None) -> list[dict]:
    """Every unanswered *.question.json across runs/*, newest first."""
    root = root or tools.runs_root()
    out = []
    for q in sorted(root.glob("*/escalations/*.question.json")) if root.is_dir() else ():
        eid = q.name[:-len(".question.json")]
        if list(q.parent.glob(f"{eid}.reply.*")):
            continue
        try:
            payload = json.loads(q.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        out.append({"run_id": q.parent.parent.name, "escalation_id": eid,
                    "mtime": q.stat().st_mtime, **payload})
    return sorted(out, key=lambda e: e["mtime"], reverse=True)


def resolve_target(run_id: str | None = None, node: str | None = None,
                   latest: bool = False) -> dict | None:
    """Pick the escalation a reply addresses: newest pending matching run_id/node, or newest overall."""
    for e in pending_escalations():
        if latest:
            return e
        if run_id and e["run_id"] != run_id:
            continue
        if node and e["node"] != node:
            continue
        return e
    return None


def write_reply(run_id: str, escalation_id: str, text: str) -> Path:
    """Atomic (tmp + rename): the driver/bridge poll this path and must never see a torn write.
    The dot-prefixed tmp name cannot match the <esc_id>.reply.* glob."""
    d = tools.esc_dir(tools.run_dir(run_id))
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{escalation_id}.reply.txt"
    tmp = d / f".{escalation_id}.tmp"
    tmp.write_text(text)
    os.replace(tmp, p)
    return p


def poll_for_reply(rd: Path, esc_id: str, interval: float = 30.0,
                   renotify_after: float = 24 * 3600) -> dict:
    """Block until <esc_id>.reply.* appears (spec 7.1 step 5: 30s poll; 7.4: re-notify at 24h,
    never auto-default — waits as long as needed)."""
    waited = 0.0
    while True:
        r = tools.read_reply(rd, esc_id)
        if r is not None:
            return r
        time.sleep(interval)
        waited += interval
        if waited >= renotify_after:
            waited = 0.0
            tools.notify(f"still waiting on escalation {esc_id} in {rd.name}")


def format_inbox(entries: list[dict]) -> str:
    if not entries:
        return "inbox empty — no pending escalations."
    lines = [f"{len(entries)} pending escalation(s):", ""]
    for e in entries:
        lines.append(f"[{e['run_id']}:{e['node']}] {e['escalation_id']}  ({e.get('created_at', '?')})")
        lines.append(f"  Q: {e['question']}")
        for o in e.get("options") or []:
            mark = "  (recommended)" if o.get("id") == e.get("recommendation") else ""
            lines.append(f"    {o['id']}. {o['label']}{mark}")
        lines.append(f"  reply: python -m pipeline.cli reply --run-id {e['run_id']} "
                     f"--node {e['node']} --option N | --message '...'")
        lines.append("")
    return "\n".join(lines).rstrip()
