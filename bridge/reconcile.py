"""Rebuild bridge state after a crash / DB loss (spec 13.4 note + 13.9 + 13.11).

threads.sqlite is recoverable from two sources:
  1. Slack history — bot messages carry metadata {run_launched|escalation_posted}.
  2. Filesystem walk — *.question.json with a sibling *.reply.* means "resolved".
Runs at every bridge startup; safe to re-run (all inserts are INSERT OR IGNORE).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger("bridge.reconcile")


def _meta(msg: dict) -> tuple[str, dict]:
    meta = msg.get("metadata") or {}
    return meta.get("event_type", ""), meta.get("event_payload") or {}


def reconcile_from_slack(db, client, channel: str, max_pages: int = 3) -> None:
    """Scan channel history for our run/escalation anchor messages (13.9)."""
    cursor = None
    for _ in range(max_pages):
        try:
            resp = client.conversations_history(
                channel=channel, limit=200, include_all_metadata=True, cursor=cursor)
        except Exception as exc:
            log.warning("history scan failed (%s); relying on FS walk only", exc)
            return
        for msg in resp.get("messages", []):
            etype, payload = _meta(msg)
            if etype == "run_launched" and payload.get("run_id"):
                db.insert_run(payload["run_id"], msg.get("thread_ts") or msg["ts"], channel)
            if msg.get("reply_count"):
                _scan_thread(db, client, channel, msg["ts"])
        cursor = (resp.get("response_metadata") or {}).get("next_cursor")
        if not cursor:
            return


def _scan_thread(db, client, channel: str, thread_ts: str) -> None:
    try:
        resp = client.conversations_replies(
            channel=channel, ts=thread_ts, limit=200, include_all_metadata=True)
    except Exception as exc:
        log.warning("replies scan failed for %s: %s", thread_ts, exc)
        return
    for msg in resp.get("messages", []):
        etype, payload = _meta(msg)
        if etype == "run_launched" and payload.get("run_id"):
            db.insert_run(payload["run_id"], thread_ts, channel)
        elif etype == "escalation_posted" and payload.get("escalation_id"):
            run_id = payload.get("run_id", "")
            if run_id and not db.get_run(run_id):
                db.insert_run(run_id, thread_ts, channel)
            db.insert_escalation(payload["escalation_id"], run_id, msg["ts"])


def reconcile_from_fs(db, runs_root: Path) -> None:
    """Mark escalations resolved when a reply file already exists; refresh run status."""
    for qfile in sorted(runs_root.glob("*/escalations/*.question.json")):
        esc_id = qfile.name.removesuffix(".question.json")
        esc = db.get_escalation(esc_id)
        if esc is None or esc["resolved_at"]:
            continue  # unknown -> bridge initial_scan posts it; resolved -> done
        for reply in qfile.parent.glob(f"{esc_id}.reply.*"):
            try:
                payload = reply.read_text().strip()
            except OSError:
                payload = ""
            db.mark_resolved(esc_id, "file_drop", payload)
            db.set_run_status(esc["run_id"], "running")
            break
    for tfile in runs_root.glob("*/transitions.jsonl"):
        run_id = tfile.parent.name
        if not db.get_run(run_id):
            continue
        last = _last_json_line(tfile)
        status = str((last or {}).get("status", "")).lower()
        if status == "failed":
            db.set_run_status(run_id, "failed")
        elif status == "done":
            db.set_run_status(run_id, "done")


def _last_json_line(path: Path) -> dict | None:
    try:
        lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
        return json.loads(lines[-1]) if lines else None
    except (OSError, json.JSONDecodeError):
        return None


def reconcile(db, client, runs_root: Path, requests_channel: str) -> None:
    reconcile_from_slack(db, client, requests_channel)
    reconcile_from_fs(db, runs_root)
    log.info("reconcile complete")
