"""Inbound routing (spec 13.5) + filesystem mailbox handling (13.6) + security (13.8).

Slack -> filesystem: mention -> request.txt + detached `python -m pipeline.cli run`;
thread message / button click -> <escalation_id>.reply.txt.
Filesystem -> Slack: question.json -> Block Kit thread post; reply file -> ack;
transitions.jsonl line -> status channel broadcast.
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .config import BridgeConfig
from .render import redact, render_escalation, render_transition
from .threads_db import ThreadsDB

log = logging.getLogger("bridge.routes")
MENTION_RE = re.compile(r"<@[A-Z0-9][A-Z0-9._|-]*>", re.IGNORECASE)
ACTION_RE = re.compile(r"^esc:(?P<esc_id>[^:]+):opt:(?P<opt>.+)$")


class Router:
    def __init__(self, cfg: BridgeConfig, db: ThreadsDB, gateway, spawn=None, spawn_eval=None):
        self.cfg, self.db, self.gateway = cfg, db, gateway
        self.spawn = spawn or self._spawn_pipeline
        self.spawn_eval = spawn_eval or self._spawn_eval

    # ---------------- Slack-originated events ----------------
    def handle_event(self, event: dict) -> None:
        etype = event.get("type")
        if etype == "app_mention":
            self.handle_app_mention(event)
        elif etype == "message":
            self.handle_message(event)

    def handle_app_mention(self, event: dict) -> None:
        channel, user, ts = event.get("channel"), event.get("user"), event.get("ts")
        thread_ts = event.get("thread_ts")
        if thread_ts and thread_ts != ts and self._operator_dir(thread_ts).is_dir():
            return  # mention inside an operator thread -> handle_message routes it
        if thread_ts and thread_ts != ts and self.db.run_for_thread(thread_ts):
            return  # mention inside a run thread -> handled as a thread reply by handle_message
        is_eval = self.cfg.eval_channel is not None and channel == self.cfg.eval_channel
        if channel != self.cfg.requests_channel and not is_eval:  # 13.9: wrong channel -> redirect
            where = f"<#{self.cfg.requests_channel}>" + (
                f" (training) or <#{self.cfg.eval_channel}> (evals)" if self.cfg.eval_channel
                else "")
            self.gateway.post_message(
                channel, f"Please post run requests in {where}.", thread_ts=ts)
            return
        if self.cfg.allowed_users is not None and user not in self.cfg.allowed_users:
            self.gateway.post_message(
                channel, f"Sorry <@{user}>, you're not on the allowed-user list for "
                "launching runs (config/slack.toml).", thread_ts=ts)
            return
        text = MENTION_RE.sub("", event.get("text", "")).strip()
        if not text:
            self.gateway.post_message(
                channel, "Tell me what to run, e.g. `@pi05-bot train pi0.5 on task-0 ...`",
                thread_ts=ts)
            return
        # Operator mode for BOTH channels: the mention starts a conversational operator
        # session for this thread; the operator launches FSM runs as its tool after
        # confirming in chat. Channel decides which operator owns the thread.
        module = "eval_domino.operator" if is_eval else "pipeline.operator"
        self._operator_deliver(ts, channel, user, text, greet=True, module=module)

    def handle_message(self, event: dict) -> None:
        if event.get("bot_id") or event.get("subtype") == "bot_message":
            return
        if event.get("subtype") == "message_changed":  # 13.9: log only, never re-record
            log.info("message edited in %s; recorded reply unchanged", event.get("channel"))
            return
        if event.get("subtype"):
            return
        thread_ts = event.get("thread_ts")
        if not thread_ts or thread_ts == event.get("ts"):
            return
        if self._operator_dir(thread_ts).is_dir():  # operator thread: all free text -> operator
            text = MENTION_RE.sub("", event.get("text", "")).strip()
            if text:
                self._operator_deliver(thread_ts, event.get("channel"), event.get("user"), text)
            return
        run = self.db.run_for_thread(thread_ts)
        if run is None:
            return  # 13.9: unknown thread -> ignore
        text_raw = MENTION_RE.sub("", event.get("text", "")).strip().lower()
        if text_raw in ("cancel", "stop", "abort"):  # cooperative cancel (eval shard boundaries)
            (self.cfg.runs_root / run["run_id"] / "CANCEL").touch()
            self.db.set_run_status(run["run_id"], "cancelling")
            self.gateway.post_message(
                event.get("channel"),
                f"🛑 Cancel recorded for `{run['run_id']}` — stopping at the next safe point "
                "(partial results will be reported).", thread_ts=thread_ts)
            return
        open_escs = self.db.unresolved(run["run_id"])
        if not open_escs:
            log.warning("in-thread reply with no open escalation: %s", run["run_id"])
            return
        if len(open_escs) > 1:  # nodes 3+4 can be escalated concurrently — never guess which
            names = ", ".join(f"`{e['escalation_id'].rsplit('_', 2)[0]}`" for e in open_escs)
            self.gateway.post_ephemeral(
                event.get("channel"), event.get("user"), thread_ts=thread_ts,
                text=f"{len(open_escs)} questions are open for this run ({names}). "
                     "Use the buttons on the question you mean — free-text replies are only "
                     "auto-routed when exactly one question is open.")
            return
        text = MENTION_RE.sub("", event.get("text", "")).strip()
        self._resolve(open_escs[0], run, text, "slack_text",
                      channel=event.get("channel"), user=event.get("user"))

    def handle_block_actions(self, payload: dict) -> None:
        action = (payload.get("actions") or [{}])[0]
        match = ACTION_RE.match(action.get("action_id", ""))
        channel = (payload.get("channel") or {}).get("id")
        user = (payload.get("user") or {}).get("id")
        if not match:
            return
        esc = self.db.get_escalation(match["esc_id"])
        if esc is None:
            log.warning("button click for unknown escalation %s", match["esc_id"])
            return
        run = self.db.get_run(esc["run_id"])
        label = ((action.get("text") or {}).get("text") or f"option {match['opt']}").strip()
        self._resolve(esc, run, match["opt"], "slack_button",
                      channel=channel, user=user, label=label)

    def _resolve(self, esc, run, payload: str, method: str,
                 channel: str | None, user: str | None, label: str | None = None) -> None:
        """Write the reply file unless someone (CLI/file drop) beat us to it (13.9 race)."""
        run_id, esc_id = esc["run_id"], esc["escalation_id"]
        thread_ts = run["slack_thread_ts"] if run else None
        if (self.cfg.allowed_users is not None and user is not None
                and user not in self.cfg.allowed_users):  # 13.8: gate escalation answers too
            if channel:
                self.gateway.post_ephemeral(
                    channel, user, thread_ts=thread_ts,
                    text="Sorry, you're not on the allowed-user list (config/slack.toml) "
                         "for answering escalations.")
            return
        channel = (run["slack_channel_id"] if run else None) or channel
        existing = self._existing_reply(run_id, esc_id)
        if esc["resolved_at"] or existing is not None:
            if existing is not None:
                self.db.mark_resolved(esc_id, "file_drop", existing)
            if channel and user:
                self.gateway.post_ephemeral(
                    channel, user, thread_ts=thread_ts,
                    text=f"This escalation (`{esc_id}`) was already answered via "
                         f"{esc['reply_method'] or 'CLI/file drop'}.")
            return
        self._write_reply_file(run_id, esc_id, payload)
        self.db.mark_resolved(esc_id, method, payload)
        self.db.set_run_status(run_id, "running")
        shown = label or payload
        self.gateway.post_message(
            channel, f"✅ Reply recorded: {redact(shown)}. Continuing `{esc['escalation_id'].rsplit('_', 2)[0]}`.",
            thread_ts=thread_ts)

    # ---------------- filesystem-originated events ----------------
    def handle_question_file(self, path: Path) -> None:
        esc_id = path.name.removesuffix(".question.json")
        run_id = path.parent.parent.name
        if self.db.get_escalation(esc_id):
            return  # already posted (dedup across restarts / double FS events)
        esc = self._read_json(path)
        if esc is None:
            return
        channel, thread_ts = self._ensure_thread(run_id)
        msg_ts = self.gateway.post_message(
            channel, redact(f"❓ {esc.get('node', '?')} escalation: {esc.get('question', '')}"),
            blocks=render_escalation(esc, esc_id), thread_ts=thread_ts,
            metadata={"event_type": "escalation_posted",
                      "event_payload": {"run_id": run_id, "escalation_id": esc_id}})
        self.db.insert_escalation(esc_id, run_id, msg_ts, posted_at=esc.get("created_at"))
        self.db.set_run_status(run_id, "escalated")

    def handle_reply_file(self, path: Path) -> None:
        """A *.reply.* appeared. If we didn't write it (CLI/file drop), ack in thread (13.6)."""
        esc_id = path.name.split(".reply")[0]
        esc = self.db.get_escalation(esc_id)
        if esc is None or esc["resolved_at"]:
            return  # unknown escalation, or our own write / already acked
        try:
            payload = path.read_text().strip()
        except OSError:
            payload = ""
        self.db.mark_resolved(esc_id, "file_drop", payload)
        self.db.set_run_status(esc["run_id"], "running")
        run = self.db.get_run(esc["run_id"])
        if run:
            self.gateway.post_message(
                run["slack_channel_id"],
                f"\U0001F4E8 Got a reply via CLI/file drop: {redact(payload[:300]) or '(empty)'}. Continuing.",
                thread_ts=run["slack_thread_ts"])

    def handle_transition(self, event: dict) -> None:
        run_id = event.get("run_id")
        status = str(event.get("status", "")).lower()
        if self.cfg.status_channel:
            self.gateway.post_message(self.cfg.status_channel, render_transition(event))
        if run_id and (status == "failed"
                       or (event.get("node") == "aggregate_report" and status == "succeeded")
                       or (event.get("node") == "run" and status == "done")):
            self.notify_operator_of_run_event(run_id, event)
        if run_id and self.db.get_run(run_id):
            if status == "failed":
                self.db.set_run_status(run_id, "failed")
            elif status == "escalated":
                self.db.set_run_status(run_id, "escalated")
            elif status == "done" or (status == "succeeded" and event.get("node") in (None, "run", "pipeline")):
                self.db.set_run_status(run_id, "done")

    # ---------------- operator threads ----------------
    def _operator_dir(self, thread_ts: str) -> Path:
        return self.cfg.runs_root / "_operator" / str(thread_ts)

    def _operator_deliver(self, thread_ts: str, channel: str | None, user: str | None,
                          text: str, greet: bool = False, module: str | None = None) -> None:
        """Drop a message into the thread's operator inbox and spawn the handler process.
        The handler resumes the thread's SDK session and writes its reply to outbox/.
        `module` (recorded once, at thread creation) picks which operator owns the thread:
        eval_domino.operator (eval channel) or pipeline.operator (training channel)."""
        d = self._operator_dir(thread_ts)
        (d / "inbox").mkdir(parents=True, exist_ok=True)
        (d / "outbox").mkdir(parents=True, exist_ok=True)
        mod_p = d / ".module"
        if not mod_p.exists():
            mod_p.write_text(module or "eval_domino.operator")
        meta = d / "meta.json"
        if not meta.exists():
            meta.write_text(json.dumps({"channel": channel, "opened_by": user,
                                        "opened_at": datetime.now(timezone.utc).isoformat()}))
        body = (f"<@{user}>: {text}" if user else text)
        p = d / "inbox" / f"{time.time_ns()}.txt"
        tmp = d / "inbox" / f".{time.time_ns()}.tmp"
        tmp.write_text(body + "\n")
        os.replace(tmp, p)
        if greet and channel and user:
            self.gateway.post_ephemeral(channel, user, thread_ts=thread_ts,
                                        text="On it — thinking…")
        self._spawn_operator(str(thread_ts))

    def _spawn_operator(self, thread_ts: str) -> None:
        mod_p = self._operator_dir(thread_ts) / ".module"
        module = mod_p.read_text().strip() if mod_p.exists() else "eval_domino.operator"
        subprocess.Popen(
            [sys.executable, "-m", module, "handle", thread_ts],
            cwd=str(self.cfg.repo_root), stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, start_new_session=True)

    def handle_operator_out(self, path: Path) -> None:
        """runs/_operator/<thread>/outbox/*.md -> post to the thread, then mark consumed."""
        if not path.exists() or path.suffix != ".md":
            return
        thread_ts = path.parent.parent.name
        meta_p = path.parent.parent / "meta.json"
        channel = None
        if meta_p.exists():
            channel = self._read_json(meta_p).get("channel") if self._read_json(meta_p) else None
        channel = channel or self.cfg.eval_channel or self.cfg.requests_channel
        try:
            text = path.read_text()
        except OSError:
            return
        self.gateway.post_message(channel, redact(text), thread_ts=thread_ts)
        path.rename(path.with_suffix(".posted"))

    def notify_operator_of_run_event(self, run_id: str, event: dict) -> None:
        """Terminal run events flow back into the originating operator thread so the operator
        posts a human summary (results, failure diagnosis)."""
        marker = self.cfg.runs_root / run_id / ".operator_thread"
        if not marker.exists():
            return
        thread_ts = marker.read_text().strip()
        if not self._operator_dir(thread_ts).is_dir():
            return
        self._operator_deliver(
            thread_ts, None, None,
            f"[system notification — not a user message] run `{run_id}` event: "
            f"{json.dumps(event)}. If terminal, read eval_results and post a concise summary; "
            f"if it failed, diagnose briefly and say what you'll do.")

    # ---------------- helpers ----------------
    def _existing_reply(self, run_id: str, esc_id: str) -> str | None:
        for p in (self.cfg.runs_root / run_id / "escalations").glob(f"{esc_id}.reply.*"):
            try:
                return p.read_text().strip()
            except OSError:
                return ""
        return None

    def _write_reply_file(self, run_id: str, esc_id: str, payload: str) -> Path:
        esc_dir = self.cfg.runs_root / run_id / "escalations"
        esc_dir.mkdir(parents=True, exist_ok=True)
        final = esc_dir / f"{esc_id}.reply.txt"
        tmp = esc_dir / f".{esc_id}.tmp"  # tmp name must NOT match *.reply.* (pipeline glob)
        tmp.write_text(payload + "\n")
        os.replace(tmp, final)  # atomic: pipeline never sees a partial reply
        return final

    def _ensure_thread(self, run_id: str) -> tuple[str, str]:
        """Runs launched outside Slack (CLI) get a thread anchor on first contact."""
        run = self.db.get_run(run_id)
        if run:
            return run["slack_channel_id"], run["slack_thread_ts"]
        ts = self.gateway.post_message(
            self.cfg.requests_channel,
            f"\U0001F4CC Tracking run `{run_id}` (launched outside Slack). "
            "Escalations will appear in this thread.",
            metadata={"event_type": "run_launched", "event_payload": {"run_id": run_id}})
        self.db.insert_run(run_id, ts, self.cfg.requests_channel)
        return self.cfg.requests_channel, ts

    def _propose_run_id(self, text: str) -> str:
        words = re.findall(r"[a-z0-9]+", text.lower())
        slug = "_".join(words[:4])[:40] or "run"
        run_id = f"{slug}_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}"
        while (self.cfg.runs_root / run_id).exists():
            run_id = f"{slug}_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:4]}"
        return run_id

    def _spawn_pipeline(self, run_id: str) -> None:
        # Subprocess only — the bridge never imports pipeline code (spec 13.1).
        subprocess.Popen(
            [sys.executable, "-m", "pipeline.cli", "run", "--detach", run_id],
            cwd=str(self.cfg.repo_root), stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, start_new_session=True)

    def _spawn_eval(self, run_id: str) -> None:
        subprocess.Popen(
            [sys.executable, "-m", "eval_domino.cli", "run", "--detach", run_id],
            cwd=str(self.cfg.repo_root), stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, start_new_session=True)

    @staticmethod
    def _read_json(path: Path) -> dict | None:
        for _ in range(3):  # tolerate watcher firing mid-write
            try:
                return json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                time.sleep(0.2)
        log.warning("unreadable question file: %s", path)
        return None
