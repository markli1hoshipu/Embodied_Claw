"""Main bridge service: Socket Mode listener + watchdog FS watcher (spec 13.1, 13.6, 13.9).

All Slack/socket/watchdog threads only enqueue; one main loop dispatches, so DB and
mailbox writes are serialized. Outbound posts honor 429 Retry-After and are queued
in memory while Slack is unreachable.
"""
from __future__ import annotations

import json
import logging
import queue
import threading
import time
from collections import deque
from pathlib import Path

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from slack_sdk.socket_mode import SocketModeClient
from slack_sdk.socket_mode.response import SocketModeResponse
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from .render import redact_deep
from .routes import Router

log = logging.getLogger("bridge")


class SlackGateway:
    """All outbound Slack traffic. Redaction choke point (13.8) + rate limits (13.9)."""

    def __init__(self, client: WebClient, max_retries: int = 5):
        self.client = client
        self.max_retries = max_retries
        self.pending: deque = deque()  # (method, kwargs) queued while Slack unreachable

    def post_message(self, channel, text, blocks=None, thread_ts=None, metadata=None) -> str:
        kwargs = {"channel": channel, "text": redact_deep(text)}
        if blocks:
            kwargs["blocks"] = redact_deep(blocks)
        if thread_ts:
            kwargs["thread_ts"] = thread_ts
        if metadata:
            kwargs["metadata"] = redact_deep(metadata)  # run_id slugs derive from raw user text
        resp = self._call("chat_postMessage", **kwargs)
        return resp["ts"] if resp else ""

    def post_ephemeral(self, channel, user, text, thread_ts=None) -> None:
        kwargs = {"channel": channel, "user": user, "text": redact_deep(text)}
        if thread_ts:
            kwargs["thread_ts"] = thread_ts
        self._call("chat_postEphemeral", **kwargs)

    def _call(self, method: str, **kwargs):
        for attempt in range(self.max_retries):
            try:
                return getattr(self.client, method)(**kwargs)
            except SlackApiError as exc:
                status = getattr(exc.response, "status_code", None)
                if status == 429:  # honor Retry-After (13.9)
                    delay = int(exc.response.headers.get("Retry-After", 1) or 1)
                    log.warning("rate limited; retrying %s in %ss", method, delay)
                    time.sleep(min(delay, 120))
                    continue
                log.error("slack api error on %s: %s", method, exc)
                return None
            except Exception as exc:  # network down -> queue for later (13.9)
                log.warning("slack unreachable (%s); queueing %s", exc, method)
                self.pending.append((method, kwargs))
                return None
        log.error("gave up on %s after %d retries", method, self.max_retries)
        return None

    def flush_pending(self) -> None:
        for _ in range(len(self.pending)):
            method, kwargs = self.pending.popleft()
            self._call(method, **kwargs)


def classify_path(path: str) -> str | None:
    p = Path(path)
    if p.parent.name == "escalations" and p.name.endswith(".question.json"):
        return "question"
    if p.parent.name == "escalations" and ".reply." in p.name and not p.name.startswith("."):
        return "reply"
    if p.name == "transitions.jsonl":
        return "transitions"
    if (p.parent.name == "outbox" and p.suffix == ".md"
            and p.parent.parent.parent.name == "_operator"):
        return "operator_out"   # runs/_operator/<thread_ts>/outbox/*.md -> post to thread
    return None


class MailboxHandler(FileSystemEventHandler):
    """Watchdog handler: classifies mailbox paths and enqueues work items."""

    def __init__(self, events: queue.Queue):
        self.events = events

    def _enqueue(self, path: str) -> None:
        kind = classify_path(path)
        if kind:
            self.events.put((f"fs_{kind}", Path(path)))

    def on_created(self, event):
        if not event.is_directory:
            self._enqueue(event.src_path)

    def on_moved(self, event):
        if not event.is_directory:
            self._enqueue(event.dest_path)

    def on_modified(self, event):
        if not event.is_directory and classify_path(event.src_path) == "transitions":
            self.events.put(("fs_transitions", Path(event.src_path)))


class Bridge:
    def __init__(self, cfg, db, gateway: SlackGateway, router: Router,
                 socket_client: SocketModeClient | None = None):
        self.cfg, self.db, self.gateway, self.router = cfg, db, gateway, router
        self.socket_client = socket_client
        self.events: queue.Queue = queue.Queue()
        self.stop_event = threading.Event()
        self._offsets: dict[Path, int] = {}  # transitions.jsonl read positions
        self._backoff = 1
        if socket_client is not None:
            socket_client.socket_mode_request_listeners.append(self._on_socket_request)

    # ---- Socket Mode (listener thread: ack + enqueue only) ----
    def _on_socket_request(self, client, req) -> None:
        client.send_socket_mode_response(SocketModeResponse(envelope_id=req.envelope_id))
        if req.type == "events_api":
            self.events.put(("slack_event", req.payload.get("event", {})))
        elif req.type == "interactive" and req.payload.get("type") == "block_actions":
            self.events.put(("block_actions", req.payload))

    def _ensure_connected(self) -> None:
        if self.socket_client is None or self.socket_client.is_connected():
            self._backoff = 1
            return
        try:  # reconnect with exponential backoff (13.9)
            log.info("socket disconnected; reconnecting (backoff %ss)", self._backoff)
            self.socket_client.connect()
            self._backoff = 1
        except Exception as exc:
            log.warning("reconnect failed: %s", exc)
            time.sleep(self._backoff)
            self._backoff = min(self._backoff * 2, 60)

    # ---- startup catch-up ----
    def initial_scan(self) -> None:
        """Queue unseen questions/replies; skip transition history (reconcile covers state)."""
        for p in sorted(self.cfg.runs_root.glob("*/escalations/*.question.json")):
            self.events.put(("fs_question", p))
        for p in sorted(self.cfg.runs_root.glob("*/escalations/*.reply.*")):
            self.events.put(("fs_reply", p))
        for p in sorted(self.cfg.runs_root.glob("_operator/*/outbox/*.md")):
            self.events.put(("fs_operator_out", p))
        for p in self.cfg.runs_root.glob("*/transitions.jsonl"):
            self._offsets[p] = p.stat().st_size  # only broadcast lines written from now on

    # ---- transitions tail ----
    def drain_transitions(self, path: Path) -> None:
        offset = self._offsets.get(path, 0)
        try:
            with open(path, "r") as fh:
                fh.seek(offset)
                chunk = fh.read()
        except OSError:
            return
        consumed = 0
        for line in chunk.splitlines(keepends=True):
            if not line.endswith("\n"):
                break  # partial line; re-read on the next modify event
            consumed += len(line)
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                log.warning("bad transitions line in %s: %.80s", path, line)
                continue
            event.setdefault("run_id", path.parent.name)
            self.router.handle_transition(event)
        self._offsets[path] = offset + consumed

    # ---- main loop ----
    def dispatch(self, kind: str, payload) -> None:
        if kind == "slack_event":
            self.router.handle_event(payload)
        elif kind == "block_actions":
            self.router.handle_block_actions(payload)
        elif kind == "fs_question":
            self.router.handle_question_file(payload)
        elif kind == "fs_reply":
            self.router.handle_reply_file(payload)
        elif kind == "fs_operator_out":
            self.router.handle_operator_out(payload)
        elif kind == "fs_transitions":
            self.drain_transitions(payload)

    def run_forever(self) -> None:
        self.cfg.runs_root.mkdir(parents=True, exist_ok=True)
        observer = Observer()
        observer.schedule(MailboxHandler(self.events), str(self.cfg.runs_root), recursive=True)
        observer.start()
        self.initial_scan()
        if self.socket_client is not None:
            self._ensure_connected()
        log.info("bridge up: watching %s", self.cfg.runs_root)
        try:
            while not self.stop_event.is_set():
                try:
                    kind, payload = self.events.get(timeout=5)
                except queue.Empty:
                    self._ensure_connected()
                    self.gateway.flush_pending()
                    continue
                try:
                    self.dispatch(kind, payload)
                except Exception:
                    log.exception("error handling %s event", kind)
        finally:
            observer.stop()
            observer.join(timeout=5)
            if self.socket_client is not None:
                self.socket_client.close()
