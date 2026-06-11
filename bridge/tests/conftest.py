"""Shared fixtures: tmpdir mailbox, in-memory-style DB, fully mocked Slack gateway.

No network, no real tokens, no pipeline imports anywhere in these tests.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from bridge.config import BridgeConfig
from bridge.routes import Router
from bridge.threads_db import ThreadsDB


class FakeGateway:
    """Stands in for SlackGateway; records every outbound call."""

    def __init__(self):
        self.posts: list[dict] = []
        self.ephemerals: list[dict] = []
        self._ts = 1000.0

    def post_message(self, channel, text, blocks=None, thread_ts=None, metadata=None) -> str:
        self._ts += 1
        ts = f"{self._ts:.6f}"
        self.posts.append({"channel": channel, "text": text, "blocks": blocks,
                           "thread_ts": thread_ts, "metadata": metadata, "ts": ts})
        return ts

    def post_ephemeral(self, channel, user, text, thread_ts=None) -> None:
        self.ephemerals.append({"channel": channel, "user": user, "text": text,
                                "thread_ts": thread_ts})

    def flush_pending(self) -> None:
        pass


@pytest.fixture
def cfg(tmp_path: Path) -> BridgeConfig:
    runs = tmp_path / "runs"
    runs.mkdir()
    return BridgeConfig(
        bot_token="xoxb-REPLACE-ME", app_token="xapp-REPLACE-ME",
        requests_channel="C_REQ", status_channel="C_STATUS",
        repo_root=tmp_path, runs_root=runs, db_path=tmp_path / "threads.sqlite")


@pytest.fixture
def db(cfg) -> ThreadsDB:
    return ThreadsDB(cfg.db_path)


@pytest.fixture
def gateway() -> FakeGateway:
    return FakeGateway()


@pytest.fixture
def spawned() -> list[str]:
    return []


@pytest.fixture
def router(cfg, db, gateway, spawned) -> Router:
    return Router(cfg, db, gateway, spawn=spawned.append)


def mention_event(text="train pi0.5 on task-0", channel="C_REQ", user="U_ALICE", ts="1.000001"):
    return {"type": "app_mention", "text": f"<@U_BOT> {text}",
            "channel": channel, "user": user, "ts": ts}


def write_question(runs_root: Path, run_id: str, esc_id: str, **overrides) -> Path:
    esc = {
        "node": "filter_build", "agent": "data_agent",
        "question": "PCA at p98 would drop episodes 102, 103, 137. OK?",
        "context": "v3 source has 132 clips.",
        "options": [{"id": 1, "label": "p98 (recommended)"},
                    {"id": 2, "label": "p95 (more aggressive)"},
                    {"id": 3, "label": "Skip PCA"}],
        "recommendation": 1,
        "created_at": "2026-06-09T18:42:17Z",
    }
    esc.update(overrides)
    path = runs_root / run_id / "escalations" / f"{esc_id}.question.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(esc))
    return path


def click_payload(esc_id: str, option: int, label="p98 (recommended)", user="U_ALICE",
                  channel="C_REQ", thread_ts="1.000001"):
    return {
        "type": "block_actions",
        "user": {"id": user},
        "channel": {"id": channel},
        "message": {"ts": "9.0", "thread_ts": thread_ts},
        "actions": [{"action_id": f"esc:{esc_id}:opt:{option}",
                     "value": str(option),
                     "text": {"type": "plain_text", "text": label}}],
    }
