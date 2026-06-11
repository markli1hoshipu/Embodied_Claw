"""Bridge configuration: env vars (.env), optional config/slack.toml allow-list (spec 13.8)."""
from __future__ import annotations

import logging
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("bridge.config")

REQUIRED_ENV = ("SLACK_BOT_TOKEN", "SLACK_APP_TOKEN", "SLACK_REQUESTS_CHANNEL_ID")


class MissingConfig(Exception):
    def __init__(self, missing: list[str]):
        self.missing = missing
        super().__init__(f"missing env vars: {', '.join(missing)}")


@dataclass
class BridgeConfig:
    bot_token: str
    app_token: str
    requests_channel: str
    status_channel: str | None
    repo_root: Path
    runs_root: Path
    db_path: Path
    allowed_users: set[str] | None = field(default=None)  # None = everyone may launch runs
    eval_channel: str | None = field(default=None)        # mentions here launch eval_domino runs


def load_env_file(path: Path) -> None:
    """Parse a KEY=VALUE .env file into os.environ (existing env wins). Never logs values."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def load_allowed_users(repo_root: Path) -> set[str] | None:
    """Optional allow-list from config/slack.toml (spec 13.8). Tolerates absence/garbage."""
    path = repo_root / "config" / "slack.toml"
    if not path.is_file():
        return None
    try:
        data = tomllib.loads(path.read_text())
    except Exception as exc:  # malformed toml must not kill the bridge
        log.warning("could not parse %s (%s); allow-list disabled", path, exc)
        return None
    users = data.get("allowed_users") or data.get("slack", {}).get("allowed_users")
    return {str(u) for u in users} if users else None


def load_config(environ: dict, repo_root: Path) -> BridgeConfig:
    missing = [k for k in REQUIRED_ENV if not environ.get(k)]
    if missing:
        raise MissingConfig(missing)
    return BridgeConfig(
        bot_token=environ["SLACK_BOT_TOKEN"],
        app_token=environ["SLACK_APP_TOKEN"],
        requests_channel=environ["SLACK_REQUESTS_CHANNEL_ID"],
        status_channel=environ.get("SLACK_STATUS_CHANNEL_ID") or None,
        repo_root=repo_root,
        # Falls back to the pipeline's EMBODIED_CLAW_RUNS so one env var can't split the mailbox
        runs_root=Path(environ.get("BRIDGE_RUNS_ROOT") or environ.get("EMBODIED_CLAW_RUNS")
                       or repo_root / "runs"),
        db_path=Path(environ.get("BRIDGE_DB_PATH") or repo_root / "bridge" / "threads.sqlite"),
        allowed_users=load_allowed_users(repo_root),
        eval_channel=environ.get("SLACK_EVAL_CHANNEL_ID") or None,
    )
