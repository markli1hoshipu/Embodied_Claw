"""Entry point: `python -m bridge.service` (spec 13.10). Standalone — no pipeline imports."""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from .config import MissingConfig, load_config, load_env_file

SETUP_MSG = """\
Slack bridge is not configured. Missing environment variables: {missing}

Create {root}/.env (copy .env.example) with:
    SLACK_BOT_TOKEN=xoxb-...           (Bot User OAuth Token)
    SLACK_APP_TOKEN=xapp-...           (App-Level Token, connections:write)
    SLACK_REQUESTS_CHANNEL_ID=C0...    (#pi05-requests channel id)
    SLACK_STATUS_CHANNEL_ID=C0...      (optional, #pi05-status)

Full Slack-app setup walkthrough: {root}/bridge/README.md
"""


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=os.environ.get("BRIDGE_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    repo_root = Path(os.environ.get("EMBODIED_CLAW_ROOT") or Path(__file__).resolve().parents[1])
    load_env_file(repo_root / ".env")
    try:
        cfg = load_config(os.environ, repo_root)
    except MissingConfig as exc:
        sys.stderr.write(SETUP_MSG.format(missing=", ".join(exc.missing), root=repo_root))
        return 2

    # Imports deferred so the setup-error path above never needs slack connectivity.
    from slack_sdk import WebClient
    from slack_sdk.socket_mode import SocketModeClient

    from .reconcile import reconcile
    from .routes import Router
    from .slack_bridge import Bridge, SlackGateway
    from .threads_db import ThreadsDB

    db = ThreadsDB(cfg.db_path)
    web = WebClient(token=cfg.bot_token)
    gateway = SlackGateway(web)
    router = Router(cfg, db, gateway)
    socket_client = SocketModeClient(app_token=cfg.app_token, web_client=web)
    bridge = Bridge(cfg, db, gateway, router, socket_client=socket_client)
    reconcile(db, web, cfg.runs_root, cfg.requests_channel)
    bridge.run_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
