# Slack bridge for the Embodied_Claw pipeline

A standalone service that bolts a Slack chat interface onto the pipeline's
**filesystem mailbox** (spec section "## 12", subsections 13.1–13.11). It:

- watches `runs/*/escalations/*.question.json` and posts each as an interactive
  Block Kit message in the run's Slack thread;
- listens via **Socket Mode** for thread replies and button clicks, and writes
  `<escalation_id>.reply.txt` files back into the mailbox;
- tails `runs/*/transitions.jsonl` and broadcasts one status line per stage
  transition to `#pi05-status`;
- handles `@bot` mentions in `#pi05-requests` to launch new runs (writes
  `runs/<run_id>/request.txt`, then spawns `python -m pipeline.cli run --detach <run_id>`
  as a subprocess).

The bridge **never imports pipeline code** (spec 13.1). If the bridge dies, the
pipeline keeps working via CLI / file drop; if the pipeline crashes, the bridge
keeps the channel alive.

## Layout

| File | Purpose |
|---|---|
| `service.py` | entry point: `python -m bridge.service` |
| `slack_bridge.py` | main service — Socket Mode listener + watchdog FS watcher + outbound gateway (429/redaction/offline queue) |
| `routes.py` | mention → new run; thread message → reply; button click → reply; mailbox file → Slack |
| `render.py` | escalation JSON → Block Kit (spec 13.7) + secret redaction (13.8) |
| `threads_db.py` | `threads.sqlite` schema (13.4) + migration (`python -m bridge.threads_db`) |
| `reconcile.py` | rebuild `threads.sqlite` from Slack history + FS walk after a crash (13.9) |
| `embodied_claw_bridge.service` | systemd user unit (13.10) |
| `tests/` | fully mocked test suite (no network, no real tokens) |

## 1. Slack app setup (one-time, spec 13.2)

1. https://api.slack.com/apps → **Create New App** → "From scratch" → name
   `pi05-pipeline` → choose workspace.
2. **OAuth & Permissions** → *Bot Token Scopes*, grant:
   - `chat:write` — post messages
   - `chat:write.public` — post in channels the bot isn't a member of (optional)
   - `channels:history` — read messages in public channels
   - `groups:history` — same for private channels (if you'll use one)
   - `app_mentions:read` — receive @mentions
   - `users:read` — resolve user IDs to names in transcripts
   - `files:write` — upload long log snippets as files
3. **Socket Mode** → enable. Generate an **App-Level Token** with
   `connections:write` — this is the `xapp-...` token.
4. **Event Subscriptions** → enable. Subscribe to bot events: `app_mention`,
   `message.channels`, `message.groups`.
5. **Interactivity & Shortcuts** → enable (needed for option buttons). No
   Request URL needed in Socket Mode.
6. **Install to Workspace** → grab the Bot User OAuth Token (`xoxb-...`).
7. Create the two channels and `/invite @pi05-pipeline` into both:
   - `#pi05-requests` — colleagues post new requests; the bot replies in thread.
   - `#pi05-status` (optional) — one-line broadcasts of every stage transition.
   Get each channel's ID from *channel name → View channel details → bottom of
   the About tab* (looks like `C0XXXXXXX`).
8. Copy `.env.example` → `.env` at the repo root (gitignored) and fill in:
   ```
   SLACK_BOT_TOKEN=xoxb-...
   SLACK_APP_TOKEN=xapp-...
   SLACK_REQUESTS_CHANNEL_ID=C0XXXXXX
   SLACK_STATUS_CHANNEL_ID=C0XXXXXX
   ```

Optional access control (spec 13.8): copy `config/slack.toml.example` to
`config/slack.toml` and set
```toml
allowed_users = ["U0AAAAAAA", "U0BBBBBBB"]
```
Only these Slack user IDs may launch runs **and** answer escalations (thread
replies + button clicks gate repo overwrites etc.). If the file is absent,
anyone in `#pi05-requests` may do both.

## 2. Run it

Foreground (debugging):
```bash
cd /work/markhsp/Embodied_Claw
.venv/bin/python -m bridge.service
```
With no tokens configured it exits with setup instructions (no traceback).

### systemd user service (spec 13.10)

```bash
mkdir -p ~/.config/systemd/user
cp bridge/embodied_claw_bridge.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now embodied_claw_bridge
journalctl --user -u embodied_claw_bridge -f     # logs
```
`Restart=always` gives crash recovery; on every start the bridge **reconciles**:
scans the requests channel history (bot messages carry Slack `metadata` with
`run_id`/`escalation_id`) plus walks `runs/*/escalations/` so questions answered
via CLI while it was down get acked and `threads.sqlite` can be rebuilt from
scratch if deleted.

## 3. Debugging

- `BRIDGE_LOG_LEVEL=DEBUG .venv/bin/python -m bridge.service` — verbose logs.
- `sqlite3 bridge/threads.sqlite 'SELECT * FROM runs; SELECT * FROM escalations;'`
  — what the bridge thinks is open. Safe to delete the DB; reconcile rebuilds it.
- Bot doesn't react to mentions → check Event Subscriptions has `app_mention`
  and the app was **reinstalled** after scope changes; check the bot is in the
  channel; check `SLACK_REQUESTS_CHANNEL_ID` matches (mentions elsewhere get a
  polite redirect).
- Buttons do nothing → Interactivity must be enabled; Socket Mode must show
  "connected" in the logs.
- Question not appearing in Slack → confirm the file is
  `runs/<run_id>/escalations/<id>.question.json` and valid JSON; the watcher
  logs `unreadable question file` on parse failure.
- Duplicate/missed transitions → the bridge only broadcasts lines appended
  *after* it started; historical lines are intentionally skipped on restart.
- Run mailbox/state for one run: `ls runs/<run_id>/escalations/`,
  `tail runs/<run_id>/transitions.jsonl`.
- Tests: `.venv/bin/python -m pytest bridge/tests/ -q` (fully mocked; never
  needs network or tokens).

## 4. Security notes (spec 13.8)

- Only `SLACK_REQUESTS_CHANNEL_ID` accepts mention-triggered runs.
- All outbound text/blocks pass through a redaction filter
  (`hf_...`, `xoxb-...`, `xapp-...`, `ghp_...`, `sk-ant-...` → `[REDACTED]`),
  applied both at render time and again at the gateway choke point.
- Tokens live in `.env` (gitignored); they are never logged or echoed.

## 5. Adding another transport (Discord, email, ...)

The pipeline contract is *files only*, so a second transport is another small
service beside this one — no pipeline changes:

1. Reuse `routes.Router` with your own gateway object: anything implementing
   `post_message(channel, text, blocks=None, thread_ts=None, metadata=None) -> ts`
   and `post_ephemeral(channel, user, text, thread_ts=None)` (see
   `bridge/tests/conftest.py::FakeGateway` for the minimal shape).
2. Map your transport's events into the three inbound calls:
   `handle_event` (new-run requests / thread replies), `handle_block_actions`
   (structured option picks), or call `Router._write_reply_file` equivalents by
   writing `runs/<run_id>/escalations/<escalation_id>.reply.txt` yourself —
   a single integer means "option id", anything else is free-form (spec 7.2).
3. Reuse `slack_bridge.MailboxHandler`/`Bridge.drain_transitions` for the
   outbound (filesystem → transport) direction, or simply watch the same paths.
4. Give it its own state DB (or a `transport` column); never share locks with
   the pipeline — the mailbox is the only meeting point.
