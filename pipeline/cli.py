"""Driver CLI (spec sections 7.2 + 13):

  python -m pipeline.cli run <run_id> [--request "..."] [--detach] [--poll N]
  python -m pipeline.cli reply --run-id X --node N (--option N | --message "...")
  python -m pipeline.cli reply --latest --message "..."
  python -m pipeline.cli inbox
"""
from __future__ import annotations

import argparse
import fcntl
import json
import subprocess
import sys
import time

from langgraph.types import Command

from pipeline import inbox, tools
from pipeline.graph import build_graph, finalize_run, open_checkpointer, pending_interrupts
from pipeline.state import STAGES, init_state, pending


def _index_link(run_id: str) -> None:
    """Global run index: pipeline_runs/<id> -> runs/<id> (spec section 10 layout)."""
    try:
        idx = tools.runs_root().parent / "pipeline_runs"
        idx.mkdir(parents=True, exist_ok=True)
        link = idx / run_id
        if not link.is_symlink():
            link.symlink_to(tools.run_dir(run_id))
    except OSError:
        pass


def drive(graph, conf, payload, run_id: str, poll: float = 30.0) -> dict:
    """invoke / resume loop. Pending interrupts are matched to mailbox replies by escalation_id
    and resumed per-interrupt-id (verified on langgraph 1.2.4), so concurrently escalated nodes
    (3+4) resume independently. No reply -> poll every `poll` seconds; never auto-default."""
    rd = tools.run_dir(run_id)
    while True:
        if payload is None:
            ints = pending_interrupts(graph, conf)
            if ints:
                resume, waited = {}, 0.0
                while not resume:
                    for i in ints:
                        r = tools.read_reply(rd, i.value["escalation_id"])
                        if r is not None:
                            resume[i.id] = r
                    if not resume:
                        time.sleep(poll)
                        waited += poll
                        if waited >= 24 * 3600:  # spec 7.4: re-notify, keep waiting, no defaults
                            waited = 0.0
                            for i in ints:
                                tools.notify(f"[{run_id}] still awaiting reply: "
                                             f"{i.value.get('question', i.value['escalation_id'])}")
                payload = Command(resume=resume)
            elif not graph.get_state(conf).next:
                break
        result = graph.invoke(payload, conf, durability="sync")
        payload = None
        if "__interrupt__" not in result and not graph.get_state(conf).next:
            break
    values = graph.get_state(conf).values
    finalize_run(run_id, values)
    return values


def cmd_run(args) -> None:
    run_id = args.run_id
    rd = tools.run_dir(run_id)
    rd.mkdir(parents=True, exist_ok=True)
    if args.request:
        (rd / "request.txt").write_text(args.request)
    if not (rd / "request.txt").exists():
        sys.exit(f"no request found: write {rd / 'request.txt'} or pass --request '...'")
    if args.detach:  # setsid self-relaunch; the run survives this shell (used by the bridge)
        cmd = [sys.executable, "-m", "pipeline.cli", "run", run_id]
        with open(rd / "driver.log", "ab") as log:
            subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, start_new_session=True)
        print(f"detached driver started for '{run_id}'; log: {rd / 'driver.log'}")
        return
    _index_link(run_id)
    print(f"runs root: {tools.runs_root()}")  # must match the bridge's resolved root
    lock = open(rd / ".driver.lock", "w")  # two drivers on one run could double-launch training
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        sys.exit(f"another driver holds {rd}/.driver.lock; refusing to run '{run_id}' twice")
    try:
        graph = build_graph(open_checkpointer(run_id))
        conf = {"configurable": {"thread_id": run_id}}
        snap = graph.get_state(conf)
        if not snap.values:                      # fresh run
            payload = init_state(run_id)
        elif snap.next:                          # crashed or escalated mid-run -> resume
            payload = None
        else:                                    # finished earlier: retry non-succeeded stages
            redo = {s: pending() for s in STAGES
                    if (snap.values.get(s) or {}).get("status") != "succeeded"}
            if not redo:
                print(f"'{run_id}' already completed; nothing to do")
                finalize_run(run_id, snap.values)
                _print_table(run_id, snap.values)
                return
            payload = redo
        try:
            values = drive(graph, conf, payload, run_id, poll=args.poll)
        except RuntimeError as e:  # e.g. missing ANTHROPIC_API_KEY: one clear line, no traceback
            sys.exit(f"error: {e}")
        _print_table(run_id, values)
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def _print_table(run_id: str, values: dict) -> None:
    print(f"=== {run_id} ===")
    for s in STAGES:
        st = values.get(s) or {}
        extra = st.get("error") or "; ".join(st.get("artifact_paths") or [])
        print(f"  {s:<15} {st.get('status', 'pending'):<10} {extra}")


def cmd_reply(args) -> None:
    if args.option is None and not args.message:
        sys.exit("provide --option N or --message '...'")
    target = inbox.resolve_target(run_id=args.run_id, node=args.node, latest=args.latest)
    if target is None:
        sys.exit("no pending escalation matches (see `python -m pipeline.cli inbox`)")
    text = str(args.option) if args.option is not None else args.message
    p = inbox.write_reply(target["run_id"], target["escalation_id"], text)
    print(f"reply recorded for [{target['run_id']}:{target['node']}] -> {p}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="pipeline.cli", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run / resume a pipeline run")
    r.add_argument("run_id")
    r.add_argument("--request", help="write runs/<id>/request.txt from this text first")
    r.add_argument("--detach", action="store_true", help="relaunch detached (setsid) and return")
    r.add_argument("--poll", type=float, default=30.0, help="escalation reply poll seconds")
    p = sub.add_parser("reply", help="answer a pending escalation",
                       description="Target the newest pending escalation matching the filters "
                                   "(spec 7.2A). One of --option/--message is required.")
    p.add_argument("--run-id", help="only answer an escalation from this run")
    p.add_argument("--node", help="only answer an escalation from this node (e.g. filter_build)")
    p.add_argument("--latest", action="store_true",
                   help="target the newest pending escalation across ALL runs")
    p.add_argument("--option", type=int, help="select option id N from the question's options")
    p.add_argument("--message", help="free-form reply text (anything that is not an option id)")
    sub.add_parser("inbox", help="list pending escalations")
    args = ap.parse_args(argv)
    if args.cmd == "run":
        cmd_run(args)
    elif args.cmd == "reply":
        cmd_reply(args)
    else:
        print(inbox.format_inbox(inbox.pending_escalations()))


if __name__ == "__main__":
    main()
