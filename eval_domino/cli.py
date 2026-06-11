"""Eval driver CLI — mirrors pipeline.cli byte-for-byte semantics on the eval graph:

  python -m eval_domino.cli run <run_id> [--request "..."] [--detach] [--poll N]
  python -m eval_domino.cli reply --run-id X --node N (--option N | --message "...")
  python -m eval_domino.cli inbox

reply/inbox are shared with pipeline.cli (same runs root + mailbox), kept here so eval users
need only one module name.
"""
from __future__ import annotations

import argparse
import fcntl
import subprocess
import sys
import time

from langgraph.types import Command

from eval_domino import tools
from eval_domino.graph import build_graph, finalize_run, open_checkpointer, pending_interrupts
from eval_domino.state import STAGES, init_state
from pipeline import inbox
from pipeline.state import pending


def drive(graph, conf, payload, run_id: str, poll: float = 30.0) -> dict:
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
                        if waited >= 24 * 3600:
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
    if args.detach:
        cmd = [sys.executable, "-m", "eval_domino.cli", "run", run_id]
        with open(rd / "driver.log", "ab") as log:
            subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, start_new_session=True)
        print(f"detached eval driver started for '{run_id}'; log: {rd / 'driver.log'}")
        return
    print(f"runs root: {tools.runs_root()}")
    lock = open(rd / ".driver.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        sys.exit(f"another driver holds {rd}/.driver.lock; refusing to run '{run_id}' twice")
    try:
        graph = build_graph(open_checkpointer(run_id))
        conf = {"configurable": {"thread_id": run_id}}
        snap = graph.get_state(conf)
        if not snap.values:
            payload = init_state(run_id)
        elif snap.next:
            payload = None
        else:
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
        except RuntimeError as e:
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
        print(f"  {s:<17} {st.get('status', 'pending'):<10} {extra}")


def cmd_reply(args) -> None:
    if args.option is None and not args.message:
        sys.exit("provide --option N or --message '...'")
    target = inbox.resolve_target(run_id=args.run_id, node=args.node, latest=args.latest)
    if target is None:
        sys.exit("no pending escalation matches (see `python -m eval_domino.cli inbox`)")
    text = str(args.option) if args.option is not None else args.message
    p = inbox.write_reply(target["run_id"], target["escalation_id"], text)
    print(f"reply recorded for [{target['run_id']}:{target['node']}] -> {p}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="eval_domino.cli", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run / resume an eval run")
    r.add_argument("run_id")
    r.add_argument("--request", help="write runs/<id>/request.txt from this text first")
    r.add_argument("--detach", action="store_true", help="relaunch detached (setsid) and return")
    r.add_argument("--poll", type=float, default=30.0, help="escalation reply poll seconds")
    p = sub.add_parser("reply", help="answer a pending escalation")
    p.add_argument("--run-id")
    p.add_argument("--node")
    p.add_argument("--latest", action="store_true")
    p.add_argument("--option", type=int)
    p.add_argument("--message")
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
