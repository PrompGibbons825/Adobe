"""Dev tool: a small ssh/rsync fleet of tagged EC2 boxes (Project=chameleon-2026-10-30).

    python tools/fleet.py hosts
    python tools/fleet.py sync <local_tree> [--name X]            # -> ~/trees/X on every host, .venv symlinked
    python tools/fleet.py run --tree X --shards N -- <cmd with {shard} and {nshards}>
    python tools/fleet.py exec -- <cmd>                           # same command on every host, in sequence
    python tools/fleet.py pull <remote_dir> <local_dir>           # merge <remote_dir> of every host into <local_dir>
    python tools/fleet.py down [--purge]                          # terminate everything tagged (+ key pair, SG)

Plain ssh and rsync via subprocess; the aws CLI only finds the hosts. Stdlib only.
Env: AWS_PROFILE (default vlatlabs), FLEET_REGIONS (default eu-north-1), FLEET_KEY, FLEET_TAG.
Remote layout: ~/chameleon/.venv (shared venv), ~/trees/<name>/ (synced trees), cases under
/data/agents/work/crello-wide (the same path as on the ThinkPad). Shard i of N goes to a host in
proportion to its vCPU count; a shard runs as ``cd ~/trees/<tree> && <cmd>`` with {shard}=i, {nshards}=N.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

HOME = Path("/data/agents/work/designer/fleet")
KEY = os.environ.get("FLEET_KEY", str(HOME / "key.pem"))
TAG = os.environ.get("FLEET_TAG", "chameleon-2026-10-30")
PROFILE = os.environ.get("AWS_PROFILE", "vlatlabs")
REGIONS = os.environ.get("FLEET_REGIONS", "eu-north-1").split(",")
SSH = ["ssh", "-i", KEY, "-o", "StrictHostKeyChecking=accept-new", "-o", f"UserKnownHostsFile={HOME / 'known_hosts'}",
       "-o", "ServerAliveInterval=30", "-o", "ConnectTimeout=15", "-o", "LogLevel=ERROR"]
EXCLUDES = [".venv", "output", "__pycache__", ".git", "*.pyc", ".pytest_cache", "node_modules"]


def aws(region, *args):
    r = subprocess.run(["aws", "--profile", PROFILE, "--region", region, *args, "--output", "json"],
                       capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"aws {' '.join(args[:2])} failed in {region}: {r.stderr.strip()[:300]}")
    return json.loads(r.stdout) if r.stdout.strip() else {}


def hosts(states=("running",)):
    out = []
    for reg in REGIONS:
        d = aws(reg, "ec2", "describe-instances", "--filters", f"Name=tag:Project,Values={TAG}",
                f"Name=instance-state-name,Values={','.join(states)}")
        for r in d.get("Reservations", []):
            for i in r["Instances"]:
                cpu = i["CpuOptions"]
                out.append(dict(id=i["InstanceId"], type=i["InstanceType"], ip=i.get("PublicIpAddress"), region=reg,
                                vcpu=cpu["CoreCount"] * cpu["ThreadsPerCore"], spot=i.get("InstanceLifecycle") == "spot",
                                state=i["State"]["Name"]))
    return sorted(out, key=lambda h: (-h["vcpu"], h["id"]))


def ssh_cmd(h, remote):
    return SSH + [f"ubuntu@{h['ip']}", remote]


def rsync_e():
    return " ".join(SSH)


def cmd_hosts(a):
    hs = hosts()
    for h in hs:
        print(f"{h['id']}  {h['type']:<12} {h['ip']:<16} {h['vcpu']:>3} vCPU  {'spot' if h['spot'] else 'on-demand'}  {h['region']}")
    print(f"total {sum(h['vcpu'] for h in hs)} vCPU on {len(hs)} hosts")


def par(fn, items):
    """Run fn(item) in a thread per item; return results in order, exceptions re-raised after all finish."""
    res, errs = [None] * len(items), []

    def go(k, it):
        try:
            res[k] = fn(it)
        except Exception as e:  # noqa: BLE001
            errs.append(e)

    ts = [threading.Thread(target=go, args=(k, it)) for k, it in enumerate(items)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    if errs:
        raise errs[0]
    return res


def cmd_sync(a):
    src = Path(a.tree).resolve()
    name = a.name or src.name
    hs = hosts()

    def one(h):
        ex = [x for p in EXCLUDES for x in ("--exclude", p)]
        subprocess.run(["rsync", "-a", "--delete", "-e", rsync_e(), *ex, f"{src}/", f"ubuntu@{h['ip']}:trees/{name}/"], check=True)
        subprocess.run(ssh_cmd(h, f"ln -sfn ~/chameleon/.venv ~/trees/{name}/.venv"), check=True)
        return h["id"]

    t0 = time.time()
    par(one, hs)
    print(f"synced {src} -> ~/trees/{name} on {len(hs)} hosts in {time.time() - t0:.1f}s")


def allocate(hs, n):
    """Shard indices per host, in proportion to vCPU (largest remainder)."""
    tot = sum(h["vcpu"] for h in hs)
    q = [n * h["vcpu"] / tot for h in hs]
    cnt = [int(x) for x in q]
    for k in sorted(range(len(hs)), key=lambda k: q[k] - cnt[k], reverse=True)[: n - sum(cnt)]:
        cnt[k] += 1
    out, s = [], 0
    for c in cnt:
        out.append(list(range(s, s + c)))
        s += c
    return out


def cmd_run(a):
    cmd = " ".join(a.command[1:] if a.command and a.command[0] == "--" else a.command)
    if "{shard}" not in cmd or "{nshards}" not in cmd:
        raise SystemExit("the command must contain {shard} and {nshards}")
    hs = hosts()
    alloc = allocate(hs, a.shards)
    logs = HOME / "logs" / f"{a.tree}-{time.strftime('%H%M%S')}"
    logs.mkdir(parents=True, exist_ok=True)
    jobs = [(h, i) for h, idx in zip(hs, alloc) for i in idx]
    for h, idx in zip(hs, alloc):
        print(f"{h['id']} {h['ip']}: shards {idx[0]}-{idx[-1]} ({len(idx)})" if idx else f"{h['id']}: none")
    print(f"logs: {logs}", flush=True)
    procs = []
    for h, i in jobs:
        remote = (f"export PATH=$HOME/.local/bin:$PATH CHAMELEON_FLEET_SHARD={i}; cd ~/trees/{a.tree} && "
                  + cmd.format(shard=i, nshards=a.shards))
        f = open(logs / f"shard-{i:03d}.log", "w")
        procs.append((i, h, subprocess.Popen(ssh_cmd(h, remote), stdout=f, stderr=subprocess.STDOUT), f))
    t0, failed = time.time(), []
    for i, h, p, f in procs:
        rc = p.wait()
        f.close()
        if rc:
            failed.append(i)
            print(f"shard {i} on {h['id']} FAILED rc={rc}: tail of {logs / f'shard-{i:03d}.log'}")
            print("   " + "\n   ".join((logs / f"shard-{i:03d}.log").read_text().splitlines()[-5:]))
    print(f"{len(jobs) - len(failed)}/{len(jobs)} shards ok in {time.time() - t0:.0f}s; logs {logs}")
    sys.exit(1 if failed else 0)


def cmd_exec(a):
    cmd = " ".join(a.command[1:] if a.command and a.command[0] == "--" else a.command)

    def one(h):
        r = subprocess.run(ssh_cmd(h, f"export PATH=$HOME/.local/bin:$PATH; {cmd}"), capture_output=True, text=True)
        return h, r

    rc = 0
    for h, r in par(one, hosts()):
        print(f"== {h['id']} {h['ip']} rc={r.returncode}\n{r.stdout}{r.stderr}", end="")
        rc |= bool(r.returncode)
    sys.exit(rc)


def cmd_pull(a):
    Path(a.local_dir).mkdir(parents=True, exist_ok=True)
    rd = a.remote_dir.rstrip("/")
    rd = rd if rd.startswith("/") else f"{rd}"

    def one(h):
        subprocess.run(["rsync", "-a", "-e", rsync_e(), f"ubuntu@{h['ip']}:{rd}/", f"{a.local_dir.rstrip('/')}/"], check=True)

    par(one, hosts())
    print(f"pulled {rd} from all hosts into {a.local_dir}")


def cmd_down(a):
    for reg in REGIONS:
        ids = [i["id"] for i in hosts(("pending", "running", "stopping", "stopped")) if i["region"] == reg]
        if ids:
            aws(reg, "ec2", "terminate-instances", "--instance-ids", *ids)
            print(f"{reg}: terminating {' '.join(ids)}")
            subprocess.run(["aws", "--profile", PROFILE, "--region", reg, "ec2", "wait", "instance-terminated", "--instance-ids", *ids])
        if a.purge:
            sgs = aws(reg, "ec2", "describe-security-groups", "--filters", f"Name=tag:Project,Values={TAG}")["SecurityGroups"]
            for sg in sgs:
                aws(reg, "ec2", "delete-security-group", "--group-id", sg["GroupId"])
                print(f"{reg}: deleted {sg['GroupId']}")
            subprocess.run(["aws", "--profile", PROFILE, "--region", reg, "ec2", "delete-key-pair", "--key-name", TAG])
    if not a.purge:
        print("instances terminated; SG and key pair kept (use --purge to delete them)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="sub", required=True)
    sub.add_parser("hosts")
    p = sub.add_parser("sync")
    p.add_argument("tree")
    p.add_argument("--name")
    p = sub.add_parser("run")
    p.add_argument("--tree", required=True)
    p.add_argument("--shards", type=int, required=True)
    p.add_argument("command", nargs=argparse.REMAINDER)
    p = sub.add_parser("exec")
    p.add_argument("command", nargs=argparse.REMAINDER)
    p = sub.add_parser("pull")
    p.add_argument("remote_dir")
    p.add_argument("local_dir")
    p = sub.add_parser("down")
    p.add_argument("--purge", action="store_true")
    a = ap.parse_args()
    {"hosts": cmd_hosts, "sync": cmd_sync, "run": cmd_run, "exec": cmd_exec, "pull": cmd_pull, "down": cmd_down}[a.sub](a)


if __name__ == "__main__":
    main()
