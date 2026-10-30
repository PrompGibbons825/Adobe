"""Dev bench (bet B0): engine versions on the wide Crello dev set. Never sealed data, never the gate.

    loop/capped .venv/bin/python -m tools.bench run    --engine <worktree> --tag main
    .venv/bin/python -m tools.bench judge  --a sprint --b main        # blind pairwise panel, changed tiles only
    .venv/bin/python -m tools.bench parity --tag main                 # the same panel against the designer's resize
    .venv/bin/python -m tools.bench table  --tags main,sprint,night   # one markdown table

Data: ``tools.crello_wide`` (index -> sample -> fetch) leaves $CRELLO_WIDE/cases/<family>.json
(source design, the designer's other sizes, assets/). Every dev task is the designer's source size
-> the designer's other size; the designer's own resize is the reference. Results land in
$CRELLO_WIDE/bench/<tag>/ (results.json, summary.json, png/) and bench/judge_*.json, bench/parity_*.json.

Per tile, ``run`` records: rule faults (hard_rule_errors + promise_errors, with capacity-exempt
small print counted apart), audit flags (tools.audit.flags), hidden elements (tools.gate.visible),
and ``sim``, the mean IoU (in canvas fractions) of our element boxes with the designer's, matching
texts by content and other elements by kind and order.

The judge is the gate panel's machinery (prompt v2, gray mats with per-image offsets, forced choice
with the A/B labels randomised per ask, both orders): Claude family only, ``claude -p --model
claude-sonnet-5-5``. A tile's score is the share of its valid orders that picked ``a``; the win rate
is the mean tile score with a 95% bootstrap CI over tiles (ties count one half).

``BENCH_JUDGE=bedrock`` sends the same prompts and images to Bedrock (``BedrockJudge``: the ``aws bedrock-runtime
invoke-model`` CLI, 12 asks at once) instead of ``claude -p``; the reply is parsed by the same ``parse_batch``. Env:
BEDROCK_MODEL (global.anthropic.claude-sonnet-5-5), BEDROCK_REGION (us-east-1), BENCH_BEDROCK_PROFILE (vlatlabs),
BENCH_BEDROCK_WORKERS (12), BENCH_BEDROCK_ENDPOINT (test only). Every call's tokens go to
/data/agents/work/designer/status/bedrock_tokens.jsonl. ``judge`` and ``parity`` take ``--out`` to write elsewhere.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

WIDE = Path(os.environ.get("CRELLO_WIDE", "/data/agents/work/crello-wide"))
CASES = WIDE / "cases"
BENCH = WIDE / "bench"
MODEL = "claude-sonnet-5-5"
BATCH = 5
FONTS = ROOT / "brand_kit" / "fonts"
BEDROCK_TOKENS = Path(os.environ.get("BEDROCK_TOKENS", "/data/agents/work/designer/status/bedrock_tokens.jsonl"))


# ---- tasks ---------------------------------------------------------------------------------------

def aspect_class(W: float, H: float) -> str:
    ar = W / H
    return ("extreme" if ar >= 6 or ar <= 0.35 else "strip" if ar >= 3.5 else "wide" if ar >= 1.8 else
            "tall" if ar < 0.75 else "near-square")


def load_tasks(limit: Optional[int] = None, names_file: Optional[str] = None) -> Dict[str, dict]:
    """name -> {source, target (canvas), reference (designer's design), family, photo}."""
    tasks = {}
    for p in sorted(CASES.glob("*.json")):
        f = json.loads(p.read_text())
        for k, ref in enumerate(f["targets"]):
            tasks[f"{f['family']}_{k}"] = dict(source=f["source"], target=dict(ref["canvas"]), reference=ref,
                                               family=f["family"], photo=f.get("photo", False))
    names = sorted(tasks)
    if names_file:
        keep = set(json.loads(Path(names_file).read_text())["tasks"])
        names = [n for n in names if n in keep]
    if limit:
        names = names[:limit]
    return {n: tasks[n] for n in names}


def _norm(s) -> str:
    return " ".join(str(s or "").lower().split())


def similarity(ours: dict, ref: dict) -> Optional[float]:
    """Mean IoU of our element boxes with the designer's, in canvas fractions. Texts are matched
    by content (duplicates in order). Other elements are matched greedily, in z order, to the
    unused designer element of the same type with the closest aspect ratio and canvas share (the
    designer re-orders layers between sizes). None if nothing matches. A crude closeness measure,
    not a quality score: the designer is one good answer among many."""
    import math

    def boxes(d):
        W, H = d["canvas"]["width"], d["canvas"]["height"]
        return [(e, (e["x"] / W, e["y"] / H, (e["x"] + e["width"]) / W, (e["y"] + e["height"]) / H)) for e in d["elements"]]

    def area(b):
        return max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))

    def asp(b):
        return max(1e-6, (b[2] - b[0])) / max(1e-6, (b[3] - b[1]))

    def iou(b, r):
        iw = max(0.0, min(b[2], r[2]) - max(b[0], r[0]))
        ih = max(0.0, min(b[3], r[3]) - max(b[1], r[1]))
        inter = iw * ih
        uni = area(b) + area(r) - inter
        return inter / uni if uni > 0 else 0.0

    txt, oth = {}, {}
    for e, b in boxes(ref):
        if e.get("type") == "text":
            txt.setdefault(_norm(e.get("content")), []).append(b)
        else:
            oth.setdefault(e.get("type"), []).append(b)
    seen, ious = {}, []
    for e, b in sorted(boxes(ours), key=lambda eb: eb[0].get("z_index", 0)):
        if e.get("type") == "text":
            k = _norm(e.get("content"))
            i = seen.get(k, 0)
            seen[k] = i + 1
            if i < len(txt.get(k, [])):
                ious.append(iou(b, txt[k][i]))
        else:
            cand = oth.get(e.get("type"), [])
            if cand:
                r = min(cand, key=lambda c: abs(math.log(asp(b) / asp(c))) + 0.3 * abs(math.log(area(b) / area(c))))
                cand.remove(r)
                ious.append(iou(b, r))
    return sum(ious) / len(ious) if ious else None


# ---- run -----------------------------------------------------------------------------------------

def tile_metrics(t: dict, out: dict) -> dict:
    from tools import audit
    from tools.gate.visible import hidden_elements
    from tools.rules import capacity_errors, hard_rule_errors, promise_errors

    src, tgt = t["source"], t["target"]
    os.environ.setdefault("CHAMELEON_BRAND_KIT", str(CASES))  # the cases' assets (roles, alpha), not brand_kit's
    m = dict(hard=[], promise=[], capacity=[], audit=[], hidden=[], sim=None)
    m["hard"] = hard_rule_errors(src, out, tgt)
    if not m["hard"]:
        for k, fn in (("promise", lambda: promise_errors(src, out, tgt)),
                      ("capacity", lambda: capacity_errors(out, tgt)),
                      ("audit", lambda: audit.flags(src, out, tgt)),
                      ("hidden", lambda: hidden_elements(src, out, tgt))):
            try:
                m[k] = fn()
            except Exception as ex:  # noqa: BLE001
                m[k] = [f"check crashed: {type(ex).__name__}: {ex}"]
    m["sim"] = similarity(out, t["reference"])
    return m


def cmd_tournament(a):
    """Fix the tournament set once: <=a.n tasks, equal share per aspect class of the target."""
    tasks = load_tasks()
    by = {}
    for n, t in tasks.items():
        by.setdefault(aspect_class(t["target"]["width"], t["target"]["height"]), []).append(n)
    per = a.n // len(by)
    pick = {c: sorted(_rng("tournament", c).sample(v, min(per, len(v)))) for c, v in sorted(by.items())}
    out = dict(n=sum(map(len, pick.values())), available={c: len(v) for c, v in by.items()},
               per_class={c: len(v) for c, v in pick.items()}, tasks=sorted(x for v in pick.values() for x in v))
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "tasks"}))


def cmd_run(a):
    from tools.gate.engine_run import run_jobs

    tasks = load_tasks(a.limit, a.set)
    d = BENCH / a.tag
    d.mkdir(parents=True, exist_ok=True)
    jobs = [dict(name=n, source=t["source"], target=t["target"]) for n, t in tasks.items()]
    t0 = time.time()
    res = run_jobs(Path(a.engine), jobs, kit=CASES, workers=a.workers)
    rows = {}
    for n, r in res.items():
        row = dict(error=r["error"], fell_back=r["fell_back"], secs=r["secs"], out=r["out"])
        if r["out"] is not None:
            row["m"] = tile_metrics(tasks[n], r["out"])
        rows[n] = row
    (d / "results.json").write_text(json.dumps(rows))
    s = summarise(rows, tasks)
    s["engine"] = str(a.engine)
    s["wall_secs"] = round(time.time() - t0)
    try:
        import subprocess
        s["commit"] = subprocess.run(["git", "-C", str(a.engine), "rev-parse", "--short", "HEAD"],
                                     capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    (d / "summary.json").write_text(json.dumps(s, indent=1))
    print(json.dumps(s, indent=1))


def summarise(rows: dict, tasks: dict) -> dict:
    ok = {n: r for n, r in rows.items() if r.get("m")}
    cnt = lambda k: sum(len(r["m"][k]) for r in ok.values())  # noqa: E731
    tiles = lambda k: sum(1 for r in ok.values() if r["m"][k])  # noqa: E731
    secs = sorted(r["secs"] for r in rows.values())
    sims = [r["m"]["sim"] for r in ok.values() if r["m"]["sim"] is not None]
    shapes = {}
    for n, r in ok.items():
        W, H = tasks[n]["target"]["width"], tasks[n]["target"]["height"]
        c = "wide" if W / H >= 2.4 else "tall" if W / H <= 0.62 else "mid"
        shapes.setdefault(c, []).append(r["m"]["sim"] or 0)
    return dict(
        tiles=len(rows), crashes=sum(1 for r in rows.values() if r["error"]), fallbacks=sum(1 for r in rows.values() if r["fell_back"]),
        hard_tiles=tiles("hard"), hard=cnt("hard"), promise_tiles=tiles("promise"), promise=cnt("promise"),
        capacity_tiles=tiles("capacity"), audit_tiles=tiles("audit"), audit=cnt("audit"),
        hidden_tiles=tiles("hidden"), hidden=cnt("hidden"),
        sim_mean=round(sum(sims) / len(sims), 4) if sims else None,
        sim_by_shape={k: round(sum(v) / len(v), 4) for k, v in shapes.items()},
        secs_median=secs[len(secs) // 2] if secs else None, secs_p95=secs[int(len(secs) * 0.95)] if secs else None)


# ---- judge ---------------------------------------------------------------------------------------

def _render(design: dict, path: Path) -> Path:
    if not path.exists():
        from renderer.converter.convert import render_design_to_png

        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.tmp.png")
        render_design_to_png(design, CASES / "assets", str(tmp), fonts_dir=FONTS)
        tmp.replace(path)
    return path


def _results(tag: str) -> dict:
    p = BENCH / tag / "results.json"
    if not p.exists():
        raise SystemExit(f"no results for tag {tag!r}: run `bench run` first")
    return json.loads(p.read_text())


def _canon(o) -> str:
    return json.dumps(o, sort_keys=True, separators=(",", ":"))


def _rng(*parts):
    return random.Random(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest())


# ---- Bedrock judge -------------------------------------------------------------------------------

def use_bedrock() -> bool:
    return os.environ.get("BENCH_JUDGE", "").lower() == "bedrock"


def judge_model() -> str:
    return os.environ.get("BEDROCK_MODEL", "global.anthropic.claude-sonnet-5-5") if use_bedrock() else MODEL


class BedrockJudge:
    """The gate panel's ``ClaudeJudge`` contract (``judge(prompt, model) -> (text, usage)``) over the
    ``aws bedrock-runtime invoke-model`` CLI (Anthropic Messages body; the v1 CLI double-encodes the image
    bytes of ``converse`` JSON input, ``--body fileb://`` is exact on v1 and v2). The image files named in the prompt are attached inline, each
    after a text label with its path, so the prompt's file names still identify them. Retries throttling
    and 5xx with backoff; any other error raises (an account that may not call Bedrock must stop the run,
    not return empty panels)."""

    OPEN = "Read every image below with the Read tool (several files per call is fine)."
    ATTACH = ("The images are attached after this message, each preceded by a label giving its file path; "
              "the paths in the list below refer to those labelled images.")
    SYSTEM = "You are a careful, blind design judge. You only look at the images attached and answer in JSON."
    lock = threading.Lock()

    def __init__(self):
        self.model = judge_model()
        self.region = os.environ.get("BEDROCK_REGION", "us-east-1")
        self.profile = os.environ.get("BENCH_BEDROCK_PROFILE", os.environ.get("AWS_PROFILE", "vlatlabs"))
        self.endpoint = os.environ.get("BENCH_BEDROCK_ENDPOINT")
        self.aws = shutil.which("aws") or "aws"

    @staticmethod
    def _image(path: str) -> bytes:
        data = Path(path).read_bytes()
        if len(data) > 3_500_000:  # Converse limit is 3.75 MB per image
            import io

            from PIL import Image

            with Image.open(path) as im:
                im = im.convert("RGB")
                im.thumbnail((1568, 1568))
                buf = io.BytesIO()
                im.save(buf, "PNG", optimize=True)
                data = buf.getvalue()
        return data

    def _call(self, prompt: str, paths: List[str], td: Path):
        content = [{"type": "text", "text": prompt.replace(self.OPEN, self.ATTACH)}]
        for p in paths:
            content.append({"type": "text", "text": f"[{p}]"})
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                         "data": base64.b64encode(self._image(p)).decode()}})
        body = td / "body.json"
        body.write_text(json.dumps({"anthropic_version": "bedrock-2023-05-31", "max_tokens": 8000, "system": self.SYSTEM,
                                    "messages": [{"role": "user", "content": content}]}))
        cmd = [self.aws, "--profile", self.profile, "--region", self.region, "bedrock-runtime", "invoke-model",
               "--model-id", self.model, "--content-type", "application/json", "--accept", "application/json",
               "--body", f"fileb://{body}", "--cli-read-timeout", "600"]
        if self.endpoint:
            cmd += ["--endpoint-url", self.endpoint]
        out = td / "reply.json"
        return subprocess.run(cmd + [str(out)], capture_output=True, text=True, timeout=700, stdin=subprocess.DEVNULL), out

    def __call__(self, prompt: str, model: str = "") -> tuple:
        paths = []
        for m in re.finditer(r"(/[^\s\"']+?\.png)", prompt):
            if m.group(1) not in paths:
                paths.append(m.group(1))
        err = ""
        with tempfile.TemporaryDirectory() as td:
            for k in range(6):
                r, out = self._call(prompt, paths, Path(td))
                if r.returncode == 0:
                    break
                err = (r.stderr or r.stdout or "")[-400:]
                if re.search(r"Throttl|TooManyRequests|ServiceUnavailable|InternalServer|ModelTimeout|ModelNotReady|timed out", err):
                    time.sleep(min(60, 2 ** k * 2 + random.random() * 2))
                    continue
                raise RuntimeError(f"bedrock invoke-model failed: {err.strip()}")
            else:
                raise RuntimeError(f"bedrock invoke-model failed after retries: {err.strip()}")
            j = json.loads(out.read_text())
        text = "".join(b.get("text", "") for b in j.get("content", []) if b.get("type") == "text")
        u = j.get("usage") or {}
        usage = {"input": u.get("input_tokens", 0), "output": u.get("output_tokens", 0), "stop": j.get("stop_reason")}
        try:
            BEDROCK_TOKENS.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(dict(ts=round(time.time()), model=self.model, region=self.region, input=usage["input"],
                                   output=usage["output"], images=len(paths), pid=os.getpid(),
                                   who=os.environ.get("BENCH_WHO", "bench"))) + "\n"
            with self.lock, open(BEDROCK_TOKENS, "a") as fh:
                fh.write(line)
        except OSError:
            pass
        return text, usage


def panel(tiles: Dict[str, dict], label: str, workers: int = 3) -> Dict[str, dict]:
    """tiles: name -> {ref: png, a: png, b: png}. Every tile is asked in both orders. Returns
    name -> {score_a (0..1), orders: [...]}; tiles with no valid verdict are omitted."""
    from tools.gate import panel as P

    jdir = BENCH / "judge_files"
    jdir.mkdir(parents=True, exist_ok=True)
    asks = []
    for n, t in sorted(tiles.items()):
        for order in (0, 1):
            r = _rng(label, n, order)
            pads = ["x", "y"]
            r.shuffle(pads)
            a_img, b_img = (t["a"], t["b"]) if order == 0 else (t["b"], t["a"])
            asks.append(dict(id=f"c{len(asks):03d}", tile=n, a_is=("a" if order == 0 else "b"),
                             ref=str(P.matted(t["ref"], P.MAT_PADS["ref"], jdir)),
                             A=str(P.matted(a_img, P.MAT_PADS[pads[0]], jdir)),
                             B=str(P.matted(b_img, P.MAT_PADS[pads[1]], jdir))))
    _rng(label, "batches").shuffle(asks)
    batches = [asks[i:i + BATCH] for i in range(0, len(asks), BATCH)]
    if use_bedrock():
        judge = BedrockJudge()
        workers = int(os.environ.get("BENCH_BEDROCK_WORKERS", "12"))
    else:
        judge = P.ClaudeJudge(jdir)
    verdicts: Dict[str, Optional[dict]] = {}

    def one(batch):
        todo = batch
        for _ in range(2):
            ids = [x["id"] for x in todo]
            text, usage = judge(P.prompt_for(todo, "claude", "v2", mat=True), MODEL)
            got = P.parse_batch(text, ids, {x["id"]: x["A"] for x in todo})
            for i, v in got.items():
                if v:
                    verdicts[i] = v
            todo = [x for x in todo if x["id"] not in verdicts]
            if not todo:
                break

    capped = None
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(one, b) for b in batches]
        for f in futs:
            try:
                f.result()
            except P.CapReached as e:
                capped = str(e)[:200]
    if capped:
        print("usage limit reached, partial panel:", capped)
    by_tile: Dict[str, list] = {}
    for x in asks:
        v = verdicts.get(x["id"])
        if not v:
            continue
        a_side = "A" if x["a_is"] == "a" else "B"
        by_tile.setdefault(x["tile"], []).append(dict(pick="a" if v["overall"] == a_side else "b",
                                                      conf=v.get("confidence"), why=v.get("why", "")))
    return {n: dict(score_a=sum(o["pick"] == "a" for o in os_) / len(os_), orders=os_) for n, os_ in by_tile.items()}


def _ci(values: List[float], seed: int = 0, n: int = 4000):
    from tools.gate.stats import boot_means

    if not values:
        return None, None, None
    b = boot_means(values, n, seed)
    return float(sum(values) / len(values)), float(__import__("numpy").quantile(b, 0.025)), float(__import__("numpy").quantile(b, 0.975))


def _summ(scores: Dict[str, dict]) -> dict:
    v = [s["score_a"] for s in scores.values()]
    mean, lo, hi = _ci(v)
    return dict(tiles=len(v), win_rate=mean, ci95=[lo, hi], wins=sum(x > 0.5 for x in v), losses=sum(x < 0.5 for x in v),
                ties=sum(x == 0.5 for x in v), share_ge=(sum(x >= 0.5 for x in v) / len(v)) if v else None)


def cmd_judge(a):
    ra, rb = _results(a.a), _results(a.b)
    tasks = load_tasks()
    both = [n for n in sorted(ra) if n in rb and ra[n].get("out") and rb[n].get("out") and n in tasks]
    changed = [n for n in both if _canon(ra[n]["out"]) != _canon(rb[n]["out"])]
    r = _rng("pick", a.a, a.b)
    pick = sorted(r.sample(changed, min(a.max_tiles, len(changed))))
    tiles = {}
    for n in pick:
        t = tasks[n]
        tiles[n] = dict(ref=_render(t["source"], BENCH / "ref" / f"{n}.src.png"),
                        a=_render(ra[n]["out"], BENCH / a.a / "png" / f"{n}.png"),
                        b=_render(rb[n]["out"], BENCH / a.b / "png" / f"{n}.png"))
    scores = panel(tiles, f"judge|{a.a}|{a.b}", a.workers)
    s = _summ(scores)
    s.update(a=a.a, b=a.b, common=len(both), changed=len(changed), sampled=len(pick), model=judge_model())
    Path(a.out or BENCH / f"judge_{a.a}_vs_{a.b}.json").write_text(json.dumps(dict(summary=s, tiles=scores), indent=1))
    print(json.dumps(s, indent=1))


def cmd_parity(a):
    ra = _results(a.tag)
    tasks = load_tasks()
    names = [n for n in sorted(ra) if ra[n].get("out") and n in tasks]
    pick = sorted(_rng("parity", a.tag).sample(names, min(a.max_tiles, len(names))))
    tiles = {}
    for n in pick:
        t = tasks[n]
        tiles[n] = dict(ref=_render(t["source"], BENCH / "ref" / f"{n}.src.png"),
                        a=_render(ra[n]["out"], BENCH / a.tag / "png" / f"{n}.png"),
                        b=_render(t["reference"], BENCH / "designer" / f"{n}.png"))
    scores = panel(tiles, f"parity|{a.tag}", a.workers)
    s = _summ(scores)
    v = [x["score_a"] >= 0.5 for x in scores.values()]
    m, lo, hi = _ci([float(x) for x in v], seed=1)
    s.update(tag=a.tag, share_ge_designer=m, share_ci95=[lo, hi], sampled=len(pick), model=judge_model())
    Path(a.out or BENCH / f"parity_{a.tag}.json").write_text(json.dumps(dict(summary=s, tiles=scores), indent=1))
    print(json.dumps(s, indent=1))


# ---- table ---------------------------------------------------------------------------------------

def _f(x, nd=3):
    return "-" if x is None else f"{x:.{nd}f}"


def cmd_table(a):
    tags = a.tags.split(",")
    sums = {t: json.loads((BENCH / t / "summary.json").read_text()) for t in tags}
    L = ["| tag | commit | tiles | crash/fallback | hard (tiles) | promise (tiles) | capacity tiles | audit flagged tiles | hidden (tiles) | sim | p95 s |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for t in tags:
        s = sums[t]
        L.append(f"| {t} | {s.get('commit', '')} | {s['tiles']} | {s['crashes']}/{s['fallbacks']} | {s['hard']} ({s['hard_tiles']}) | "
                 f"{s['promise']} ({s['promise_tiles']}) | {s['capacity_tiles']} | {s['audit_tiles']} | {s['hidden']} ({s['hidden_tiles']}) | "
                 f"{_f(s['sim_mean'])} | {_f(s['secs_p95'], 2)} |")
    pw = [p for p in sorted(BENCH.glob("judge_*_vs_*.json"))]
    rows = []
    for p in pw:
        s = json.loads(p.read_text())["summary"]
        if s["a"] in tags and s["b"] in tags:
            rows.append(f"| {s['a']} vs {s['b']} | {s['sampled']} of {s['changed']} changed | {_f(s['win_rate'], 2)} "
                        f"[{_f(s['ci95'][0], 2)}, {_f(s['ci95'][1], 2)}] | {s['wins']}/{s['ties']}/{s['losses']} |")
    if rows:
        L += ["", "| pairwise (a vs b) | tiles | a win rate [95% CI] | a wins/ties/losses |", "|---|---|---|---|"] + rows
    par = []
    for t in tags:
        p = BENCH / f"parity_{t}.json"
        if p.exists():
            s = json.loads(p.read_text())["summary"]
            par.append(f"| {t} | {s['sampled']} | {_f(s['share_ge_designer'], 2)} [{_f(s['share_ci95'][0], 2)}, {_f(s['share_ci95'][1], 2)}] | {_f(s['win_rate'], 2)} |")
    if par:
        L += ["", "| parity vs designer | tiles | share judged >= designer [95% CI] | mean score |", "|---|---|---|---|"] + par
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--engine", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--set", default="", help="a task-list json, e.g. the tournament set")
    p.add_argument("--workers", type=int, default=4)
    p = sub.add_parser("judge")
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--max-tiles", type=int, default=60)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--out", default="", help="result json (default: bench/judge_<a>_vs_<b>.json or parity_<tag>.json)")
    p = sub.add_parser("parity")
    p.add_argument("--tag", required=True)
    p.add_argument("--max-tiles", type=int, default=60)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--out", default="", help="result json (default: bench/judge_<a>_vs_<b>.json or parity_<tag>.json)")
    p = sub.add_parser("tournament")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--out", default=str(WIDE / "tournament_set.json"))
    p = sub.add_parser("table")
    p.add_argument("--tags", required=True)
    a = ap.parse_args()
    {"run": cmd_run, "judge": cmd_judge, "parity": cmd_parity, "table": cmd_table, "tournament": cmd_tournament}[a.cmd](a)


if __name__ == "__main__":
    main()
