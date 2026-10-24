"""Gate tool: the blind judge panel, two model families (loop/STEERING.md S9, S10).

Every judge call is one fresh CLI process run from the gate's judge folder
``~/judge-adobe-hackathon-2026/<gate-id>/`` so the loop's usage meters count it:

* the **claude** family: ``claude -p --model claude-sonnet-5-5`` with the Read tool only;
* the **cursor** family: ``agent -p --trust --mode=ask --model gpt-5.6-sol-high`` (Cursor CLI,
  read-only ask mode), run from the gate's ``asks/`` folder so the images are opened by name;
* the **tiebreak** judge: ``agent ... --model gemini-3.7-flash-high``, a third family asked only
  on a split.

The judge sees images under random names (``tools.judge.views.blind_ask``): per comparison REF,
the source design, and A/B, two resizes. Which side is which lives only in the gate's private
folder. **Batching:** one call carries ``batch`` comparisons (6-12; S10), so the Cursor daily cap
counts calls, not tiles.

Per changed tile:

* one ask to each family; the candidate sits on A for one family and on B for the other, and
  which family gets which order is random per tile (never fixed), so position bias cancels
  within each family across tiles;
* when the two families split on overall, one tie-break ask goes to the third family;
* forced choice on the per-tile brief criteria (collisions, readable, brand, intentional) and
  overall, plus a 1-10 score per criterion for each side. (The brief's fifth row, consistency
  across formats, is not a per-tile question.)

**Tile score** (``score_tile``), per criterion over the *active* families: both agree -> that
vote; a split with a tie-break ask -> the majority of three (the tie-break's vote); a split with
no tie-break (criteria other than overall) -> 0.5; one family only -> its vote. Each family's own
vote is kept too (``family``), for the per-family harm test and agreement rate in
tools.gate.stats.

**Sentinels** with a known answer go to every active family: the good design (the designer's
resize where the task has one, else the base output) against a copy with one unambiguous defect
(``sentinel_kinds``: the headline cut past the edge, the headline moved onto the photo) or, only
where a designer's resize is the good side, fit-and-centre. run.py checks each family's accuracy
on its own; a family below ``sentinel_min`` is dropped from the gate run. **Report-only
sentinels** (``report_kinds``: the S6 small defects plus the 20 px misalignment and low-contrast
moves, which both families missed about half the time in G6's first run) are measured, never
decisive. ``python -m tools.gate.sentinel_check`` measures the sentinels on dev data.

**Cursor cap:** every ``agent`` call is logged in ``<judge root>/agent_calls.jsonl`` before it
starts. When a UTC day already holds ``agent_daily_cap`` calls, or either CLI reports a rate or
usage limit (Claude's weekly or five-hour limit included), the panel raises ``CapReached``; the
limited call labels nothing, so a ``--resume`` (or run.py's ``--wait``) continues exactly where it
stopped. The gate never falls back to one family because the other ran out of quota.

**What decides:** the pairwise votes. The 1-10 scores are only reported.

A judge is a callable ``judge(prompt, model) -> (reply_text, usage_dict)``; unit tests pass a
fake one (one callable for every family, or a dict ``{family: callable}``). The real ones keep
context small (a one-line system prompt, no CLAUDE.md, no MCP, read-only tools).
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from tools.gate.baselines import fit_centre

CRITERIA = ["collisions", "readable", "brand", "intentional", "overall"]
TIEBREAK = "tiebreak"
CLIS = ("claude", "agent")

DEFAULT_PANEL = {
    "protocol": "two-family",
    "families": {"claude": {"cli": "claude", "model": "claude-sonnet-5-5"},
                 "cursor": {"cli": "agent", "model": "gpt-5.6-sol-high"}},
    "tiebreak": {"cli": "agent", "model": "gemini-3.7-flash-high"},
    "batch": 8,              # comparisons per call (S10: 6-12)
    "concurrency": 4,        # claude calls at once
    "agent_concurrency": 2,  # agent calls at once
    "agent_daily_cap": 200,  # agent calls per UTC day over every gate (S9)
    "view": 1024,            # long side of every image the judge sees
    "timeout": 900,          # seconds per call (a batch)
    "retries": 1,            # extra calls for the comparisons a reply left out
    "min_sentinels": 8,      # decisive sentinel asks per family per look, at least
    "small_sentinels": 0.5,  # report-only sentinel asks per decisive sentinel ask
    "sentinel_kinds": ["p_crop", "p_onart", "fit_centre"],  # decisive; fit_centre only vs a designer resize
    "report_kinds": ["p_mis20", "p_lowcon", "p_hide_one", "p_clip_letter", "p_edge_touch", "p_typescale"],
}
PANEL_KEYS = set(DEFAULT_PANEL)

RUBRIC = """  - No unwanted collisions: overlaps are fine when they're intentional (layered graphics, deliberate stacking); what matters is that elements that shouldn't overlap, don't, and text never sits on a busy image.
  - Everything is readable: text is large enough and has enough contrast; nothing is cut off.
  - It stays on brand: colours, fonts, and the logo survive the resize intact and in proportion.
  - It looks intentional: things line up, spacing is even, the most important element still stands out, pictures keep their proportions."""

PROMPT = """You judge a design-resize competition. A resize engine re-lays out a design for a new canvas size; pictures are never regenerated, only moved, scaled or cropped, and text may change size but never wording or font.

The rubric (from the challenge brief):
{rubric}

There are {n} independent comparisons. Each has three images:
  REF  the original design (what the designer intended)
  A    one resize of REF to a new canvas
  B    another resize of REF to the same canvas
{open_line}
{items}
Open nothing else and run nothing else. Judge only what you see; do not guess where a version came from. Judge every comparison on its own.

For each comparison decide which of A and B is the better resize of its REF, per criterion:
  collisions   unwanted overlaps; text on a busy image
  readable     text large enough, enough contrast, nothing cut off
  brand        colours, fonts and logo intact and in proportion
  intentional  alignment, even spacing, the most important element stands out, picture proportions
  overall      which one is better on the whole rubric
Forced choice: always "A" or "B", even when they look alike. Also score each side 1-10 per criterion (10 = a demanding designer would ship it as is). confidence: 1 coin flip, 2 clear, 3 obvious.

Reply with exactly one JSON object and nothing else, with one entry per comparison id:
{{"{first}": {{"collisions": "A", "readable": "A", "brand": "B", "intentional": "A", "overall": "A", "confidence": 2, "scores": {{"A": {{"collisions": 7, "readable": 8, "brand": 9, "intentional": 6, "overall": 7}}, "B": {{"collisions": 6, "readable": 8, "brand": 9, "intentional": 5, "overall": 6}}}}, "why": "<at most 20 words naming the decisive difference>"}}{more}}}"""

OPEN_LINE = {
    "claude": "Read every image below with the Read tool (several files per call is fine).",
    "agent": "Open every image file below (they are in the current folder).",
}

SYSTEM = "You are a careful, blind design judge. You only read the image files you are told to read and answer in JSON."

LIMIT_RE = re.compile(r"rate.?limit|usage limit|quota|too many requests|\b429\b|limit (?:reached|exceeded)|"
                      r"out of (?:credits|usage)|hit your (?:\w+ )?limit", re.I)
CLAUDE_RETRY_SECS = 3600  # Claude's limits reset on rolling windows, not at UTC midnight


class CapReached(Exception):
    """The Cursor daily cap or a rate/usage limit: pause the gate, resume later."""

    def __init__(self, msg: str, reset_at: Optional[float] = None):
        super().__init__(msg)
        self.reset_at = reset_at if reset_at is not None else next_utc_day()


def next_utc_day(now: Optional[float] = None) -> float:
    t = _dt.datetime.fromtimestamp(now if now is not None else time.time(), _dt.timezone.utc)
    d = (t + _dt.timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return d.timestamp()


def judge_root() -> Path:
    return Path(os.environ.get("CHAMELEON_JUDGE_ROOT", Path.home() / "judge-adobe-hackathon-2026"))


def agent_ledger() -> Path:
    return judge_root() / "agent_calls.jsonl"


def _utc_day(t: Optional[float] = None) -> str:
    return _dt.datetime.fromtimestamp(t if t is not None else time.time(), _dt.timezone.utc).strftime("%Y-%m-%d")


def agent_calls_today(day: Optional[str] = None) -> int:
    p = agent_ledger()
    if not p.exists():
        return 0
    day = day or _utc_day()
    n = 0
    for ln in p.read_text().splitlines():
        try:
            n += json.loads(ln).get("day") == day
        except ValueError:
            continue
    return n


def prompt_for(items: List[dict], cli: str = "claude") -> str:
    """One call's prompt: ``items`` are dicts with id, ref, A, B (paths). The ``agent`` CLI runs
    from the asks folder and opens files by name; ``claude`` reads absolute paths."""
    def p(x):
        return Path(x).name if cli == "agent" else str(x)

    lines = "\n".join(f"  {it['id']}:  REF {p(it['ref'])}   A {p(it['A'])}   B {p(it['B'])}" for it in items)
    more = "".join(f', "{it["id"]}": {{...}}' for it in items[1:3]) + (", ..." if len(items) > 3 else "")
    return PROMPT.format(rubric=RUBRIC, n=len(items), open_line=OPEN_LINE.get(cli, OPEN_LINE["claude"]),
                         items=lines, first=items[0]["id"], more=more)


def parse_reply(r) -> Optional[dict]:
    """One comparison's verdict (a dict, or JSON text holding one), or None when it is not a
    complete forced choice."""
    if isinstance(r, str):
        if "{" not in r:
            return None
        try:
            r = json.loads(r[r.index("{"): r.rindex("}") + 1])
        except ValueError:
            return None
    if not isinstance(r, dict) or any(r.get(c) not in ("A", "B") for c in CRITERIA):
        return None
    out = {c: r[c] for c in CRITERIA}
    try:
        out["confidence"] = int(r.get("confidence", 1))
    except (TypeError, ValueError):
        out["confidence"] = 1
    sc = r.get("scores")
    if isinstance(sc, dict) and all(isinstance(sc.get(s), dict) for s in "AB"):
        try:
            out["scores"] = {s: {c: float(sc[s][c]) for c in CRITERIA} for s in "AB"}
        except (KeyError, TypeError, ValueError):
            pass
    out["why"] = str(r.get("why", ""))[:200]
    return out


def parse_batch(text: str, ids: List[str]) -> Dict[str, Optional[dict]]:
    """Per comparison id, its verdict or None. Accepts an object keyed by id, a list of objects
    carrying ``id``, and (for a batch of one) a bare verdict."""
    out: Dict[str, Optional[dict]] = {i: None for i in ids}
    if not text or "{" not in text:
        return out
    j = None
    for a, b in (("{", "}"), ("[", "]")):
        if a in text and b in text:
            try:
                j = json.loads(text[text.index(a): text.rindex(b) + 1])
                break
            except ValueError:
                continue
    if isinstance(j, list):
        j = {str(x.get("id")): x for x in j if isinstance(x, dict)}
    if not isinstance(j, dict):
        return out
    if len(ids) == 1 and ids[0] not in j and parse_reply(j):
        return {ids[0]: parse_reply(j)}
    for i in ids:
        if i in j:
            out[i] = parse_reply(j[i])
    return out


# ---- real judges ---------------------------------------------------------------------------------

def _bin(name: str) -> str:
    return shutil.which(name) or str(Path.home() / ".local" / "bin" / name)


class ClaudeJudge:
    """One fresh ``claude -p`` per call, from the gate's judge folder. Raises CapReached on a
    usage or rate limit, so a spent weekly limit pauses the gate instead of failing every Claude
    ask (which would leave the cursor family judging alone)."""

    def __init__(self, cwd: Path, timeout: float = 900):
        self.cwd, self.timeout = Path(cwd), timeout

    def __call__(self, prompt: str, model: str) -> Tuple[str, dict]:
        cmd = [_bin("claude"), "-p", prompt, "--model", model, "--allowedTools", "Read", "--tools", "Read",
               "--strict-mcp-config", "--disable-slash-commands", "--output-format", "json",
               "--system-prompt", SYSTEM]
        env = dict(os.environ, CLAUDE_CODE_DISABLE_CLAUDE_MDS="1", CLAUDE_CODE_DISABLE_AUTO_MEMORY="1",
                   CLAUDE_CODE_DISABLE_GIT_INSTRUCTIONS="1")
        p = subprocess.run(cmd, cwd=self.cwd, env=env, capture_output=True, text=True, timeout=self.timeout,
                           stdin=subprocess.DEVNULL)
        try:
            j = json.loads(p.stdout)
        except ValueError:
            j = None
        if not isinstance(j, dict) or j.get("is_error") or p.returncode != 0:
            res = str(j.get("result", "")) if isinstance(j, dict) else ""
            blob = res[-1000:] + (p.stdout or "")[-1000:] + (p.stderr or "")[-1000:]
            if LIMIT_RE.search(blob):
                raise CapReached(f"claude {model}: {blob.strip()[-200:]}", time.time() + CLAUDE_RETRY_SECS)
            if not isinstance(j, dict):
                return p.stdout[-2000:], {"error": (p.stderr or "")[-300:], "exit": p.returncode}
        u = j.get("usage") or {}
        usage = {"input": u.get("input_tokens", 0), "cache_read": u.get("cache_read_input_tokens", 0),
                 "cache_write": u.get("cache_creation_input_tokens", 0), "output": u.get("output_tokens", 0),
                 "turns": j.get("num_turns"), "cost_usd": j.get("total_cost_usd")}
        return str(j.get("result", "")), usage


class CursorJudge:
    """One fresh ``agent -p --mode=ask`` (Cursor CLI, read-only) per call, from the asks folder.
    Raises CapReached on a rate or usage limit."""

    def __init__(self, cwd: Path, timeout: float = 900):
        self.cwd, self.timeout = Path(cwd), timeout

    def __call__(self, prompt: str, model: str) -> Tuple[str, dict]:
        cmd = [_bin("agent"), "-p", "--trust", "--mode=ask", "--model", model, "--output-format", "json", prompt]
        p = subprocess.run(cmd, cwd=self.cwd, capture_output=True, text=True, timeout=self.timeout,
                           stdin=subprocess.DEVNULL)
        try:
            j = json.loads(p.stdout.strip().splitlines()[-1]) if p.stdout.strip() else None
        except ValueError:
            j = None
        if not isinstance(j, dict) or j.get("is_error") or p.returncode != 0:
            blob = (p.stdout or "")[-1000:] + (p.stderr or "")[-1000:]
            if LIMIT_RE.search(blob):
                raise CapReached(f"agent {model}: {blob.strip()[-200:]}")
            return (p.stdout or "")[-2000:], {"error": (p.stderr or "")[-300:], "exit": p.returncode}
        u = j.get("usage") or {}
        usage = {"input": u.get("inputTokens", 0), "cache_read": u.get("cacheReadTokens", 0),
                 "cache_write": u.get("cacheWriteTokens", 0), "output": u.get("outputTokens", 0),
                 "secs_api": round((j.get("duration_api_ms") or 0) / 1000, 1)}
        return str(j.get("result", "")), usage


def _seeded(*parts) -> random.Random:
    return random.Random(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest())


def _other(side: str) -> str:
    return "B" if side == "A" else "A"


# ---- tile scores -----------------------------------------------------------------------------

def score_tile(rec: dict, active: List[str]) -> Optional[dict]:
    """``rec`` (from Panel.judge_tiles) scored over the active families: adds ``score`` (pooled),
    ``family`` (each active family's own vote) and ``delta``. None when no active family voted."""
    votes = rec.get("votes") or {}
    prim = [f for f in active if votes.get(f)]
    if not prim:
        return None
    tb = votes.get(TIEBREAK) if len(prim) == 2 else None
    score = {}
    for c in CRITERIA:
        v = [votes[f][c] for f in prim]
        if len(v) == 1 or v[0] == v[1]:
            score[c] = v[0]
        elif tb is not None:
            score[c] = tb[c]  # the two split: the third vote is the majority of three
        else:
            score[c] = 0.5
    ds = [rec["deltas"][f] for f in prim if (rec.get("deltas") or {}).get(f)]
    delta = {c: sum(d[c] for d in ds if c in d) / sum(1 for d in ds if c in d)
             for c in CRITERIA if any(c in d for d in ds)}
    return dict(rec, score=score, family={f: dict(votes[f]) for f in prim}, delta=delta,
                tiebreak_used=tb is not None and votes[prim[0]]["overall"] != votes[prim[1]]["overall"])


def score_tiles(recs: Iterable[dict], active: List[str]) -> List[dict]:
    return [s for s in (score_tile(r, active) for r in recs) if s is not None]


class Panel:
    """Asks, labels and views of one gate run. ``private`` holds everything that ties an ask to a
    tile (never shown to a judge); the judge folder holds only the blind images."""

    def __init__(self, gate_id: str, private: Path, cfg: Optional[dict] = None, judge=None, seed: int = 0):
        self.cfg = dict(DEFAULT_PANEL, **(cfg or {}))
        self.gate_id, self.seed = gate_id, seed
        self.dir = judge_root() / gate_id
        self.ask_dir = self.dir / "asks"
        self.ask_dir.mkdir(parents=True, exist_ok=True)
        self.private = Path(private)
        self.private.mkdir(parents=True, exist_ok=True)
        os.chmod(self.private, 0o700)
        self.views_dir = self.private / "views"
        self.roster: Dict[str, dict] = dict(self.cfg["families"])
        if self.cfg.get("tiebreak"):
            self.roster[TIEBREAK] = dict(self.cfg["tiebreak"])
        self.families: List[str] = list(self.cfg["families"])
        self.judges = {f: self._judge_for(f, judge) for f in self.roster}
        for j in {id(j): j for j in self.judges.values()}.values():
            if hasattr(j, "bind"):  # test judges look up the hidden side; real ones never can
                j.bind(self)
        self.rng = _seeded(seed, gate_id, "names")
        self._lock = threading.Lock()
        self.asks: Dict[str, dict] = {}
        self.labels: Dict[str, dict] = {}
        self.calls: List[dict] = []
        self._index: Dict[tuple, List[str]] = {}
        for a in self._read("asks.jsonl"):
            self._add_ask(a, write=False)
        for lab in self._read("labels.jsonl"):
            self.labels[lab["ask"]] = lab
        self.calls = self._read("calls.jsonl")

    def _judge_for(self, fam: str, judge):
        if isinstance(judge, dict):
            return judge.get(fam) or judge["*"]
        if judge is not None:
            return judge
        cli = self.roster[fam]["cli"]
        if cli == "agent":
            return CursorJudge(self.ask_dir, self.cfg["timeout"])
        return ClaudeJudge(self.dir, self.cfg["timeout"])

    # ---- persistence -------------------------------------------------------------------------

    def _read(self, name: str) -> List[dict]:
        p = self.private / name
        return [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()] if p.exists() else []

    def _append(self, name: str, rec: dict) -> None:
        with self._lock, (self.private / name).open("a") as f:
            f.write(json.dumps(rec) + "\n")

    def _add_ask(self, a: dict, write: bool = True) -> None:
        self.asks[a["ask"]] = a
        self._index.setdefault((a["kind"], a["key"], a["family"]), []).append(a["ask"])
        if write:
            self._append("asks.jsonl", a)

    # ---- views ---------------------------------------------------------------------------------

    def view(self, design: dict, name: str, kit: Optional[Path] = None, reencode: bool = False) -> Path:
        """The judge's view of ``design`` (cached under ``name``). Rendering reads the kit's assets
        through CHAMELEON_BRAND_KIT, so views are made one at a time."""
        from tools.judge.views import views

        if kit:
            os.environ["CHAMELEON_BRAND_KIT"] = str(Path(kit).resolve())
        size = int(self.cfg["view"])
        p = views(design, name, self.views_dir, sizes=(size,))[size]
        if reencode:
            q = p.with_name(p.stem + ".re.png")
            if not q.exists():
                import io

                from PIL import Image

                buf = io.BytesIO()
                with Image.open(p) as im:
                    im.convert("RGB").save(buf, "JPEG", quality=92)
                buf.seek(0)
                with Image.open(buf) as im:
                    im.save(q)
            p = q
        return p

    # ---- asks ----------------------------------------------------------------------------------

    def pair(self, kind: str, key: str, ref: Path, x: Path, y: Path, family: str, side: str, batch: int) -> str:
        """One ask of x against y for ``family`` with x on ``side``. Reuses the ask already made
        for (kind, key, family), so a resumed run asks nothing twice."""
        have = self._index.get((kind, key, family))
        if have:
            return have[0]
        from tools.judge.views import blind_ask

        a, b = (x, y) if side == "A" else (y, x)
        ask = blind_ask(ref, a, b, self.ask_dir, self.rng, set(self.asks))
        rec = {"ask": ask["id"], "kind": kind, "key": key, "x_side": side, "family": family,
               "model": self.roster[family]["model"], "batch": batch,
               "ref": str(ask["ref"]), "A": str(ask["A"]), "B": str(ask["B"])}
        self._add_ask(rec)
        return ask["id"]

    def _sides(self, kind: str, key: str, fams: List[str]) -> Dict[str, str]:
        """Per family, the side x sits on: random per key, and opposite for the two families."""
        r = _seeded(self.seed, self.gate_id, "order", kind, key)
        first = r.choice("AB")
        order = list(fams)
        r.shuffle(order)
        return {f: (first if i % 2 == 0 else _other(first)) for i, f in enumerate(order)}

    def _cap_check(self) -> None:
        cap = int(self.cfg.get("agent_daily_cap") or 0)
        if cap and agent_calls_today() >= cap:
            raise CapReached(f"agent daily cap: {cap} calls already made today (UTC)")

    def _call(self, fam: str, items: List[str], abort: threading.Event) -> None:
        """Judge ``items`` (ask ids of one family) in one call, retrying the ones a reply left out."""
        cli = self.roster[fam]["cli"]
        model = self.roster[fam]["model"]
        left, errs, calls = list(items), {}, []
        for _ in range(1 + int(self.cfg["retries"])):
            if not left or abort.is_set():
                break
            if cli == "agent":
                with self._lock:
                    self._cap_check()
                    agent_ledger().parent.mkdir(parents=True, exist_ok=True)
                    with agent_ledger().open("a") as f:
                        f.write(json.dumps({"day": _utc_day(), "at": round(time.time()), "gate": self.gate_id,
                                            "model": model, "items": len(left)}) + "\n")
            its = [{"id": a, "ref": self.asks[a]["ref"], "A": self.asks[a]["A"], "B": self.asks[a]["B"]} for a in left]
            t0 = time.time()
            cid = f"{fam}.{len(self.calls)}.{os.getpid()}.{threading.get_ident() % 100000}.{int(t0 * 1000) % 10**8}"
            try:
                text, u = self.judges[fam](prompt_for(its, cli), model)
                got = parse_batch(text, left)
                err = None
            except CapReached:
                raise
            except Exception as ex:  # noqa: BLE001 - a failed call is recorded, not fatal
                got, u, err = {a: None for a in left}, {}, f"{type(ex).__name__}: {ex}"[:200]
            call = {"call": cid, "family": fam, "cli": cli, "model": model, "items": len(left),
                    "ok": sum(1 for v in got.values() if v), "usage": u, "secs": round(time.time() - t0, 1),
                    "error": err, "day": _utc_day()}
            self._append("calls.jsonl", call)
            with self._lock:
                self.calls.append(call)
            calls.append(cid)
            for a, lab in got.items():
                if lab:
                    rec = {"ask": a, "ok": True, "label": lab, "calls": list(calls), "error": None}
                    self._append("labels.jsonl", rec)
                    with self._lock:
                        self.labels[a] = rec
                else:
                    errs[a] = err or "missing or unparsable in the reply"
            left = [a for a in left if a not in self.labels]
        if abort.is_set():
            return
        for a in left:
            rec = {"ask": a, "ok": False, "label": None, "calls": calls, "error": errs.get(a)}
            self._append("labels.jsonl", rec)
            with self._lock:
                self.labels[a] = rec

    def run(self, ask_ids: Iterable[str]) -> None:
        """Judge every ask that has no label yet: ``batch`` asks per call, grouped by family;
        claude calls ``concurrency`` at once, agent calls ``agent_concurrency`` at once. Raises
        CapReached (after the calls in flight finish) when the agent cap or a limit is hit."""
        todo = [a for a in dict.fromkeys(ask_ids) if a not in self.labels]
        if not todo:
            return
        n = max(1, int(self.cfg["batch"]))
        chunks: Dict[str, List[Tuple[str, List[str]]]] = {c: [] for c in CLIS}
        for fam in self.roster:
            ids = [a for a in todo if self.asks[a]["family"] == fam]
            for i in range(0, len(ids), n):
                chunks[self.roster[fam]["cli"]].append((fam, ids[i: i + n]))
        abort = threading.Event()
        caps: List[CapReached] = []

        def go(job):
            if abort.is_set():
                return
            try:
                self._call(job[0], job[1], abort)
            except CapReached as e:
                abort.set()
                caps.append(e)

        pools = []
        for cli, w in (("claude", self.cfg["concurrency"]), ("agent", self.cfg["agent_concurrency"])):
            if chunks[cli]:
                ex = ThreadPoolExecutor(max(1, min(4, int(w))))
                pools.append((ex, [ex.submit(go, j) for j in chunks[cli]]))
        for ex, futs in pools:
            for f in futs:
                f.result()
            ex.shutdown()
        if caps:
            raise caps[0]

    def vote(self, aid: Optional[str]) -> Optional[Dict[str, float]]:
        """Per criterion 1.0 when the ask chose x, 0.0 for y; None when it failed or is missing."""
        lab = self.labels.get(aid) if aid else None
        if not lab or not lab["ok"]:
            return None
        side = self.asks[aid]["x_side"]
        return {c: 1.0 if lab["label"][c] == side else 0.0 for c in CRITERIA}

    def votes(self, ask_ids: Iterable[str]) -> Dict[str, List[float]]:
        out: Dict[str, List[float]] = {c: [] for c in CRITERIA}
        for aid in ask_ids:
            v = self.vote(aid)
            if v:
                for c in CRITERIA:
                    out[c].append(v[c])
        return out

    def delta(self, aid: Optional[str]) -> Optional[Dict[str, float]]:
        lab = self.labels.get(aid) if aid else None
        if not lab or not lab["ok"] or "scores" not in lab["label"]:
            return None
        xs = self.asks[aid]["x_side"]
        sc = lab["label"]["scores"]
        return {c: sc[xs][c] - sc[_other(xs)][c] for c in CRITERIA}

    # ---- a batch of changed tiles ------------------------------------------------------------

    def tile_views(self, t: dict, reencode_cand: bool = False) -> Tuple[Path, Path, Path]:
        n = f"{t['set']}.{t['tile']}"
        kit = t.get("kit")
        return (self.view(t["source"], f"{n}.src", kit), self.view(t["base"], f"{n}.base", kit),
                self.view(t["cand"], f"{n}.cand", kit, reencode=reencode_cand))

    def sentinels(self, pool: List[dict], n: int, batch: int, fams: List[str], small: bool = False
                  ) -> Dict[str, List[str]]:
        """``n`` sentinel comparisons, each asked of every family in ``fams``: a good design
        against a known-worse one, x = the good side. ``small``: the small-defect kinds instead
        (ask kind ``sentinel_small``, report only). Returns ask ids per family."""
        from tools.judge.perturb import perturb

        out: Dict[str, List[str]] = {f: [] for f in fams}
        if not pool or n <= 0:
            return out
        ask_kind = "sentinel_small" if small else "sentinel"
        rng = _seeded(self.seed, self.gate_id, "small_sentinels" if small else "sentinels", batch)
        kinds = list(self.cfg["report_kinds"] if small else self.cfg["sentinel_kinds"])
        made, tries = 0, 0
        while made < n and tries < 20 * n:
            tries += 1
            t = rng.choice(pool)
            kind = rng.choice(kinds)
            good = t.get("reference") or t["base"]
            tgt = {"width": good["canvas"]["width"], "height": good["canvas"]["height"]}
            if kind == "fit_centre":
                if not t.get("reference"):
                    continue  # the engine's own output is no ground truth against fit-and-centre
                bad = fit_centre(t["source"], tgt)
                if abs(t["source"]["canvas"]["width"] / t["source"]["canvas"]["height"]
                       - tgt["width"] / tgt["height"]) < 0.05:
                    continue  # same shape: fit-and-centre is not worse
            else:
                bad = perturb(kind, good, random.Random(rng.random()))
            if bad is None:
                continue
            key = f"{batch}.{made}.{kind}"
            n0 = f"{t['set']}.{t['tile']}"
            kit = t.get("kit")
            ref = self.view(t["source"], f"{n0}.src", kit)
            gv = self.view(good, f"{n0}.good", kit)
            bv = self.view(bad, f"{'small' if small else 'sent'}.{key}", kit)
            for f, side in self._sides(ask_kind, key, fams).items():
                out[f].append(self.pair(ask_kind, key, ref, gv, bv, f, side, batch))
            made += 1
        return out

    def sentinel_accuracy(self, ask_ids: Iterable[str]) -> dict:
        """Overall-vote accuracy on sentinel asks, in total and per perturbation kind."""
        by: Dict[str, List[float]] = {}
        for aid in ask_ids:
            v = self.vote(aid)
            if v:
                by.setdefault(self.asks[aid]["key"].rsplit(".", 1)[-1], []).append(v["overall"])
        allv = [x for v in by.values() for x in v]
        return {"n": len(allv), "correct": int(sum(allv)), "accuracy": (sum(allv) / len(allv)) if allv else None,
                "by_kind": {k: round(sum(v) / len(v), 3) for k, v in sorted(by.items())}}

    def sentinel_kinds(self) -> Dict[str, Dict[str, dict]]:
        """Per ask kind (``sentinel``, ``sentinel_small``), family and perturbation kind: how many
        sentinel asks were answered and how many chose the good side (overall)."""
        out: Dict[str, Dict[str, dict]] = {}
        for aid, a in self.asks.items():
            if a["kind"] not in ("sentinel", "sentinel_small"):
                continue
            v = self.vote(aid)
            if not v:
                continue
            m = out.setdefault(a["kind"], {}).setdefault(a["family"], {}).setdefault(
                a["key"].rsplit(".", 1)[-1], {"n": 0, "correct": 0})
            m["n"] += 1
            m["correct"] += int(v["overall"])
        return out

    def judge_tiles(self, tiles: List[dict], batch: int, pool: Optional[List[dict]] = None,
                    share: float = 0.10, reencode_cand: bool = False, families: Optional[List[str]] = None) -> dict:
        """Judge a batch of changed tiles (dicts with set, tile, design, tier, source, base, cand,
        kit, optional reference) with the active ``families`` (default: all). Returns raw tile
        records (``votes`` per family and tie-break; score them with ``score_tiles``) and each
        family's sentinel accuracy on this batch."""
        fams = list(families) if families is not None else list(self.families)
        first: Dict[str, Dict[str, str]] = {}
        views: Dict[str, tuple] = {}
        for t in tiles:
            k = f"{t['set']}/{t['tile']}"
            views[k] = ref, bv, cv = self.tile_views(t, reencode_cand)
            first[k] = {f: self.pair("pair", k, ref, cv, bv, f, side, batch)
                        for f, side in self._sides("pair", k, fams).items()}
        n_sent = 0
        if share > 0:
            # per family as many sentinels as a two-asks-per-tile panel at this share
            n_sent = max(int(self.cfg["min_sentinels"]), math.ceil(share / (1 - share) * 2 * len(tiles)))
        src = pool if pool is not None else tiles
        sent = self.sentinels(src, n_sent, batch, fams)
        small = self.sentinels(src, math.ceil(float(self.cfg.get("small_sentinels", 0) or 0) * n_sent), batch,
                               fams, small=True)
        order = ([a for v in first.values() for a in v.values()] + [a for v in sent.values() for a in v]
                 + [a for v in small.values() for a in v])
        _seeded(self.seed, self.gate_id, "mix", batch).shuffle(order)
        self.run(order)
        tb: Dict[str, str] = {}
        if len(fams) >= 2 and TIEBREAK in self.roster:
            for k, d in first.items():
                v = [self.vote(d[f]) for f in fams[:2]]
                if v[0] and v[1] and v[0]["overall"] != v[1]["overall"]:
                    ref, bv, cv = views[k]
                    side = _seeded(self.seed, self.gate_id, "tiebreak", k).choice("AB")
                    tb[k] = self.pair("pair", k, ref, cv, bv, TIEBREAK, side, batch)
            self.run(list(tb.values()))
        recs = []
        for t in tiles:
            k = f"{t['set']}/{t['tile']}"
            ids = dict(first[k], **({TIEBREAK: tb[k]} if k in tb else {}))
            votes = {f: self.vote(a) for f, a in ids.items()}
            if not any(votes.get(f) for f in fams):
                continue  # every ask failed: no evidence either way
            recs.append({"tile": t["tile"], "set": t["set"], "design": t["design"], "tier": t["tier"],
                         "bleed": t.get("bleed"), "votes": {f: v for f, v in votes.items() if v},
                         "deltas": {f: d for f, d in ((f, self.delta(a)) for f, a in ids.items()) if d},
                         "asks": len(ids), "tiebreak": k in tb})
        asked = order + list(tb.values())
        return {"tiles": recs,
                "sentinels": {f: self.sentinel_accuracy(sent[f]) for f in fams},
                "small_sentinels": {f: self.sentinel_accuracy(small[f]) for f in fams},
                "asks": len(asked), "tiebreaks": len(tb),
                "failed": sum(1 for a in asked if a in self.labels and not self.labels[a]["ok"])}

    def compare(self, kind: str, items: List[dict], batch: int, families: Optional[List[str]] = None) -> List[dict]:
        """Anchor asks: x vs y per item (dicts with key, source, x, y, kit), one ask per family in
        opposite orders. Returns per item the share of votes for x on overall."""
        fams = list(families) if families is not None else list(self.families)
        ids: Dict[str, List[str]] = {}
        for it in items:
            kit = it.get("kit")
            n = f"{kind}.{it['key']}"
            ref = self.view(it["source"], f"{n}.src", kit)
            xv = it["x_png"] if it.get("x_png") else self.view(it["x"], f"{n}.x", kit)
            yv = it["y_png"] if it.get("y_png") else self.view(it["y"], f"{n}.y", kit)
            ids[it["key"]] = [self.pair(kind, it["key"], ref, Path(xv), Path(yv), f, side, batch)
                              for f, side in self._sides(kind, it["key"], fams).items()]
        self.run([a for v in ids.values() for a in v])
        out = []
        for k, v in ids.items():
            o = self.votes(v)["overall"]
            if o:
                out.append({"key": k, "x_share": sum(o) / len(o)})
        return out

    def usage(self) -> dict:
        """Calls and asks made and context spent, in total and per family. ``cursor_asks`` counts
        ``agent`` calls (the unit of the Cursor cap)."""
        def ctx(u):
            return (u.get("input", 0) or 0) + (u.get("cache_read", 0) or 0) + (u.get("cache_write", 0) or 0)

        fam: Dict[str, dict] = {}
        for c in self.calls:
            f = fam.setdefault(c["family"], {"calls": 0, "items": 0, "ok": 0, "ctx": [], "cost_usd": 0.0})
            f["calls"] += 1
            f["items"] += c["items"]
            f["ok"] += c["ok"]
            f["ctx"].append(ctx(c.get("usage") or {}))
            f["cost_usd"] += (c.get("usage") or {}).get("cost_usd") or 0
        per = {k: {"calls": v["calls"], "items": v["items"], "ok": v["ok"],
                   "context_tokens_mean": round(sum(v["ctx"]) / len(v["ctx"])) if v["ctx"] else 0,
                   "context_tokens_max": max(v["ctx"]) if v["ctx"] else 0,
                   "context_per_ask": round(sum(v["ctx"]) / v["items"]) if v["items"] else 0,
                   "cost_usd": round(v["cost_usd"], 3)} for k, v in sorted(fam.items())}
        toks = [x for v in fam.values() for x in v["ctx"]]
        return {"asks": len(self.labels), "failed": sum(1 for lab in self.labels.values() if not lab["ok"]),
                "calls": len(self.calls),
                "cursor_asks": sum(1 for c in self.calls if c["cli"] == "agent"),
                "context_tokens_mean": round(sum(toks) / len(toks)) if toks else 0,
                "context_tokens_max": max(toks) if toks else 0,
                "cost_usd": round(sum(v["cost_usd"] for v in fam.values()), 3), "by_family": per}

    def clean_images(self) -> None:
        """Remove the blind images once the run is decided (labels stay in the private folder)."""
        shutil.rmtree(self.ask_dir, ignore_errors=True)
