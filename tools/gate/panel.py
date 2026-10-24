"""Gate tool: the blind judge panel.

Every ask is one fresh ``claude -p`` call (Sonnet, the Read tool only) run from
``~/judge-adobe-hackathon-2026/<gate-id>/`` so the loop's usage meter counts it. The judge sees
three images under random names (``tools.judge.views.blind_ask``): REF, the source design, and
A/B, two resizes. Which side is which lives only in the gate's private folder.

Per changed tile:

* the source is the reference image; left/right is random per tile;
* one judge asks both orders; a second judge (both orders again) only when the first two
  disagree on overall;
* forced choice on the per-tile brief criteria (collisions, readable, brand, intentional) and
  overall, plus a 1-10 score per criterion for each side. (The brief's fifth row, consistency
  across formats, is not a per-tile question.)

About ``sentinel_share`` of the asks in each batch are **sentinels** with a known answer: the
good design (the designer's resize where the task has one, else the base output) against a
perturbation of it (``tools.judge.perturb.SENTINELS``) or against fit-and-centre. A batch whose
sentinel accuracy on overall is below ``sentinel_min`` is discarded by run.py.

The judge is a callable ``judge(prompt, model) -> (reply_text, usage_dict)``; unit tests pass
a fake one. ``ClaudeJudge`` is the real one; its context is kept small (a one-line system
prompt, no CLAUDE.md, no MCP, only the Read tool).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from tools.gate.baselines import fit_centre

CRITERIA = ["collisions", "readable", "brand", "intentional", "overall"]

DEFAULT_PANEL = {
    "model": "claude-sonnet-5-5",
    "second_model": "claude-sonnet-5-5",
    "concurrency": 4,
    "view": 1024,          # long side of every image the judge sees
    "timeout": 300,
    "retries": 1,
    "min_sentinels": 4,
}

RUBRIC = """  - No unwanted collisions: overlaps are fine when they're intentional (layered graphics, deliberate stacking); what matters is that elements that shouldn't overlap, don't, and text never sits on a busy image.
  - Everything is readable: text is large enough and has enough contrast; nothing is cut off.
  - It stays on brand: colours, fonts, and the logo survive the resize intact and in proportion.
  - It looks intentional: things line up, spacing is even, the most important element still stands out, pictures keep their proportions."""

PROMPT = """You judge a design-resize competition. A resize engine re-lays out a design for a new canvas size; pictures are never regenerated, only moved, scaled or cropped, and text may change size but never wording or font.

The rubric (from the challenge brief):
{rubric}

Read these three images with the Read tool, all three in one call:
  REF  {ref}   the original design (what the designer intended)
  A    {a}   one resize of REF to the new canvas
  B    {b}   another resize of REF to the same canvas
Open nothing else and run nothing else. Judge only what you see; do not guess where a version came from.

Decide which of A and B is the better resize of REF, per criterion:
  collisions   unwanted overlaps; text on a busy image
  readable     text large enough, enough contrast, nothing cut off
  brand        colours, fonts and logo intact and in proportion
  intentional  alignment, even spacing, the most important element stands out, picture proportions
  overall      which one is better on the whole rubric
Forced choice: always "A" or "B", even when they look alike. Also score each side 1-10 per criterion (10 = a demanding designer would ship it as is). confidence: 1 coin flip, 2 clear, 3 obvious.

Reply with exactly one JSON object and nothing else:
{{"collisions": "A", "readable": "A", "brand": "B", "intentional": "A", "overall": "A", "confidence": 2, "scores": {{"A": {{"collisions": 7, "readable": 8, "brand": 9, "intentional": 6, "overall": 7}}, "B": {{"collisions": 6, "readable": 8, "brand": 9, "intentional": 5, "overall": 6}}}}, "why": "<at most 20 words naming the decisive difference>"}}"""

SYSTEM = "You are a careful, blind design judge. You only read the image files you are told to read and answer in JSON."


def judge_root() -> Path:
    return Path(os.environ.get("CHAMELEON_JUDGE_ROOT", Path.home() / "judge-adobe-hackathon-2026"))


def prompt_for(ref: Path, a: Path, b: Path) -> str:
    return PROMPT.format(rubric=RUBRIC, ref=ref, a=a, b=b)


def parse_reply(text: str) -> Optional[dict]:
    """The judge's JSON, or None when it is not a complete forced choice."""
    if not text or "{" not in text:
        return None
    try:
        r = json.loads(text[text.index("{"): text.rindex("}") + 1])
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


def _claude_bin() -> str:
    return shutil.which("claude") or str(Path.home() / ".local" / "bin" / "claude")


class ClaudeJudge:
    """One fresh ``claude -p`` per ask, from the gate's judge folder."""

    def __init__(self, cwd: Path, timeout: float = 300):
        self.cwd, self.timeout = Path(cwd), timeout

    def __call__(self, prompt: str, model: str) -> Tuple[str, dict]:
        cmd = [_claude_bin(), "-p", prompt, "--model", model, "--allowedTools", "Read", "--tools", "Read",
               "--strict-mcp-config", "--disable-slash-commands", "--output-format", "json",
               "--system-prompt", SYSTEM]
        env = dict(os.environ, CLAUDE_CODE_DISABLE_CLAUDE_MDS="1", CLAUDE_CODE_DISABLE_AUTO_MEMORY="1",
                   CLAUDE_CODE_DISABLE_GIT_INSTRUCTIONS="1")
        p = subprocess.run(cmd, cwd=self.cwd, env=env, capture_output=True, text=True, timeout=self.timeout,
                           stdin=subprocess.DEVNULL)
        try:
            j = json.loads(p.stdout)
        except ValueError:
            return p.stdout[-2000:], {"error": (p.stderr or "")[-300:], "exit": p.returncode}
        u = j.get("usage") or {}
        usage = {"input": u.get("input_tokens", 0), "cache_read": u.get("cache_read_input_tokens", 0),
                 "cache_write": u.get("cache_creation_input_tokens", 0), "output": u.get("output_tokens", 0),
                 "turns": j.get("num_turns"), "cost_usd": j.get("total_cost_usd")}
        return str(j.get("result", "")), usage


def _seeded(*parts) -> random.Random:
    return random.Random(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest())


class Panel:
    """Asks, labels and views of one gate run. ``private`` holds everything that ties an ask to a
    tile (never shown to a judge); the judge folder holds only the blind images."""

    def __init__(self, gate_id: str, private: Path, cfg: Optional[dict] = None,
                 judge: Optional[Callable[[str, str], Tuple[str, dict]]] = None, seed: int = 0):
        self.cfg = dict(DEFAULT_PANEL, **(cfg or {}))
        self.gate_id, self.seed = gate_id, seed
        self.dir = judge_root() / gate_id
        self.ask_dir = self.dir / "asks"
        self.ask_dir.mkdir(parents=True, exist_ok=True)
        self.private = Path(private)
        self.private.mkdir(parents=True, exist_ok=True)
        os.chmod(self.private, 0o700)
        self.views_dir = self.private / "views"
        self.judge = judge or ClaudeJudge(self.dir, self.cfg["timeout"])
        if hasattr(self.judge, "bind"):  # test judges look up the hidden side; real ones never can
            self.judge.bind(self)
        self.rng = _seeded(seed, gate_id, "names")
        self._lock = threading.Lock()
        self.asks: Dict[str, dict] = {}
        self.labels: Dict[str, dict] = {}
        self._index: Dict[tuple, List[str]] = {}
        for a in self._read("asks.jsonl"):
            self._add_ask(a, write=False)
        for lab in self._read("labels.jsonl"):
            self.labels[lab["ask"]] = lab

    # ---- persistence -------------------------------------------------------------------------

    def _read(self, name: str) -> List[dict]:
        p = self.private / name
        return [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()] if p.exists() else []

    def _append(self, name: str, rec: dict) -> None:
        with self._lock, (self.private / name).open("a") as f:
            f.write(json.dumps(rec) + "\n")

    def _add_ask(self, a: dict, write: bool = True) -> None:
        self.asks[a["ask"]] = a
        self._index.setdefault((a["kind"], a["key"], a["judge"]), []).append(a["ask"])
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

    def pair(self, kind: str, key: str, ref: Path, x: Path, y: Path, judge: int, batch: int,
             orders: int = 2) -> List[str]:
        """``orders`` asks of x against y (both orders when 2; x's side random per tile).
        Reuses asks already made for (kind, key, judge), so a resumed run asks nothing twice."""
        have = self._index.get((kind, key, judge))
        if have:
            return list(have)
        from tools.judge.views import blind_ask

        first = _seeded(self.seed, self.gate_id, kind, key, judge).choice("AB")
        sides = [first, "B" if first == "A" else "A"][:orders]
        ids = []
        model = self.cfg["model"] if judge == 0 else self.cfg["second_model"]
        for side in sides:
            a, b = (x, y) if side == "A" else (y, x)
            ask = blind_ask(ref, a, b, self.ask_dir, self.rng, set(self.asks))
            rec = {"ask": ask["id"], "kind": kind, "key": key, "x_side": side, "judge": judge, "model": model,
                   "batch": batch, "ref": str(ask["ref"]), "A": str(ask["A"]), "B": str(ask["B"])}
            self._add_ask(rec)
            ids.append(ask["id"])
        return ids

    def run(self, ask_ids: Iterable[str]) -> None:
        """Judge every ask that has no label yet, at most ``concurrency`` at once."""
        todo = [a for a in ask_ids if a not in self.labels]
        if not todo:
            return

        def one(aid: str) -> None:
            a = self.asks[aid]
            prompt = prompt_for(Path(a["ref"]), Path(a["A"]), Path(a["B"]))
            lab, usage, t0, err = None, [], time.time(), None
            for _ in range(1 + int(self.cfg["retries"])):
                try:
                    text, u = self.judge(prompt, a["model"])
                    usage.append(u)
                    lab = parse_reply(text)
                    if lab:
                        break
                    err = "unparsable reply"
                except Exception as ex:  # noqa: BLE001 - a failed ask is recorded, not fatal
                    err = f"{type(ex).__name__}: {ex}"[:200]
            rec = {"ask": aid, "ok": lab is not None, "label": lab, "usage": usage, "error": None if lab else err,
                   "secs": round(time.time() - t0, 1)}
            self._append("labels.jsonl", rec)
            with self._lock:
                self.labels[aid] = rec

        workers = max(1, min(4, int(self.cfg["concurrency"])))
        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(one, todo))

    def votes(self, ask_ids: Iterable[str]) -> Dict[str, List[float]]:
        """Per criterion, 1.0 for every ask that chose x, 0.0 for y (failed asks are skipped)."""
        out: Dict[str, List[float]] = {c: [] for c in CRITERIA}
        for aid in ask_ids:
            lab = self.labels.get(aid)
            if not lab or not lab["ok"]:
                continue
            side = self.asks[aid]["x_side"]
            for c in CRITERIA:
                out[c].append(1.0 if lab["label"][c] == side else 0.0)
        return out

    def deltas(self, ask_ids: Iterable[str]) -> Dict[str, float]:
        acc: Dict[str, List[float]] = {}
        for aid in ask_ids:
            lab = self.labels.get(aid)
            if not lab or not lab["ok"] or "scores" not in lab["label"]:
                continue
            xs = self.asks[aid]["x_side"]
            ys = "B" if xs == "A" else "A"
            for c in CRITERIA:
                acc.setdefault(c, []).append(lab["label"]["scores"][xs][c] - lab["label"]["scores"][ys][c])
        return {c: sum(v) / len(v) for c, v in acc.items()}

    # ---- a batch of changed tiles ------------------------------------------------------------

    def tile_views(self, t: dict, reencode_cand: bool = False) -> Tuple[Path, Path, Path]:
        n = f"{t['set']}.{t['tile']}"
        kit = t.get("kit")
        return (self.view(t["source"], f"{n}.src", kit), self.view(t["base"], f"{n}.base", kit),
                self.view(t["cand"], f"{n}.cand", kit, reencode=reencode_cand))

    def sentinels(self, pool: List[dict], n: int, batch: int) -> List[str]:
        """``n`` sentinel asks: a good design against a known-worse one, x = the good side."""
        from tools.judge.perturb import SENTINELS, perturb

        if not pool or n <= 0:
            return []
        rng = _seeded(self.seed, self.gate_id, "sentinels", batch)
        kinds = list(SENTINELS) + ["fit_centre"]
        ids: List[str] = []
        tries = 0
        while len(ids) < n and tries < 20 * n:
            tries += 1
            t = rng.choice(pool)
            kind = rng.choice(kinds)
            good = t.get("reference") or t["base"]
            tgt = {"width": good["canvas"]["width"], "height": good["canvas"]["height"]}
            if kind == "fit_centre":
                bad = fit_centre(t["source"], tgt)
                if abs(t["source"]["canvas"]["width"] / t["source"]["canvas"]["height"]
                       - tgt["width"] / tgt["height"]) < 0.05:
                    continue  # same shape: fit-and-centre is not worse
            else:
                bad = perturb(kind, good, random.Random(rng.random()))
            if bad is None:
                continue
            key = f"{batch}.{len(ids)}.{kind}"
            n0 = f"{t['set']}.{t['tile']}"
            kit = t.get("kit")
            ref = self.view(t["source"], f"{n0}.src", kit)
            gv = self.view(good, f"{n0}.good", kit)
            bv = self.view(bad, f"sent.{key}", kit)
            ids += self.pair("sentinel", key, ref, gv, bv, 0, batch, orders=1)
        return ids

    def judge_tiles(self, tiles: List[dict], batch: int, pool: Optional[List[dict]] = None,
                    share: float = 0.10, reencode_cand: bool = False) -> dict:
        """Judge a batch of changed tiles (dicts with set, tile, design, tier, source, base, cand,
        kit, optional reference). Returns tile records for tools.gate.stats and the batch's
        sentinel accuracy."""
        first: Dict[str, List[str]] = {}
        views: Dict[str, tuple] = {}
        for t in tiles:
            k = f"{t['set']}/{t['tile']}"
            views[k] = ref, bv, cv = self.tile_views(t, reencode_cand)
            first[k] = self.pair("pair", k, ref, cv, bv, 0, batch)
        n_sent = 0
        if share > 0:
            n_sent = max(int(self.cfg["min_sentinels"]),
                         math.ceil(share / (1 - share) * sum(len(v) for v in first.values())))
        sent = self.sentinels(pool if pool is not None else tiles, n_sent, batch)
        order = [a for v in first.values() for a in v] + sent
        _seeded(self.seed, self.gate_id, "mix", batch).shuffle(order)
        self.run(order)
        second: Dict[str, List[str]] = {}
        for k, ids in first.items():
            v = self.votes(ids)["overall"]
            if len(v) == 2 and v[0] != v[1]:
                ref, bv, cv = views[k]
                second[k] = self.pair("pair", k, ref, cv, bv, 1, batch)
        self.run([a for v in second.values() for a in v])
        recs = []
        for t in tiles:
            k = f"{t['set']}/{t['tile']}"
            ids = first[k] + second.get(k, [])
            v = self.votes(ids)
            if not v["overall"]:
                continue  # every ask failed: no evidence either way
            recs.append({"tile": t["tile"], "set": t["set"], "design": t["design"], "tier": t["tier"],
                         "score": {c: sum(x) / len(x) for c, x in v.items() if x},
                         "delta": self.deltas(ids), "asks": len(ids), "second_judge": k in second})
        sv = self.votes(sent)["overall"]
        return {"tiles": recs, "sentinels": {"n": len(sv), "correct": int(sum(sv)),
                                              "accuracy": (sum(sv) / len(sv)) if sv else None},
                "asks": len(order) + sum(len(v) for v in second.values()),
                "failed": sum(1 for a in order if a in self.labels and not self.labels[a]["ok"])}

    def compare(self, kind: str, items: List[dict], batch: int) -> List[dict]:
        """Anchor asks: x vs y per item (dicts with key, source, x, y, kit), both orders, one judge.
        Returns per item the share of votes for x on overall."""
        ids = {}
        for it in items:
            kit = it.get("kit")
            n = f"{kind}.{it['key']}"
            ref = self.view(it["source"], f"{n}.src", kit)
            xv = it["x_png"] if it.get("x_png") else self.view(it["x"], f"{n}.x", kit)
            yv = it["y_png"] if it.get("y_png") else self.view(it["y"], f"{n}.y", kit)
            ids[it["key"]] = self.pair(kind, it["key"], ref, Path(xv), Path(yv), 0, batch)
        self.run([a for v in ids.values() for a in v])
        out = []
        for k, v in ids.items():
            o = self.votes(v)["overall"]
            if o:
                out.append({"key": k, "x_share": sum(o) / len(o)})
        return out

    def usage(self) -> dict:
        """Asks made and tokens spent (per ask, summed over retries)."""
        n, toks, cost = 0, [], 0.0
        for lab in self.labels.values():
            n += 1
            t = sum((u.get("input", 0) or 0) + (u.get("cache_read", 0) or 0) + (u.get("cache_write", 0) or 0)
                    for u in lab.get("usage") or [])
            toks.append(t)
            cost += sum(u.get("cost_usd") or 0 for u in lab.get("usage") or [])
        return {"asks": n, "failed": sum(1 for lab in self.labels.values() if not lab["ok"]),
                "context_tokens_mean": round(sum(toks) / len(toks)) if toks else 0,
                "context_tokens_max": max(toks) if toks else 0, "cost_usd": round(cost, 3)}

    def clean_images(self) -> None:
        """Remove the blind images once the run is decided (labels stay in the private folder)."""
        shutil.rmtree(self.ask_dir, ignore_errors=True)
