"""Pairwise labels from a VLM judge (dev tool, nothing here ships in the engine).

    .venv/bin/python -m tools.judge.label prompt batch_00   # the judge prompt for one blind batch
    .venv/bin/python -m tools.judge.label api --round 1     # label with the Anthropic API (needs
                                                            # ANTHROPIC_API_KEY and the SDK in .venv-dev)
    .venv/bin/python -m tools.judge.label merge             # raw judge lines -> tools/judge/data/labels.jsonl

Each ask shows the source render (REF) and two resizes (A, B) at the view the
judge would get (512 or 1024 px long side). Every pair is asked twice, in both
orders, in different batches, so a fresh judge never sees the same pair twice.
On 2026-10-06 the judge was Claude (Opus 5.5) run as blind sub-agents reading
the images with the Read tool; the API path is the same prompt for later reuse.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "output" / "judge"
DATA = Path(__file__).resolve().parent / "data"

RUBRIC = """Anyone can judge a resized design by eye, and the qualities that make one good are the same ones our scoring rewards:
  ● No unwanted collisions — overlaps are fine when they're intentional (layered graphics, deliberate stacking); what matters is that elements that shouldn't overlap, don't, and text never sits on a busy image.
  ● Everything is readable — text is large enough and has enough contrast; nothing is cut off.
  ● It stays on brand — colours, fonts, and the logo survive the resize intact and in proportion.
  ● It looks intentional — things line up, spacing is even, the most important element still stands out, pictures keep their proportions.
  ● It holds up everywhere — good across many different designs and shapes, not just one lucky example."""

CRITERIA = ["collisions", "readable", "brand", "intentional", "overall"]

TASK = """You are the automated judge of a design-resize competition. A resize engine takes a design and re-lays it out for a new canvas size; pictures are never regenerated, only moved, scaled, cropped (full-bleed backgrounds) or left out, and text may change size but never wording or font.

THE RUBRIC (verbatim from the challenge brief, section 2.5 "What a good result looks like"):
{rubric}

Each ask has three images:
  REF  the original design, for reference (what the designer intended)
  A, B two different resizes of REF to the same new canvas size, at the resolution the judge sees

Decide which of A and B is the better resize of REF under the rubric, per criterion and overall:
  collisions   unwanted overlaps; text on a busy image
  readable     text large enough, enough contrast, nothing cut off
  brand        colours, fonts and logo intact and in proportion
  intentional  alignment, even spacing, the most important element stands out, picture proportions
  overall      which one scores higher on the whole rubric (the fifth row is about many designs; ignore it here)
Forced choice: always answer "A" or "B", even when they look the same on a criterion; use confidence 1 when the overall choice was a coin flip, 2 for a clear preference, 3 for an obvious one."""

AGENT = """{task}

Your asks ({n}), in this order: {asks}
Image folder: {adir}

For each ask: Read {adir}/<ask>_ref.png, {adir}/<ask>_A.png and {adir}/<ask>_B.png with the Read tool (you may issue the three reads together), look at them carefully, then decide. Judge every ask on its own; do not try to work out where a version came from.
Open no other files and run nothing else. When all asks are judged, write all lines at once with the Write tool to
{out}
one JSON object per line, exactly these keys:
{{"ask": "<ask>", "collisions": "A", "readable": "A", "brand": "B", "intentional": "A", "overall": "A", "confidence": 2, "why": "<at most 20 words naming the decisive difference>"}}
Avoid double quotes inside "why". Then reply with the same lines as your final report."""


def agent_prompt(batch: str) -> str:
    asks = json.loads((WORK / "batches" / f"{batch}.json").read_text())
    out = WORK / "labels_raw" / f"{batch}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    return AGENT.format(task=TASK.format(rubric=RUBRIC), n=len(asks), asks=", ".join(asks), adir=WORK / "asks", out=out)


def cmd_prompt(a) -> None:
    print(agent_prompt(a.batch))


def cmd_api(a) -> None:  # pragma: no cover - needs a key; same prompt, one ask per call
    import anthropic  # dev-only: install into W/.venv-dev, never into the engine

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set")
    client = anthropic.Anthropic()
    asks = json.loads((WORK / "asks.json").read_text())
    out = WORK / "labels_raw" / f"api_round{a.round}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["ask"] for l in out.read_text().splitlines()} if out.exists() else set()

    def img(p: Path) -> dict:
        return {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                            "data": base64.b64encode(p.read_bytes()).decode()}}

    for aid, ask in asks.items():
        if ask["round"] != a.round or aid in done:
            continue
        content = [{"type": "text", "text": TASK.format(rubric=RUBRIC)}, {"type": "text", "text": "REF:"},
                   img(WORK / "asks" / f"{aid}_ref.png"), {"type": "text", "text": "A:"}, img(WORK / "asks" / f"{aid}_A.png"),
                   {"type": "text", "text": "B:"}, img(WORK / "asks" / f"{aid}_B.png"),
                   {"type": "text", "text": 'Reply with one JSON object only: {"collisions": "A|B", "readable": "A|B", '
                                            '"brand": "A|B", "intentional": "A|B", "overall": "A|B", "confidence": 1-3, '
                                            '"why": "<20 words"}'}]
        msg = client.messages.create(model=a.model, max_tokens=300, messages=[{"role": "user", "content": content}])
        text = msg.content[0].text
        rec = json.loads(text[text.index("{"):text.rindex("}") + 1])
        rec["ask"] = aid
        with out.open("a") as f:
            f.write(json.dumps(rec) + "\n")


def cmd_merge(a) -> None:
    asks = json.loads((WORK / "asks.json").read_text())
    pairs = {json.loads(l)["pair"]: json.loads(l) for l in (DATA / "pairs.jsonl").read_text().splitlines() if l.strip()}
    path = DATA / "labels.jsonl"
    have = {json.loads(l)["ask"] for l in path.read_text().splitlines() if l.strip()} if path.exists() else set()
    new = bad = 0
    with path.open("a") as f:
        for raw in sorted((WORK / "labels_raw").glob("*.jsonl")):
            for line in raw.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    bad += 1
                    continue
                aid = r.get("ask")
                if aid not in asks or aid in have or any(r.get(c) not in ("A", "B") for c in CRITERIA):
                    bad += aid not in have
                    continue
                ask = asks[aid]
                p = pairs[ask["pair"]]
                rec = {"ask": aid, "pair": ask["pair"], "case": p["case"], "view": ask["view"], "shown_A": ask["A"],
                       "shown_B": ask["B"], "judge": a.judge, "batch": raw.stem}
                for c in CRITERIA:
                    rec[c] = ask[r[c]]  # the winning candidate id
                rec["confidence"] = int(r.get("confidence", 1))
                rec["why"] = str(r.get("why", ""))[:200]
                f.write(json.dumps(rec) + "\n")
                have.add(aid)
                new += 1
    print(new, "new labels,", bad, "rejected lines;", len(have), "labels in", path)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prompt")
    p.add_argument("batch")
    p = sub.add_parser("api")
    p.add_argument("--round", type=int, default=1)
    p.add_argument("--model", required=True, help="a current Claude model id with vision")
    p = sub.add_parser("merge")
    p.add_argument("--judge", default="claude-opus-5.5-subagent")
    a = ap.parse_args()
    {"prompt": cmd_prompt, "api": cmd_api, "merge": cmd_merge}[a.cmd](a)


if __name__ == "__main__":
    main()
