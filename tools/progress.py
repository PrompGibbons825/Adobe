"""Build the step-by-step progress page: every pair across every step.

usage (repo root): python tools/progress.py <out_dir> review/notes_progress.json
Needs review/steps/<step>/<case>.png from tools/snapshot.sh, and the final
per-pair panels from tools/crello_report.py.
"""

import hashlib
import html
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

out = Path(sys.argv[1])
notes = json.loads(Path(sys.argv[2]).read_text())
img = out / "img"
img.mkdir(parents=True, exist_ok=True)
STEPS = [("0_before", "Before today"), ("A", "A · grouping by contact"), ("B1", "B1 · keep or break groups"),
         ("B2", "B2 · re-stack groups"), ("C", "C · cards resize, share, hierarchy"),
         ("D", "D · rotated text strips"), ("E", "E · contrast"), ("F", "F · rotation-correct geometry, no thumbnails"), ("G", "G · faces, fill, frame corners, relative costs"), ("H", "H · grid fallback, upright text"), ("I", "I · real faces, edge pinning, picture coverage"), ("J", "J · rigid structured frames, compositions kept (final)")]
try:
    LF = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 26)
except Exception:
    LF = ImageFont.load_default()


def jpg(src, name, max_w=1100):
    im = Image.open(src).convert("RGB")
    if im.width > max_w:
        im = im.resize((max_w, round(im.height * max_w / im.width)), Image.LANCZOS)
    im.save(img / name, "JPEG", quality=78, optimize=True)
    return f"img/{name}"


def pair_img(a, b, name, labels):
    """Two panels side by side (source | ours) in one image."""
    ia, ib = Image.open(a).convert("RGB"), Image.open(b).convert("RGB")
    h = max(ia.height, ib.height)
    S = Image.new("RGB", (ia.width + ib.width + 30, h + 40), "white")
    d = ImageDraw.Draw(S)
    d.text((0, 4), labels[0], fill=(80, 80, 80), font=LF)
    d.text((ia.width + 30, 4), labels[1], fill=(80, 80, 80), font=LF)
    S.paste(ia, (0, 40))
    S.paste(ib, (ia.width + 30, 40))
    S.save(img / name, "JPEG", quality=78, optimize=True)
    return f"img/{name}"


def digest(p):
    return hashlib.md5(Image.open(p).convert("RGB").tobytes()).hexdigest()


def fig(path, label):
    return (f'<figure><a href="{path}" target="_blank" rel="noopener"><img src="{path}" alt="{html.escape(label)}" '
            f'loading="lazy"></a><figcaption>{html.escape(label)}</figcaption></figure>')


cases = []
for root, pre, n in (("crello_holdout", "h", 40), ("crello", "d", 7)):
    idx = json.loads(Path(f"{root}/index.json").read_text())
    for k in range(n):
        a, b = idx[k][0], idx[k][1]
        cases.append(dict(key=f"{pre}{k}", root=root, k=k, title=a["title"], src=(a["w"], a["h"]), dst=(b["w"], b["h"]),
                          fmt=b.get("format", ""), set="Holdout" if pre == "h" else "Dev"))
for n, (s, t) in {"L1": ((1080, 1080), (1080, 1350)), "L2": ((1080, 1080), (1600, 400)), "L3": ((1080, 1350), (728, 90))}.items():
    cases.append(dict(key=n, root=None, title=f"Official Level {n[1]} sample (Solstice)", src=s, dst=t, fmt="", set="Official"))

VERD = {"better": "Better", "mixed": "Mixed", "same": "Same", "worse": "Worse"}
sections, tally = [], {}
for c in cases:
    key = c["key"]
    v, note = notes.get(key, ["same", ""])
    tally[v] = tally.get(v, 0) + 1
    files = []
    if c["root"]:
        R = Path(c["root"]) / "report"
        p = f"p{c['k']}"
        top = fig(jpg(R / f"{p}_source.png", f"{key}_orig.jpg"), "Original") + \
            fig(jpg(R / f"{p}_designer.png", f"{key}_designer.jpg"), "Designer's resize")
    else:
        top = fig(jpg(Path(f"review/steps/{key}_source.png"), f"{key}_orig.jpg"), "Original")
    steps, prev = [], None
    tmpl = {s: json.loads(Path(f"review/steps/{s}/templates.json").read_text()).get(key, "") for s, _ in STEPS}
    for s, label in STEPS:
        f = Path(f"review/steps/{s}/{key}.png")
        if not f.exists():
            continue
        dg = digest(f)
        if dg == prev:
            steps.append(f'<div class="nochange"><span>{html.escape(label)}</span><em>No change</em></div>')
        else:
            steps.append(fig(jpg(f, f"{key}_{s}.jpg"), f"{label} · {tmpl[s]}"))
        prev = dg
    extra = ""
    if c["root"]:
        R = Path(c["root"]) / "report"
        p = f"p{c['k']}"
        meta = json.loads((R / f"{p}_meta.json").read_text())
        g = pair_img(R / f"{p}_groups_src.png", R / f"{p}_groups_ours.png", f"{key}_groups.jpg", ("Source, grouped", "Final, groups placed"))
        legend = "".join(f'<li><span class="sw" style="background:rgb{tuple(col)}"></span>{html.escape(t)}</li>'
                         for t, col in zip(meta["legend"], meta["palette"]))
        w = ""
        if (R / f"{p}_weight_src.png").exists() and (R / f"{p}_weight_ours.png").exists():
            w = fig(pair_img(R / f"{p}_weight_src.png", R / f"{p}_weight_ours.png", f"{key}_weight.jpg", ("Source", "Final")),
                    "Visual weight: circles = weight (blue text, orange images, red photo subject); black + = centre of mass; grey ring = canvas centre")
        wl = "".join(f'<div class="wcol"><h5>{lab}</h5><pre>{html.escape(chr(10).join(meta["weights"].get(tag, [])))}</pre></div>'
                     for tag, lab in (("src", "Source"), ("ours", "Final")) if tag in meta["weights"])
        extra = f"""
  <details>
    <summary>Final grouping and visual weight</summary>
    <div class="glayout">{fig(g, 'Grouping: same colour = same group in the source and the final layout')}<ul class="legend">{legend}</ul></div>
    {w}
    <div class="weights">{wl}</div>
  </details>"""
    sections.append(f"""
<section class="pair" id="{key}" data-v="{v}">
  <header class="pair-head">
    <div class="pid">{key}</div>
    <div class="ptitle"><h3>{html.escape(c['title'])}</h3>
      <p class="dims">{c['set']} · {c['src'][0]}×{c['src'][1]} → {c['dst'][0]}×{c['dst'][1]}{(' · ' + html.escape(c['fmt'])) if c['fmt'] else ''}</p></div>
    <span class="pill v-{v}">{VERD[v]}</span>
  </header>
  <p class="note">{html.escape(note)}</p>
  <div class="row top">{top}</div>
  <h4>Each step</h4>
  <div class="row steps">{''.join(steps)}</div>{extra}
</section>""")

order = ["better", "mixed", "same", "worse"]
filters = '<button class="chip on" data-f="all" type="button">All ' + str(len(cases)) + "</button>" + "".join(
    f'<button class="chip v-{k}" data-f="{k}" type="button">{VERD[k]} {tally[k]}</button>' for k in order if k in tally)
STEP_LIST = "".join(f"<li><b>{html.escape(l.split(' · ')[0])}</b> {html.escape(l.split(' · ')[1]) if ' · ' in l else ''}</li>" for _, l in STEPS[1:])
page = f"""<title>Chameleon Step Progress</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* Layout: a review column; each pair reads original/designer, then the step timeline, then final diagnostics. */
:root {{
  --bg: #f6f6f3; --surface: #ffffff; --fg: #1d1f22; --muted: #5e6268; --line: #d9dad5; --accent: #c8102e;
  --good: #1f7a4d; --good-bg: #e2f2e9; --bad: #b3261e; --bad-bg: #fbe5e3;
  --mix: #8a5a00; --mix-bg: #fbf0d9; --same: #4a5059; --same-bg: #e9eaec;
  --sans: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, "SF Mono", Menlo, monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg: #16181b; --surface: #1f2226; --fg: #e8e9ea; --muted: #a0a5ad; --line: #33373d; --accent: #ff5a6e;
  --good: #6fd19c; --good-bg: #183428; --bad: #ff8a80; --bad-bg: #3a1d1b;
  --mix: #f2c46b; --mix-bg: #352a12; --same: #c3c7cd; --same-bg: #2a2e33; color-scheme: dark; }} }}
:root[data-theme="dark"] {{
  --bg: #16181b; --surface: #1f2226; --fg: #e8e9ea; --muted: #a0a5ad; --line: #33373d; --accent: #ff5a6e;
  --good: #6fd19c; --good-bg: #183428; --bad: #ff8a80; --bad-bg: #3a1d1b;
  --mix: #f2c46b; --mix-bg: #352a12; --same: #c3c7cd; --same-bg: #2a2e33; color-scheme: dark; }}
body {{ background: var(--bg); color: var(--fg); font-family: var(--sans); font-size: 15px; line-height: 1.5;
  padding-inline: clamp(16px, 4vw, 48px); padding-block: 32px 64px; }}
.wrap {{ max-width: 1560px; margin: 0 auto; display: grid; gap: 24px; }}
h1 {{ font-size: clamp(26px, 3vw, 34px); font-weight: 600; margin: 0; text-wrap: balance; letter-spacing: -0.01em; }}
.eyebrow {{ font: 500 12px/1 var(--mono); letter-spacing: .08em; text-transform: uppercase; color: var(--accent); margin: 0 0 10px; }}
.intro {{ display: grid; gap: 14px; max-width: 960px; }}
.intro p {{ margin: 0; }}
.steplist {{ margin: 0; padding-left: 20px; display: grid; gap: 4px; }}
.filters {{ display: flex; flex-wrap: wrap; gap: 8px; position: sticky; top: env(safe-area-inset-top, 0px); z-index: 2;
  background: var(--bg); padding-block: 10px; }}
.chip, .pill {{ font: 500 12px/1 var(--mono); padding: 7px 11px; border-radius: 999px; white-space: nowrap; border: 1px solid transparent; }}
.chip {{ cursor: pointer; background: var(--same-bg); color: var(--fg); }}
.chip.on {{ border-color: var(--fg); }}
.chip:focus-visible, a:focus-visible, summary:focus-visible {{ outline: 2px solid var(--accent); outline-offset: 2px; }}
.v-better {{ background: var(--good-bg); color: var(--good); }}
.v-worse {{ background: var(--bad-bg); color: var(--bad); }}
.v-mixed {{ background: var(--mix-bg); color: var(--mix); }}
.v-same {{ background: var(--same-bg); color: var(--same); }}
.pair {{ background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: clamp(16px, 2.5vw, 28px);
  display: grid; gap: 12px; scroll-margin-top: 60px; }}
.pair-head {{ display: flex; gap: 14px; align-items: flex-start; flex-wrap: wrap; }}
.pid {{ font: 600 20px/1.2 var(--mono); color: var(--accent); min-width: 3ch; }}
.ptitle {{ flex: 1; min-width: 0; }}
.ptitle h3 {{ margin: 0; font-size: 18px; font-weight: 600; text-wrap: balance; }}
.dims {{ margin: 2px 0 0; color: var(--muted); font-size: 13px; font-variant-numeric: tabular-nums; }}
.note {{ margin: 0; max-width: 80ch; }}
h4 {{ margin: 4px 0 0; font: 500 12px/1 var(--mono); letter-spacing: .08em; text-transform: uppercase; color: var(--muted); }}
.row {{ display: grid; gap: 14px; align-items: start; }}
.top {{ grid-template-columns: repeat(auto-fit, minmax(min(100%, 300px), 1fr)); }}
.steps {{ grid-template-columns: repeat(auto-fill, minmax(min(100%, 230px), 1fr)); }}
figure {{ margin: 0; display: grid; gap: 6px; min-width: 0; }}
figure img {{ display: block; width: 100%; height: auto; max-height: 520px; object-fit: contain; object-position: left top;
  background: repeating-conic-gradient(var(--line) 0 25%, transparent 0 50%) 0 0 / 16px 16px; outline: 1px solid var(--line); }}
figcaption {{ font-size: 12px; color: var(--muted); overflow-wrap: anywhere; }}
.nochange {{ display: grid; gap: 4px; align-content: start; padding: 12px; border: 1px dashed var(--line); border-radius: 6px;
  font-size: 12px; color: var(--muted); }}
.nochange em {{ font-style: normal; font-family: var(--mono); }}
details {{ border-top: 1px solid var(--line); padding-top: 10px; display: grid; gap: 12px; }}
details[open] {{ gap: 12px; }}
summary {{ cursor: pointer; font: 500 13px/1.4 var(--sans); color: var(--fg); }}
.glayout {{ display: grid; gap: 14px; grid-template-columns: minmax(0, 3fr) minmax(220px, 1fr); align-items: start; margin-top: 10px; }}
@media (max-width: 760px) {{ .glayout {{ grid-template-columns: 1fr; }} }}
.legend {{ list-style: none; margin: 0; padding: 0; display: grid; gap: 6px; font-size: 13px; min-width: 0; }}
.legend li {{ display: flex; gap: 8px; align-items: baseline; }}
.sw {{ width: 12px; height: 12px; border-radius: 2px; flex: none; transform: translateY(1px); }}
.weights {{ display: grid; gap: 14px; grid-template-columns: repeat(auto-fit, minmax(min(100%, 320px), 1fr)); }}
.wcol h5 {{ margin: 0 0 4px; font-size: 13px; }}
.wcol pre {{ margin: 0; font: 12px/1.55 var(--mono); color: var(--muted); white-space: pre-wrap; overflow-wrap: anywhere; }}
@media (prefers-reduced-motion: no-preference) {{ html {{ scroll-behavior: smooth; }} }}
</style>
<div class="wrap">
  <div class="intro">
    <p class="eyebrow">Chameleon engine · 40 holdout, 7 dev, 3 official cases</p>
    <h1>Every pair across every step</h1>
    <p>For each case: the original and the designer's resize, then our render after each step. A dashed box means that step changed nothing for this case. The verdict compares the final render with the one from before today. Open "Final grouping and visual weight" under a case for its diagnostics.</p>
    <ul class="steplist">{STEP_LIST}</ul>
  </div>
  <div class="filters" role="group" aria-label="Filter by verdict">{filters}</div>
  {''.join(sections)}
</div>
<script>
(function () {{
  var chips = document.querySelectorAll('.chip[data-f]');
  var pairs = document.querySelectorAll('.pair');
  chips.forEach(function (c) {{
    c.addEventListener('click', function () {{
      var f = c.getAttribute('data-f');
      chips.forEach(function (x) {{ x.classList.toggle('on', x === c); }});
      pairs.forEach(function (p) {{ p.hidden = !(f === 'all' || p.getAttribute('data-v') === f); }});
    }});
  }});
}})();
</script>
"""
(out / "index.html").write_text(page)
files = sorted(p.name for p in img.iterdir())
print(out / "index.html", len(files), "images", sum((img / f).stat().st_size for f in files) // 1024, "KB", tally)
