"""Build a review gallery page for pairs that changed in a step.

Each pair gets: original | designer | before | after at full size, the
grouping overlay (source and ours) with its legend, and the visual-weight
overlay (source and ours) with its numbers.

usage (repo root, after tools/compare.sh <root> has rendered the pairs):
  python tools/gallery.py <root> <out_dir> <notes.json>
notes.json: {"title": ..., "summary": [...], "pairs": {"p5": {"verdict": "better", "note": "..."}, ...}}
"""

import html
import json
import sys
from pathlib import Path

from PIL import Image

root, out, notes_path = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
notes = json.loads(notes_path.read_text())
R = root / "report"
img_dir = out / "img"
img_dir.mkdir(parents=True, exist_ok=True)
index = json.loads((root / "index.json").read_text())


def save(src, name, max_w=1400):
    im = Image.open(src).convert("RGB")
    if im.width > max_w:
        im = im.resize((max_w, round(im.height * max_w / im.width)), Image.LANCZOS)
    im.save(img_dir / name, "JPEG", quality=86, optimize=True)
    return f"img/{name}"


def fig(path, label, cls=""):
    if path is None:
        return f'<figure class="{cls} missing"><div class="ph">not available</div><figcaption>{label}</figcaption></figure>'
    return (f'<figure class="{cls}"><a href="{path}" target="_blank" rel="noopener">'
            f'<img src="{path}" alt="{html.escape(label)}" loading="lazy"></a><figcaption>{label}</figcaption></figure>')


VERDICT = {"better": "Better", "worse": "Worse", "mixed": "Mixed", "same": "Same"}
sections, nav, tally = [], [], {}
for p, info in notes["pairs"].items():
    k = int(p[1:])
    a, b = index[k][0], index[k][1]
    meta = json.loads((R / f"{p}_meta.json").read_text())
    v = info.get("verdict", "same")
    tally[v] = tally.get(v, 0) + 1
    old = Path(f"/tmp/claude-old/{root.name}-{p}.png")
    imgs = {
        "orig": save(R / f"{p}_source.png", f"{p}_orig.jpg"),
        "des": save(R / f"{p}_designer.png", f"{p}_designer.jpg"),
        "before": save(old, f"{p}_before.jpg") if old.exists() else None,
        "after": save(R / f"{p}_ours.png", f"{p}_after.jpg"),
        "gs": save(R / f"{p}_groups_src.png", f"{p}_groups_src.jpg"),
        "go": save(R / f"{p}_groups_ours.png", f"{p}_groups_ours.jpg"),
        "ws": save(R / f"{p}_weight_src.png", f"{p}_weight_src.jpg") if (R / f"{p}_weight_src.png").exists() else None,
        "wo": save(R / f"{p}_weight_ours.png", f"{p}_weight_ours.jpg") if (R / f"{p}_weight_ours.png").exists() else None,
    }
    legend = "".join(
        f'<li><span class="sw" style="background:rgb{tuple(c)}"></span>{html.escape(t)}</li>'
        for t, c in zip(meta["legend"], meta["palette"]))
    weights = "".join(
        f'<div class="wcol"><h5>{lab}</h5><pre>{html.escape(chr(10).join(meta["weights"].get(tag, [])))}</pre></div>'
        for tag, lab in (("src", "Source"), ("ours", "Ours")) if tag in meta["weights"])
    nav.append(f'<a class="chip v-{v}" href="#{p}">{p}</a>')
    sections.append(f"""
<section class="pair" id="{p}">
  <header class="pair-head">
    <div class="pid">{p}</div>
    <div class="ptitle">
      <h3>{html.escape(a['title'])}</h3>
      <p class="dims">{a['w']}×{a['h']} → {b['w']}×{b['h']} · {html.escape(b.get('format', ''))} · layout <code>{html.escape(meta['template'])}</code></p>
    </div>
    <span class="pill v-{v}">{VERDICT.get(v, v)}</span>
  </header>
  <p class="note">{html.escape(info.get('note', ''))}</p>
  <div class="row four">
    {fig(imgs['orig'], 'Original')}{fig(imgs['des'], "Designer's resize")}{fig(imgs['before'], 'Before this step')}{fig(imgs['after'], 'After this step')}
  </div>
  <h4>Grouping</h4>
  <div class="row two-legend">
    {fig(imgs['gs'], 'Source, grouped')}{fig(imgs['go'], 'Ours, groups placed')}
    <ul class="legend">{legend}</ul>
  </div>
  <h4>Visual weight</h4>
  <p class="key">Circles show weight (area = weight): blue text, orange images, red the photo's subject. Black + is the centre of visual mass, grey ring the canvas centre, red heat the saliency map, faded area the cropped-away background.</p>
  <div class="row two">
    {fig(imgs['ws'], 'Source')}{fig(imgs['wo'], 'Ours')}
  </div>
  <div class="weights">{weights}</div>
</section>""")

order = ["better", "mixed", "same", "worse"]
tally_html = "".join(f'<span class="pill v-{k}">{tally[k]} {VERDICT[k].lower()}</span>' for k in order if k in tally)
summary = "".join(f"<li>{html.escape(s)}</li>" for s in notes.get("summary", []))
page = f"""<title>{html.escape(notes['title'])}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* Layout: one long review column; each pair is a block of image rows read top to bottom. */
:root {{
  --bg: #f6f6f3; --surface: #ffffff; --fg: #1d1f22; --muted: #5e6268; --line: #d9dad5;
  --accent: #c8102e;
  --good: #1f7a4d; --good-bg: #e2f2e9; --bad: #b3261e; --bad-bg: #fbe5e3;
  --mix: #8a5a00; --mix-bg: #fbf0d9; --same: #4a5059; --same-bg: #e9eaec;
  --sans: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, "SF Mono", Menlo, monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg: #16181b; --surface: #1f2226; --fg: #e8e9ea; --muted: #a0a5ad; --line: #33373d;
  --accent: #ff5a6e;
  --good: #6fd19c; --good-bg: #183428; --bad: #ff8a80; --bad-bg: #3a1d1b;
  --mix: #f2c46b; --mix-bg: #352a12; --same: #c3c7cd; --same-bg: #2a2e33; color-scheme: dark; }} }}
:root[data-theme="dark"] {{
  --bg: #16181b; --surface: #1f2226; --fg: #e8e9ea; --muted: #a0a5ad; --line: #33373d;
  --accent: #ff5a6e;
  --good: #6fd19c; --good-bg: #183428; --bad: #ff8a80; --bad-bg: #3a1d1b;
  --mix: #f2c46b; --mix-bg: #352a12; --same: #c3c7cd; --same-bg: #2a2e33; color-scheme: dark; }}
body {{ background: var(--bg); color: var(--fg); font-family: var(--sans); font-size: 15px; line-height: 1.5;
  padding-inline: clamp(16px, 4vw, 48px); padding-block: 32px 64px; }}
.wrap {{ max-width: 1500px; margin: 0 auto; display: grid; gap: 28px; }}
h1 {{ font-size: clamp(26px, 3vw, 34px); font-weight: 600; margin: 0; text-wrap: balance; letter-spacing: -0.01em; }}
.eyebrow {{ font: 500 12px/1 var(--mono); letter-spacing: .08em; text-transform: uppercase; color: var(--accent); margin: 0 0 10px; }}
.intro {{ display: grid; gap: 14px; max-width: 900px; }}
.intro ul {{ margin: 0; padding-left: 20px; display: grid; gap: 6px; }}
.tally, .nav {{ display: flex; flex-wrap: wrap; gap: 8px; }}
.pill, .chip {{ font: 500 12px/1 var(--mono); padding: 6px 10px; border-radius: 999px; white-space: nowrap; }}
.chip {{ text-decoration: none; }}
.chip:focus-visible, a:focus-visible {{ outline: 2px solid var(--accent); outline-offset: 2px; }}
.v-better {{ background: var(--good-bg); color: var(--good); }}
.v-worse {{ background: var(--bad-bg); color: var(--bad); }}
.v-mixed {{ background: var(--mix-bg); color: var(--mix); }}
.v-same {{ background: var(--same-bg); color: var(--same); }}
.pair {{ background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: clamp(16px, 2.5vw, 28px);
  display: grid; gap: 14px; scroll-margin-top: 16px; }}
.pair-head {{ display: flex; gap: 14px; align-items: flex-start; flex-wrap: wrap; }}
.pid {{ font: 600 20px/1.2 var(--mono); color: var(--accent); min-width: 3ch; }}
.ptitle {{ flex: 1; min-width: 0; }}
.ptitle h3 {{ margin: 0; font-size: 18px; font-weight: 600; text-wrap: balance; }}
.dims {{ margin: 2px 0 0; color: var(--muted); font-size: 13px; font-variant-numeric: tabular-nums; }}
code {{ font-family: var(--mono); font-size: 12px; }}
.note {{ margin: 0; max-width: 75ch; }}
h4 {{ margin: 6px 0 0; font: 500 12px/1 var(--mono); letter-spacing: .08em; text-transform: uppercase; color: var(--muted); }}
.key {{ margin: 0; color: var(--muted); font-size: 13px; max-width: 110ch; }}
.row {{ display: grid; gap: 14px; align-items: start; }}
.four {{ grid-template-columns: repeat(auto-fit, minmax(min(100%, 260px), 1fr)); }}
.two {{ grid-template-columns: repeat(auto-fit, minmax(min(100%, 360px), 1fr)); }}
.two-legend {{ grid-template-columns: repeat(auto-fit, minmax(min(100%, 320px), 1fr)); }}
figure {{ margin: 0; display: grid; gap: 6px; min-width: 0; }}
figure img {{ display: block; width: 100%; height: auto; max-height: 560px; object-fit: contain; object-position: left top;
  background: repeating-conic-gradient(var(--line) 0 25%, transparent 0 50%) 0 0 / 16px 16px; outline: 1px solid var(--line); }}
figcaption {{ font-size: 12px; color: var(--muted); }}
.ph {{ display: grid; place-items: center; min-height: 120px; color: var(--muted); border: 1px dashed var(--line); font-size: 13px; }}
.legend {{ list-style: none; margin: 0; padding: 0; display: grid; gap: 6px; font-size: 13px; align-content: start; min-width: 0; }}
.legend li {{ display: flex; gap: 8px; align-items: baseline; }}
.sw {{ width: 12px; height: 12px; border-radius: 2px; flex: none; transform: translateY(1px); }}
.weights {{ display: grid; gap: 14px; grid-template-columns: repeat(auto-fit, minmax(min(100%, 320px), 1fr)); }}
.wcol h5 {{ margin: 0 0 4px; font-size: 13px; }}
.wcol pre {{ margin: 0; font: 12px/1.55 var(--mono); color: var(--muted); white-space: pre-wrap; overflow-wrap: anywhere; }}
</style>
<div class="wrap">
  <div class="intro">
    <p class="eyebrow">Chameleon review · {html.escape(notes.get('set', root.name))}</p>
    <h1>{html.escape(notes['title'])}</h1>
    <ul>{summary}</ul>
    <div class="tally">{tally_html}</div>
    <nav class="nav" aria-label="Pairs">{''.join(nav)}</nav>
  </div>
  {''.join(sections)}
</div>
"""
(out / "index.html").write_text(page)
print(out / "index.html", sum(f.stat().st_size for f in img_dir.iterdir()) // 1024, "KB of images")
