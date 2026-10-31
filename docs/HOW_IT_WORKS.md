[← Back to the README](../README.md)

# How it works, in depth

About 6,000 lines of Python in `engine/`. Entry point: `resize()` in
[`engine/resize.py`](../engine/resize.py), called by [`engine/main.py`](../engine/main.py).

### 1. Analyze: what is this design made of? ([`analyze.py`](../engine/analyze.py))

The input has no semantic labels, so roles are **inferred**:

- **Text roles** (`headline`, `subhead`, `body`, `legal`, `cta`) come from font size,
  position, and the words themselves ([`textsem.py`](../engine/textsem.py) recognises
  dates, prices, phone numbers, quotes, attributions, imperatives like "Shop now").
- **Image roles** (`logo`, `hero`, `minor`, `bg`) come from size and coverage, plus
  CLIP (a small image-text model) rating how logo-like a compact image is.
- **Grouping**: elements that touch, share edges, sit on the same line, or live
  inside a card or frame are joined into **units** that move together. A badge row
  is one unit; a button and its label are one unit; a card and the copy inside it
  are one unit with children. Two readings are tried (tight and loose grouping) and
  both feed the candidate pool.
- **Priorities** rank units (headline 10, logo 9, call-to-action 8.5, hero 7 … legal
  1.2) so the biggest decisions go to the elements that matter.
- **Images are analysed** ([`saliency.py`](../engine/saliency.py), classical image
  processing, no network): spectral-residual saliency for where the eye goes, edge
  density for where text would be hard to read, alpha channels as object masks for
  cut-outs, and face detection (YuNet, with OpenCV Haar cascades as a fallback) with
  an estimated **eye line**.
- Pieces entirely off the source canvas, and images whose asset file does not exist,
  are noted as units that are never placed (see [Known limitations](../README.md#known-limitations)).

### 2. Propose: many candidate layouts ([`resize.py`](../engine/resize.py), [`layout.py`](../engine/layout.py))

Layout is a **discrete choice of arrangement** followed by a **continuous fit**
(the same split professional tools use). `_candidates()` yields the arrangements:

| Candidate | Idea | Used when |
|---|---|---|
| `same` (center / pinned / spread) | The designer's own arrangement, refit. Pieces keep their edges (pinned) or spread to the new proportions. | always, at near-same shapes |
| `fill` (`same-fill`) | Same design, but large pieces keep their share of the new canvas (photos cropped or kept whole), copy re-wraps in place and all type grows by one factor until it would collide. | aspect change ≥ 1.15x |
| `bgturn` | Same design with its background photo turned a quarter so a calm side opens up. | aspect change ≥ 1.5x |
| `turn` | The whole design turned a quarter and stretched edge to edge, with only the words set upright. | real strip conversions only (≥ 8x change) |
| `stack` | One column, in the source's reading order (top / bottom / left / right regions). | tall and square canvases |
| `split` | Hero picture in its own column, text beside it. | wide canvases |
| `backdrop` | A full-bleed hero photo becomes the background; copy goes on its calm region. | designs led by a photo |
| `card_panel`, `panel` | A card, or a cropped photo, becomes a full-bleed band on one side with copy on the rest. | wide / tall changes |
| `row` | Everything on one line: logo, hero, copy, groups, badges, button. Optional second column, rotated text. | ultra-wide banners |
| `grid` | Units split into 2–4 columns, optionally under a headline tier. | last resort, only when no faithful layout reads |

Each layout is built from **blocks** ([`blocks.py`](../engine/blocks.py)): text blocks,
image blocks, rigid groups, badge rows that wrap, frames with their contents, and
re-stackable compositions. A block knows its **visual size** (inked text, opaque image
pixels) rather than its padded box, so gaps and alignment follow what the eye sees.

Text is measured with the **renderer's own fonts and wrapping**
([`textfit.py`](../engine/textfit.py)), so a line we think fits is a line that fits.

### 3. Finalize: crop, decorate, anchor, clamp ([`finalize.py`](../engine/finalize.py))

- **Background crop**: the crop window of a photo weighs the copy against the subject.
  It tries to keep the subject and eyes clear of copy, place the focal point on a
  third, and avoid cropping out what the source's own crop showed.
- Decoration scales with the canvas, borders keep hugging the edge, corner pieces
  stay in their corner, ribbons and small images float to a free corner, separators
  keep their place relative to their copy.
- **Type floor**: no text below about 14 px when the design is seen at 768 px on its
  long side, widening the box if needed.
- Everything is clamped into the canvas.

### 4. Critique: pick the winner ([`score.py`](../engine/score.py), [`legible.py`](../engine/legible.py))

[`score()`](../engine/score.py) approximates the judge's rubric as a sum of terms:
readability, contrast, text on busy ground, collisions, faces and eyes, subject
hidden, picture cover, arrangement kept, grouping, hierarchy, prominence, balance
and share of canvas. Two layers sit on top of the raw score:

- **Same-design preference** (`_prefer_same`): the best "same design" layout beats the
  rebuilt ones unless they have clearly fewer defects. The margin fades as the shape
  change grows (full at ≤ 1.5x, gone by 4x), so a standard format change stays
  recognisable and an extreme one is free to restructure.
- **Read check** ([`legible.py`](../engine/legible.py)): the top candidate is rendered at
  the judge's view (long side 768 px) and read with OCR. If its headline or button
  has lost a third or more of the words that the *source's own render* reads, the
  next candidates are tried in score order, subject to keeping the reading order and
  not shrinking the headline below 80%. It is a veto, not a score term.

### 5. Build and guards ([`resize.py`](../engine/resize.py))

- Every changed picture box is switched to `cover` (cropped) for solid pictures or
  `contain` (shown whole) for drawings, so **no image is ever stretched**.
- `_fit_lines` re-measures each line in the renderer's fonts; a line wider than its
  box widens it where there is room, otherwise the type comes down until it fits.
- Letter spacing scales with the type.
- Any exception falls back to a uniform contain-scale, so a bad input produces a
  mediocre answer, never a crash.

### Models

Everything runs on CPU, offline, with small models:

| Model | Used for | Where |
|---|---|---|
| YuNet (ONNX) | Face and eye-line detection | in git, `engine/models/` |
| OpenCV Haar cascades | Face detection fallback | bundled with OpenCV |
| CLIP ViT-B/32 (quantised ONNX) | Logo vs. photo, "does this photo have a natural up?" | downloaded by `tools/fetch_models.py` |
| PP-OCRv4 mobile via RapidOCR | The read check | bundled in the pip wheel |

None is fine-tuned, so there are no adapter weights to publish. Missing models
degrade the engine quietly rather than breaking it.

---

## Repo map and settings

```
engine/
  main.py       orchestrator: input/*.json -> resize() -> output/ (JSON + PNG) + self-check
  resize.py     candidates, selection, read check, build, fallback
  analyze.py    roles, grouping, units, priorities
  layout.py     templates: same, fill, turn, stack, split, backdrop, panel, row, grid
  blocks.py     blocks that know their visual size and place their elements
  finalize.py   background crop, decoration, anchors, clamping, type floor
  score.py      the critic
  legible.py    render + OCR read check
  saliency.py   saliency, busyness, faces (classical)
  vision.py     small ONNX models: CLIP, OCR, optional subject mask
  textfit.py    text measurement with the renderer's own fonts
  textsem.py    what text says: dates, prices, CTAs, quotes
  models/       YuNet in git; CLIP downloaded by tools/fetch_models.py
renderer/       the official renderer, vendored
brand_kit/      sample assets and fonts
input/          sample tasks (one per difficulty level)
output/         results land here
tools/          evaluation: held-out Crello runs, scoreboards, comparison sheets
setup.sh        one-command local setup
```

**Environment variables** (all optional):

| Variable | Default | Effect |
|---|---|---|
| `CHAMELEON_DEBUG` | off | `1` prints every candidate's score and any swallowed error |
| `CHAMELEON_TURN_MIN` | `8` | aspect-ratio change from which the whole design may turn |
| `CHAMELEON_READ_CHECK` | on | `0` disables the OCR veto |
| `CHAMELEON_JUDGE_SIDE` | `768` | long side, in px, at which readability is judged |
| `CHAMELEON_UNWRAP` | on | `0` keeps every text string exactly as given |
| `CHAMELEON_TIERS` | off | `1` enables the tiered critic |
| `CHAMELEON_SUBJECT_MASK` | off | `1` enables the learned subject mask |
| `CHAMELEON_ASSETS_DIR`, `CHAMELEON_BRAND_KIT`, `CHAMELEON_FONTS_DIR` | `brand_kit/…` | point the engine at another asset and font folder |
