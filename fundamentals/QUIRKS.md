# What the official renderer actually draws (QUIRKS)

Read from `renderer/converter/*.py` (vendored converter) and the pinned cr-renderer
(`CyberAgentAILab/cr-renderer@a17e1fb`, `cr_renderer/{renderer,text_utils,fonts,color_utils}.py`),
skia-python 144.1, Pillow 13.0.0. Same versions in the venv and in the Docker image, both on Linux.
Every item marked **[probed]** was confirmed by rendering a probe design with the real renderer on
2026-10-30. Line references are to those files. `fundamentals/` implements this model; the
conformance suite (`tests/conformance/`) pins it.

Pipeline: `render_design_to_png` -> `convert_chameleon_to_crello` (converter: wraps text, rasterises
shapes and fitted images with PIL) -> `CrelloV5Renderer.render(max_size=2048, png)` (skia draws on a
white canvas).

## Canvas and scale
- Canvas cleared to **white** (`skia.ColorWHITE`). Transparent areas of the top-most pixels show white.
- Output size: `s = min(1, 2048 / max(W, H))`, PNG is `round(W*s) x round(H*s)`. Canvases with a
  side > 2048 are **downscaled** (3000x500 -> 2048x341) **[probed]**. Every hidden format we know is
  <= 2048 except e.g. 2560x1440 YouTube channel art: there everything (text too) is drawn at s < 1.
- No clipping to the canvas other than the surface edge. Anything off-canvas is simply cut.

## Z-order
- Elements are sorted by `z_index` (missing = 0) with Python's **stable** sort: ties keep list order.
  Later = on top. No groups, no blend modes.

## Geometry and rotation
- Each element: `translate(x, y)`, then if `rotation != 0`: `rotate(deg, w/2, h/2)`; degrees,
  **clockwise** (y-down), about the element's box centre. For text the box is the element's
  `width x height`, not the drawn text, so rotated text pivots around the box centre even when the
  text overflows the box **[probed]**.
- x, y, width, height are floats; skia anti-aliases sub-pixel edges.

## Text: wrapping (converter, `wrap.py`, `convert._append_text`)
- **Greedy word wrap at spaces only.** Per-character advances from
  `skia.Font(typeface, font_size).getWidths(textToGlyphs(content))`: one glyph per char, **no
  kerning, no shaping, no ligatures**. A line breaks when `line_width + w(char) > width` and a
  space was seen on the line; that last space becomes `"\n"` in place. Trailing spaces count
  toward the width while deciding. A single word wider than `width` is **never broken**: it
  overflows the box to the right (or both sides when centred).
- Existing `"\n"` in content are hard breaks.
- Wrap is measured with the element-level weight/style only. It ignores:
  - **`letter_spacing`** (drawn later, so positive spacing overflows the box: 200 px box, 8 px
    spacing drew to 239 px) **[probed]**;
  - **`capitalize`** (drawn upper-cased, so wider than measured) **[probed]**;
  - **bold/italic `style_runs`** (a bold run is wider than measured).
- Wrap measures at the **float** `font_size`; drawing uses **`int(font_size)`** (truncation, see
  below). Drawn lines are therefore never wider than measured from size alone, but line breaks are
  decided for a slightly bigger font.

## Text: drawing (`text_utils.render_text`)
- Glyphs are drawn with `skia.Font(typeface, int(fontSize))`: **font size is truncated to an
  integer** (20.99 draws exactly like 20) **[probed]**. `font_size < 1` draws nothing.
- Hinted, non-subpixel FreeType: **advances are whole pixels** at the integer size (Solstice Sans
  Regular at 30 px: H=20, e=18, l=7...). This is the main Linux vs macOS difference: CoreText on a
  Mac gives fractional advances, so a Mac-local wrap can differ by a word. The scorer runs Linux.
- Line `i` has its **top at `y + i * font_size * line_height`** (float font_size here) and its
  baseline at that plus `-fAscent` of the int-size font. Line height default 1.2.
  - Solstice Sans: ascent 1.05 em, descent 0.35 em (font box 1.40 em > 1.2 line pitch).
  - Solstice Serif: ascent 1.006 em, descent 0.274 em.
  - So ink starts below the box top (cap height 0.70 em, about 0.35 em of air above caps), and
    the drawn font box of n lines is `(n-1)*fs*lh + (asc+desc)*int(fs)`.
- **Element `height` is ignored for text.** No vertical alignment, no clipping: lines past the
  box bottom are drawn below it. `max_lines` is ignored.
- Alignment per line: `left` -> 0, `center` -> `(width - line_w)/2`, `right` -> `width - line_w`,
  where `line_w = sum(advances) + letter_spacing*(glyphs-1)` per style span. **`justify` (and any
  other value) -> left** with a warning **[probed]**. Overwide lines go negative: centred overflow
  spills on both sides, right-aligned overflow spills left.
- `letter_spacing` (px) is added after every glyph but the last, per style span.
- **Text `opacity` is ignored** (the converter never passes it into `TextElement`) **[probed]**.
  Only the alpha in `color` fades text.
- Colours via `cr_renderer.color_utils.parse_color`: `rgba(...)`, `rgb(...)`, `#rgb`, `#rrggbb`
  work. **Named colours crash the render** (`"red"` -> ValueError) and so does a leading space
  (`" rgba(...)"` -> NotImplementedError) **[probed]**.
- `style_runs`: the schema says `end` is exclusive, the converter treats it as **inclusive**
  (`range(start, end + 1)`): a run colours one extra character **[probed]**. Runs can change color,
  bold and italic.
- `capitalize: true` (not in the schema; the converter reads it) upper-cases at draw time.
- Text effects (shadow, outline...) are not reachable from the schema.
- **Render crashes** (pydantic validation in `cr_renderer.schema.TextElement`; one bad text kills
  the whole PNG): text `width <= 0` or `height <= 0`, `font_size <= 0`, `line_height <= 0`, and any
  `text_align` outside left/center/right/justify. Images and shapes with a 0 size just draw nothing.
- **HTML entities are unescaped at draw time** (`html.unescape` on `TextElement.text`) but the wrap,
  line map and colour map were computed on the raw string: `"A &amp; B words"` draws `&` and the
  line breaks shift by the length difference (mid-word breaks, a drawn `\n` glyph)
  **[probed: "A &amp; B words here" -> lines "A & B\nwor" / "ds here"]**.

## Fonts (`brand_fonts.py`, `cr_renderer/fonts.py`)
- Only the brand kit's files are loaded: `Solstice Sans` (Regular, Bold, Italic, BoldItalic) and
  `Solstice Serif` (Regular, Bold, Italic; **no BoldItalic**). No system fonts.
- Family lookup: `normalize_family(name)` (`.title()`, strips " Bold/Regular/Light/Italic/Medium"
  suffixes) -> else "Montserrat" (absent) -> else **the first family loaded = Solstice Sans**.
  So any unknown family (`Montserrat`, `Lora`, `SolsticeSans` without a space) renders as Solstice
  Sans **[probed]**.
- Weight/style: the drawer only knows bold/not-bold and italic/not-italic. `font_weight` in
  {bold, semibold, extrabold, black} -> Bold face; everything else (light, medium, thin...) ->
  Regular face. A family name ending in a weight word ("Solstice Sans Bold") forces that weight.
- **Wrap/draw face mismatch for non-schema weights**: the wrap measures with
  `lookup(family, weight, style)`; a weight with no exact face (light, medium, semibold...) falls
  back to the family's *first file* (`-Bold.ttf`, sorted), so wrap measures **Bold** while
  `light`/`medium` draw **Regular** **[probed: "light" -> "fallback to bold bold"]**. Schema
  weights (regular, bold) are consistent. Serif bold+italic falls back to Serif Bold.

## Images (`assets.py`)
- The asset is loaded with PIL, converted to RGBA, and **pre-fitted by the converter to a
  `round(w) x round(h)` bitmap** (LANCZOS); skia then draws that bitmap into the float `w x h` box
  with `Alphaf = opacity`.
  - `contain` (the default): scale = min(w/iw, h/ih); centred with **transparent** padding
    (offsets use integer `//2`).
  - `cover`: scale = max(...), **centre crop** (no focal point).
  - `fill` or any other value: stretched to the box (aspect not kept).
- **Missing asset**: a grey (214,214,214) box with a dark red border and the label "missing
  asset: <id>", the full element size, with opacity applied **[probed]**. A warning is returned.
- Image `opacity` works (paint alpha).

## Shapes (`shapes.py`)
- Rasterised by PIL to `round(w) x round(h)` **without anti-aliasing** (ellipse and rounded corners
  are jagged before skia scales them), then drawn like an image.
- `fill_color` is parsed by the converter's own `parse_rgba`, which accepts **only `rgba()/rgb()`**.
  **A hex fill (`#00ff00`) renders grey (200,200,200)** **[probed]**.
- Fill alpha and `opacity` multiply (both become paint alpha; rgba(...,0.5) and opacity 0.5 render
  identically) **[probed]**.
- `rectangle` honours `corner_radius` (clamped to half the side). **`stroke_color`/`stroke_width`
  are ignored on rectangles and ellipses** **[probed]**.
- `line`: a horizontal line through `height//2`, colour `stroke_color` (else `fill_color`), width
  `max(1, round(stroke_width))`; the rest of the box is transparent. Rotation is the only way to
  get a non-horizontal line.
- Unknown `type`: a transparent box (nothing drawn), with a warning.

## Things that do NOT matter to the drawing
- `group_id`, `max_lines`, element `height` for text, `font_style` other than "italic", unknown keys.

## Measured conformance (fundamentals vs the renderer)
`python -m tests.conformance.run --n 2000 --seed 1` (feature cases + 2000 random texts):
see the commit message / `loop/sprint` board for the latest numbers. Wrapped strings must match
the converter's char for char; ink boxes within 1.5 px (rotated: never smaller than the ink, at
most 8% of the font size larger).

## Implications for engines and checkers (one line each)
1. Measure text with the renderer's own wrap at the float size, then draw extents at `int(size)`;
   prefer integer font sizes so measure and draw agree.
2. Include `letter_spacing`, `capitalize` and bold runs in the width yourself, or text overflows.
3. The drawn text block's height is `(n-1)*fs*lh + (asc+desc)*int(fs)` from the box top; reserve
   it, since the renderer never clips to `height`.
4. Never emit named colours or hex shape fills; emit `rgba(r,g,b,a)` with no leading space.
5. Text fading must go through the colour alpha, not `opacity`.
6. Off-canvas and overlap checks must use the drawn box (incl. overflow of long words, letter
   spacing, centred spill and rotation), not the element box.
