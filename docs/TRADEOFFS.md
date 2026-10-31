[← Back to the README](../README.md)

# Tradeoffs and the decisions behind them

Each of these is a place we chose one thing over another. We list what it costs.

### 1. Classical analysis and small offline models, not a generative or large vision model

**Decision.** Layout is solved by geometry, measured text, image statistics and a
few small CPU models. No generative model, no API, no LLM in the loop.

**Why.**
- The rules make **image regeneration a disqualifying violation**, and stage-one
  scoring runs with **no network**. A generative pipeline invites both.
- The scoring machine is a laptop-class Apple chip, and speed and reliability are a
  rubric row. Ours is 1–2 s per design after warm-up.
- Every decision is inspectable. When a layout is wrong we can read the score
  breakdown and fix the term, which is hard to do with a model.

**Cost.** We cannot "understand" a design the way a person or a VLM does. Roles are
guessed from geometry and wording, so an unusual design can be misread. CLIP,
OCR and face detection give narrow, cheap slices of understanding where they pay
off most.

### 2. Generate many candidates and critique them, not one rule-based pass

**Decision.** Try roughly 10–40 arrangements per design (9, 35 and 41 on the three samples) and let our own critic choose.

**Why.** No single template is best across square→strip, portrait→landscape and
photo-led vs. type-led designs. Searching over arrangements is robust to designs we
did not anticipate, and adding a new arrangement never breaks the others.

**Cost.** The critic is the weak point: a weighted sum can let a soft virtue (big
type, even balance) buy off a hard defect (cropped eyes), and small weight changes
can flip winners on unrelated designs. We have mitigations (below) but not a cure.

### 3. Faithfulness first: "same design" must be beaten, not merely matched

**Decision.** For a standard format change (square ↔ portrait, 16:9 ↔ 9:16,
Instagram post/story, YouTube, A1), keep the original's look. A rebuilt layout wins
only if it has clearly fewer defects *and* reads at least as well.

**Why.** The rubric weights faithfulness to the original design's intent, and the
organizers want a standard format change (e.g. 16:9 ↔ 9:16, Instagram post ↔ story)
to keep the original's look. A defect-free, readable, recognisably-the-same design
is a safer bet with an AI judge than a clever new arrangement. The scoring penalty
for departing from the original is real; the cost of keeping it is usually small.

**Cost.** The engine can be conservative. When the original arrangement is mediocre
at the new shape, we stay close to it rather than fix it.

### 4. Restructure only when the shape change is extreme

**Decision.** The designer's arrangement binds for moderate changes and loosens as
the aspect ratio change grows (full weight up to 1.5x, none from 4x). Turning the
whole design a quarter turn is offered only for real strip conversions (≥ 8x,
tunable with `CHAMELEON_TURN_MIN`).

**Why.** A square into a 12:1 leaderboard cannot keep a vertical stack and stay
legible; rows, columns or turned text are the only way to get usable type. But a 2:1
portrait into a 1:2 landscape is an ordinary orientation change and should stay
upright. The fade makes that transition smooth, not a cliff.

**Cost.** Strip layouts are a different design from the source by necessity. The
Level 3 sample (portrait → 728×90) reads at small sizes; the critic itself rates
its readability poorly (see [Known limitations](../README.md#known-limitations)).

### 5. Pictures are cropped or shown whole, never stretched, and eyes come first

**Decision.** Every resized picture is `cover` (cropped) or `contain` (whole). Crops
are chosen around the subject, and **never cut or cover a person's eyes**; the lower
face may take a card, but the eyes may not.

**Why.** The brief says "a naive program stretches the picture". Distorted pictures
are the most visible defect an AI judge catches. Faces are the first thing viewers
look at; a card across the eyes, or a crop through them, reads as an error.

**Cost.** Sometimes a crop is conservative, or a picture is shown whole with unused
space beside it, when a bolder crop might look better. Face detection can miss a
face (we have a Haar fallback, and a later pass for people the first finder missed),
and the eye line is an estimate.

### 6. Keep every element, even when we cannot show it

**Decision.** The output always has exactly the source's element ids, as the rules
require. The engine places every element **except in two cases**: a piece that lay
entirely off the source canvas (never visible in the original), and an image whose
asset file does not exist (nothing to draw). Those are kept in the JSON but placed at
zero opacity (text is tucked at 1 px under the background, since the renderer
ignores opacity on text).

**Why.** The id check must pass, and the missing-asset case is a deliberate test in
the Level 3 sample: the spec says to handle it "gracefully (e.g. skip it,
placeholder it)", not to crash.

**Cost.** This is the one place we do not show an element. The renderer would draw a
labelled placeholder if we placed it; we currently do not (see
[Known limitations](../README.md#known-limitations)). We audited every image element in the
samples and all seven Crello sets (thousands of elements): exactly one asset is missing, and
it is the planted one (we checked the files on disk, not engine output).

### 7. Words are fixed; only size and line breaks change

**Decision.** Content and font family are never rewritten (the rules treat both as
fixed for stage one). Font size changes. Where a paragraph re-flows into a new
width, we drop line breaks that only wrapped it by hand (`textfit.unwrap_breaks`),
but never one that ends a sentence, starts a new piece of information, or sits in
display type. The headline keeps its designed lines. Set `CHAMELEON_UNWRAP=0` to
keep every string byte-for-byte.

**Why.** A hand-wrapped paragraph with its old breaks in a new box wraps ragged and
ugly. Re-flowing it is how designers do it.

**Cost.** The output string can differ from the source by whitespace in those
paragraphs. No word is added, removed or reordered.

### 8. A hand-built critic, with defects ranked over taste

**Decision.** Our critic is a hand-weighted sum approximating the rubric. On top of
it: (a) defect terms are listed separately from taste terms, and the same-design
comparison counts defects only (taste terms such as big type or even spread are
no reason to give up the designer's arrangement); (b) a read check can veto a
winner.

**Why.** We have no ground-truth scores for the hidden judge. We do have designer
resizes (Crello) as weak evidence, which we use to compare engines and catch
regressions rather than to fit weights.

**Cost.** Weights are tuned by hand and can overfit what we looked at. A **tiered
critic** (check visible defects first, then readability, then arrangement, then the
rest in rounded bands) is built (`CHAMELEON_TIERS=1`) but **off by default**, because
the terms are not yet clean defect measures. A learned critic is the obvious next step; we did not have the data or time
to make it trustworthy.

### 9. OCR as a veto, not a score term

**Decision.** Readability is checked by rendering the top candidates and running OCR,
and only as a veto on a headline or button that has lost a third or more of its words.

**Why.** The judge is an AI viewing the rendered image, so "would it read?" is better
answered by reading it than by a type-size rule. In our renders, below about 4 px of
letter height most words are lost, turned text reads poorly at any size, and among
larger texts the misses are text on busy or low-contrast ground, which no size rule
sees. A text only has to read if it reads in the *source* rendered the same way, so
a script signature or decorative face is not held against us.

**Cost.** OCR is slow and imperfect, so it is only applied to the top few candidates
and only to the headline and button. Words OCR cannot read in the source itself are
never held against a candidate.

### 10. Photo "subject mask" is off

**Decision.** A learned foreground mask (IS-Net) is implemented in
[`vision.py`](../engine/vision.py) but disabled (`CHAMELEON_SUBJECT_MASK=1` enables it).
Classical saliency, alpha masks and face detection are used instead.

**Why.** The critic cannot yet use it without flipping layout choices on unrelated
designs, so it stays off until the critic is trustworthy enough to take advantage
of it.

**Cost.** On textured photos where the subject is an object on marble or wood, we
rely on tone-based heuristics, which are coarser than a real mask.

### 11. Quality over speed, within a laptop budget

**Decision.** Search broadly (many candidates, the best few rendered for OCR) rather than
return the first acceptable layout.

**Why.** The rules say a slower, higher-quality result is not thrown out for being
slower, though speed is a rubric row.

**Cost.** About 1–2 s per design, plus about 30 s on the very first design (one-time
warm-up). Fast enough for batch use, not real-time.

### 12. Never crash

**Decision.** Each candidate is built in its own try/except, and the whole resize
falls back to a plain uniform scale on failure.

**Why.** An unusual hidden design should score badly, not zero.

**Cost.** A real bug can hide behind the fallback. `CHAMELEON_DEBUG=1` prints
tracebacks for exactly this reason.

### 13. How we evaluate, and why we do not trust one number

**Decision.** We compare against **designer-made resizes** (Crello) as evidence for
the elements both versions kept, and view side-by-side sheets of *original | each
engine*, never the designer's resize as a reference image.

**Why.** Of 49 Crello designer resizes only 24 keep the same elements, so a
designer's resize is a noisy target. And the judge compares to the *original*. Scores
are computed on shared elements only, and we always judge in real fonts, never a
fallback face.

**Cost.** No single metric tells us we are winning. Visual review of the sheets
remains part of the loop, which is slow.
