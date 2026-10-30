# Origin — independent Chameleon research engine

Origin takes the official design JSON and target canvas and emits the same
schema. It is a separate implementation built from the input/output contract.
Only the starter's renderer, schema, assets, Dockerfile and task runner are
reused. No `engine.smart` code is imported or copied.

The central representation is a scene graph with uncertain grouping: explicit
and inferred rigid islands, elastic text/panels, canvas backgrounds, and a frame
of ornamental artwork around usable negative space. Multiple grouping hypotheses
are retained. Four approaches compete:

1. **Transport:** solve two least-squares deformation fields and project them to monotone
   mappings, assigning
   stretch to whitespace while protecting the spans of connected artwork.
2. **Hierarchy:** reconstruct the source separators, with alternative orientations.
3. **Constraint search:** explore separator, orientation and space allocation with
   bounded lookahead, using a cheap typography bound before exact realization.
4. **Geometry transfer (experimental):** retrieve coordinate arrangements from
   unrelated designer families. The entire evaluation family set is excluded
   before compiling the library; no asset pixels or text content are stored.

All proposals pass through the same realization and finalization. One objective
balances readable copy, image prominence, source relationships, collisions,
hierarchy and contrast. Appearance scoring uses a bounded thumbnail compositor;
final previews use the untouched official renderer. The input's copy, font
families, assets, style runs, opacity and rotation remain unchanged. Elements are
never intentionally hidden or removed, even when a tiny target cannot make all
copy comfortably readable. This limitation is reported, not disguised.

## Run

Python 3.11:

```bash
pip install -r requirements.txt
python -m engine.main
```

This reads `input/*.json` and writes the official `output/` JSON and PNGs.
`ORIGIN_METHOD=elastic`, `hierarchy`, `search`, or `portfolio` chooses a strategy.
`ORIGIN_PRIOR=0` disables the experimental geometry-transfer proposals in the
portfolio (enabled in the evaluated default configuration). `ORIGIN_ASSETS` can point to a development asset directory.

The existing starter Dockerfile builds the same offline interface:

```bash
docker build --platform linux/amd64 -t origin-chameleon .
docker run --rm --network none -v "$PWD/input:/app/input" -v "$PWD/output:/app/output" origin-chameleon
```

## Evaluation and isolation

The research scripts use public development designs only, with a deterministic
family split. `lab/PLAN.txt` records the comparison plan. `lab.tournament` evaluates
method ablations on identical tasks, `lab.baselines` checks the saved incumbent
outputs on the same IDs, `lab.gallery` creates visual comparisons, and `lab.oracle`
shows alternate candidates to distinguish selection errors from missing layouts.
`lab.check_external` calls the existing checker in a separate evaluator process;
it is not a runtime engine dependency.

Reliability and appearance are separate acceptance criteria. Structural tests,
metamorphic tests and stress tests do not establish visual superiority. An
internal score, larger font, or lower geometric error is not an Adobe evaluator
verdict. See `RESULTS.txt` for measured outcomes and remaining limitations.

The sprint directory is an independent Git repository. It has no shared worktree
registration, engine-repository remote, loop hook, timer, ledger entry or gate run.
All outputs and locks are private to this directory. The ThinkPad launcher limits
execution to one CPU, 1500 MB memory, zero swap, low priority and a wall timeout.
Only one job can acquire its private compute lock at a time.

## Sources and originality

The implementation and experiments are original to this sprint. Monotone
transport, scene graphs, slicing floorplans and retrieval are established ideas;
no claim of research priority is made. The contribution is their concrete
combination around uncertain scene relationships and preservation of negative
space, plus one finalized-output selection path and bounded execution.

Conceptual background: O'Donovan, Agarwala and Hertzmann, *Learning Layouts for
Single-Page Graphic Designs* (TVCG 2014), https://research.adobe.com/publication/learning-layouts-for-single-page-graphic-designs/.
The optional geometry library is compiled from CyberAgent Crello multi-size
families under CDLA-Permissive-2.0; `engine/data/provenance.json` records its
construction and evaluation-family exclusions. Renderer code and asset rights
remain with their original owners. No generated asset imagery is used.
