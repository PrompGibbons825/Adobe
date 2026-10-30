# Testing the engine (and comparing engines)

Branch: `engine-v1`.

## 1. Run the engine

```bash
git clone git@github.com:PrompGibbons825/Adobe.git && cd Adobe && git checkout engine-v1
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python tools/fetch_models.py      # CLIP for logo detection (~150 MB); optional
.venv/bin/python -m engine.main             # input/*.json -> output/
```

As scored:

```bash
docker build --platform linux/amd64 -t my-chameleon-engine .
docker run --rm --network none -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output my-chameleon-engine
```

Everything optional degrades quietly: without the CLIP files there is no logo
detection, without `rapidocr-onnxruntime` there is no OCR read check.

## 2. Held-out designer pairs

Crello designs the designer made in several sizes, that neither engine was
tuned on (our 46 tuning titles and the design ids in your gate manifests are
excluded). Needs the Crello parquet shards and `fonts.pickle` locally:

```bash
# validation split: 62 designs, 78 resize tasks
.venv/bin/python tools/crello_fresh.py crello_val \
    --parquet <crello>/validation/*.parquet --fonts <crello>/resources/fonts.pickle \
    --manifests <chameleon>/tools/gate/manifests
```

## 3. Run any engine on it

```bash
tools/run_fresh.sh crello_val          # ours + every engine checkout under ~/vs-wt/v_*
```

or one engine by hand (run from that engine's checkout):

```bash
# this engine
CHAMELEON_ASSETS_DIR=$PWD/crello_val/assets CHAMELEON_FONTS_PICKLE=$PWD/crello_val/fonts_subset.pickle \
    .venv/bin/python tools/h2h.py . ours crello_val review/crello_val/ours
# a chameleon engine (brand-kit fonts)
(cd <chameleon checkout> && CHAMELEON_BRAND_KIT=<Adobe>/crello_val \
    <Adobe>/.venv/bin/python <Adobe>/tools/h2h.py . v_mine <Adobe>/crello_val <Adobe>/review/crello_val/v_mine)
```

## 4. Score and look

```bash
.venv/bin/python tools/scoreboard.py review/crello_val/ours review/crello_val/v_mine \
    --roots crello_val --read --struct --json review/crello_val/scores.json
.venv/bin/python tools/sheet.py crello_val review/crello_val/compare \
    review/crello_val/ours,review/crello_val/v_mine --scores review/crello_val/scores.json
```

Scoreboard columns (designer resizes are evidence only for the elements they kept):

| column | meaning |
|---|---|
| dist | distance to the designer's resize on shared elements (position, size, order, type rank); lower is better |
| read | share of words OCR reads at a 768 px view (texts the source's own render reads) |
| lowread | share of those texts read under 40% |
| align kept | share of the source's shared edges / centre lines kept |
| cover+ | copy over the photo's subject beyond the source's |

A directory named `v_*` or `sprint` is treated as a chameleon engine and rendered
in the brand-kit fonts it measured with.

## Results (2026-10-30, distance to designer / words read, lower distance is better)

| set | engine-v1 | chameleon ws/strategies |
|---|---|---|
| challenging (crello_hard) | 0.543 / 90.5% | 0.674 / 83.0% |
| training split (crello_train) | 0.584 / 90.4% | 0.689 / 80.2% |
| validation split (crello_val) | 0.625 / 90.7% | 0.805 / 86.1% |
| first held-out (crello_fresh) | 0.461 / 93.3% | 0.560 / 84.4% |
