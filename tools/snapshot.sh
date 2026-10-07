#!/bin/zsh
# Render all holdout, dev and official cases with the current engine into review/steps/<tag>/.
tag=$1; py=.venv/bin/python
for root in crello_holdout crello; do
  CHAMELEON_ASSETS_DIR=$root/assets CHAMELEON_FONTS_PICKLE=$root/fonts_subset.pickle $py tools/snapshot.py $tag $root 2>/dev/null | tail -1
done
$py tools/snapshot.py $tag official 2>/dev/null | tail -1
