#!/bin/bash
# Every engine over a Crello pair folder (default crello_fresh): ours in this
# repo and at step L, line/jeff-rules, and each distinct engine on Vlatko's
# branches (worktrees under ~/vs-wt). Outputs land in review/<root>/<engine>.
A=/Users/jeffreyyu/Adobe; R=${1:-crello_fresh}; O=$A/review/$R; WT=/Users/jeffreyyu/vs-wt
mkdir -p $O
run_ours () {  # name repo
  rm -rf $O/$1; (cd $2 && CHAMELEON_ASSETS_DIR=$A/$R/assets CHAMELEON_FONTS_PICKLE=$A/$R/fonts_subset.pickle \
    $A/.venv/bin/python $A/tools/h2h.py . $1 $A/$R $O/$1 > $O/$1.log 2>&1); echo "$1: $(tail -1 $O/$1.log)"; }
run_ours ours_now $A
run_ours ours_stepL /Users/jeffreyyu/adobe-stepL
run_ours jeffrules $WT/v_line-jeff-rules
for d in $WT/v_*; do
  n=$(basename $d); [ $n = v_line-jeff-rules ] && continue
  rm -rf $O/$n; (cd $d && CHAMELEON_BRAND_KIT=$A/$R $A/.venv/bin/python $A/tools/h2h.py . $n $A/$R $O/$n > $O/$n.log 2>&1)
  echo "$n: $(grep -v -i font $O/$n.log | tail -1)"
done
echo ALLDONE
