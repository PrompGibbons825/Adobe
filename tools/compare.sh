#!/bin/zsh
# Diff winning templates against the last commit and build an
# original | designer | old | new sheet for every pair that changed.
# usage: tools/compare.sh crello_holdout [out.png]
root=$1; out=${2:-review/changed_$root.png}
export CHAMELEON_ASSETS_DIR=$root/assets CHAMELEON_FONTS_PICKLE=$root/fonts_subset.pickle
py=.venv/bin/python
cp tools/choices.py /tmp/claude-choices.py      # same script for both sides
$py /tmp/claude-choices.py $root 2>/dev/null > /tmp/claude-new-$root.txt
git stash -q
$py /tmp/claude-choices.py $root 2>/dev/null > /tmp/claude-old-$root.txt
changed=(${(f)"$(diff <(cat /tmp/claude-old-$root.txt) <(cat /tmp/claude-new-$root.txt) | grep '^>' | awk '{print $2}')"})
if [[ -n "$changed" ]]; then
  $py tools/crello_report.py $root $changed >/dev/null 2>&1
  mkdir -p /tmp/claude-old; for p in $changed; do cp $root/report/${p}_ours.png /tmp/claude-old/$root-$p.png; done
fi
git stash pop -q
paste -d' ' /tmp/claude-old-$root.txt /tmp/claude-new-$root.txt | awk '$2!=$5 {print $1": "$3" -> "$6}'
[[ -z "$changed" ]] && exit 0
$py tools/crello_report.py $root $changed >/dev/null 2>&1
$py - $root $out $changed <<'PY'
import sys, json
from PIL import Image, ImageDraw, ImageFont
root, out, ps = sys.argv[1], sys.argv[2], sys.argv[3:]
idx = json.load(open(f'{root}/index.json'))
try:
    F = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 28)
except Exception:
    F = ImageFont.load_default()
rows = []
for p in ps:
    a, b = idx[int(p[1:])][0], idx[int(p[1:])][1]
    R = f'{root}/report/'
    ims = [Image.open(R + p + '_source.png'), Image.open(R + p + '_designer.png'),
           Image.open(f'/tmp/claude-old/{root}-{p}.png'), Image.open(R + p + '_ours.png')]
    fit = []
    for im in ims:
        im = im.convert('RGB'); s = min(300 / im.height, 520 / im.width)
        im = im.resize((int(im.width * s), int(im.height * s)))
        f = Image.new('RGB', (im.width + 4, im.height + 4), (40, 40, 40)); f.paste(im, (2, 2)); fit.append(f)
    r = Image.new('RGB', (4 * 540 + 20, max(f.height for f in fit) + 50), 'white')
    d = ImageDraw.Draw(r)
    d.text((10, 6), f"{p} · {a['title'][:60]} · {a['w']}x{a['h']} -> {b['w']}x{b['h']}", fill=(20, 20, 20), font=F)
    for k, f in enumerate(fit):
        r.paste(f, (10 + k * 540, 44))
    rows.append(r)
head = Image.new('RGB', (rows[0].width, 44), 'white'); d = ImageDraw.Draw(head)
for k, l in enumerate(['Original', 'Designer', 'Before', 'After']):
    d.text((10 + k * 540, 8), l, fill=(90, 90, 90), font=F)
rows = [head] + rows
S = Image.new('RGB', (rows[0].width, sum(r.height for r in rows)), 'white'); y = 0
for r in rows:
    S.paste(r, (0, y)); y += r.height
S.save(out); print(out, S.size)
PY
