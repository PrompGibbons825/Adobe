"""Run resize on input/*.json + tools/cases/*.json x the sheet catalogue; dump outputs + timings.
usage: check.py ROOT OUTDIR [SHARD NSHARDS]"""
import glob, json, sys, time, statistics
from pathlib import Path
root = Path(sys.argv[1]).resolve(); out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
sh, nsh = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (0, 1)
sys.path.insert(0, str(root))
from engine.resize import resize
from tools.sheet import CATALOGUE
secs = {}
jobs = []
for f in sorted(glob.glob(str(root / "input/*.json")) + glob.glob(str(root / "tools/cases/*.json"))):
    t = json.load(open(f))
    tg = [("given", t["target_canvas"])] + [(k, {"width": w, "height": h}) for k, (w, h) in CATALOGUE.items()]
    for k, c in tg:
        jobs.append((f"{Path(f).stem}__{k}", t["source"], c))
for j, (n, s, c) in enumerate(jobs):
    if j % nsh != sh:
        continue
    t0 = time.perf_counter(); o = resize(s, c); secs[n] = time.perf_counter() - t0
    (out / f"{n}.json").write_text(json.dumps(o, sort_keys=True))
json.dump(secs, open(out / f"_secs_{sh}.json", "w"))
print(f"shard {sh}: n={len(secs)}")
