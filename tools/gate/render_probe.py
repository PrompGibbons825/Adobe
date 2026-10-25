"""Gate tool: per-task memory and time probe of one engine on one sealed set.

    .venv/bin/python -m tools.gate.render_probe <version> [--engine ROOT] [--mem-mb 4096] [--timeout 300]

Each task runs in its own subprocess (``engine_run.run_isolated``) under a resident-memory
cap and a timeout. The report is by class only (canvas sizes, element counts, asset pixel
sizes, peak RSS, the engine frame active at the peak), never names, ids or content, so it
may be printed from a sealed set (loop/ROUND.md rule 5).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tools.gate import sealed
from tools.gate.engine_run import GATE_ROOT, run_isolated


def _asset_px(kit: Path, source: dict) -> int:
    """Largest asset of the design in pixels (header only, nothing decoded)."""
    try:
        from PIL import Image
    except Exception:  # pragma: no cover
        return 0
    best = 0
    for e in source.get("elements", []):
        a = e.get("asset_id") or e.get("src") or e.get("asset")
        if not isinstance(a, str):
            continue
        p = kit / "assets" / a
        if p.is_file():
            try:
                with Image.open(p) as im:
                    best = max(best, im.size[0] * im.size[1])
            except Exception:
                pass
    return best


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("version")
    ap.add_argument("--engine", default=str(GATE_ROOT))
    ap.add_argument("--mem-mb", type=int, default=2500)
    ap.add_argument("--timeout", type=float, default=300)
    ap.add_argument("--only", default="", help="comma-separated task indices")
    a = ap.parse_args(argv)
    kit = sealed.assets_dir(a.version).parent
    tasks = list(sealed.iter_tasks(a.version))
    rows = []
    only = {int(x) for x in a.only.split(",") if x}
    for i, t in enumerate(tasks):
        if only and i not in only:
            continue
        job = {"name": f"t{i}", "source": t["source"], "target": t["target_canvas"]}
        r = run_isolated(Path(a.engine), job, kit=kit, mem_mb=a.mem_mb, timeout=a.timeout)
        src, tc = t["source"], t["target_canvas"]
        kinds = {}
        for e in src.get("elements", []):
            kinds[e.get("type")] = kinds.get(e.get("type"), 0) + 1
        rows.append({"i": i, "src": f"{src.get('canvas', {}).get('width')}x{src.get('canvas', {}).get('height')}",
                     "tgt": f"{tc.get('width')}x{tc.get('height')}", "kinds": kinds,
                     "asset_mpx": round(_asset_px(kit, src) / 1e6, 1), "rss_mb": r["rss_mb"],
                     "secs": r["secs"], "status": r["status"], "peak_at": r.get("peak_at"),
                     "err": (r.get("error") or "").split(":")[0][:60]})
        print(json.dumps(rows[-1]), flush=True)
    rows.sort(key=lambda r: -(r["rss_mb"] or 0))
    print("top by peak RSS:", file=sys.stderr)
    for r in rows[:10]:
        print(json.dumps(r), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
