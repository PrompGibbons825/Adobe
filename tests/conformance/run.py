"""Run the conformance suite: every feature case, then the property set.

    python -m tests.conformance.run [--n 2000] [--seed 1]
    docker run --rm --network none -v $(pwd):/repo -w /repo --entrypoint python <image> -m tests.conformance.run
"""

from __future__ import annotations

import argparse
import logging
import platform
import sys
import time

from tests.conformance import prop
from tests.conformance.features import CASES


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    logging.disable(logging.WARNING)
    import skia  # type: ignore

    print(f"platform {platform.platform()} python {platform.python_version()} skia {skia.__version__}")
    bad = 0
    for name, fn in CASES.items():
        t = time.time()
        try:
            errs = fn()
        except Exception as x:  # noqa: BLE001
            errs = [f"crashed: {type(x).__name__}: {x}"]
        bad += bool(errs)
        print(f"{'ok  ' if not errs else 'FAIL'} {name} ({time.time() - t:.1f}s)")
        for m in errs[:6]:
            print(f"       {m}")
    print(f"features: {len(CASES) - bad}/{len(CASES)} conform")
    rc = prop.main(["--n", str(args.n), "--seed", str(args.seed)])
    return 1 if bad or rc else 0


if __name__ == "__main__":
    sys.exit(main())
