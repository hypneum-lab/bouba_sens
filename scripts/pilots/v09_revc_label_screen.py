"""Raw-feature screen of the rev. C two-channel labels and splits (2026-10-28).

For both worlds (CC-film 10 fps, primary; Studyforrest, secondary): the
label provenance and class counts (``v09_revc.two_channel_labels``), the
frozen purged segment split (``SegmentSplitWorld.describe``) and the raw
linear-probe balanced accuracies of y, a and b per modality and for the
concatenation (10 contiguous time folds, 64-frame purge;
``v09_revc_label_feasibility.probe_all``). No network, no seed.

Run from the repository root on macM1 (2026-10-28):

  uv run python scripts/pilots/v09_revc_label_screen.py > \
    reports/pilots/v09_revc_label_screen.json

Data: ~/hypneum-lab/bouba_sens/data/{ccfilm_real_10fps,studyforrest_real}.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import v09_revc_label_feasibility as feas

from bouba_sens.v09_revc import SegmentSplitWorld, two_channel_labels

DATA = Path.home() / "hypneum-lab/bouba_sens/data"


def main() -> None:
    out = {}
    for src, sub in (("ccfilm", "ccfilm_real_10fps"), ("studyforrest", "studyforrest_real")):
        d = DATA / sub
        x, _ = feas.load(d)
        y, info = two_channel_labels(src, x)
        a, b = y // 2, y % 2
        w = SegmentSplitWorld(src, d)
        out[src] = {
            "label": info,
            "split": w.describe(),
            "raw_bacc_y": feas.probe_all(x, y, 64),
            "raw_bacc_a": feas.probe_all(x, a, 64),
            "raw_bacc_b": feas.probe_all(x, b, 64),
        }
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
