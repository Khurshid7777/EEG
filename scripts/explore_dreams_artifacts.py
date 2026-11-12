"""Increment 61, exploratory only (not used to tune anything): why did the pre-registered segmentation miss the
expert's artifacts on DREAMS? Per-second feature medians inside annotated seconds of each type vs. seconds with no
annotation of any EEG type. The pre-registered thresholds are NOT changed by this analysis.

    PYTHONPATH=src python scripts/explore_dreams_artifacts.py DIR OUT.json
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_dreams_artifacts as r  # noqa: E402


def features(x: np.ndarray, fs: float):
    n = int(len(x) // round(fs))
    w = int(round(fs))
    nyq = fs / 2
    band = sosfiltfilt(butter(3, [0.5 / nyq, min(35.0, 0.9 * nyq) / nyq], btype="band", output="sos"), x)
    slow = sosfiltfilt(butter(2, 0.5 / nyq, btype="low", output="sos"), x)
    std = band[: n * w].reshape(n, w).std(axis=1)
    sw = slow[: n * w].reshape(n, w)
    p2p = np.array([sw[max(0, i - 1):min(n, i + 3)].max() - sw[max(0, i - 1):min(n, i + 3)].min() for i in range(n)])
    k = max(1, int(np.ceil(0.02 * fs)))
    jump = np.concatenate([np.abs(x[k:] - x[:-k]), np.zeros(k)])[: n * w].reshape(n, w).max(axis=1)
    return n, {"std_0.5_35": std, "slow_p2p": p2p, "jump_20ms": jump}


def main() -> int:
    d, out = Path(sys.argv[1]), Path(sys.argv[2])
    acc: dict[str, dict[str, list]] = {}
    for k in range(1, 21):
        x, fs, _ = r.read_edf_channel(d / f"excerpt{k}.edf", ("CZ-A", "C3-A", "CZ2-A"))
        n, feats = features(x, fs)
        iv = r.intervals(d / f"Visual_scoring1_excerpt{k}.txt")
        clean = ~r.seconds_mask(iv, n)
        # features relative to the record median (what the pre-registered rules use) and absolute
        for name, v in feats.items():
            med = np.median(v)
            for tag in ("ondE", "mvtE", "highE", "transE"):
                m = r.seconds_mask(iv, n, {tag})
                if m.sum() >= 5:
                    acc.setdefault(tag, {}).setdefault(name, []).append((float(np.median(v[m])), float(np.median(v[m]) / med)))
            if clean.sum() >= 5:
                acc.setdefault("no_artifact", {}).setdefault(name, []).append((float(np.median(v[clean])), float(np.median(v[clean]) / med)))
    res = {tag: {name: {"median_absolute": round(float(np.median([a for a, _ in vals])), 2),
                        "median_ratio_to_record_median": round(float(np.median([b for _, b in vals])), 2), "excerpts": len(vals)}
                 for name, vals in feats.items()} for tag, feats in acc.items()}
    out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
