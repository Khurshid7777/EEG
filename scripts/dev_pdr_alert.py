"""Increment 44, development only: per-epoch occipital spectra for the ds005385 development sample
of increment 43 (100 people, eyes closed before / after 2 h of tasks), to choose how to measure
the posterior dominant rhythm on alert epochs before its pre-registration. The increment-43 test
sample and the fresh sample are not touched.

    PYTHONPATH=src python scripts/dev_pdr_alert.py run data/pdr_dev.npz.d [--workers 7]
    PYTHONPATH=src python scripts/dev_pdr_alert.py explore data/pdr_dev.npz.d
"""

import argparse
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_ds005385_serial import BASE, STABLE, _curl, split  # noqa: E402

EPOCH_S = 4.0


def occipital_epochs(analysis) -> tuple[np.ndarray, np.ndarray]:
    """(freqs, psd[n_epochs, n_freqs]) of the O1/O2 mean, 4-s non-overlapping epochs."""
    from scipy.signal import periodogram

    fs = analysis.sampling_rate_hz
    x = np.asarray(analysis.signal, float)[[analysis.channel_names.index(c) for c in ("O1", "O2")]].mean(axis=0)
    n = int(EPOCH_S * fs)
    segs = np.stack([x[k:k + n] for k in range(0, x.size - n + 1, n)])
    f, p = periodogram(segs, fs=fs, window="hann", axis=-1)
    keep = (f >= 1.0) & (f <= 30.0)
    return f[keep], p[:, keep]


def _one(item):
    sub, ses, task, acq, out_dir = item
    from neurolens.layer1_ingest.registry import ingest
    from neurolens.layer2_preprocess.filters import apply_filters
    from neurolens.layer2_preprocess.reref import rereference
    from neurolens.pipeline.config_loader import load_configs

    name = f"{sub}_ses-{ses}_task-{task}_acq-{acq}_eeg.edf"
    dst = Path(out_dir) / f"{sub}_{acq}.npz"
    if dst.exists():
        return str(dst)
    try:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / name
            _curl(f"{BASE}/{sub}/ses-{ses}/eeg/{name}", p)
            sig = ingest(p)
        idx = [sig.channel_names.index(c) for c in
               ("Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz", "C4", "T4", "T5", "P3", "Pz",
                "P4", "T6", "O1", "O2") if c in sig.channel_names]
        sig = sig.model_copy(update={"signal": np.ascontiguousarray(sig.signal[idx]),
                                     "channel_names": [sig.channel_names[i] for i in idx]})
        a = rereference(apply_filters(sig, load_configs().filters), scheme="average")
        f, p = occipital_epochs(a)
        np.savez_compressed(dst, f=f, p=p.astype(np.float32))
        return str(dst)
    except Exception as exc:  # noqa: BLE001
        return f"ERROR {sub} {acq}: {exc!r}"


def pdr_candidates(f: np.ndarray, p: np.ndarray) -> dict:
    from neurolens.layer3_features.peak_utils import peak_from_psd

    thr = np.log10(2.0)
    per = [peak_from_psd(f, row) for row in p]
    freq = np.array([x[0] for x in per])
    prom = np.array([x[1] for x in per])

    def peak(rows):
        fr, pr = peak_from_psd(f, p[rows].mean(axis=0))
        return fr if pr >= thr else 0.0

    alpha = p[:, (f >= 8) & (f <= 13)].sum(axis=1) / p.sum(axis=1)
    out = {"all_epochs": peak(np.ones(len(p), bool))}
    for q in (0.5, 0.75):
        out[f"prom_q{q}"] = peak(prom >= np.quantile(prom, q))
        out[f"alpha_q{q}"] = peak(alpha >= np.quantile(alpha, q))
    g = freq[prom >= thr]
    out["median_genuine"] = float(np.median(g)) if g.size >= 6 else out["all_epochs"]
    return out


def explore(d: Path) -> None:
    files = {p.stem: p for p in d.glob("*.npz")}
    subs = sorted({k.rsplit("_", 1)[0] for k in files})
    res = {}
    for s in subs:
        if f"{s}_pre" in files and f"{s}_post" in files:
            res[s] = [pdr_candidates(*(lambda z: (z["f"], z["p"]))(np.load(files[f"{s}_{a}"]))) for a in ("pre", "post")]
    for m in res[subs[0]][0]:
        pre = np.array([v[0][m] for v in res.values()])
        post = np.array([v[1][m] for v in res.values()])
        both = (pre > 0) & (post > 0)
        dlt = (post - pre)[both]
        lim = 2.576 * np.sqrt(np.mean(dlt ** 2))
        print(f"{m:15s} found {np.mean(pre > 0):.2f}/{np.mean(post > 0):.2f}  limit {lim:.2f} Hz  "
              f"|Δ|>1 Hz {np.mean(np.abs(dlt) > 1):.2f}  |Δ|>1.14 Hz {np.mean(np.abs(dlt) > 1.1362):.2f}  "
              f"median pre {np.median(pre[pre > 0]):.2f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "explore"])
    ap.add_argument("dir")
    ap.add_argument("--workers", type=int, default=7)
    a = ap.parse_args()
    d = Path(a.dir)
    if a.cmd == "explore":
        explore(d)
        return 0
    d.mkdir(parents=True, exist_ok=True)
    todo = [(p["participant_id"], *rec, str(d)) for p in split()["dev"] for rec in STABLE]
    with ProcessPoolExecutor(a.workers) as ex:
        for r in ex.map(_one, todo):
            if r.startswith("ERROR"):
                print(r, flush=True)
    print(f"{len(list(d.glob('*.npz')))} of {len(todo)} saved", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
