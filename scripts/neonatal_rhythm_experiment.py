"""Increment 27 (development): does a rhythmicity feature help the neonatal seizure
detector? Neonatal seizures are sustained repetitive discharges; the current epoch
features (amplitude vs own baseline, spectral concentration, dominant frequency,
relative band power) do not measure repetition.

For every Helsinki recording: the v1 epoch features (as the neonatal model uses them) and,
per channel, the maximum autocorrelation of the 0.5-8 Hz signal at lags 0.15-2 s over the
8 s ending at each epoch (causal), aggregated over channels. Grouped (by patient) 10-fold
cross-validation of the same gradient-boosted model with and without the rhythmicity
features; event-level sensitivity and false alarms per hour after the same post-processing
search on the out-of-fold probabilities.

    PYTHONPATH=src python scripts/neonatal_rhythm_experiment.py OUT.json [WORKERS]
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

CACHE = Path("data/physionet/feature_cache")
RHY_CACHE = Path("data/physionet/rhythm_cache")
WIN_S, FS_LOW, LAG_MIN_S, LAG_MAX_S = 8.0, 64.0, 0.15, 2.0
RHY_NAMES = ["rhy_max", "rhy_top3", "rhy_frac_ge_0.6", "rhy_top3_vs_base", "rhy_period_s"]


def rhythm_features(path: str, epoch_times: np.ndarray) -> np.ndarray:
    from scipy.signal import butter, resample_poly, sosfiltfilt

    from neurolens.layer1_ingest.electrodes import is_eeg_channel
    from neurolens.layer1_ingest.registry import ingest
    from neurolens.layer2_preprocess.reref import rereference

    sig = rereference(ingest(Path(path)), "average")
    fs = float(sig.sampling_rate_hz)
    idx = [i for i, n in enumerate(sig.channel_names) if is_eeg_channel(n)]
    x = sig.signal[idx].astype(np.float64)
    x = sosfiltfilt(butter(4, [0.5, 8.0], "band", fs=fs, output="sos"), x, axis=1)
    up, down = int(FS_LOW), int(round(fs))
    g = np.gcd(up, down)
    x = resample_poly(x, up // g, down // g, axis=1)
    w = int(WIN_S * FS_LOW)
    lo, hi = int(LAG_MIN_S * FS_LOW), int(LAG_MAX_S * FS_LOW)
    out = np.zeros((len(epoch_times), len(RHY_NAMES)))
    rho_hist = []
    for k, t in enumerate(epoch_times):
        e = int(round((t + 1.0) * FS_LOW))  # epoch end (2-s epochs, times are centres)
        s = e - w
        if s < 0 or e > x.shape[1]:
            rho_hist.append(0.0)
            continue
        seg = x[:, s:e]
        seg = seg - seg.mean(axis=1, keepdims=True)
        n = 2 * w
        f = np.fft.rfft(seg, n, axis=1)
        ac = np.fft.irfft(f * np.conj(f), n, axis=1)[:, :hi + 1]
        ac = ac / np.maximum(ac[:, :1], 1e-12)
        r = ac[:, lo:hi + 1]
        rho = r.max(axis=1)
        per = (lo + r.argmax(axis=1)) / FS_LOW
        top = np.sort(rho)[::-1]
        top3 = float(top[:3].mean())
        rho_hist.append(top3)
        base = float(np.median(rho_hist[-300:]))  # ~5 min causal baseline
        out[k] = [top[0], top3, float((rho >= 0.6).mean()), top3 - base, float(per[np.argmax(rho)])]
    return out


def _rhythm_cached(item):
    path, times = item
    RHY_CACHE.mkdir(parents=True, exist_ok=True)
    cp = RHY_CACHE / (Path(path).stem + ".npy")
    if cp.exists():
        a = np.load(cp)
        if a.shape[0] == len(times):
            return a
    a = rhythm_features(path, np.asarray(times))
    np.save(cp, a)
    return a


def main() -> int:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    from neurolens.evaluation.ml import Dataset, postproc_search, score_probs
    from neurolens.evaluation.runner import default_config, load_items, prepare_records

    out = Path(sys.argv[1])
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    items = load_items("helsinki", ["dev"], only_cached=True) + load_items("helsinki", ["test"], only_cached=True)
    recs = prepare_records(items, default_config(), cache_dir=CACHE, workers=workers)
    ds = Dataset.build(recs)
    with ProcessPoolExecutor(workers) as ex:
        R = list(ex.map(_rhythm_cached, [(str(r.path), list(t)) for r, t in zip(ds.records, ds.times)]))
    groups = np.array([r.annotation.subject for r in ds.records])
    variants = {"v1": [x for x in ds.X], "v1+rhythm": [np.column_stack([x, r]) for x, r in zip(ds.X, R)]}
    gkf = GroupKFold(10)
    idx_all = np.arange(len(ds.records))
    result = {"n_records": len(ds.records), "n_seizures": int(sum(len(r.annotation.seizures) for r in ds.records)),
              "rhythm_features": RHY_NAMES, "variants": {}}
    for name, Xs in variants.items():
        probs = [None] * len(ds.records)
        for tr, te in gkf.split(idx_all, groups=groups):
            X = np.vstack([Xs[i][ds.keep[i]] for i in tr])
            y = np.concatenate([ds.y[i][ds.keep[i]] for i in tr])
            clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=31,
                                                 class_weight="balanced", random_state=0).fit(X, y)
            for i in te:
                probs[i] = clf.predict_proba(Xs[i])[:, 1]
        yy = np.concatenate([ds.y[i][ds.keep[i]] for i in idx_all])
        pp = np.concatenate([probs[i][ds.keep[i]] for i in idx_all])
        row = {"epoch_auc_oof": float(roc_auc_score(yy, pp))}
        for fa in (1.0, 2.0, 4.0):
            tr_ = postproc_search(ds, probs, fa_target=fa, causal=True)
            b = tr_.best
            row[f"fa_target_{fa}"] = {"params": b.params, "sensitivity": b.score.sensitivity,
                                      "fa_per_hour": b.score.fa_per_hour,
                                      "latency_median_s": b.score.latency_median_s}
        result["variants"][name] = row
        print(name, json.dumps(row), flush=True)
    out.write_text(json.dumps(result, indent=1, default=float) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
