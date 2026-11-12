"""Increment 61 (pre-registered, docs/preregistration_increment61.md): the time-resolved artifact segmentation on
the DREAMS Artifacts Database (CC BY-NC-ND 3.0, evaluation only). Needs the extracted archive:

    PYTHONPATH=src python scripts/run_dreams_artifacts.py DIR OUT.json     # DIR = .../DatabaseArtifacts
"""

import json
import re
import sys
from pathlib import Path

import numpy as np

SEED, N_BOOT = 61, 2000
EEG_SUFFIX = "E"            # sections whose type ends with E refer to the EEG (ondE, mvtE, ..., 50HzE, satE)
TYPE_MAP = {"mvtE": ("muscle", "high_amp"), "transE": ("pop",), "ondE": ("drift",), "highE": ("high_amp",)}


def read_edf_channel(path: Path, wanted: tuple[str, ...]):
    """(signal in µV, fs, label) of the first channel whose label starts with one of ``wanted`` (case-insensitive)."""
    raw = path.read_bytes()
    ns = int(raw[252:256])
    labs = [raw[256 + 16 * i:272 + 16 * i].decode("latin-1").strip() for i in range(ns)]
    o = 256 + ns * 16 + ns * 80 + ns * 8                      # after labels, transducers, physical dimensions
    field = lambda base, i: float(raw[base + 8 * i:base + 8 * i + 8])  # noqa: E731
    pmin = [field(o, i) for i in range(ns)]
    pmax = [field(o + ns * 8, i) for i in range(ns)]
    dmin = [field(o + 2 * ns * 8, i) for i in range(ns)]
    dmax = [field(o + 3 * ns * 8, i) for i in range(ns)]
    o2 = o + 4 * ns * 8 + ns * 80                              # after prefilter fields
    spr = [int(raw[o2 + 8 * i:o2 + 8 * i + 8]) for i in range(ns)]
    nrec = int(raw[236:244]); dur = float(raw[244:252])
    idx = next(i for w in wanted for i, lab in enumerate(labs) if lab.upper().startswith(w))
    rec_samples = sum(spr)
    data = np.frombuffer(raw, dtype="<i2", offset=256 * (1 + ns)).reshape(nrec, rec_samples)
    off = sum(spr[:idx])
    x = data[:, off:off + spr[idx]].reshape(-1).astype(np.float64)
    gain = (pmax[idx] - pmin[idx]) / (dmax[idx] - dmin[idx])
    x = (x - dmin[idx]) * gain + pmin[idx]
    return x, spr[idx] / dur, labs[idx]


def intervals(path: Path, eeg_only: bool = True) -> list[tuple[str, float, float]]:
    out, cur = [], None
    for ln in path.read_text(errors="replace").splitlines():
        m = re.match(r"\[(?:vis\d*_)?(?:atf_)?([A-Za-z0-9]+)/", ln.strip(), re.I)
        if m:
            cur = m.group(1)
            continue
        p = ln.split()
        if cur is not None and len(p) == 2:
            try:
                out.append((cur, float(p[0]), float(p[1])))
            except ValueError:
                pass
    if eeg_only:
        out = [(t, a, d) for t, a, d in out if t.endswith(EEG_SUFFIX) and t.lower() not in {"atf_eeg", "eeg"}]
    return out


def seconds_mask(iv, n: int, types=None) -> np.ndarray:
    m = np.zeros(n, bool)
    for t, a, d in iv:
        if types is not None and t not in types:
            continue
        for s in range(max(0, int(np.floor(a))), min(n, int(np.ceil(a + d)))):
            if min(a + d, s + 1) - max(a, s) >= 0.5:
                m[s] = True
    return m


def counts(pred: np.ndarray, truth: np.ndarray):
    return int((pred & truth).sum()), int((~pred & truth).sum()), int((pred & ~truth).sum()), int((~pred & ~truth).sum())


def metrics(c):
    tp, fn, fp, tn = c
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else float("nan")
    return sens, spec, f1


def main() -> int:
    from neurolens.layer2_preprocess.artifact_segments import segment_channel

    d, out = Path(sys.argv[1]), Path(sys.argv[2])
    per, rows = {}, []
    for k in range(1, 21):
        edf = d / f"excerpt{k}.edf"
        if not edf.exists():
            rows.append({"excerpt": k, "error": "no EDF"}); continue
        try:
            x, fs, lab = read_edf_channel(edf, ("CZ-A", "C3-A", "CZ2-A"))
            masks = segment_channel(x, fs)
            n = len(next(iter(masks.values())))
            truth_iv = intervals(d / f"Visual_scoring1_excerpt{k}.txt")
            auto_iv = intervals(d / f"Automatic_detection_excerpt{k}.txt")
            truth = seconds_mask(truth_iv, n)
            union = np.logical_or.reduce(list(masks.values()))
            auto = seconds_mask(auto_iv, n)
            row = {"excerpt": k, "channel": lab, "fs": fs, "n": n, "prevalence": float(truth.mean()),
                   "pred_fraction": float(union.mean()), "ours": counts(union, truth), "authors": counts(auto, truth),
                   "by_type": {}}
            for t, ours in TYPE_MAP.items():
                tm = seconds_mask(truth_iv, n, {t})
                if tm.any():
                    pm = np.logical_or.reduce([masks[o] for o in ours])
                    row["by_type"][t] = {"truth_seconds": int(tm.sum()), "recall": float((pm & tm).sum() / tm.sum())}
            rows.append(row)
        except Exception as exc:  # noqa: BLE001 - listed and excluded, as pre-registered
            rows.append({"excerpt": k, "error": repr(exc)})
    ok = [r for r in rows if "error" not in r]
    tot = lambda key: tuple(int(sum(r[key][i] for r in ok)) for i in range(4))  # noqa: E731
    s, sp, f = metrics(tot("ours"))
    sa, spa, fa = metrics(tot("authors"))
    rng = np.random.default_rng(SEED)
    bs = []
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(ok), len(ok))
        c = tuple(int(sum(ok[i]["ours"][j] for i in pick)) for j in range(4))
        ca = tuple(int(sum(ok[i]["authors"][j] for i in pick)) for j in range(4))
        m, ma = metrics(c), metrics(ca)
        bs.append((*m, ma[2]))
    bs = np.array(bs)
    ci = lambda col: [round(float(q), 3) for q in np.nanpercentile(bs[:, col], [2.5, 97.5])]  # noqa: E731
    f1_diff = bs[:, 2] - bs[:, 3]
    by_type = {}
    for t in TYPE_MAP:
        rec = [(r["by_type"][t]["recall"], r["by_type"][t]["truth_seconds"]) for r in ok if t in r["by_type"]]
        if rec:
            by_type[t] = {"excerpts": len(rec), "truth_seconds": int(sum(w for _, w in rec)),
                          "recall_pooled": round(sum(r * w for r, w in rec) / sum(w for _, w in rec), 3)}
    status = ("confirmed" if (s >= 0.60 and sp >= 0.80) else ("limited" if (s >= 0.40 and sp >= 0.70) else "failed"))
    res = {"n_excerpts": len(ok), "errors": [(r["excerpt"], r["error"]) for r in rows if "error" in r],
           "prevalence": round(float(np.mean([r["prevalence"] for r in ok])), 3),
           "ours": {"sensitivity": round(s, 3), "specificity": round(sp, 3), "f1": round(f, 3),
                    "sensitivity_ci95": ci(0), "specificity_ci95": ci(1), "f1_ci95": ci(2)},
           "authors_algorithm": {"sensitivity": round(sa, 3), "specificity": round(spa, 3), "f1": round(fa, 3)},
           "f1_difference_ours_minus_authors": round(f - fa, 3), "f1_difference_ci95": [round(float(q), 3) for q in np.nanpercentile(f1_diff, [2.5, 97.5])],
           "by_type_recall": by_type,
           "per_excerpt": [{k: r[k] for k in ("excerpt", "fs", "prevalence", "pred_fraction")} for r in ok],
           "H1": status, "H2_f1_not_below_authors_minus_0.05": bool(f - fa >= -0.05)}
    out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
