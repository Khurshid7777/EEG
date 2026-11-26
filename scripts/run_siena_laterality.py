"""Increment 75 (pre-registered, docs/preregistration_increment75.md): seizure-onset lateralisation on Siena.
Only the needed EDF records are fetched (HTTP range requests).

    PYTHONPATH=src python scripts/run_siena_laterality.py run OUT.jsonl
    PYTHONPATH=src python scripts/run_siena_laterality.py assess OUT.jsonl OUT.json
"""

import csv
import io
import json
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np
from scipy.signal import welch

BASE = "https://physionet-open.s3.amazonaws.com/siena-scalp-eeg/1.0.0"
PRE_S, POST_S, ICTAL_S, BASE_FROM, BASE_TO, MIN_ONSET = 90, 20, 10, 90, 30, 100
BAND = (3.0, 12.0)
LEFT, RIGHT = ("F7", "T3", "T5"), ("F8", "T4", "T6")
MARGIN = 0.2
UNILATERAL_TEMPORAL = ["PN00", "PN01", "PN03", "PN05", "PN06", "PN07", "PN09", "PN11", "PN12", "PN13", "PN14", "PN16", "PN17"]


def _get(url, rng=None):
    req = urllib.request.Request(url, headers={"Range": f"bytes={rng[0]}-{rng[1]}"} if rng else {})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def read_edf_segment(url: str, t0: float, t1: float):
    """(names, fs, data[ch, samples] in uV) for [t0, t1) seconds; channels sharing the modal sampling rate."""
    h = _get(url, (0, 255))
    hdr_bytes, n_rec, dur, ns = int(h[184:192]), int(h[236:244]), float(h[244:252]), int(h[252:256])
    sh = _get(url, (256, 256 + ns * 256 - 1))
    f = lambda off, w: [sh[off + i * w: off + (i + 1) * w].decode("latin-1").strip() for i in range(ns)]  # noqa: E731
    o = 0
    labels = f(o, 16); o += 16 * ns
    o += 80 * ns
    dims = f(o, 8); o += 8 * ns
    pmin = np.array(f(o, 8), float); o += 8 * ns
    pmax = np.array(f(o, 8), float); o += 8 * ns
    dmin = np.array(f(o, 8), float); o += 8 * ns
    dmax = np.array(f(o, 8), float); o += 8 * ns
    o += 80 * ns
    nsamp = np.array(f(o, 8), int)
    rec_samples = nsamp.sum()
    r0, r1 = int(t0 // dur), min(n_rec, int(np.ceil(t1 / dur)))
    raw = _get(url, (hdr_bytes + r0 * rec_samples * 2, hdr_bytes + r1 * rec_samples * 2 - 1))
    arr = np.frombuffer(raw, dtype="<i2").reshape(r1 - r0, rec_samples).astype(float)
    cuts = np.concatenate([[0], np.cumsum(nsamp)])
    fs = np.bincount(nsamp).argmax() / dur
    out, names = [], []
    from neurolens.layer1_ingest.electrodes import normalize_channel_name
    for i in range(ns):
        if nsamp[i] / dur != fs or labels[i].startswith("EDF Annot"):
            continue
        x = arr[:, cuts[i]:cuts[i + 1]].reshape(-1)
        x = (x - dmin[i]) * (pmax[i] - pmin[i]) / (dmax[i] - dmin[i]) + pmin[i]
        if dims[i].lower().startswith("mv"):
            x = x * 1000.0
        out.append(x)
        names.append(normalize_channel_name(labels[i]))
    off = int((t0 - r0 * dur) * fs)
    return names, fs, np.array(out)[:, off:off + int((t1 - t0) * fs)]


def _score(x, names, fs, win, ref_idx):
    from neurolens.layer3_features.regional import STANDARD_1020
    a, b = int(win[0] * fs), int(win[1] * fs)
    seg = x[:, a:b] - x[ref_idx].mean(axis=0, keepdims=True)
    f, p = welch(seg, fs=fs, nperseg=int(2 * fs), axis=-1)
    m = (f >= BAND[0]) & (f < BAND[1])
    return {n: p[i, m].sum() for i, n in enumerate(names)}


def one_seizure(subject, fname, onset):
    names, fs, x = read_edf_segment(f"{BASE}/{subject}/{fname}", onset - PRE_S, onset + POST_S)
    from neurolens.layer3_features.regional import STANDARD_1020
    ref_idx = [i for i, n in enumerate(names) if n in STANDARD_1020]
    base = _score(x, names, fs, (PRE_S - BASE_FROM, PRE_S - BASE_TO), ref_idx)
    out = {}
    for tag, dur in (("ictal10", ICTAL_S), ("ictal20", POST_S)):
        ict = _score(x, names, fs, (PRE_S, PRE_S + dur), ref_idx)
        lr = lambda grp: float(np.mean([np.log(ict[c] / base[c]) for c in grp if c in ict]))  # noqa: E731
        out[tag] = {"S_left": lr(LEFT), "S_right": lr(RIGHT), "delta": lr(LEFT) - lr(RIGHT)}
    return out


def run(out_path):
    from neurolens.datasets.siena import parse_siena_seizure_list

    info = {r["patient_id"].strip(): {k.strip(): v.strip() for k, v in r.items()}
            for r in csv.DictReader(io.StringIO(_get(f"{BASE}/subject_info.csv").decode()), skipinitialspace=True)}
    rows = []
    for subject in sorted(info):
        text = _get(f"{BASE}/{subject}/Seizures-list-{subject}.txt").decode("latin-1")
        for rec in parse_siena_seizure_list(text, subject):
            for k, sz in enumerate(rec.seizures):
                base = {"subject": subject, "file": rec.file, "k": k, "onset_s": sz.onset_s,
                        "localization": info[subject]["localization"], "lateralization": info[subject]["lateralization"]}
                if sz.onset_s < MIN_ONSET:
                    rows.append({**base, "error": "onset < 100 s from file start"})
                    continue
                try:
                    rows.append({**base, **one_seizure(subject, rec.file.split("/")[1], sz.onset_s)})
                except Exception as exc:  # noqa: BLE001 - excluded and counted, as pre-registered
                    rows.append({**base, "error": repr(exc)})
                print(rows[-1]["subject"], rows[-1]["file"], rows[-1].get("error", round(rows[-1]["ictal10"]["delta"], 2)), flush=True)
    Path(out_path).write_text("".join(json.dumps(r) + "\n" for r in rows))


def _call(delta):
    return "L" if delta > MARGIN else ("R" if delta < -MARGIN else "?")


def assess(path):
    rows = [json.loads(l) for l in Path(path).read_text().splitlines()]
    ok = [r for r in rows if "error" not in r]
    res = {"n_seizures": len(rows), "n_excluded": len(rows) - len(ok), "errors": [(r["subject"], r["file"], r["error"]) for r in rows if "error" in r]}

    def patient_calls(window):
        out = {}
        for s in sorted({r["subject"] for r in ok}):
            c = [_call(r[window]["delta"]) for r in ok if r["subject"] == s]
            l, rr = c.count("L"), c.count("R")
            out[s] = "L" if l > rr else ("R" if rr > l else "?")
        return out

    truth = {r["subject"]: r["lateralization"] for r in rows}
    primary = [s for s in UNILATERAL_TEMPORAL]
    for window in ("ictal10", "ictal20"):
        calls = patient_calls(window)
        cm = {s: calls.get(s, "?") for s in primary}
        acc = {c: np.mean([cm[s] == c for s in primary if truth[s] == c]) for c in ("L", "R")}
        bal = float(np.mean(list(acc.values())))
        res[window] = {"patient_calls": cm, "truth": {s: truth[s] for s in primary}, "acc_left": float(acc["L"]),
                       "acc_right": float(acc["R"]), "balanced_accuracy": bal}
        if window == "ictal10":
            rng = np.random.default_rng(75)
            labels = np.array([truth[s] for s in primary])
            perm = []
            for _ in range(10000):
                sh = rng.permutation(labels)
                perm.append(np.mean([np.mean([cm[s] == c for s, t in zip(primary, sh) if t == c]) for c in ("L", "R")]))
            res[window]["permutation_p"] = float((np.sum(np.array(perm) >= bal - 1e-12) + 1) / 10001)
            # seizure level
            sl = [r for r in ok if r["subject"] in primary]
            called = [r for r in sl if _call(r[window]["delta"]) != "?"]
            correct = lambda rs: np.mean([_call(r[window]["delta"]) == truth[r["subject"]] for r in rs]) if rs else float("nan")  # noqa: E731
            bs = []
            subs = sorted({r["subject"] for r in sl})
            for _ in range(2000):
                pick = rng.choice(subs, len(subs))
                rs = [r for s in pick for r in sl if r["subject"] == s]
                cl = [r for r in rs if _call(r[window]["delta"]) != "?"]
                bs.append((correct(cl), len(cl) / len(rs)))
            res["seizure_level"] = {"n": len(sl), "coverage": len(called) / len(sl), "accuracy_among_called": float(correct(called)),
                                    "accuracy_ci95": [round(float(np.nanpercentile([b[0] for b in bs], q)), 3) for q in (2.5, 97.5)],
                                    "coverage_ci95": [round(float(np.percentile([b[1] for b in bs], q)), 3) for q in (2.5, 97.5)]}
    res["PN10_descriptive"] = [round(r["ictal10"]["delta"], 2) for r in ok if r["subject"] == "PN10"]
    res["H1"] = bool(res["ictal10"]["balanced_accuracy"] >= 0.70 and res["ictal10"]["permutation_p"] < 0.05)
    return res


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2])
    else:
        r = assess(sys.argv[2])
        Path(sys.argv[3]).write_text(json.dumps(r, indent=1, default=float) + "\n")
        print(json.dumps(r, indent=1, default=float))
