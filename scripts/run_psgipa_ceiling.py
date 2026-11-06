"""Increment 57 (pre-registered, docs/preregistration_increment57.md): expert-agreement ceiling for sleep staging on
PSG-IPA (PhysioNet, 5 nights, 12 independent manual scorers + semi-automatic + automatic scoring). Only the
annotation text files are read (no signals); the product model is not run (the montage has Cz instead of C4).

    PYTHONPATH=src python scripts/run_psgipa_ceiling.py run OUT.json
"""

import json
import subprocess
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

BASE = "https://physionet-open.s3.amazonaws.com/psg-ipa/1.0.0/Sleep_stages/Annotations"
NIGHTS, SCORERS, SEED, N_BOOT = range(1, 6), range(1, 13), 57, 2000
STAGE = {"Sleep stage W": 0, "Sleep stage N1": 1, "Sleep stage N2": 2, "Sleep stage N3": 3, "Sleep stage R": 4}
NAMES = ["W", "N1", "N2", "N3", "R"]


def _get(url: str) -> str:
    return subprocess.run(["curl", "-sSfL", "--retry", "5", "--retry-delay", "4", url], check=True,
                          capture_output=True).stdout.decode("utf-8", "replace")


def parse(text: str) -> dict[int, int]:
    """{epoch index: stage} for 30-s stage lines; '?' and non-stage annotations are skipped."""
    out = {}
    for ln in text.splitlines()[1:]:
        p = [x.strip() for x in ln.split(",")]
        if len(p) >= 5 and p[4] in STAGE and float(p[3]) == 30:
            out[int(round(float(p[2]) / 30.0))] = STAGE[p[4]]
    return out


def kappa(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0:
        return float("nan")
    cm = np.zeros((5, 5))
    np.add.at(cm, (a, b), 1)
    n = cm.sum()
    po = np.trace(cm) / n
    pe = (cm.sum(0) * cm.sum(1)).sum() / n**2
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def f1(truth: np.ndarray, pred: np.ndarray) -> list[float]:
    out = []
    for c in range(5):
        tp, fp, fn = np.sum((pred == c) & (truth == c)), np.sum((pred == c) & (truth != c)), np.sum((pred != c) & (truth == c))
        out.append(float(2 * tp / (2 * tp + fp + fn)) if tp + fp + fn else float("nan"))
    return out


def load() -> dict:
    d = {"manual": {}, "semiauto": {}, "auto": {}}
    for n in NIGHTS:
        for s in SCORERS:
            for kind in ("manual", "semiauto"):
                d[kind][(n, s)] = parse(_get(f"{BASE}/{kind}/SN{n}_SleepStages_{kind}_scorer{s}.txt"))
        d["auto"][(n, 0)] = parse(_get(f"{BASE}/auto/SN{n}_SleepStages_auto.txt"))
    return d


def _vec(night_scores: list[dict], keys) -> np.ndarray:
    return np.array(keys)


def pooled_pairs(d: dict, nights) -> dict:
    """kappa of every manual pair, epochs of the given nights pooled."""
    res = {}
    for s1, s2 in combinations(SCORERS, 2):
        a, b = [], []
        for n in nights:
            x, y = d["manual"][(n, s1)], d["manual"][(n, s2)]
            common = sorted(set(x) & set(y))
            a += [x[k] for k in common]
            b += [y[k] for k in common]
        res[(s1, s2)] = kappa(np.array(a), np.array(b))
    return res


def consensus(d: dict, kind_scorers: list[int], n: int) -> dict[int, int]:
    """Majority stage of the given manual scorers; epochs with a tie or fewer than 3 votes are dropped."""
    votes: dict[int, list[int]] = {}
    for s in kind_scorers:
        for k, c in d["manual"][(n, s)].items():
            votes.setdefault(k, []).append(c)
    out = {}
    for k, v in votes.items():
        if len(v) < 3:
            continue
        cnt = np.bincount(v, minlength=5)
        top = np.flatnonzero(cnt == cnt.max())
        if len(top) == 1:
            out[k] = int(top[0])
    return out


def against_consensus(d: dict, kind: str, scorer: int, nights, others: list[int]):
    a, b = [], []
    for n in nights:
        cons = consensus(d, others, n)
        own = d[kind][(n, scorer)]
        common = sorted(set(cons) & set(own))
        a += [cons[k] for k in common]
        b += [own[k] for k in common]
    return np.array(a), np.array(b)


def assess(d: dict) -> dict:
    nights = list(NIGHTS)
    pairs = pooled_pairs(d, nights)
    med_pair = float(np.median(list(pairs.values())))
    rng = np.random.default_rng(SEED)
    boots = []
    for _ in range(N_BOOT):
        pick = [int(x) for x in rng.integers(1, 6, 5)]
        boots.append(float(np.median(list(pooled_pairs(d, pick).values()))))
    ci = [round(float(q), 3) for q in np.percentile(boots, [2.5, 97.5])]
    per_scorer, f1s, a_all, b_all = {}, [], [], []
    for s in SCORERS:
        a, b = against_consensus(d, "manual", s, nights, [x for x in SCORERS if x != s])
        per_scorer[s] = round(kappa(a, b), 3)
        f1s.append(f1(a, b))
        a_all.append(a); b_all.append(b)
    per_night = {n: round(float(np.median([kappa(*against_consensus(d, "manual", s, [n], [x for x in SCORERS if x != s]))
                                           for s in SCORERS])), 3) for n in nights}
    allscorers = list(SCORERS)
    other = {}
    for kind in ("semiauto", "auto"):
        ks = []
        for s in (SCORERS if kind == "semiauto" else [0]):
            a, b = [], []
            for n in nights:
                cons = consensus(d, allscorers, n)
                own = d[kind][(n, s)]
                common = sorted(set(cons) & set(own))
                a += [cons[k] for k in common]
                b += [own[k] for k in common]
            ks.append(kappa(np.array(a), np.array(b)))
        other[kind] = {"median_kappa_vs_consensus_all12": round(float(np.median(ks)), 3),
                       "range": [round(float(min(ks)), 3), round(float(max(ks)), 3)]}
    f1_med = {NAMES[c]: round(float(np.nanmedian([f[c] for f in f1s])), 3) for c in range(5)}
    med_cons = float(np.median(list(per_scorer.values())))
    return {"n_nights": 5, "n_scorers": 12,
            "pairwise_manual_median_kappa": round(med_pair, 3), "pairwise_manual_ci95_by_night": ci,
            "pairwise_manual_range": [round(min(pairs.values()), 3), round(max(pairs.values()), 3)],
            "scorer_vs_consensus_of_other_11": per_scorer, "scorer_vs_consensus_median": round(med_cons, 3),
            "scorer_vs_consensus_median_by_night": per_night, "f1_median_by_stage": f1_med,
            "semiauto_and_auto": other, "model_reference_kappa": 0.635,
            "H1_pairwise_in_0.65_0.85": 0.65 <= med_pair <= 0.85,
            "H2_ceiling_above_model": med_cons > 0.635,
            "H3_n1_lowest_below_0.60_and_W_R_above_0.80": bool(f1_med["N1"] == min(f1_med.values()) and f1_med["N1"] < 0.60
                                                              and f1_med["W"] > 0.80 and f1_med["R"] > 0.80)}


def main() -> int:
    if sys.argv[1] != "run":
        return 2
    res = assess(load())
    Path(sys.argv[2]).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
