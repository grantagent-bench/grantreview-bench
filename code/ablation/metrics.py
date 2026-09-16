"""Evaluation metrics with honest uncertainty for a small-N controlled study.

Pure-Python (no scipy dependency).  Positive class = "awarded".

Includes bootstrap CIs on accuracy and a paired McNemar test between each system
and the full pipeline, because with N in the low hundreds the interesting question
is not "which point estimate is higher" but "is the difference distinguishable from
noise".  This is the statistical backbone of a "when it fails" argument.
"""
from __future__ import annotations

import math
import random
from typing import Optional


def _rate(num: float, den: float) -> float:
    return num / den if den else 0.0


def classification(preds: list[str], labels: list[str]) -> dict:
    """preds/labels are 'awarded'/'declined', aligned."""
    tp = fp = fn = tn = 0
    correct = []
    for p, y in zip(preds, labels):
        correct.append(1 if p == y else 0)
        if y == "awarded":
            tp += (p == "awarded"); fn += (p == "declined")
        else:
            fp += (p == "awarded"); tn += (p == "declined")
    precision = _rate(tp, tp + fp)
    recall = _rate(tp, tp + fn)
    f1 = _rate(2 * precision * recall, precision + recall) if (precision + recall) else 0.0
    n = len(labels)
    return {
        "n": n, "accuracy": _rate(sum(correct), n),
        "precision": precision, "recall": recall, "f1": f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "_correct": correct,
    }


def _percentile(sorted_vals: list[float], q: float) -> float:
    # linearly interpolated percentile (numpy 'linear' method); input pre-sorted
    pos = q * (len(sorted_vals) - 1)
    lo_i = int(pos)
    hi_i = min(lo_i + 1, len(sorted_vals) - 1)
    frac = pos - lo_i
    return sorted_vals[lo_i] * (1 - frac) + sorted_vals[hi_i] * frac


def bootstrap_ci(correct: list[int], n_boot: int = 10000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float]:
    if not correct:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(correct)
    means = []
    for _ in range(n_boot):
        s = sum(correct[rng.randrange(n)] for _ in range(n))
        means.append(s / n)
    means.sort()
    return (_percentile(means, alpha / 2), _percentile(means, 1 - alpha / 2))


def balanced_accuracy(tp: int, fp: int, fn: int, tn: int) -> float:
    """Mean of per-class recall (TPR on awarded, TNR on declined).

    Robust to class imbalance; an all-one-class predictor scores 0.5, not the
    majority-class base rate.
    """
    tpr = _rate(tp, tp + fn)
    tnr = _rate(tn, tn + fp)
    return (tpr + tnr) / 2


def mcc(tp: int, fp: int, fn: int, tn: int) -> float:
    """Matthews correlation coefficient in [-1, 1] from the confusion counts.

    Returns 0.0 when any marginal is empty (zero denominator) — the standard
    convention for a degenerate predictor.
    """
    denom = math.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denom == 0:
        return 0.0
    return (tp * tn - fp * fn) / denom


def bootstrap_auc_mcc(preds: list[str], labels: list[str],
                      scores: list[Optional[float]], n_boot: int = 2000,
                      alpha: float = 0.05, seed: int = 0) -> dict:
    """Percentile-bootstrap 95% CIs for AUC and MCC by resampling items jointly.

    (pred, label, score) rows are resampled with replacement under a fixed seed,
    so both CIs come from the same 2000 resamples.  AUC is undefined on a
    single-class resample; such draws are skipped for the AUC CI.  MCC uses the
    0.0 zero-denominator guard, so every draw contributes.  Returns
    {"auc_ci95": [lo, hi] | None, "mcc_ci95": [lo, hi] | None}; auc_ci95 is None
    when the point AUC itself is undefined (no usable scores).
    """
    n = len(labels)
    if n == 0:
        return {"auc_ci95": None, "mcc_ci95": None}
    rng = random.Random(seed)
    have_auc = roc_auc(scores, labels) is not None
    aucs: list[float] = []
    mccs: list[float] = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        rl = [labels[i] for i in idx]
        tp = fp = fn = tn = 0
        for i, y in zip(idx, rl):
            if y == "awarded":
                tp += (preds[i] == "awarded"); fn += (preds[i] == "declined")
            else:
                fp += (preds[i] == "awarded"); tn += (preds[i] == "declined")
        mccs.append(mcc(tp, fp, fn, tn))
        if have_auc:
            a = roc_auc([scores[i] for i in idx], rl)
            if a is not None:
                aucs.append(a)

    def _ci(vals: list[float]) -> Optional[list[float]]:
        if not vals:
            return None
        vals = sorted(vals)
        return [round(_percentile(vals, alpha / 2), 4),
                round(_percentile(vals, 1 - alpha / 2), 4)]

    return {"auc_ci95": _ci(aucs) if have_auc else None, "mcc_ci95": _ci(mccs)}


def mann_whitney_u(a: list[float], b: list[float]) -> dict:
    """Two-sided Mann-Whitney U with normal approximation and tie correction.

    a = awarded scores, b = declined scores.  Also returns rank-biserial effect size.
    """
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return {"U": 0.0, "p": 1.0, "effect": 0.0}
    combined = sorted([(v, 0) for v in a] + [(v, 1) for v in b], key=lambda t: t[0])
    # average ranks with ties
    ranks = [0.0] * len(combined)
    i = 0
    while i < len(combined):
        j = i
        while j + 1 < len(combined) and combined[j + 1][0] == combined[i][0]:
            j += 1
        avg = (i + j) / 2 + 1  # 1-based average rank
        for t in range(i, j + 1):
            ranks[t] = avg
        i = j + 1
    r1 = sum(ranks[idx] for idx, (_, grp) in enumerate(combined) if grp == 0)
    u1 = r1 - na * (na + 1) / 2
    u = min(u1, na * nb - u1)
    mu = na * nb / 2
    # tie-corrected sigma
    n = na + nb
    tie_term = 0.0
    i = 0
    while i < len(combined):
        j = i
        while j + 1 < len(combined) and combined[j + 1][0] == combined[i][0]:
            j += 1
        tt = j - i + 1
        tie_term += tt ** 3 - tt
        i = j + 1
    sigma = math.sqrt((na * nb / 12) * ((n + 1) - tie_term / (n * (n - 1)))) if n > 1 else 0.0
    if sigma == 0:
        p = 1.0
    else:
        z = (u - mu) / sigma
        p = 2 * (1 - _norm_cdf(abs(z)))
    # Rank-biserial correlation, signed: positive when group a (awarded) scores higher.
    # (min-U must be used only for the z-score; using it here would erase direction.)
    effect = 2 * u1 / (na * nb) - 1
    return {"U": u1, "p": max(0.0, min(1.0, p)), "effect": effect}


def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def mcnemar(correct_a: list[int], correct_b: list[int]) -> dict:
    """Paired McNemar test (exact binomial on discordant pairs).

    b = A right & B wrong ; c = A wrong & B right.  p from Binomial(b+c, 0.5).
    """
    b = c = 0
    for ca, cb in zip(correct_a, correct_b):
        if ca == 1 and cb == 0:
            b += 1
        elif ca == 0 and cb == 1:
            c += 1
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "p": 1.0}
    k = min(b, c)
    # two-sided exact binomial
    p = 0.0
    for i in range(0, k + 1):
        p += math.comb(n, i) * (0.5 ** n)
    p = min(1.0, 2 * p)
    return {"b": b, "c": c, "p": p}


def roc_auc(scores: list[Optional[float]], labels: list[str]) -> Optional[float]:
    """AUC via the Mann-Whitney relationship. Positive = awarded. None if unscored."""
    pos = [s for s, y in zip(scores, labels) if y == "awarded" and s is not None]
    neg = [s for s, y in zip(scores, labels) if y == "declined" and s is not None]
    if not pos or not neg:
        return None
    mw = mann_whitney_u(pos, neg)
    u1 = mw["U"]
    return u1 / (len(pos) * len(neg))


def score_gap(scores: list[Optional[float]], labels: list[str]) -> dict:
    pos = [s for s, y in zip(scores, labels) if y == "awarded" and s is not None]
    neg = [s for s, y in zip(scores, labels) if y == "declined" and s is not None]
    ap = sum(pos) / len(pos) if pos else 0.0
    an = sum(neg) / len(neg) if neg else 0.0
    out = {"awarded_mean": ap, "declined_mean": an, "gap": ap - an}
    if pos and neg:
        out.update(mann_whitney_u(pos, neg))
    return out


def ece(scores: list[Optional[float]], labels: list[str], bins: int = 10) -> Optional[float]:
    """Expected Calibration Error using score/5 as p(award)."""
    pairs = [(s / 5.0, 1 if y == "awarded" else 0)
             for s, y in zip(scores, labels) if s is not None]
    if not pairs:
        return None
    n = len(pairs)
    total = 0.0
    for bi in range(bins):
        lo, hi = bi / bins, (bi + 1) / bins
        bucket = [(p, y) for p, y in pairs if (lo <= p < hi) or (bi == bins - 1 and p == 1.0)]
        if not bucket:
            continue
        conf = sum(p for p, _ in bucket) / len(bucket)
        acc = sum(y for _, y in bucket) / len(bucket)
        total += (len(bucket) / n) * abs(conf - acc)
    return total


def summarize(system: str, preds: list[str], labels: list[str],
              scores: list[Optional[float]], calls: list[int], tokens: list[int],
              full_correct: Optional[list[int]] = None) -> dict:
    cls = classification(preds, labels)
    correct = cls.pop("_correct")
    lo, hi = bootstrap_ci(correct)
    boot = bootstrap_auc_mcc(preds, labels, scores)
    out = {
        "system": system,
        **{k: round(v, 4) if isinstance(v, float) else v for k, v in cls.items()},
        "acc_ci95": [round(lo, 4), round(hi, 4)],
        "balanced_accuracy": round(
            balanced_accuracy(cls["tp"], cls["fp"], cls["fn"], cls["tn"]), 4),
        "mcc": round(mcc(cls["tp"], cls["fp"], cls["fn"], cls["tn"]), 4),
        "mcc_ci95": boot["mcc_ci95"],
        "auc": roc_auc(scores, labels),
        "auc_ci95": boot["auc_ci95"],
        "score_gap": score_gap(scores, labels),
        "ece": ece(scores, labels),
        "compute": {
            "total_calls": sum(calls), "total_tokens": sum(tokens),
            "mean_calls": round(sum(calls) / len(calls), 2) if calls else 0,
            "mean_tokens": round(sum(tokens) / len(tokens), 1) if tokens else 0,
        },
        "_correct": correct,
    }
    if out["auc"] is not None:
        out["auc"] = round(out["auc"], 4)
    if out["ece"] is not None:
        out["ece"] = round(out["ece"], 4)
    for kk in ("awarded_mean", "declined_mean", "gap", "p", "effect", "U"):
        if kk in out["score_gap"] and isinstance(out["score_gap"][kk], float):
            out["score_gap"][kk] = round(out["score_gap"][kk], 4)
    if full_correct is not None:
        out["mcnemar_vs_full"] = mcnemar(correct, full_correct)
        out["mcnemar_vs_full"]["p"] = round(out["mcnemar_vs_full"]["p"], 4)
    return out
