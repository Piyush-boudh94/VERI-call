"""Accuracy, precision, recall, F1 and ROC -- at thresholds set on held-out data.

The project reports EER and Cllr everywhere, and the reason is not squeamishness:
EER and AUC are threshold-free, and accuracy, precision, recall and F1 are not.
Quoting an F1 without saying where the threshold came from is quoting a number
somebody chose, and if they chose it on the same data they then measured, they
chose the flattering one.

So: every threshold here is set on the **calib** fold and applied unchanged to
**eval_hindi**. The folds are speaker-disjoint, so this is the honest version of
these metrics and the numbers are lower than the tuned-on-eval version would be.

Reported at four operating points, because "accuracy" alone hides the trade:

  EER      -- where false alarms and misses are equally frequent
  FPR 1%   -- a bank that will tolerate one false alarm in a hundred good calls
  FPR 5%   -- the budget docs/RESULTS.md uses elsewhere
  FPR 10%  -- a triage setting where a human reviews anything flagged

Also per channel, because a single pooled number across clean, G.711 and a
noisy line is an average over conditions a real call is never an average of.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vericall.data.cache import CachedFold
from vericall.fusion.config import FusionConfig, load_weights
from vericall.fusion.calibration import eer, load_calibration


sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_fairness import branch_llr  # noqa: E402  (same fused-score path)


def confusion(score, y, thr):
    """spoof is the positive class: it is the thing being detected."""
    pred = score >= thr
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
    tn = int((~pred & (y == 0)).sum()); fn = int((~pred & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {
        "threshold": float(thr),
        "accuracy": (tp + tn) / max(1, len(y)),
        "precision": prec,
        "recall": rec,
        "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "miss_rate": fn / (fn + tp) if fn + tp else 0.0,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def roc(score, y, n=512):
    """TPR/FPR over the full threshold range, plus AUC by the trapezoid rule."""
    order = np.argsort(-score)
    y = np.asarray(y)[order]
    tp = np.cumsum(y == 1); fp = np.cumsum(y == 0)
    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
    tpr = np.concatenate([[0.0], tp / max(1, npos)])
    fpr = np.concatenate([[0.0], fp / max(1, nneg)])
    auc = float(np.trapezoid(tpr, fpr)) if hasattr(np, "trapezoid") \
        else float(np.trapz(tpr, fpr))
    idx = np.unique(np.linspace(0, len(fpr) - 1, n).astype(int))
    return auc, fpr[idx].tolist(), tpr[idx].tolist()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default="/scratch/vericall-cache-ch")
    ap.add_argument("--dataset", default="hindi_paired")
    ap.add_argument("--threshold-fold", default="calib",
                    help="thresholds are SET here and never measured here")
    ap.add_argument("--eval-fold", default="eval_hindi")
    ap.add_argument("--calib-dir", default="calib_hindi_ch")
    ap.add_argument("--a-ckpt", default="checkpoints/branch_Ahindich_seed0.pt")
    ap.add_argument("--a2-ckpt", default="checkpoints/branch_A2hindich_seed0.pt")
    ap.add_argument("--fpr", type=float, nargs="+", default=[0.01, 0.05, 0.10])
    ap.add_argument("--out", default="docs/eval_metrics_hindi.json")
    ap.add_argument("--plot", default="docs/roc_hindi.png",
                    help="ROC curve; empty string to skip")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cdir = Path(args.calib_dir)
    cal_a = load_calibration(cdir / "A@pooled.json")
    cal_a2 = load_calibration(cdir / "A2@pooled.json")
    w = load_weights(cdir / "weights.json", FusionConfig().w)

    def fused(fold_name):
        fold = CachedFold.load(Path(args.cache) / args.dataset / fold_name)
        a = branch_llr(fold, dev, "A", args.a_ckpt, cal_a)
        a2 = branch_llr(fold, dev, "A2", args.a2_ckpt, cal_a2)
        # There is no `channel` attribute. The name lives in the per-row meta
        # and the integer in `codec`; prefer the name, fall back to the
        # integer. Reading a missing attribute silently collapsed this table
        # into a single "clean" row covering every row, which is how it read
        # before being checked -- and it looked plausible.
        try:
            ch = [m["channel"] for m in fold.meta]
        except (AttributeError, KeyError, TypeError):
            ch = [f"codec_{c}" for c in np.asarray(fold.codec).tolist()]
        return (w["A"] * a + w["A2"] * a2, np.asarray(fold.label),
                np.asarray(ch, dtype=object))

    t_s, t_y, _ = fused(args.threshold_fold)
    e_s, e_y, e_ch = fused(args.eval_fold)
    print(f"thresholds set on {args.threshold_fold} ({len(t_y):,} rows), "
          f"applied to {args.eval_fold} ({len(e_y):,} rows)")
    print(f"fusion w = {dict(w)}  calib = {cdir.name}\n")

    # the operating points, each fixed on the threshold fold
    t_bona = t_s[t_y == 0]
    e_eer, _ = eer(e_s, e_y)
    t_eer, t_thr = eer(t_s, t_y)
    points = {"EER": float(t_thr)}
    for f in args.fpr:
        points[f"FPR{int(f * 100)}%"] = float(np.quantile(t_bona, 1.0 - f))

    auc, fpr_c, tpr_c = roc(e_s, e_y)
    out = {
        "dataset": args.dataset, "eval_fold": args.eval_fold,
        "threshold_fold": args.threshold_fold, "calib_dir": cdir.name,
        "fusion_weights": dict(w),
        "n_eval": int(len(e_y)),
        "n_bona": int((e_y == 0).sum()), "n_spoof": int((e_y == 1).sum()),
        "roc_auc": auc, "eer_eval": float(e_eer), "eer_threshold_fold": float(t_eer),
        "roc_curve": {"fpr": fpr_c, "tpr": tpr_c},
        "operating_points": {}, "per_channel": {},
    }

    print(f"ROC AUC {auc:.4f}   EER {e_eer*100:.2f}% "
          f"(threshold fold {t_eer*100:.2f}%)\n")
    hdr = ("operating point", "thr", "accuracy", "precision", "recall", "F1",
           "FPR", "miss")
    print(f"{hdr[0]:<16}{hdr[1]:>7}{hdr[2]:>10}{hdr[3]:>11}"
          f"{hdr[4]:>8}{hdr[5]:>7}{hdr[6]:>8}{hdr[7]:>8}")
    print("-" * 75)
    for name, thr in points.items():
        m = confusion(e_s, e_y, thr)
        out["operating_points"][name] = m
        print(f"{name:<16}{thr:>7.2f}{m['accuracy']*100:>9.1f}%"
              f"{m['precision']*100:>10.1f}%{m['recall']*100:>7.1f}%"
              f"{m['f1']*100:>6.1f}%{m['fpr']*100:>7.1f}%{m['miss_rate']*100:>7.1f}%")

    # per channel, at the 5% budget -- the one the rest of the docs use
    thr5 = points.get("FPR5%", points["EER"])
    print(f"\nper channel at FPR5% (thr {thr5:.2f}), and its own EER:")
    print(f"{'channel':<14}{'n':>7}{'EER':>8}{'accuracy':>10}"
          f"{'recall':>8}{'F1':>7}{'AUC':>7}")
    print("-" * 61)
    for c in sorted(set(e_ch.tolist())):
        sel = e_ch == c
        if sel.sum() < 20:
            continue
        m = confusion(e_s[sel], e_y[sel], thr5)
        c_eer, _ = eer(e_s[sel], e_y[sel])
        c_auc, _, _ = roc(e_s[sel], e_y[sel], n=32)
        out["per_channel"][c] = {**m, "eer": float(c_eer), "auc": c_auc,
                                 "n": int(sel.sum())}
        print(f"{c:<14}{int(sel.sum()):>7}{c_eer*100:>7.2f}%"
              f"{m['accuracy']*100:>9.1f}%{m['recall']*100:>7.1f}%"
              f"{m['f1']*100:>6.1f}%{c_auc:>7.3f}")

    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"\nwrote {args.out}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5.2, 5.0), dpi=160)
        ax.plot([0, 1], [0, 1], "--", lw=0.8, color="#999",
                label="chance (AUC 0.500)")
        ax.plot(fpr_c, tpr_c, lw=1.8, color="#1f4e79",
                label=f"pooled channels (AUC {auc:.3f})")
        for c, m in out["per_channel"].items():
            sel = e_ch == c
            _, f_, t_ = roc(e_s[sel], e_y[sel])
            ax.plot(f_, t_, lw=1.2, alpha=0.85,
                    label=f"{c} (AUC {m['auc']:.3f})")
        for name in ("FPR1%", "FPR5%", "FPR10%"):
            if name in out["operating_points"]:
                m = out["operating_points"][name]
                ax.plot(m["fpr"], m["recall"], "o", ms=5, color="#c00", zorder=5)
                ax.annotate(name, (m["fpr"], m["recall"]),
                            textcoords="offset points", xytext=(6, -9),
                            fontsize=7, color="#c00")
        ax.set_xlabel("false positive rate  (genuine calls flagged)")
        ax.set_ylabel("true positive rate  (clones caught)")
        ax.set_title(f"VeriCall ROC -- {args.dataset}:{args.eval_fold}\n"
                     f"thresholds set on {args.threshold_fold}", fontsize=9)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.grid(alpha=0.25, lw=0.5)
        ax.legend(loc="lower right", fontsize=7, framealpha=0.95)
        fig.tight_layout()
        fig.savefig(args.plot)
        print(f"wrote {args.plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
