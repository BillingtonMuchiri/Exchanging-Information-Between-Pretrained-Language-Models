# compare_text_hint.py - compares the text-hint arm before and after the indexing fix.
# Usage: python compare_text_hint.py OLD_RESULTS.csv NEW_RESULTS.csv
#   Experiment 1:  exp/results_old.csv exp/results.csv        (column "mech")
#   Experiment 4:  exp4/results4_old.csv exp4/results4.csv    (column "mech")
#   Experiment 5:  exp5/results5_old.csv exp5/results5.csv    (column "arm")
import sys, numpy as np, pandas as pd
from scipy import stats as sps
old, new = pd.read_csv(sys.argv[1]), pd.read_csv(sys.argv[2])
col = "arm" if "arm" in old.columns else "mech"
def table(R, label):
    t = R.groupby([col, "task"]).acc.mean().unstack().round(4); t.columns = [f"{label}_{c}" for c in t.columns]; return t
both = table(old, "old").join(table(new, "new"), how="outer")
print("\nMean accuracy by arm and task, before (old) and after (new) the fix:\n"); print(both.to_string())
def gains(R, hint, base):
    P = R[R.task == "arce"].pivot_table(index=["src", "rcv", "seed"], columns=col, values="acc")
    G = (P[hint] - P[base]).groupby(["src", "rcv"]).mean(); return G
for hint in [a for a in old[col].unique() if "text_hint" in a]:
    suffix = hint.replace("text_hint", ""); base = "receiver_alone" if "receiver_alone" in old[col].unique() else f"receiver{suffix}"
    go, gn = gains(old, hint, base), gains(new, hint, base)
    print(f"\n{hint} - {base} on ARC-Easy (mean over seeds, per ordered pair)")
    print(f"  old: mean {go.mean():+.4f}, positive in {int((go > 0).sum())} of {len(go)} pairs")
    print(f"  new: mean {gn.mean():+.4f}, positive in {int((gn > 0).sum())} of {len(gn)} pairs")
    print(f"  Spearman(old gain, new gain) over pairs = {sps.spearmanr(go, gn)[0]:.3f}")
    if "source_alone" in old[col].unique():
        P = new[new.task == "arce"].pivot_table(index=["src", "rcv", "seed"], columns=col, values="acc")
        print(f"  new: mean |{hint} - source_alone| = {(P[hint] - P['source_alone']).abs().mean():.4f}  (near 0 means the arm just copies the source)")
# Experiment 1 / 4: is the text hint still the best exchange mechanism in each pair?
ex = [m for m in ["text_hint", "kd", "bridge_ridge", "bridge_procrustes", "bridge_nonlinear"] if m in old[col].unique()]
if col == "mech" and len(ex) == 5:
    for lab, R in [("old", old), ("new", new)]:
        A = R.groupby(["src", "rcv", "mech"]).acc.mean().unstack()[ex]
        print(f"\n{lab}: text hint has the highest mean accuracy (3 tasks) in {int((A.idxmax(1) == 'text_hint').sum())} of {len(A)} pairs")
        m, n = A.shape; Rk = np.apply_along_axis(lambda r: sps.rankdata(-r), 1, A.values); S = ((Rk.sum(0) - Rk.sum(0).mean()) ** 2).sum()
        print(f"{lab}: Kendall's W = {12 * S / (m ** 2 * (n ** 3 - n)):.3f}")
