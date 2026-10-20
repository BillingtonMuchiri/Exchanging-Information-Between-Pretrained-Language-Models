# analysis_sensitivity.py - regenerates Section 5.9 (dependence checks), Table 8 baselines and Appendix B tables from the per-run CSVs.
# Usage: python analysis_sensitivity.py --exp1 DIR --exp3 DIR --exp4 DIR --out DIR
#   exp1 DIR must contain results.csv ; exp3 DIR: results3.csv ; exp4 DIR: results4.csv, predictors4.csv, pair_stats4.csv
import argparse, os, json, numpy as np, pandas as pd
from scipy import stats as sps
ap = argparse.ArgumentParser(); ap.add_argument("--exp1", default="exp"); ap.add_argument("--exp3", default="exp3"); ap.add_argument("--exp4", default="exp4"); ap.add_argument("--out", default="analysis_out")
a_ = ap.parse_args(); os.makedirs(a_.out, exist_ok=True)
def boot(clusters, n=2000):
    rs = np.random.RandomState(0); arr = [np.asarray(v) for v in clusters]; k = len(arr)
    m = [np.mean(np.concatenate([arr[i] for i in rs.randint(0, k, k)])) for _ in range(n)]
    return np.mean(np.concatenate(arr)), np.percentile(m, 2.5), np.percentile(m, 97.5)
def kw(M):
    m, n = M.shape; Rk = np.apply_along_axis(lambda r: sps.rankdata(-r), 1, M); S = ((Rk.sum(0) - Rk.sum(0).mean()) ** 2).sum(); return 12 * S / (m ** 2 * (n ** 3 - n))
def contrasts(df, pairs):
    df = df.copy(); df["u"] = df.apply(lambda r: "|".join(sorted([r.src, r.rcv])), axis=1); models = sorted(set(df.src) | set(df.rcv)); rows = []
    for name, (x, y) in pairs.items():
        df["d"] = df[x] - df[y]; o = boot(df.groupby(["src", "rcv"]).d.apply(list).values); u = boot(df.groupby("u").d.apply(list).values)
        loo = [df[(df.src != m) & (df.rcv != m)].d.mean() for m in models]
        rows.append(dict(contrast=name, mean=df.d.mean(), ordered_lo=o[1], ordered_hi=o[2], unordered_lo=u[1], unordered_hi=u[2], loo_min=min(loo), loo_max=max(loo), pairs_ahead=int((df.d > 0).sum()), n_pairs=len(df)))
    return pd.DataFrame(rows)
# ---- Experiment 1
R = pd.read_csv(f"{a_.exp1}/results.csv"); E = ["text_hint", "kd", "bridge_ridge", "bridge_procrustes", "bridge_nonlinear"]
A = R[R.task == "arce"].pivot_table(index=["src", "rcv", "seed"], columns="mech", values="acc").reset_index().groupby(["src", "rcv"]).mean(numeric_only=True).reset_index()
c1 = contrasts(A, {f"text_hint - {m}": ("text_hint", m) for m in ["receiver_alone", "ensemble", "kd", "bridge_ridge", "bridge_procrustes", "bridge_nonlinear"]}); c1.to_csv(f"{a_.out}/sens_exp1.csv", index=False)
allt = R[R.mech.isin(E)].groupby(["src", "rcv", "mech"]).acc.mean().unstack()[E].reset_index(); allt["u"] = allt.apply(lambda r: "|".join(sorted([r.src, r.rcv])), axis=1)
W = {"W_ordered": kw(allt[E].values), "W_unordered": kw(allt.groupby("u")[E].mean().values)}
for m in sorted(set(allt.src) | set(allt.rcv)): W[f"W_drop_{m}"] = kw(allt[(allt.src != m) & (allt.rcv != m)][E].values)
json.dump(W, open(f"{a_.out}/kendall_w.json", "w"), indent=1); print(c1.round(3).to_string()); print(W)
A.merge(R[R.task == "arce"].pivot_table(index=["src", "rcv", "seed"], columns="mech", values="acc").reset_index().groupby(["src", "rcv"]).text_hint.agg(["min", "max"]).reset_index(), on=["src", "rcv"]).to_csv(f"{a_.out}/tableB1_exp1_per_pair.csv", index=False)
# ---- Experiment 3
if os.path.exists(f"{a_.exp3}/results3.csv"):
    R3 = pd.read_csv(f"{a_.exp3}/results3.csv"); T = R3[R3.task == "arce"].pivot_table(index=["src", "rcv", "seed"], columns="arm", values="acc").reset_index().groupby(["src", "rcv"]).mean(numeric_only=True).reset_index()
    contrasts(T, {"proj - proj_shuf": ("proj", "proj_shuf"), "proj - prompt_only": ("proj", "prompt_only"), "proj - text_hint": ("proj", "text_hint"), "text_hint - prompt_only": ("text_hint", "prompt_only")}).to_csv(f"{a_.out}/sens_exp3.csv", index=False)
    T.to_csv(f"{a_.out}/tableB2_exp3_per_pair.csv", index=False)
# ---- Experiment 4: predictor baselines with model-level bootstrap
R4 = pd.read_csv(f"{a_.exp4}/results4.csv"); PR = pd.read_csv(f"{a_.exp4}/predictors4.csv"); PS = pd.read_csv(f"{a_.exp4}/pair_stats4.csv")
A4 = R4[R4.task == "arce"].pivot_table(index=["src", "rcv", "seed"], columns="mech", values="acc").reset_index(); A4["gain"] = A4.text_hint - A4.receiver_alone
D = A4.merge(PR, on=["src", "rcv", "seed"]).groupby(["src", "rcv"]).mean(numeric_only=True).reset_index().merge(PS, on=["src", "rcv"]); y = D.gain.values
sets = {"Mean only (baseline)": [], "Source accuracy alone": ["acc_src_val"], "Receiver accuracy alone": ["acc_rcv_val"], "Source + receiver accuracy": ["acc_src_val", "acc_rcv_val"],
        "Mismatch (1 - CKA)": ["mismatch"], "Mismatch + vocabulary + dimension ratio": ["mismatch", "vocab_jaccard", "log_dim_ratio"], "Accuracy gap": ["gap"], "Headroom": ["headroom"],
        "Gap + headroom": ["gap", "headroom"], "Source + receiver accuracy + headroom (post hoc)": ["acc_src_val", "acc_rcv_val", "headroom"],
        "Gap + headroom + error overlap (post hoc)": ["gap", "headroom", "overlap"], "All six predictors": ["gap", "headroom", "overlap", "mismatch", "vocab_jaccard", "log_dim_ratio"]}
P = {}
for nm, cols in sets.items():
    pr = np.zeros(len(D))
    for i, r in D.iterrows():
        tr = ((D.src != r.src) & (D.rcv != r.src) & (D.src != r.rcv) & (D.rcv != r.rcv)).values
        X = np.c_[np.ones(tr.sum()), D.loc[tr, cols].values] if cols else np.ones((tr.sum(), 1)); b = np.linalg.solve(X.T @ X + 1e-6 * np.eye(X.shape[1]), X.T @ D.loc[tr, "gain"].values)
        pr[i] = (np.r_[1, r[cols].values.astype(float)] if cols else np.ones(1)) @ b
    P[nm] = pr
mods = sorted(set(D.src) | set(D.rcv)); rs = np.random.RandomState(0); Ws = []
for _ in range(2000):
    cnt = pd.Series(rs.choice(mods, len(mods)).tolist()).value_counts(); w = np.array([cnt.get(s, 0) * cnt.get(r, 0) for s, r in zip(D.src, D.rcv)], float)
    if (w > 0).sum() >= 5: Ws.append(w)
def wR2(pr, w): yb = np.average(y, weights=w); return 1 - np.average((y - pr) ** 2, weights=w) / np.average((y - yb) ** 2, weights=w)
def auc(s, l): p, n = s[l], s[~l]; return float(sps.mannwhitneyu(p, n, alternative="greater").statistic / (len(p) * len(n)))
rows = []
for nm, pr in P.items():
    bs = [wR2(pr, w) for w in Ws]; rows.append(dict(set=nm, R2=1 - ((y - pr) ** 2).sum() / ((y - y.mean()) ** 2).sum(), ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5),
                                                 RMSE=np.sqrt(((y - pr) ** 2).mean()), MAE=np.abs(y - pr).mean(), spearman=sps.spearmanr(pr, y)[0], AUC=auc(pr, y > 0.01)))
pd.DataFrame(rows).to_csv(f"{a_.out}/table8_predictors.csv", index=False); print(pd.DataFrame(rows).round(3).to_string())
for a, b in [("Headroom", "Mismatch (1 - CKA)"), ("Gap + headroom", "Accuracy gap"), ("Headroom", "Source + receiver accuracy")]:
    ds = [wR2(P[a], w) - wR2(P[b], w) for w in Ws]; print(a, "-", b, round(wR2(P[a], np.ones(len(y))) - wR2(P[b], np.ones(len(y))), 3), np.percentile(ds, [2.5, 97.5]).round(2))
D.to_csv(f"{a_.out}/tableB3_exp4_per_pair.csv", index=False)
