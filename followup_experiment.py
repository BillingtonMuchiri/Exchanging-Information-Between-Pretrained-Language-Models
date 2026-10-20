# =====================================================================================
# FOLLOW-UP EXPERIMENT 2 - "Can better label-free alignment close the bridge gap?"
# Stand-alone, resumable. Reuses the cached features from kaggle_experiment.py (Experiment 1).
#
# KAGGLE: if Experiment-1 files are found under /kaggle/input the run is CPU-only (~20-40 min). If NOT found, the script
#         rebuilds data + features itself (~35 min extra): then turn Internet ON and use a GPU.
# INPUT : add the OUTPUT of your Experiment-1 notebook version as an Input (it contains /exp/features and
#         /exp/data_full.json). The script finds it automatically. Without features it stops with a message.
# RESUME: re-run the cell; finished (pair, seed, arm) rows in exp2/results2.csv are skipped.
# OUTPUT: /kaggle/working/exp2/results2.csv, report2/ (tables, figures, REPORT2.md), exp2_results.zip
#
# Experiment 1 found: text hint > KD > ridge > Procrustes > nonlinear bridges (bridges 12-25 points below the
# receiver). Hypotheses here (pre-register before running):
#   F1  Ridge-bridge accuracy rises with the amount of UNLABELED paired data (scaling curve).
#   F2  Shared-space methods (CCA, relative representations) and reduced-rank maps beat the plain ridge map.
#   F3  A few labels (20-400) used only to re-fit the receiver's probe on mapped features close most of the
#       remaining gap to the receiver; report the label-efficiency curve.
# =====================================================================================
import os, sys, json, time, glob, shutil, itertools, warnings, zipfile
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy import stats as sps
warnings.filterwarnings("ignore")

DRY_RUN = False                 # True = synthetic features (logic test only)
FRESH_START      = True         # True = ignore ALL earlier files: wipe exp2 results and start from zero
REBUILD_FEATURES = True         # True = also rebuild data + features from the models (needs GPU + Internet, ~35 min)
                                # set False to reuse features found in /kaggle/input or /kaggle/working/exp (CPU-only, ~30 min)
SEEDS   = [0, 1, 2]
MODELS  = ["qwen05", "smol360", "tiny11", "pythia410"]     # keys used by Experiment 1
TASKS   = ["arce", "arcc", "sciq"]
N_TRAIN = 1200                  # labeled items per seed (same as Experiment 1)
N_BOOT  = 2000
L2_GRID, RIDGE_GRID = [1e-4, 1e-3, 1e-2, 1e-1], [1e1, 1e2, 1e3, 1e4]
SCALE_N = [300, 600, 1200, 1600]          # unlabeled items for the scaling curve (pool = 1600 ARC-Easy train items)
RR_RANKS, CCA_K, REL_K, FT_M = [16, 64, 256], [32, 128], [128, 512], [20, 100, 400]

W = "/kaggle/working" if os.path.exists("/kaggle") else os.path.abspath(".")
E1 = f"{W}/exp"; E2 = f"{W}/exp2"; os.makedirs(f"{E2}/report2", exist_ok=True)
FEAT = f"{E1}/features" if not DRY_RUN else f"{E2}/features_dry"
os.makedirs(FEAT, exist_ok=True)

def restore_previous():
    """Find Experiment-1 files anywhere under /kaggle/input (any depth): exp/, exp2/, features/, data_full.json."""
    n = 0
    def cp(s_, d_):
        nonlocal n
        if not os.path.exists(d_): os.makedirs(os.path.dirname(d_), exist_ok=True); shutil.copy2(s_, d_); n += 1
    for root, dirs, files in os.walk("/kaggle/input"):
        base = os.path.basename(root)
        for f in files:
            s_ = os.path.join(root, f)
            if f == "data_full.json": cp(s_, f"{E1}/data_full.json")
            elif base == "features" and f.endswith(".npy"): cp(s_, f"{E1}/features/{f}")
            elif f == "results2.csv" and not FRESH_START: cp(s_, f"{E2}/results2.csv")
            elif f == "results.csv" and "exp2" not in root: cp(s_, f"{E1}/results.csv")
    if n: print(f"[resume] restored {n} files from /kaggle/input")
    else: print("[resume] nothing found under /kaggle/input; will rebuild data/features if needed (needs Internet + GPU)")
if FRESH_START and not DRY_RUN:
    shutil.rmtree(E2, ignore_errors=True); os.makedirs(f"{E2}/report2", exist_ok=True)
    if REBUILD_FEATURES:
        shutil.rmtree(f"{E1}/features", ignore_errors=True); os.makedirs(f"{E1}/features", exist_ok=True)
        if os.path.exists(f"{E1}/data_full.json"): os.remove(f"{E1}/data_full.json")
    print(f"[fresh] results wiped; rebuild features = {REBUILD_FEATURES}")
    if not REBUILD_FEATURES: restore_previous()      # only reuse data/features, never old results
    for f in (f"{E2}/results2.csv",):
        if os.path.exists(f): os.remove(f)
elif not DRY_RUN: restore_previous()

# ------------------------------------ helpers ----------------------------------------
def softmax(z): z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)
def fit_head(X, y=None, soft=None, l2=1e-2, seed=0, iters=150):
    n, _, D = X.shape; X = X.astype(np.float32); T = np.eye(4, dtype=np.float32)[y] if soft is None else soft.astype(np.float32)
    w0 = (np.random.RandomState(seed).randn(D) * 0.01).astype(np.float64)
    def f(w):
        p = softmax((X @ w.astype(np.float32)).astype(np.float64)); loss = -(T * np.log(p + 1e-12)).sum(1).mean() + 0.5 * l2 * (w @ w)
        return loss, (np.einsum("ncd,nc->d", X, (p - T).astype(np.float32)) / n + l2 * w).astype(np.float64)
    return minimize(f, w0, jac=True, method="L-BFGS-B", options={"maxiter": iters}).x.astype(np.float32)
def head_probs(X, w): return softmax((X.astype(np.float32) @ w).astype(np.float64))
def auto_l2(X, y, seed):
    n = len(y); idx = np.random.RandomState(seed).permutation(n); k = int(0.8 * n); tr, va = idx[:k], idx[k:]; best, bl = 1e9, L2_GRID[1]
    for l2 in L2_GRID:
        p = head_probs(X[va], fit_head(X[tr], y[tr], l2=l2, seed=seed)); nll = -np.log(p[np.arange(len(va)), y[va]] + 1e-12).mean()
        if nll < best: best, bl = nll, l2
    return bl
def ece(P, y, bins=15):
    conf, pred = P.max(1), P.argmax(1); acc = (pred == y).astype(float); e = 0.0; edges = np.linspace(0, 1, bins + 1)
    for a, b in zip(edges[:-1], edges[1:]):
        m = (conf > a) & (conf <= b)
        if m.any(): e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return e
def metrics(P, y): return dict(acc=float((P.argmax(1) == y).mean()), ece=float(ece(P, y)), nll=float(-np.log(P[np.arange(len(y)), y] + 1e-12).mean()))
class Std:
    def __init__(self, X): F = X.reshape(-1, X.shape[-1]).astype(np.float32); self.m = F.mean(0); self.s = F.std(0) + 1e-6
    def __call__(self, X): return (X.astype(np.float32) - self.m) / self.s
def ridge_fit(X, Y, lam): return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1], dtype=np.float32), X.T @ Y).astype(np.float32)
def r2(Y, Yh): return float(1 - ((Y - Yh) ** 2).sum() / (((Y - Y.mean(0)) ** 2).sum() + 1e-9))
def best_ridge(Xf, Yf, seed):
    n = len(Xf); idx = np.random.RandomState(seed).permutation(n); k = int(0.8 * n); best, bl = -1e9, RIDGE_GRID[1]
    for lam in RIDGE_GRID:
        sc = r2(Yf[idx[k:]], Xf[idx[k:]] @ ridge_fit(Xf[idx[:k]], Yf[idx[:k]], lam))
        if sc > best: best, bl = sc, lam
    return ridge_fit(Xf, Yf, bl), bl
def pca_basis(X, k): _, _, V = np.linalg.svd(X - X.mean(0), full_matrices=False); return V[:k].T.astype(np.float32)
def flat(X): return X.reshape(-1, X.shape[-1])


# ---------- fallback: rebuild data + base features (identical recipe to Experiment 1) -----------
HF = {"qwen05": "Qwen/Qwen2.5-0.5B", "smol360": "HuggingFaceTB/SmolLM2-360M",
      "tiny11": "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T", "pythia410": "EleutherAI/pythia-410m"}
def build_data():
    from datasets import load_dataset
    rng = np.random.RandomState(0)
    def arc(cfg, split):
        try: ds = load_dataset("allenai/ai2_arc", cfg, split=split)
        except Exception: ds = load_dataset("ai2_arc", cfg, split=split)
        out = []
        for r in ds:
            t, l = r["choices"]["text"], r["choices"]["label"]
            if len(t) == 4 and r["answerKey"] in l: out.append(dict(q=r["question"], c=list(t), y=l.index(r["answerKey"])))
        return out
    def sciq(split):
        try: ds = load_dataset("allenai/sciq", split=split)
        except Exception: ds = load_dataset("sciq", split=split)
        out = []
        for r in ds:
            c = [r["correct_answer"], r["distractor1"], r["distractor2"], r["distractor3"]]; perm = rng.permutation(4)
            out.append(dict(q=r["question"], c=[c[i] for i in perm], y=int(np.where(perm == 0)[0][0])))
        return out
    def sub(items, n): idx = rng.permutation(len(items))[:n]; return [items[i] for i in idx]
    data = {"train": sub(arc("ARC-Easy", "train"), 1600), "arce": sub(arc("ARC-Easy", "test"), 800),
            "arcc": sub(arc("ARC-Challenge", "test"), 800), "sciq": sub(sciq("test"), 800)}
    os.makedirs(E1, exist_ok=True); json.dump(data, open(f"{E1}/data_full.json", "w")); return data

def extract_base(key, data):
    import torch
    from transformers import AutoTokenizer, AutoModel
    dev = "cuda" if torch.cuda.is_available() else "cpu"; print(f"[features] rebuilding {key} on {dev}")
    tok = AutoTokenizer.from_pretrained(HF[key]); tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModel.from_pretrained(HF[key], torch_dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev).eval()
    layer = int(round(0.75 * model.config.num_hidden_layers))
    for sp, items in data.items():
        texts = [f"Question: {it['q']}\nAnswer: {c}" for it in items for c in it["c"]]; order = np.argsort([len(t) for t in texts])
        feats = np.zeros((len(texts), model.config.hidden_size), np.float16)
        with torch.no_grad():
            for i in range(0, len(texts), 48):
                idx = order[i:i + 48]
                enc = tok([texts[j] for j in idx], return_tensors="pt", padding=True, truncation=True, max_length=160).to(dev)
                hs = model(**enc, output_hidden_states=True).hidden_states[layer].float(); m = enc["attention_mask"].unsqueeze(-1).float()
                feats[idx] = ((hs * m).sum(1) / m.sum(1)).cpu().numpy().astype(np.float16)
        np.save(f"{FEAT}/{key}_{sp}_base.npy", feats.reshape(len(items), 4, -1))
    del model; import gc; gc.collect(); torch.cuda.empty_cache() if dev == "cuda" else None

def load_feat(key, split):
    p = f"{FEAT}/{key}_{split}_base.npy"
    if not os.path.exists(p):
        if DRY_RUN: synth(); return np.load(p)
        extract_base(key, get_data())
    return np.load(p)
def synth():
    dims = {"qwen05": 96, "smol360": 80, "tiny11": 128, "pythia410": 64}; data = json.load(open(f"{E1}/data_dry.json"))
    for key, D in dims.items():
        r = np.random.RandomState(abs(hash(key)) % 2**31); Wm = r.randn(24, D) / 5
        for s, items in data.items():
            base = np.zeros((len(items), 4, D), np.float32)
            for i, it in enumerate(items):
                z = np.random.RandomState(i + (7 if s == "train" else 1000 + len(s))).randn(4, 24); z[it["y"]] += 0.8
                base[i] = np.tanh(z @ Wm) + np.random.RandomState(r.randint(1e9)).randn(4, D) * 0.6
            np.save(f"{FEAT}/{key}_{s}_base.npy", base.astype(np.float16))

def get_data():
    p = f"{E1}/data_full.json"
    if DRY_RUN:
        os.makedirs(E1, exist_ok=True); p = f"{E1}/data_dry.json"
        if not os.path.exists(p):
            rng = np.random.RandomState(0); mk = lambda n: [dict(q=f"q{i}", c=list("abcd"), y=int(rng.randint(4))) for i in range(n)]
            json.dump({"train": mk(400), "arce": mk(150), "arcc": mk(150), "sciq": mk(150)}, open(p, "w"))
    if not os.path.exists(p): return build_data()
    return json.load(open(p))

# ----------------------------------- alignment methods --------------------------------
def cca_basis(Xp, Yp, k, eps=1e-2):
    """Regularized CCA on PCA-reduced (n, d) arrays; returns projections (dx,k), (dy,k) giving unit-variance variates."""
    n = len(Xp); Xc, Yc = Xp - Xp.mean(0), Yp - Yp.mean(0)
    Cxx = Xc.T @ Xc / n + eps * np.eye(Xc.shape[1]); Cyy = Yc.T @ Yc / n + eps * np.eye(Yc.shape[1]); Cxy = Xc.T @ Yc / n
    def inv_sqrt(C): w, V = np.linalg.eigh(C); return (V / np.sqrt(np.maximum(w, 1e-8))) @ V.T
    Sx, Sy = inv_sqrt(Cxx), inv_sqrt(Cyy); U, S, Vt = np.linalg.svd(Sx @ Cxy @ Sy, full_matrices=False)
    return (Sx @ U[:, :k]).astype(np.float32), (Sy @ Vt[:k].T).astype(np.float32), S[:k]

# ----------------------------------- experiment ---------------------------------------
def run(data):
    y = {s: np.array([it["y"] for it in data[s]]) for s in data}
    F = {k: {s: load_feat(k, s) for s in data} for k in MODELS}
    path = f"{E2}/results2.csv"; done, rows = set(), []
    if os.path.exists(path): old = pd.read_csv(path); done = set(zip(old.src, old.rcv, old.seed, old.arm)); rows = old.to_dict("records")
    for seed in SEEDS:
        rs = np.random.RandomState(seed); perm = rs.permutation(len(data["train"])); tr = perm[:N_TRAIN]; ytr = y["train"][tr]
        std = {k: Std(F[k]["train"][tr]) for k in MODELS}
        Z = {k: {s: std[k](F[k][s]) for s in data} for k in MODELS}            # standardized features, all splits
        probe = {k: fit_head(Z[k]["train"][tr], ytr, l2=auto_l2(Z[k]["train"][tr], ytr, seed), seed=seed) for k in MODELS}
        for a, b in itertools.permutations(MODELS, 2):
            arms = {}
            def ev(name, fn, params=0):
                if (a, b, seed, name) in done: return
                t0 = time.perf_counter(); P = fn(); fit_s = time.perf_counter() - t0
                for s in TASKS:
                    r = dict(src=a, rcv=b, seed=seed, arm=name, task=s, fit_s=fit_s, params=params); r.update(metrics(P[s], y[s])); rows.append(r)
                done.add((a, b, seed, name))
            ev("source_alone", lambda: {s: head_probs(Z[a][s], probe[a]) for s in TASKS}); ev("receiver_alone", lambda: {s: head_probs(Z[b][s], probe[b]) for s in TASKS})
            pool_order = rs.permutation(len(data["train"]))            # unlabeled item order (pool items only; no test texts)
            def pair_rows(n): idx = pool_order[:n]; return flat(Z[a]["train"][idx]), flat(Z[b]["train"][idx])
            # F1: ridge scaling with unlabeled data
            maps = {}
            for n in SCALE_N:
                Xf, Yf = pair_rows(min(n, len(pool_order))); Wm, lam = best_ridge(Xf, Yf, seed); maps[n] = Wm
                ev(f"ridge_n{n}", (lambda Wm=Wm: {s: head_probs(Z[a][s] @ Wm, probe[b]) for s in TASKS}), Wm.size)
            # F2: reduced-rank ridge, CCA shared space, relative representations (all with n = 1600 unlabeled items)
            Xf, Yf = pair_rows(1600); Wm = maps[1600] if 1600 in maps else best_ridge(Xf, Yf, seed)[0]
            Yh = Xf @ Wm; _, _, Vt = np.linalg.svd(Yh - Yh.mean(0), full_matrices=False)
            for r_ in RR_RANKS:
                Vr = Vt[:min(r_, Vt.shape[0])].T.astype(np.float32); Wr = Wm @ Vr @ Vr.T
                ev(f"rrr_r{r_}", (lambda Wr=Wr: {s: head_probs(Z[a][s] @ Wr, probe[b]) for s in TASKS}), Vr.size)
            kmax = min(256, Xf.shape[1], Yf.shape[1]); Va, Vb = pca_basis(Xf, kmax), pca_basis(Yf, kmax)
            for k in CCA_K:
                if (a, b, seed, f"cca_k{k}") in done: continue
                A, B, _ = cca_basis(Xf @ Va, Yf @ Vb, min(k, kmax))
                trB = Z[b]["train"][tr] @ Vb @ B; hb = fit_head(trB, ytr, l2=auto_l2(trB, ytr, seed), seed=seed)     # head trained in R's canonical space
                ev(f"cca_k{k}", (lambda A=A, hb=hb: {s: head_probs(Z[a][s] @ Va @ A, hb) for s in TASKS}), A.size)
            for K in REL_K:
                if (a, b, seed, f"rel_K{K}") in done: continue
                aidx = pool_order[:K // 4 + 1][:K // 4]            # K rows = K/4 items x 4 choices (anchor texts shared by both models)
                anchA, anchB = flat(Z[a]["train"][aidx]), flat(Z[b]["train"][aidx])
                nrm = lambda X: X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-9)
                relA = lambda X: nrm(X) @ nrm(anchA).T; relB = lambda X: nrm(X) @ nrm(anchB).T
                sA, sB = Std(relA(Z[a]["train"][tr])), Std(relB(Z[b]["train"][tr]))
                trR = sB(relB(Z[b]["train"][tr])); hb = fit_head(trR, ytr, l2=auto_l2(trR, ytr, seed), seed=seed)
                ev(f"rel_K{K}", (lambda hb=hb, sA=sA: {s: head_probs(sA(relA(Z[a][s])), hb) for s in TASKS}), anchA.size)
            # F3: few labels re-fit the receiver probe on the mapped source features (ridge n=1600 map)
            for m in FT_M:
                def fn(m=m):
                    mapped_tr = Z[a]["train"][tr[:m]] @ Wm; w0 = fit_head(mapped_tr, ytr[:m], l2=1.0, seed=seed)
                    return {s: head_probs(Z[a][s] @ Wm, w0) for s in TASKS}
                ev(f"ridge_ft{m}", fn, m)
        pd.DataFrame(rows).to_csv(path + ".tmp", index=False); os.replace(path + ".tmp", path); print(f"[exp2] seed {seed} done")

# ----------------------------------- analysis -----------------------------------------
def cboot(clusters, n_boot=N_BOOT, seed=0):
    rs = np.random.RandomState(seed); arr = [np.asarray(c) for c in clusters]; k = len(arr)
    m = [np.mean(np.concatenate([arr[i] for i in rs.randint(0, k, k)])) for _ in range(n_boot)]
    return float(np.mean(np.concatenate(arr))), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))
def analyse():
    R = pd.read_csv(f"{E2}/results2.csv"); R["pair"] = R.src + ">" + R.rcv; rep = f"{E2}/report2"; L = []
    def save(df, name):
        df.to_csv(f"{rep}/{name}.csv")
        try: t = df.round(4).to_markdown()
        except Exception: t = df.round(4).to_string()
        L.append(f"\n### {name}\n\n{t}")
    arms = list(dict.fromkeys(R.arm)); rows = []
    for ar in arms:
        r = {"arm": ar}
        for s in TASKS:
            g = R[(R.arm == ar) & (R.task == s)].groupby("pair").acc.apply(list); mu, lo, hi = cboot(g.values); r[s] = f"{mu:.3f} [{lo:.3f},{hi:.3f}]"
        rows.append(r)
    save(pd.DataFrame(rows).set_index("arm"), "T4_accuracy_by_arm")
    P = R[R.task == "arce"].pivot_table(index=["pair", "seed"], columns="arm", values="acc"); base, rcv, src = P["ridge_n1200"], P["receiver_alone"], P["source_alone"]
    rows = []
    for ar in arms:
        if ar in ("source_alone", "receiver_alone"): continue
        gap = ((P[ar] - base) / (rcv - base).replace(0, np.nan)).median()
        d = (P[ar] - base).groupby(level=0).apply(list); mu, lo, hi = cboot(d.values)
        rows.append(dict(arm=ar, acc=P[ar].mean(), retention_vs_source=(P[ar] / src).mean(), diff_vs_ridge_n1200=mu, ci_lo=lo, ci_hi=hi, median_gap_closed=gap))
    save(pd.DataFrame(rows).set_index("arm"), "T5_fidelity_vs_ridge_baseline")
    # F1 scaling: Spearman of accuracy vs log n_items
    sc = [(n, P[f"ridge_n{n}"].mean()) for n in SCALE_N if f"ridge_n{n}" in P]
    rho, p = sps.spearmanr([np.log(n) for n, _ in sc], [v for _, v in sc]) if len(sc) > 2 else (np.nan, np.nan)
    L.append(f"\n**F1** ridge accuracy vs unlabeled items: " + ", ".join(f"n={n}: {v:.3f}" for n, v in sc) + f"; Spearman(log n, acc) = {rho:.2f} (p = {p:.3g}); reference receiver_alone = {rcv.mean():.3f}.")
    # F2: best shared-space/low-rank arm vs ridge n=1600
    cand = [a for a in arms if a.startswith(("rrr", "cca", "rel"))]
    if cand:
        best = max(cand, key=lambda a: P[a].mean()); d = (P[best] - P["ridge_n1600"]).groupby(level=0).apply(list); mu, lo, hi = cboot(d.values)
        L.append(f"**F2** best of reduced-rank/CCA/relative-rep = {best} ({P[best].mean():.3f}); difference vs ridge_n1600 = {mu:+.3f} (95% CI {lo:+.3f}, {hi:+.3f}).")
    # F3: label-efficiency
    fts = [(m, P[f"ridge_ft{m}"].mean()) for m in FT_M if f"ridge_ft{m}" in P]
    L.append("**F3** label-efficiency (accuracy of mapped source features with probe re-fit on m labels): " + ", ".join(f"m={m}: {v:.3f}" for m, v in fts) + f"; receiver_alone = {rcv.mean():.3f}, source_alone = {src.mean():.3f}.")
    L.append("\nNote: with few labels, 'ridge_ft' is partly a source-model probe trained on a small labeled set; compare it to source_alone, not only to receiver_alone.")
    # optional comparison with Experiment 1 text hint
    p1 = f"{E1}/results.csv"
    if os.path.exists(p1):
        R1 = pd.read_csv(p1); t = R1[(R1.task == "arce") & (R1.mech == "text_hint")].assign(pair=lambda d: d.src + ">" + d.rcv).groupby(["pair", "seed"]).acc.mean()
        L.append(f"\n**Reference** Experiment-1 text hint (ARC-Easy) = {t.mean():.3f}; best Experiment-2 arm = {P.drop(columns=['source_alone','receiver_alone']).mean().idxmax()} ({P.drop(columns=['source_alone','receiver_alone']).mean().max():.3f}).")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4)); xs = [n for n, _ in sc]; ax[0].plot(xs, [v for _, v in sc], "o-", color="#1d3b5f"); ax[0].axhline(rcv.mean(), ls="--", c="grey", label="receiver alone"); ax[0].set_xscale("log")
    ax[0].set_xlabel("unlabeled items used to fit ridge map"); ax[0].set_ylabel("ARC-Easy accuracy"); ax[0].legend()
    order = [a for a in arms if a not in ("source_alone", "receiver_alone")]; ax[1].barh(order, [P[a].mean() for a in order], color="#4875ab"); ax[1].axvline(rcv.mean(), ls="--", c="grey"); ax[1].set_xlim(0.2, max(P[order].mean().max(), rcv.mean()) + 0.05); ax[1].set_xlabel("ARC-Easy accuracy")
    plt.tight_layout(); plt.savefig(f"{rep}/fig3_scaling_and_arms.png", dpi=200); plt.close()
    open(f"{rep}/REPORT2.md", "w").write("# Experiment 2: stronger label-free bridges\n" + "\n".join(L)); print("\n".join(L))
    with zipfile.ZipFile(f"{W}/exp2_results.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(f"{E2}/results2.csv", "results2.csv")
        for f in glob.glob(f"{rep}/*"): z.write(f, "report2/" + os.path.basename(f))

if __name__ == "__main__" or "ipykernel" in sys.modules:
    t0 = time.time(); data = get_data(); run(data); analyse(); print(f"DONE in {(time.time()-t0)/60:.1f} min -> {E2}")
