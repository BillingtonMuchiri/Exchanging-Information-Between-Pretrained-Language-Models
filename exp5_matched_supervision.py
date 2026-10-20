# =====================================================================================
# EXPERIMENT 5 - MATCHED SUPERVISION: do the channel rankings survive when every channel gets the same labels?
# Stand-alone, resumable, CPU-only if the Experiment-4 features exist (otherwise it rebuilds them: GPU + Internet).
# RESUME: re-run; add the previous output as Input to carry over cached features (/kaggle/input/**/exp5 or exp4).
# FIRST run QUICK_TEST = True, then False.   Time with cached features: about 30 minutes.
#
# Experiments 1-2 compared complete configurations: the text hint and the bridges' receiver probe used gold labels, KD used none.
# Here every channel (text hint, KD, ridge bridge) is run under TWO supervision regimes, with the same 1,200 items:
#   GOLD     : the receiver is trained with gold labels (KD target = 0.5 * gold + 0.5 * source soft prediction)
#   NO-LABEL : the receiver sees no gold label; it learns only from the source's out-of-fold predictions
#              (text hint: head trained to reproduce the source's predicted answer from the hinted prompt; bridge: receiver
#               probe trained on source pseudo-labels; KD: pure soft targets). The source probe itself is always gold-trained.
# Each regime has its own receiver-alone control (gold probe vs pseudo-label probe).
# PRE-REGISTER before running:
#   M1  text hint minus KD is positive under BOTH regimes (the channel ranking does not depend on label access)
#   M2  text hint minus ridge bridge is positive under BOTH regimes
#   M3  the text hint's gain over its same-regime receiver-alone control is positive under the GOLD regime (replication) and
#       smaller or equal under NO-LABEL (without labels the hint cannot be corrected by the receiver)
# Intervals: cluster bootstrap over unordered model pairs and over models (8 models, 56 ordered pairs, 2 seeds).
# =====================================================================================
import os, sys, json, time, glob, shutil, gc, math, warnings, itertools, zipfile
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy import stats as sps
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ----------------------------------- CONFIG ------------------------------------------
QUICK_TEST   = False      # True = tiny run to check the pipeline end-to-end
DRY_RUN      = False      # True = synthetic features, no models/datasets needed (logic test)
SEEDS        = [0, 1]
N_POOL       = 1600       # ARC-Easy train items kept as pool: first N_TRAIN used for training, the rest (400) = validation for predictors
N_TRAIN      = 1200
N_TEST       = 600        # items per test task
MODELS = {                # all ungated on the Hugging Face Hub
    "qwen05": "Qwen/Qwen2.5-0.5B", "qwen15": "Qwen/Qwen2.5-1.5B",
    "smol135": "HuggingFaceTB/SmolLM2-135M", "smol360": "HuggingFaceTB/SmolLM2-360M", "smol17": "HuggingFaceTB/SmolLM2-1.7B",
    "tiny11": "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T", "pythia410": "EleutherAI/pythia-410m", "opt350": "facebook/opt-350m",
}
TASKS        = ["arce", "arcc", "sciq"]
LAYER_FRAC   = 0.75
BATCH        = 48
MAXLEN       = 160
L2_GRID      = [1e-4, 1e-3, 1e-2, 1e-1]
RIDGE_GRID   = [1e1, 1e2, 1e3, 1e4]
N_BOOT       = 2000
WORTH        = 0.01       # a pair is "worth building" if the text hint gains more than 1 point over the receiver
if QUICK_TEST:
    SEEDS, N_POOL, N_TRAIN, N_TEST, N_BOOT = [0], 240, 160, 120, 300
BASE = "/kaggle/working/exp5" if os.path.exists("/kaggle") else os.path.abspath("./exp5")
for d in ["features", "report"]:
    os.makedirs(f"{BASE}/{d}", exist_ok=True)
if DRY_RUN: BASE_FEAT = f"{BASE}/features_dry"; os.makedirs(BASE_FEAT, exist_ok=True)
else:       BASE_FEAT = f"{BASE}/features"

# ---------------------------- resume from earlier sessions ---------------------------
def restore_previous():
    n = 0
    for src in glob.glob("/kaggle/input/**/exp4", recursive=True) + glob.glob("/kaggle/input/**/exp5", recursive=True):
        for root, _, files in os.walk(src):
            for f in files:
                s = os.path.join(root, f); d = os.path.join(BASE, os.path.relpath(s, src))
                if not os.path.exists(d):
                    os.makedirs(os.path.dirname(d), exist_ok=True); shutil.copy2(s, d); n += 1
    if n: print(f"[resume] restored {n} files from a previous session")
restore_previous()

# ----------------------------------- DATA --------------------------------------------
def build_data():
    path = f"{BASE}/data_{'quick' if QUICK_TEST else 'full'}.json"
    if os.path.exists(path): return json.load(open(path))
    rng = np.random.RandomState(0)
    if DRY_RUN:
        mk = lambda n: [dict(q=f"q{i}", c=[f"a{j}" for j in range(4)], y=int(rng.randint(4))) for i in range(n)]
        data = {"train": mk(N_POOL), "arce": mk(N_TEST), "arcc": mk(N_TEST), "sciq": mk(N_TEST)}
    else:
        from datasets import load_dataset
        def arc(cfg, split):
            try: ds = load_dataset("allenai/ai2_arc", cfg, split=split)
            except Exception: ds = load_dataset("ai2_arc", cfg, split=split)
            out = []
            for r in ds:
                t, l = r["choices"]["text"], r["choices"]["label"]
                if len(t) == 4 and r["answerKey"] in l:
                    out.append(dict(q=r["question"], c=list(t), y=l.index(r["answerKey"])))
            return out
        def sciq(split):
            try: ds = load_dataset("allenai/sciq", split=split)
            except Exception: ds = load_dataset("sciq", split=split)
            out = []
            for r in ds:
                c = [r["correct_answer"], r["distractor1"], r["distractor2"], r["distractor3"]]
                perm = rng.permutation(4); out.append(dict(q=r["question"], c=[c[i] for i in perm], y=int(np.where(perm == 0)[0][0])))
            return out
        def sub(items, n): 
            idx = rng.permutation(len(items))[:n]; return [items[i] for i in idx]
        data = {"train": sub(arc("ARC-Easy", "train"), N_POOL), "arce": sub(arc("ARC-Easy", "test"), N_TEST),
                "arcc": sub(arc("ARC-Challenge", "test"), N_TEST), "sciq": sub(sciq("test"), N_TEST)}
    json.dump(data, open(path, "w")); return data

# ------------------------------ FEATURE EXTRACTION -----------------------------------
def prompts(items, hint):
    out = []
    for it in items:
        for h in (range(4) if hint else [None]):
            for c in it["c"]:
                if hint: out.append(f"Question: {it['q']}\nA peer model suggests: {it['c'][h]}\nAnswer: {c}")
                else:    out.append(f"Question: {it['q']}\nAnswer: {c}")
    return out

def extract_model(key, data):
    """Mean-pooled hidden states (layer = LAYER_FRAC*depth). Saved as float16 .npy, skipped if present."""
    need = [(s, h) for s in data for h in (False, True) if not os.path.exists(f"{BASE_FEAT}/{key}_{s}_{'hint' if h else 'base'}.npy")]
    if not need: return
    if DRY_RUN: return
    import torch
    from transformers import AutoTokenizer, AutoModel
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[features] loading {MODELS[key]} on {dev}")
    tok = AutoTokenizer.from_pretrained(MODELS[key])
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModel.from_pretrained(MODELS[key], torch_dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev).eval()
    layer = int(round(LAYER_FRAC * model.config.num_hidden_layers))
    for s, h in need:
        t0 = time.time(); texts = prompts(data[s], h); order = np.argsort([len(t) for t in texts]); D = model.config.hidden_size
        feats = np.zeros((len(texts), D), np.float16)
        with torch.no_grad():
            for i in range(0, len(texts), BATCH):
                idx = order[i:i + BATCH]
                enc = tok([texts[j] for j in idx], return_tensors="pt", padding=True, truncation=True, max_length=MAXLEN).to(dev)
                hs = model(**enc, output_hidden_states=True).hidden_states[layer].float()
                m = enc["attention_mask"].unsqueeze(-1).float()
                feats[idx] = ((hs * m).sum(1) / m.sum(1)).cpu().numpy().astype(np.float16)
        n = len(data[s]); arr = feats.reshape(n, 4, 4, D) if h else feats.reshape(n, 4, D)
        np.save(f"{BASE_FEAT}/{key}_{s}_{'hint' if h else 'base'}.npy", arr)
        print(f"[features] {key} {s} {'hint' if h else 'base'} {arr.shape} {time.time()-t0:.0f}s")
    del model; gc.collect()
    if dev == "cuda": torch.cuda.empty_cache()

def load_feat(key, split, hint=False, data=None):
    p = f"{BASE_FEAT}/{key}_{split}_{'hint' if hint else 'base'}.npy"
    if DRY_RUN and not os.path.exists(p): synth_features(key, data)
    return np.load(p)

def synth_features(key, data):
    """Dry-run only: shared latent factors + model-specific nonlinear mixing, label-informative."""
    D = 64 + 16 * (abs(hash(key)) % 5); NL = 0.25 + 0.15 * (abs(hash(key)) % 5); SIG = 0.08 + 0.1 * (abs(hash(key) // 7) % 6); r = np.random.RandomState(abs(hash(key)) % 2**31); Wm = r.randn(24, D) / 5
    for s, items in data.items():
        rs = np.random.RandomState(abs(hash(s)) % 2**31)
        base = np.zeros((len(items), 4, D), np.float32); hint = np.zeros((len(items), 4, 4, D), np.float32)
        for i, it in enumerate(items):
            z0 = np.random.RandomState(i + (7 if s == "train" else 99999 + hash(s) % 1000)).randn(4, 24)
            z0[it["y"]] += SIG * (1.5 if s != "sciq" else 1.2)
            nz = rs.randn(4, D) * NL
            base[i] = np.tanh(z0 @ Wm) + nz
            for h in range(4): hint[i, :, h] = base[i] + (0.6 if h == it["y"] else 0.0) * r.randn(D) * 0.0 + 0.3 * (h == it["y"])
        np.save(f"{BASE_FEAT}/{key}_{s}_base.npy", base.astype(np.float16)); np.save(f"{BASE_FEAT}/{key}_{s}_hint.npy", hint.astype(np.float16))

# ---------------------------- numerics: heads, bridges, metrics ----------------------
def softmax(z): z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)

def fit_head(X, y=None, soft=None, l2=1e-2, seed=0, iters=150):
    """Softmax-over-choices linear scorer; X (n,4,D) standardized. y: hard labels or soft: (n,4) targets."""
    n, _, D = X.shape; X = X.astype(np.float32)
    T = np.eye(4, dtype=np.float32)[y] if soft is None else soft.astype(np.float32)
    w0 = (np.random.RandomState(seed).randn(D) * 0.01).astype(np.float64)
    def f(w):
        z = X @ w.astype(np.float32); p = softmax(z.astype(np.float64))
        loss = -(T * np.log(p + 1e-12)).sum(1).mean() + 0.5 * l2 * (w @ w)
        g = np.einsum("ncd,nc->d", X, (p - T).astype(np.float32)) / n + l2 * w
        return loss, g.astype(np.float64)
    return minimize(f, w0, jac=True, method="L-BFGS-B", options={"maxiter": iters}).x.astype(np.float32)

def head_probs(X, w): return softmax((X.astype(np.float32) @ w).astype(np.float64))

def auto_l2(X, y, seed):
    n = len(y); idx = np.random.RandomState(seed).permutation(n); k = int(0.8 * n); tr, va = idx[:k], idx[k:]
    best, bl = 1e9, L2_GRID[1]
    for l2 in L2_GRID:
        w = fit_head(X[tr], y[tr], l2=l2, seed=seed); p = head_probs(X[va], w)
        nll = -np.log(p[np.arange(len(va)), y[va]] + 1e-12).mean()
        if nll < best: best, bl = nll, l2
    return bl

def oof_probs(X, y, l2, seed, k=5):
    n = len(y); folds = np.array_split(np.random.RandomState(seed + 17).permutation(n), k); P = np.zeros((n, 4))
    for f in folds:
        tr = np.setdiff1d(np.arange(n), f); w = fit_head(X[tr], y[tr], l2=l2, seed=seed); P[f] = head_probs(X[f], w)
    return P

def ece(P, y, bins=15):
    conf, pred = P.max(1), P.argmax(1); acc = (pred == y).astype(float); e = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for a, b in zip(edges[:-1], edges[1:]):
        m = (conf > a) & (conf <= b)
        if m.any(): e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return e

def metrics(P, y): 
    return dict(acc=float((P.argmax(1) == y).mean()), ece=float(ece(P, y)), nll=float(-np.log(P[np.arange(len(y)), y] + 1e-12).mean()))

class Std:
    def __init__(self, X): 
        F = X.reshape(-1, X.shape[-1]).astype(np.float32); self.m = F.mean(0); self.s = F.std(0) + 1e-6
    def __call__(self, X): return ((X.astype(np.float32) - self.m) / self.s)

def ridge_fit(Xs, Yt, lam):
    A = Xs.T @ Xs + lam * np.eye(Xs.shape[1], dtype=np.float32); return np.linalg.solve(A, Xs.T @ Yt).astype(np.float32)

def r2(Y, Yh): return float(1 - ((Y - Yh) ** 2).sum() / (((Y - Y.mean(0)) ** 2).sum() + 1e-9))

class Bridge:
    """Maps standardized source features (.., Ds) -> standardized receiver feature space (.., Dr)."""
    def __init__(self, kind, Xs, Yt, seed):
        self.kind = kind; Xf, Yf = Xs.reshape(-1, Xs.shape[-1]), Yt.reshape(-1, Yt.shape[-1]); n = len(Xf)
        t0 = time.perf_counter(); rs = np.random.RandomState(seed)
        if kind == "ridge":
            idx = rs.permutation(n); k = int(0.8 * n); best, bl = -1e9, RIDGE_GRID[1]
            for lam in RIDGE_GRID:
                W = ridge_fit(Xf[idx[:k]], Yf[idx[:k]], lam); sc = r2(Yf[idx[k:]], Xf[idx[k:]] @ W)
                if sc > best: best, bl = sc, lam
            self.W = ridge_fit(Xf, Yf, bl); self.params = self.W.size
        elif kind == "procrustes":
            k = min(Xf.shape[1], Yf.shape[1], 256)
            _, _, Vs = np.linalg.svd(Xf - Xf.mean(0), full_matrices=False); _, _, Vr = np.linalg.svd(Yf - Yf.mean(0), full_matrices=False)
            self.Vs, self.Vr = Vs[:k].T.astype(np.float32), Vr[:k].T.astype(np.float32)
            Xk, Yk = Xf @ self.Vs, Yf @ self.Vr; U, S, Vt = np.linalg.svd(Xk.T @ Yk, full_matrices=False)
            self.R = (U @ Vt).astype(np.float32); self.scale = float(S.sum() / ((Xk ** 2).sum() + 1e-9)); self.params = k * k
        elif kind == "nonlinear":      # random-ReLU-feature ridge on top-256 PCA of the source (closed form)
            k = min(Xf.shape[1], 256); _, _, Vs = np.linalg.svd(Xf - Xf.mean(0), full_matrices=False); self.Vs = Vs[:k].T.astype(np.float32)
            Z = Xf @ self.Vs; self.Z_s = Z.std() + 1e-6; self.Wr = (rs.randn(k, 1024) / np.sqrt(k)).astype(np.float32)
            H = np.maximum(Z / self.Z_s @ self.Wr, 0); idx = rs.permutation(n); kk = int(0.8 * n); best, bl = -1e9, RIDGE_GRID[1]
            for lam in RIDGE_GRID:
                W = ridge_fit(H[idx[:kk]], Yf[idx[:kk]], lam); sc = r2(Yf[idx[kk:]], H[idx[kk:]] @ W)
                if sc > best: best, bl = sc, lam
            self.W = ridge_fit(H, Yf, bl); self.params = self.W.size
        self.fit_s = time.perf_counter() - t0
    def __call__(self, X):
        sh = X.shape; Xf = X.reshape(-1, sh[-1]).astype(np.float32)
        if self.kind == "ridge": Y = Xf @ self.W
        elif self.kind == "procrustes": Y = self.scale * (Xf @ self.Vs @ self.R) @ self.Vr.T
        else: Y = np.maximum((Xf @ self.Vs) / self.Z_s @ self.Wr, 0) @ self.W
        return Y.reshape(*sh[:-1], -1)

def linear_cka(X, Y):
    X = X - X.mean(0); Y = Y - Y.mean(0)
    return float(np.linalg.norm(Y.T @ X) ** 2 / (np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y) + 1e-12))

_VC = {}
def vocab_jaccard(a, b):
    from transformers import AutoTokenizer
    def norm(t):
        if t not in _VC: _VC[t] = {k.replace("Ġ", "").replace("▁", "").replace("Ċ", "\n").lower() for k in AutoTokenizer.from_pretrained(t).get_vocab()} - {""}
        return _VC[t]
    A, B = norm(a), norm(b); return len(A & B) / len(A | B)

# ----------------------------------- PIPELINE (Experiment 5) --------------------------
import itertools
ARMS = ["receiver_gold", "text_hint_gold", "kd_gold", "bridge_gold", "receiver_nolabel", "text_hint_nolabel", "kd_nolabel", "bridge_nolabel"]
def run_experiments(data):
    keys = list(MODELS); y = {s: np.array([it["y"] for it in data[s]]) for s in data}
    F = {k: {s: load_feat(k, s, False, data) for s in data} for k in keys}; FH = {k: {s: load_feat(k, s, True, data) for s in data} for k in keys}
    rp = f"{BASE}/results5.csv"; rows = pd.read_csv(rp).to_dict("records") if os.path.exists(rp) else []; done = {(r["src"], r["rcv"], r["seed"]) for r in rows}
    t0 = time.time()
    for seed in SEEDS:
        perm = np.random.RandomState(seed).permutation(len(data["train"])); tr = perm[:N_TRAIN]; ytr = y["train"][tr]
        std = {k: Std(F[k]["train"][tr]) for k in keys}; Xtr = {k: std[k](F[k]["train"][tr]) for k in keys}; Xte = {k: {s: std[k](F[k][s]) for s in TASKS} for k in keys}
        l2s, probe, oof = {}, {}, {}
        for k in keys: l2s[k] = auto_l2(Xtr[k], ytr, seed); probe[k] = fit_head(Xtr[k], ytr, l2=l2s[k], seed=seed); oof[k] = oof_probs(Xtr[k], ytr, l2s[k], seed)
        pte = {k: {s: head_probs(Xte[k][s], probe[k]) for s in TASKS} for k in keys}
        for a, b in itertools.permutations(keys, 2):
            if (a, b, seed) in done: continue
            hy = oof[a].argmax(1); out = {}
            # GOLD regime
            out["receiver_gold"] = pte[b]
            Hraw = FH[b]["train"][tr][np.arange(len(tr)), :, hy, :]; sh = Std(Hraw); Htr = sh(Hraw); l2h = auto_l2(Htr, ytr, seed); w = fit_head(Htr, ytr, l2=l2h, seed=seed)
            out["text_hint_gold"] = {s: head_probs(sh(FH[b][s][np.arange(len(Xte[a][s])), :, pte[a][s].argmax(1), :]), w) for s in TASKS}
            mix = 0.5 * np.eye(4)[ytr] + 0.5 * oof[a]; w = fit_head(Xtr[b], soft=mix, l2=l2s[b], seed=seed); out["kd_gold"] = {s: head_probs(Xte[b][s], w) for s in TASKS}
            br = Bridge("ridge", Xtr[a], Xtr[b], seed); out["bridge_gold"] = {s: head_probs(br(Xte[a][s]), probe[b]) for s in TASKS}
            # NO-LABEL regime: only the source's out-of-fold predictions are available to the receiver
            l2p = auto_l2(Xtr[b], hy, seed); wp = fit_head(Xtr[b], hy, l2=l2p, seed=seed); out["receiver_nolabel"] = {s: head_probs(Xte[b][s], wp) for s in TASKS}
            l2hp = auto_l2(Htr, hy, seed); w = fit_head(Htr, hy, l2=l2hp, seed=seed)
            out["text_hint_nolabel"] = {s: head_probs(sh(FH[b][s][np.arange(len(Xte[a][s])), :, pte[a][s].argmax(1), :]), w) for s in TASKS}
            w = fit_head(Xtr[b], soft=oof[a], l2=l2s[b], seed=seed); out["kd_nolabel"] = {s: head_probs(Xte[b][s], w) for s in TASKS}
            out["bridge_nolabel"] = {s: head_probs(br(Xte[a][s]), wp) for s in TASKS}
            for m, per in out.items():
                for s in TASKS: r = dict(src=a, rcv=b, seed=seed, arm=m, task=s); r.update(metrics(per[s], y[s])); rows.append(r)
            if len(rows) % 700 < 25: pd.DataFrame(rows).to_csv(rp + ".tmp", index=False); os.replace(rp + ".tmp", rp)
        pd.DataFrame(rows).to_csv(rp + ".tmp", index=False); os.replace(rp + ".tmp", rp); print(f"[exp5] seed {seed} done ({time.time()-t0:.0f}s)")

# ----------------------------------- ANALYSIS (Experiment 5) --------------------------
def boot(clusters, n=N_BOOT, seed=0):
    rs = np.random.RandomState(seed); arr = [np.asarray(c) for c in clusters]; k = len(arr)
    m = [np.mean(np.concatenate([arr[i] for i in rs.randint(0, k, k)])) for _ in range(n)]
    return float(np.mean(np.concatenate(arr))), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))
def model_boot(D, col, n=N_BOOT, seed=0):
    rs = np.random.RandomState(seed); mods = sorted(set(D.src) | set(D.rcv)); v = D[col].values; out = []
    for _ in range(n):
        cnt = pd.Series(rs.choice(mods, len(mods)).tolist()).value_counts(); w = np.array([cnt.get(s, 0) * cnt.get(r, 0) for s, r in zip(D.src, D.rcv)], float)
        if w.sum() > 0: out.append(np.average(v, weights=w))
    return float(np.average(v)), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))
def analyse():
    import zipfile, glob
    R = pd.read_csv(f"{BASE}/results5.csv"); rep = f"{BASE}/report"; L = []
    def save(df, name):
        df.to_csv(f"{rep}/{name}.csv")
        try: t = df.round(4).to_markdown()
        except Exception: t = df.round(4).to_string()
        L.append(f"\n### {name}\n\n{t}")
    A = R.pivot_table(index=["src", "rcv", "seed", "task"], columns="arm", values="acc").reset_index()
    save(A.groupby("task")[ARMS].mean().T, "T13_mean_accuracy_by_arm_and_task")
    P = A[A.task == "arce"].groupby(["src", "rcv"]).mean(numeric_only=True).reset_index(); P["u"] = P.apply(lambda r: "|".join(sorted([r.src, r.rcv])), axis=1)
    for reg in ["gold", "nolabel"]:
        for ch in ["text_hint", "kd", "bridge"]: P[f"gain_{ch}_{reg}"] = P[f"{ch}_{reg}"] - P[f"receiver_{reg}"]
        P[f"text_minus_kd_{reg}"] = P[f"text_hint_{reg}"] - P[f"kd_{reg}"]; P[f"text_minus_bridge_{reg}"] = P[f"text_hint_{reg}"] - P[f"bridge_{reg}"]
    rows = []
    for col in [f"{c}_{r}" for r in ["gold", "nolabel"] for c in ["gain_text_hint", "gain_kd", "gain_bridge", "text_minus_kd", "text_minus_bridge"]]:
        col = col.replace("gain_text_hint", "gain_text_hint"); o = boot(P.groupby(["src", "rcv"])[col].apply(list).values); u = boot(P.groupby("u")[col].apply(list).values); m = model_boot(P, col)
        rows.append(dict(contrast=col, mean=o[0], ordered_lo=o[1], ordered_hi=o[2], unordered_lo=u[1], unordered_hi=u[2], model_lo=m[1], model_hi=m[2], pairs_positive=int((P[col] > 0).sum()), n_pairs=len(P)))
    save(pd.DataFrame(rows).set_index("contrast"), "T14_contrasts_M1_M2_M3")
    Q = A[A.task == "arce"].groupby(["src", "rcv"]).mean(numeric_only=True).reset_index()
    rk = {reg: Q[[f"text_hint_{reg}", f"kd_{reg}", f"bridge_{reg}"]].mean().rank(ascending=False).to_dict() for reg in ["gold", "nolabel"]}
    L.append(f"\nMean-accuracy ranking of the three channels (1 = best): gold {rk['gold']}; no-label {rk['nolabel']}.")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4)); w = 0.35; names = ["text_hint", "kd", "bridge"]; x = np.arange(3)
    for i, reg in enumerate(["gold", "nolabel"]): ax.bar(x + i * w, [Q[f"{n}_{reg}"].mean() for n in names], w, label=reg)
    ax.set_xticks(x + w / 2); ax.set_xticklabels(names); ax.set_ylabel("ARC-Easy accuracy"); ax.legend(); plt.tight_layout(); plt.savefig(f"{rep}/fig6_matched_supervision.png", dpi=200); plt.close()
    open(f"{rep}/REPORT5.md", "w").write("# Experiment 5: matched supervision\n" + "\n".join(L)); print("\n".join(L))
    with zipfile.ZipFile(f"{os.path.dirname(BASE)}/exp5_results.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(f"{BASE}/results5.csv", "results5.csv")
        for f in glob.glob(f"{rep}/*"): z.write(f, "report/" + os.path.basename(f))

if __name__ == "__main__" or "ipykernel" in sys.modules:
    t0 = time.time(); data = build_data(); print({k: len(v) for k, v in data.items()})
    for k in MODELS: extract_model(k, data)
    run_experiments(data); analyse(); print(f"ALL DONE in {(time.time()-t0)/60:.1f} min. Results in {BASE}")
