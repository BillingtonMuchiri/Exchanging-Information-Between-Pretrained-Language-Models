# =====================================================================================
# EXPERIMENT 4 - Can we PREDICT, before building anything, when exchanging information between two models helps?
# Stand-alone, resumable Kaggle script. 8 open models -> 56 ordered pairs (unseen-model generalization).
# KAGGLE SETTINGS: GPU (T4 x2 / P100), Internet ON. Time: ~2.5-3.5 h (features, once) + ~30 min (experiments).
# RESUME: re-run the cell (same session) or add the previous version's output as Input (auto-restores /kaggle/input/**/exp4).
# FIRST run QUICK_TEST = True (a few minutes) to check everything works, then set it to False.
#
# IDEA. Experiments 1-3 found that (i) the text channel beats latent channels and (ii) the benefit of a text hint tracks the
# source model's accuracy, not representational mismatch. But gap and gain share the receiver's accuracy term and were measured
# on the same test items. Here every PREDICTOR is computed on 400 held-out validation items that the outcome never touches:
#   gap       = acc(S) - acc(R)                      (skill gap)
#   headroom  = P(S correct and R wrong)             (what S could add to R; oracle-union gain)
#   overlap   = P(both wrong) / P(either wrong)      (error overlap)
#   mismatch  = 1 - linear CKA ; vocab Jaccard ; log dim ratio   (representational predictors)
# OUTCOME = text-hint gain over the receiver on the test tasks. Prediction uses a STRICT leave-two-models-out scheme:
# pair (a, b) is predicted from a regression trained ONLY on pairs that involve neither a nor b (unseen-model generalization).
# PRE-REGISTER before running:
#   P0  mismatch / vocabulary predictors do not predict gain (replication of Experiments 1 and 3)
#   P1  headroom-based prediction beats mismatch-based prediction (LOMO R2 difference > 0, 95% CI excludes 0)
#   P2  gap + headroom beats gap alone (95% CI of LOMO R2 difference excludes 0)
#   P3  gap + headroom classifies "worth building" pairs (text-hint gain > +1 point) with AUC >= 0.80
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
BASE = "/kaggle/working/exp4" if os.path.exists("/kaggle") else os.path.abspath("./exp4")
for d in ["features", "report"]:
    os.makedirs(f"{BASE}/{d}", exist_ok=True)
if DRY_RUN: BASE_FEAT = f"{BASE}/features_dry"; os.makedirs(BASE_FEAT, exist_ok=True)
else:       BASE_FEAT = f"{BASE}/features"

# ---------------------------- resume from earlier sessions ---------------------------
def restore_previous():
    n = 0
    for src in glob.glob("/kaggle/input/**/exp4", recursive=True):
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
            for h in range(4): hint[i, h, :] = base[i] + (0.6 if h == it["y"] else 0.0) * r.randn(D) * 0.0 + 0.3 * (h == it["y"])
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

# ----------------------------------- PIPELINE (Experiment 4) --------------------------
import itertools
def run_experiments(data):
    keys = list(MODELS); y = {s: np.array([it["y"] for it in data[s]]) for s in data}
    F = {k: {s: load_feat(k, s, False, data) for s in data} for k in keys}
    FH = {k: {s: load_feat(k, s, True, data) for s in data} for k in keys}
    rp, pp = f"{BASE}/results4.csv", f"{BASE}/predictors4.csv"
    rows = pd.read_csv(rp).to_dict("records") if os.path.exists(rp) else []; prow = pd.read_csv(pp).to_dict("records") if os.path.exists(pp) else []
    done = {(r["src"], r["rcv"], r["seed"]) for r in prow}
    sp = f"{BASE}/pair_stats4.csv"
    if not os.path.exists(sp):
        ps = []; rs = np.random.RandomState(0)
        for a, b in itertools.permutations(keys, 2):
            sel = rs.permutation(len(F[a]["train"]) * 4)[:3000]
            xa = F[a]["train"].reshape(-1, F[a]["train"].shape[-1])[sel].astype(np.float32); xb = F[b]["train"].reshape(-1, F[b]["train"].shape[-1])[sel].astype(np.float32)
            try: vj = 0.0 if DRY_RUN else vocab_jaccard(MODELS[a], MODELS[b])
            except Exception: vj = float("nan")
            ck = linear_cka(xa, xb); ps.append(dict(src=a, rcv=b, cka=ck, mismatch=1 - ck, vocab_jaccard=vj, log_dim_ratio=float(np.log(xb.shape[1] / xa.shape[1]))))
        pd.DataFrame(ps).to_csv(sp, index=False)
    t_start = time.time()
    for seed in SEEDS:
        perm = np.random.RandomState(seed).permutation(len(data["train"])); tr, va = perm[:N_TRAIN], perm[N_TRAIN:]; ytr, yva = y["train"][tr], y["train"][va]
        std = {k: Std(F[k]["train"][tr]) for k in keys}
        Xtr = {k: std[k](F[k]["train"][tr]) for k in keys}; Xva = {k: std[k](F[k]["train"][va]) for k in keys}; Xte = {k: {s: std[k](F[k][s]) for s in TASKS} for k in keys}
        l2s, probe, oof, pva, pte = {}, {}, {}, {}, {}
        for k in keys:
            l2s[k] = auto_l2(Xtr[k], ytr, seed); probe[k] = fit_head(Xtr[k], ytr, l2=l2s[k], seed=seed); oof[k] = oof_probs(Xtr[k], ytr, l2s[k], seed)
            pva[k] = head_probs(Xva[k], probe[k]); pte[k] = {s: head_probs(Xte[k][s], probe[k]) for s in TASKS}
        corr = {k: (pva[k].argmax(1) == yva) for k in keys}
        for a, b in itertools.permutations(keys, 2):
            if (a, b, seed) in done: continue
            cs, cr = corr[a], corr[b]; either_wrong = (~cs) | (~cr)
            prow.append(dict(src=a, rcv=b, seed=seed, acc_src_val=cs.mean(), acc_rcv_val=cr.mean(), gap=cs.mean() - cr.mean(), headroom=float((cs & ~cr).mean()),
                             overlap=float((~cs & ~cr).sum() / max(either_wrong.sum(), 1)), n_val=len(yva)))
            out = {"source_alone": pte[a], "receiver_alone": pte[b], "ensemble": {s: (pte[a][s] + pte[b][s]) / 2 for s in TASKS}}
            w = fit_head(Xtr[b], soft=oof[a], l2=l2s[b], seed=seed); out["kd"] = {s: head_probs(Xte[b][s], w) for s in TASKS}
            hy = oof[a].argmax(1); Htr_raw = FH[b]["train"][tr][np.arange(len(tr)), hy, :, :]; sh = Std(Htr_raw); Htr = sh(Htr_raw)
            w = fit_head(Htr, ytr, l2=auto_l2(Htr, ytr, seed), seed=seed); out["text_hint"] = {}
            for s in TASKS:
                hh = pte[a][s].argmax(1); out["text_hint"][s] = head_probs(sh(FH[b][s][np.arange(len(hh)), hh, :, :]), w)
            br = Bridge("ridge", Xtr[a], Xtr[b], seed); out["bridge_ridge"] = {s: head_probs(br(Xte[a][s]), probe[b]) for s in TASKS}
            for m, per in out.items():
                for s in TASKS: r = dict(src=a, rcv=b, seed=seed, mech=m, task=s); r.update(metrics(per[s], y[s])); rows.append(r)
            if len(prow) % 14 == 0:
                pd.DataFrame(rows).to_csv(rp + ".tmp", index=False); os.replace(rp + ".tmp", rp); pd.DataFrame(prow).to_csv(pp + ".tmp", index=False); os.replace(pp + ".tmp", pp)
        pd.DataFrame(rows).to_csv(rp + ".tmp", index=False); os.replace(rp + ".tmp", rp); pd.DataFrame(prow).to_csv(pp + ".tmp", index=False); os.replace(pp + ".tmp", pp)
        print(f"[experiments] seed {seed} done ({time.time()-t_start:.0f}s)")

# ----------------------------------- ANALYSIS (Experiment 4) --------------------------
def auc(score, label):
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    return float(sps.mannwhitneyu(pos, neg, alternative="greater").statistic / (len(pos) * len(neg)))
def analyse():
    import zipfile, glob
    R = pd.read_csv(f"{BASE}/results4.csv"); PR = pd.read_csv(f"{BASE}/predictors4.csv"); PS = pd.read_csv(f"{BASE}/pair_stats4.csv"); rep = f"{BASE}/report"; L = []
    def save(df, name, idx=True):
        df.to_csv(f"{rep}/{name}.csv", index=idx)
        try: t = df.round(4).to_markdown(index=idx)
        except Exception: t = df.round(4).to_string(index=idx)
        L.append(f"\n### {name}\n\n{t}")
    A = R.pivot_table(index=["src", "rcv", "seed", "task"], columns="mech", values="acc").reset_index()
    A["gain_text"] = A.text_hint - A.receiver_alone; A["gain_ens"] = A.ensemble - A.receiver_alone; A["kd_ret"] = A.kd / A.source_alone; A["ridge_ret"] = A.bridge_ridge / A.source_alone
    G = A.groupby(["src", "rcv", "seed"]).agg(gain_text_mean=("gain_text", "mean"), gain_ens_mean=("gain_ens", "mean")).reset_index()
    G = G.merge(A[A.task == "arce"][["src", "rcv", "seed", "gain_text", "kd_ret", "ridge_ret"]], on=["src", "rcv", "seed"]).merge(PR, on=["src", "rcv", "seed"])
    D = G.groupby(["src", "rcv"]).mean(numeric_only=True).reset_index().merge(PS, on=["src", "rcv"]); D["pair"] = D.src + ">" + D.rcv
    save(D[["pair", "gain_text", "gap", "headroom", "overlap", "mismatch", "vocab_jaccard", "log_dim_ratio"]].set_index("pair"), "T9_pair_table")
    preds = ["gap", "headroom", "overlap", "mismatch", "vocab_jaccard", "log_dim_ratio"]; rows = []
    for out in ["gain_text", "gain_text_mean", "gain_ens_mean", "kd_ret", "ridge_ret"]:
        for p in preds:
            ok = D[p].notna(); rho, pv = sps.spearmanr(D[p][ok], D[out][ok]); rows.append(dict(outcome=out, predictor=p, spearman=rho, p=pv, n_pairs=int(ok.sum())))
    save(pd.DataFrame(rows).set_index("outcome"), "T10_univariate_spearman")
    # strict leave-two-models-out prediction
    sets = {"mean only": [], "mismatch": ["mismatch"], "mismatch+vocab+dim": ["mismatch", "vocab_jaccard", "log_dim_ratio"], "gap": ["gap"], "headroom": ["headroom"],
            "gap+headroom": ["gap", "headroom"], "gap+headroom+overlap": ["gap", "headroom", "overlap"], "all": preds}
    D = D.dropna(subset=preds).reset_index(drop=True); y = D["gain_text"].values; P = {}
    for name, cols in sets.items():
        pr = np.zeros(len(D))
        for i, r in D.iterrows():
            tr = ((D.src != r.src) & (D.rcv != r.src) & (D.src != r.rcv) & (D.rcv != r.rcv)).values
            X = np.c_[np.ones(tr.sum()), D.loc[tr, cols].values] if cols else np.ones((tr.sum(), 1)); Xt = np.r_[1.0, r[cols].values.astype(float)] if cols else np.ones(1)
            beta = np.linalg.solve(X.T @ X + 1e-6 * np.eye(X.shape[1]), X.T @ D.loc[tr, "gain_text"].values); pr[i] = Xt @ beta
        P[name] = pr
    ss_tot = ((y - y.mean()) ** 2).sum(); rows = []
    for name, pr in P.items():
        rows.append(dict(set=name, LOMO_R2=1 - ((y - pr) ** 2).sum() / ss_tot, RMSE=np.sqrt(((y - pr) ** 2).mean()), spearman=sps.spearmanr(pr, y)[0],
                         AUC_worth_building=auc(pr, y > WORTH), n_pairs=len(y), n_worth=int((y > WORTH).sum())))
    save(pd.DataFrame(rows).set_index("set"), "T11_leave_two_models_out")
    # cluster bootstrap over unordered model pairs for R2 differences
    cl = D.apply(lambda r: "|".join(sorted([r.src, r.rcv])), axis=1).values; ucl = np.unique(cl); rs = np.random.RandomState(0)
    def r2_of(pr, idx): yy = y[idx]; return 1 - ((yy - pr[idx]) ** 2).sum() / (((yy - yy.mean()) ** 2).sum() + 1e-12)
    rows = []
    for a_, b_ in [("headroom", "mismatch"), ("gap+headroom", "gap"), ("gap+headroom", "mismatch"), ("gap", "mismatch"), ("headroom", "gap")]:
        diffs = []
        for _ in range(N_BOOT):
            idx = np.concatenate([np.where(cl == c)[0] for c in rs.choice(ucl, len(ucl))]); diffs.append(r2_of(P[a_], idx) - r2_of(P[b_], idx))
        rows.append(dict(contrast=f"{a_} - {b_}", delta_R2=r2_of(P[a_], np.arange(len(y))) - r2_of(P[b_], np.arange(len(y))), ci_lo=np.percentile(diffs, 2.5), ci_hi=np.percentile(diffs, 97.5)))
    save(pd.DataFrame(rows).set_index("contrast"), "T12_R2_differences_P1_P2")
    best = max(P, key=lambda k: 1 - ((y - P[k]) ** 2).sum() / ss_tot)
    L.append(f"\n**P0** Spearman(mismatch, gain) = {sps.spearmanr(D.mismatch, y)[0]:.3f}; Spearman(vocab_jaccard, gain) = {sps.spearmanr(D.vocab_jaccard, y)[0]:.3f} over {len(D)} pairs.")
    L.append(f"**P3** AUC for 'worth building' (gain > {WORTH:+.2f}) with gap+headroom = {auc(P['gap+headroom'], y > WORTH):.3f} ({int((y > WORTH).sum())} of {len(y)} pairs positive). Best predictor set: {best}.")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for i, (nm, c) in enumerate([("mismatch", "#c0504d"), ("gap", "#4875ab"), ("gap+headroom", "#1d3b5f")]):
        ax[i].scatter(P[nm], y, s=14, c=c); lim = [min(y.min(), P[nm].min()), max(y.max(), P[nm].max())]; ax[i].plot(lim, lim, "k--", lw=0.8)
        ax[i].set_xlabel("predicted gain (unseen models)"); ax[i].set_ylabel("observed gain"); ax[i].set_title(f"{nm}: LOMO R2 = {1 - ((y - P[nm]) ** 2).sum() / ss_tot:.2f}")
    plt.tight_layout(); plt.savefig(f"{rep}/fig5_prediction.png", dpi=200); plt.close()
    open(f"{rep}/REPORT4.md", "w").write("# Experiment 4: predicting when exchange helps\n" + "\n".join(L)); print("\n".join(L))
    with zipfile.ZipFile(f"{os.path.dirname(BASE)}/exp4_results.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in ["results4.csv", "predictors4.csv", "pair_stats4.csv"]: z.write(f"{BASE}/{f}", f)
        for f in glob.glob(f"{rep}/*"): z.write(f, "report/" + os.path.basename(f))

if __name__ == "__main__" or "ipykernel" in sys.modules:
    t0 = time.time(); data = build_data(); print({k: len(v) for k, v in data.items()})
    for k in MODELS: extract_model(k, data)
    run_experiments(data); analyse(); print(f"ALL DONE in {(time.time()-t0)/60:.1f} min. Results in {BASE}")
