# PATCHED: text-hint features are stored as [item, hint, choice]; the arm now fixes the hint at the source's predicted answer and scores the four choices
# =====================================================================================
# CROSS-MODEL COMPATIBILITY BENCHMARK  -  stand-alone, resumable Kaggle script
# Paste into ONE notebook cell (or several cells split at the "# %%" markers) and run.
#
# KAGGLE SETTINGS : Accelerator = GPU (T4 x2 or P100) | Internet = ON | Persistence = Files only
# RESUME          : (a) same session: just re-run, finished work is skipped.
#                   (b) new session: "Save Version -> Save & Run All", then in the next session
#                       "Add Input -> Notebook Output" of that version. The script auto-copies
#                       /kaggle/input/*/exp/* into /kaggle/working/exp and continues.
# RUNTIME (approx.): features 30-50 min on a T4 (once), experiments + analysis < 30 min.
# OUTPUTS         : /kaggle/working/exp/results.csv, pair_stats.csv, report/ (tables, figures,
#                   REPORT.md) and /kaggle/working/exp_results.zip
# =====================================================================================
import os, sys, json, time, glob, shutil, gc, math, warnings, itertools, zipfile
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy import stats as sps
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ----------------------------------- CONFIG ------------------------------------------
QUICK_TEST   = False      # True = tiny run (200 items) to check the pipeline end-to-end
DRY_RUN      = False      # True = synthetic features, no models/datasets needed (logic test)
SEEDS        = [0, 1, 2]
N_POOL       = 1600       # ARC-Easy train items kept as pool
N_TRAIN      = 1200       # items per seed used for training heads / bridges
N_TEST       = 800        # items per test task
MODELS = {                # all ungated on the Hugging Face Hub
    "qwen05":  "Qwen/Qwen2.5-0.5B",
    "smol360": "HuggingFaceTB/SmolLM2-360M",
    "tiny11":  "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
    "pythia410": "EleutherAI/pythia-410m",
}
TASKS        = ["arce", "arcc", "sciq"]       # train on ARC-Easy only; arcc/sciq = distribution shift
LAYER_FRAC   = 0.75       # hidden layer used for mean-pooled features
BATCH        = 48
MAXLEN       = 160
L2_GRID      = [1e-4, 1e-3, 1e-2, 1e-1]
RIDGE_GRID   = [1e1, 1e2, 1e3, 1e4]
N_BOOT       = 2000
if QUICK_TEST:
    SEEDS, N_POOL, N_TRAIN, N_TEST, N_BOOT = [0], 240, 200, 120, 200

BASE = "/kaggle/working/exp" if os.path.exists("/kaggle") else os.path.abspath("./exp")
for d in ["features", "report"]:
    os.makedirs(f"{BASE}/{d}", exist_ok=True)
if DRY_RUN: BASE_FEAT = f"{BASE}/features_dry"; os.makedirs(BASE_FEAT, exist_ok=True)
else:       BASE_FEAT = f"{BASE}/features"

# ---------------------------- resume from earlier sessions ---------------------------
def restore_previous():
    n = 0
    for src in glob.glob("/kaggle/input/*/exp") + glob.glob("/kaggle/input/*/*/exp"):
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
    dims = {"qwen05": 96, "smol360": 80, "tiny11": 128, "pythia410": 64}
    D = dims[key]; r = np.random.RandomState(abs(hash(key)) % 2**31); Wm = r.randn(24, D) / 5
    for s, items in data.items():
        rs = np.random.RandomState(abs(hash(s)) % 2**31)
        base = np.zeros((len(items), 4, D), np.float32); hint = np.zeros((len(items), 4, 4, D), np.float32)
        for i, it in enumerate(items):
            z0 = np.random.RandomState(i + (7 if s == "train" else 99999 + hash(s) % 1000)).randn(4, 24)
            z0[it["y"]] += 1.0 * (1.5 if s != "sciq" else 1.2)
            nz = rs.randn(4, D) * 0.5
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

# ----------------------------------- PIPELINE ----------------------------------------
EXCH = ["text_hint", "kd", "bridge_ridge", "bridge_procrustes", "bridge_nonlinear"]
REFS = ["source_alone", "receiver_alone", "ensemble", "fusion_concat"]

def run_experiments(data):
    keys = list(MODELS); y = {s: np.array([it["y"] for it in data[s]]) for s in data}
    F = {k: {s: load_feat(k, s, False, data) for s in data} for k in keys}
    FH = {k: {s: load_feat(k, s, True, data) for s in data} for k in keys}
    res_path = f"{BASE}/results.csv"; done = set(); rows = []
    if os.path.exists(res_path):
        old = pd.read_csv(res_path); done = set(zip(old.src, old.rcv, old.seed, old.mech)); rows = old.to_dict("records")
    # pair statistics (once)
    ps_path = f"{BASE}/pair_stats.csv"
    if not os.path.exists(ps_path):
        ps = []; rs = np.random.RandomState(0)
        for a, b in itertools.permutations(keys, 2):
            sel = rs.permutation(len(F[a]["train"]) * 4)[:3000]
            xa, xb = F[a]["train"].reshape(-1, F[a]["train"].shape[-1])[sel].astype(np.float32), F[b]["train"].reshape(-1, F[b]["train"].shape[-1])[sel].astype(np.float32)
            try: vj = 0.0 if DRY_RUN else vocab_jaccard(MODELS[a], MODELS[b])
            except Exception as e: vj = float("nan")
            ps.append(dict(src=a, rcv=b, cka=linear_cka(xa, xb), mismatch=1 - linear_cka(xa, xb), vocab_jaccard=vj,
                           dim_src=xa.shape[1], dim_rcv=xb.shape[1], dim_ratio=xb.shape[1] / xa.shape[1]))
        pd.DataFrame(ps).to_csv(ps_path, index=False)
    t_start = time.time()
    for seed in SEEDS:
        tr = np.random.RandomState(seed).permutation(len(data["train"]))[:N_TRAIN]; ytr = y["train"][tr]
        std = {k: Std(F[k]["train"][tr]) for k in keys}
        Xtr = {k: std[k](F[k]["train"][tr]) for k in keys}; Xte = {k: {s: std[k](F[k][s]) for s in TASKS} for k in keys}
        # unimodal probes (+ out-of-fold predictions for leakage-free teacher signals)
        probe, l2s, oof, pte = {}, {}, {}, {}
        for k in keys:
            l2s[k] = auto_l2(Xtr[k], ytr, seed); probe[k] = fit_head(Xtr[k], ytr, l2=l2s[k], seed=seed)
            oof[k] = oof_probs(Xtr[k], ytr, l2s[k], seed); pte[k] = {s: head_probs(Xte[k][s], probe[k]) for s in TASKS}
        for a, b in itertools.permutations(keys, 2):
            todo = [m for m in REFS + EXCH if (a, b, seed, m) not in done]
            if not todo: continue
            out = {}   # mech -> dict(task -> probs), plus extras
            extra = {}
            out["source_alone"] = pte[a]; out["receiver_alone"] = pte[b]
            out["ensemble"] = {s: (pte[a][s] + pte[b][s]) / 2 for s in TASKS}; extra["source_alone"] = extra["receiver_alone"] = (0.0, probe[a].size)
            extra["ensemble"] = (0.0, probe[a].size + probe[b].size)
            # fusion_concat reference
            t0 = time.perf_counter(); Xc = np.concatenate([Xtr[b], Xtr[a]], -1); l2 = auto_l2(Xc, ytr, seed); w = fit_head(Xc, ytr, l2=l2, seed=seed)
            out["fusion_concat"] = {s: head_probs(np.concatenate([Xte[b][s], Xte[a][s]], -1), w) for s in TASKS}; extra["fusion_concat"] = (time.perf_counter() - t0, w.size)
            # kd: receiver head learns only from the source's out-of-fold soft predictions
            t0 = time.perf_counter(); w = fit_head(Xtr[b], soft=oof[a], l2=l2s[b], seed=seed)
            out["kd"] = {s: head_probs(Xte[b][s], w) for s in TASKS}; extra["kd"] = (time.perf_counter() - t0, w.size)
            # text_hint: source's predicted answer is inserted as text in the receiver's prompt
            t0 = time.perf_counter(); hy = oof[a].argmax(1)
            Htr_raw = FH[b]["train"][tr][np.arange(len(tr)), hy, :, :]; sh = Std(Htr_raw); Htr = sh(Htr_raw)
            l2 = auto_l2(Htr, ytr, seed); w = fit_head(Htr, ytr, l2=l2, seed=seed); out["text_hint"] = {}
            for s in TASKS:
                hh = pte[a][s].argmax(1); Hte = sh(FH[b][s][np.arange(len(hh)), hh, :, :]); out["text_hint"][s] = head_probs(Hte, w)
            extra["text_hint"] = (time.perf_counter() - t0, w.size)
            # latent bridges: fitted on UNLABELED paired train features, then source -> receiver head ("stitching")
            align = {}
            for kind in ["ridge", "procrustes", "nonlinear"]:
                br = Bridge(kind, Xtr[a], Xtr[b], seed); m = f"bridge_{kind}"
                Yh = br(Xte[a]["arce"]); Yt = Xte[b]["arce"]; cosv = float(np.mean(np.sum(Yh * Yt, -1) / (np.linalg.norm(Yh, axis=-1) * np.linalg.norm(Yt, axis=-1) + 1e-9)))
                align[m] = (cosv, r2(Yt.reshape(-1, Yt.shape[-1]), Yh.reshape(-1, Yh.shape[-1])))
                out[m] = {s: head_probs(br(Xte[a][s]), probe[b]) for s in TASKS}; extra[m] = (br.fit_s, br.params)
            for m, per in out.items():
                if (a, b, seed, m) in done: continue
                for s in TASKS:
                    r = dict(src=a, rcv=b, seed=seed, mech=m, task=s, n=len(y[s]), fit_s=extra[m][0], params=extra[m][1],
                             align_cos=align.get(m, (np.nan, np.nan))[0] if s == "arce" else np.nan, align_r2=align.get(m, (np.nan, np.nan))[1] if s == "arce" else np.nan)
                    r.update(metrics(per[s], y[s])); rows.append(r)
                done.add((a, b, seed, m))
            pd.DataFrame(rows).to_csv(res_path + ".tmp", index=False); os.replace(res_path + ".tmp", res_path)
        print(f"[experiments] seed {seed} done ({time.time()-t_start:.0f}s)")

# ----------------------------------- ANALYSIS ----------------------------------------
def kendall_w(M):  # rows = raters (pairs), cols = objects (mechanisms); higher score = better
    m, n = M.shape; R = np.apply_along_axis(lambda r: sps.rankdata(-r), 1, M); S = ((R.sum(0) - R.sum(0).mean()) ** 2).sum()
    W = 12 * S / (m ** 2 * (n ** 3 - n)); p = sps.chi2.sf(m * (n - 1) * W, n - 1); return W, p

def cboot(vals_by_cluster, n_boot=N_BOOT, seed=0):
    rs = np.random.RandomState(seed); arr = [np.asarray(v) for v in vals_by_cluster]; k = len(arr)
    means = [np.mean(np.concatenate([arr[i] for i in rs.randint(0, k, k)])) for _ in range(n_boot)]
    return float(np.mean(np.concatenate(arr))), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

def analyse():
    R = pd.read_csv(f"{BASE}/results.csv"); PS = pd.read_csv(f"{BASE}/pair_stats.csv"); R["pair"] = R.src + ">" + R.rcv
    rep = f"{BASE}/report"; L = []
    def save(df, name, idx=True):
        df.to_csv(f"{rep}/{name}.csv", index=idx)
        try: txt = df.round(4).to_markdown(index=idx)
        except Exception: txt = df.round(4).to_string(index=idx)
        L.append(f"\n### {name}\n\n" + txt)
    # T1: accuracy by mechanism x task with cluster-bootstrap CI over ordered pairs
    rows = []
    for m in REFS + EXCH:
        r = {"mech": m}
        for s in TASKS:
            g = R[(R.mech == m) & (R.task == s)].groupby("pair").acc.apply(list)
            mu, lo, hi = cboot(g.values); r[s] = f"{mu:.3f} [{lo:.3f},{hi:.3f}]"
        rows.append(r)
    save(pd.DataFrame(rows).set_index("mech"), "T1_accuracy_by_mechanism_task")
    # fidelity: retention (acc / source_alone) and gain over receiver_alone, in-distribution and shifted
    P = R.pivot_table(index=["pair", "seed", "task"], columns="mech", values="acc").reset_index()
    for m in REFS + EXCH: P[f"ret_{m}"] = P[m] / P["source_alone"]; P[f"gain_{m}"] = P[m] - P["receiver_alone"]
    ind = P[P.task == "arce"]
    T2 = pd.DataFrame({"retention": [ind[f"ret_{m}"].mean() for m in EXCH], "gain_over_receiver": [ind[f"gain_{m}"].mean() for m in EXCH]}, index=EXCH); save(T2, "T2_fidelity_arce")
    # T3: calibration, robustness (shift drop), cost
    rob = R.pivot_table(index=["pair", "seed", "mech"], columns="task", values="acc").reset_index(); rob["shift_drop_arcc"] = rob.arce - rob.arcc; rob["shift_drop_sciq"] = rob.arce - rob.sciq
    cal = R[R.task == "arce"].groupby("mech")[["ece", "nll", "fit_s", "params"]].mean()
    save(cal.join(rob.groupby("mech")[["shift_drop_arcc", "shift_drop_sciq"]].mean()).loc[REFS + EXCH], "T3_calibration_robustness_cost")
    # H1: do rankings of exchange mechanisms differ across pairs?
    A = R[R.mech.isin(EXCH)].groupby(["pair", "mech"]).acc.mean().unstack()[EXCH]
    W, p = kendall_w(A.values); winners = A.idxmax(1); save(winners.value_counts().to_frame("n_pairs_won"), "H1_winner_counts")
    L.append(f"\n**H1** Kendall's W across {len(A)} ordered pairs = {W:.3f} (chi-square p = {p:.4g}); distinct winners = {winners.nunique()}. "
             f"W near 1 = identical rankings in every pair; low W / several winners supports H1 (rankings change across pairs).")
    # H2: fidelity vs mismatch
    D = ind.merge(PS.assign(pair=PS.src + ">" + PS.rcv), on="pair"); out = []
    for m in EXCH:
        for col in ["mismatch", "vocab_jaccard", "dim_ratio"]:
            x = D[col]; yv = D[f"ret_{m}"]; ok = x.notna()
            if ok.sum() > 5 and x[ok].nunique() > 1: rho, pv = sps.spearmanr(x[ok], yv[ok]); out.append(dict(mech=m, predictor=col, spearman=rho, p=pv))
    save(pd.DataFrame(out), "H2_spearman", idx=False)
    try:
        import statsmodels.formula.api as smf
        Z = pd.concat([D[["pair", "mismatch", f"ret_{m}"]].rename(columns={f"ret_{m}": "retention"}).assign(mech=m) for m in EXCH])
        fit = smf.mixedlm("retention ~ C(mech) * mismatch", Z, groups=Z["pair"]).fit(); L.append("\n### H2_mixed_model (retention ~ mech*mismatch + (1|pair))\n```\n" + str(fit.summary()) + "\n```")
        open(f"{rep}/H2_mixed_model.txt", "w").write(str(fit.summary()))
    except Exception as e: L.append(f"\n(mixed model skipped: {e})")
    # H3: latent bridges vs text exchange: accuracy and cost (paired by pair x seed, cluster bootstrap over pairs)
    rows = []
    for m in ["bridge_ridge", "bridge_procrustes", "bridge_nonlinear", "kd"]:
        d = ind.assign(diff=ind[m] - ind["text_hint"]).groupby("pair")["diff"].apply(list); mu, lo, hi = cboot(d.values)
        c = R[(R.task == "arce") & (R.mech == m)].fit_s.mean() - R[(R.task == "arce") & (R.mech == "text_hint")].fit_s.mean()
        rows.append(dict(mech=m, acc_diff_vs_text=mu, ci_lo=lo, ci_hi=hi, fit_seconds_diff=c))
    save(pd.DataFrame(rows).set_index("mech"), "H3_bridge_vs_text")
    # alignment quality vs downstream fidelity
    al = R[(R.task == "arce") & R.mech.str.startswith("bridge")].merge(P[P.task == "arce"][["pair", "seed", "source_alone"]], on=["pair", "seed"])
    al["retention"] = al.acc / al.source_alone; rho, pv = sps.spearmanr(al.align_r2, al.retention); L.append(f"\n**Alignment vs fidelity**: Spearman(R2 of bridge, retention) = {rho:.3f} (p = {pv:.3g}).")
    # figures
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for i, s in enumerate(TASKS):
        v = [R[(R.mech == m) & (R.task == s)].acc.mean() for m in REFS + EXCH]; ax[i].barh(REFS + EXCH, v, color=["#888"] * 4 + ["#1d3b5f"] * 5); ax[i].set_title(f"Accuracy: {s}"); ax[i].set_xlim(0.2, max(v) + 0.05)
    plt.tight_layout(); plt.savefig(f"{rep}/fig1_accuracy.png", dpi=200); plt.close()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for m in EXCH: g = D.groupby("pair").agg(x=("mismatch", "mean"), y=(f"ret_{m}", "mean")); ax.scatter(g.x, g.y, label=m, s=25)
    ax.set_xlabel("representation mismatch (1 - linear CKA)"); ax.set_ylabel("retention (acc / source acc)"); ax.legend(fontsize=7); plt.tight_layout(); plt.savefig(f"{rep}/fig2_mismatch_vs_retention.png", dpi=200); plt.close()
    open(f"{rep}/REPORT.md", "w").write("# Cross-model compatibility benchmark: auto-generated report\n" + "\n".join(L))
    print("\n".join(L))
    with zipfile.ZipFile(f"{os.path.dirname(BASE)}/exp_results.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in ["results.csv", "pair_stats.csv"]: z.write(f"{BASE}/{f}", f)
        for f in glob.glob(f"{rep}/*"): z.write(f, "report/" + os.path.basename(f))

# ------------------------------------- MAIN ------------------------------------------
if __name__ == "__main__" or "ipykernel" in sys.modules:
    t0 = time.time(); data = build_data(); print({k: len(v) for k, v in data.items()})
    for k in MODELS: extract_model(k, data)
    run_experiments(data); analyse(); print(f"ALL DONE in {(time.time()-t0)/60:.1f} min. Results in {BASE}")
