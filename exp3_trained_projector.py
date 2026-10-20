# =====================================================================================
# EXPERIMENT 3 - Trained end-to-end projector vs text hint, with matched supervision and controls,
#                on multiple-choice AND short-answer generation, over a wider mismatch range.
# Stand-alone, resumable Kaggle script (GPU required, Internet ON).  ~3-5 h on one T4 for the default run.
#
# WHY: reviewers will say Experiments 1-2 used weak, untrained bridges and unmatched supervision. Here the source model S
# (frozen) is read out as a mean-pooled hidden vector, a small trained MLP projector turns it into K soft-prompt vectors that
# are PREPENDED to the frozen receiver R's input (a simplified, C2C-inspired latent channel; NOT a reproduction of
# Cache-to-Cache), and everything is trained end-to-end on the receiver's answer loss with the SAME labeled items.
#
# ARMS (all trained on the same labeled items, receiver frozen, same prefix length K):
#   prompt_only : learned soft prompt, NO source information            (control for "any trained prefix helps")
#   proj_shuf   : same projector, source features shuffled across items (control for extra capacity)
#   proj        : same projector, true source features                  (the latent channel)
#   text_hint   : source's zero-shot answer written in the receiver's prompt + the same learned soft prompt
# Plus zero-shot S and R (no training).  Tasks: ARC-Easy (train), ARC-Challenge & SciQ-MCQ (shift), SciQ short-answer
# generation (separate training).  PRE-REGISTER before running:
#   G1 proj > prompt_only and proj > proj_shuf  (the source information is actually used)
#   G2 proj vs text_hint (two-sided: which channel is better when both are trained)
#   G3 gain(proj - prompt_only) decreases as mismatch (1 - CKA) increases (wider range: same-family to cross-family pairs)
#   G4 gain correlates with the source-receiver zero-shot accuracy gap
# RESUME: re-run; finished (pair, seed, arm, family) are skipped. New session: add the previous output as Input
#         (the script copies /kaggle/input/**/exp3 back). Set QUICK_TEST=True first (about 10 minutes) to check it runs.
# NOTE  : written without access to a GPU; the logic of the analysis was tested on synthetic results, the training code was not.
# =====================================================================================
import os, sys, json, time, glob, shutil, gc, math, warnings, zipfile, re, string
import numpy as np, pandas as pd
from scipy import stats as sps
warnings.filterwarnings("ignore")
try:
    import torch, torch.nn as nn, torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForCausalLM
    HAVE_TORCH = True
except Exception as e:
    import types
    HAVE_TORCH = False; print("torch/transformers not available (analysis-only mode):", e)
    torch = types.SimpleNamespace(no_grad=lambda: (lambda f: f)); nn = types.SimpleNamespace(Module=object)

QUICK_TEST   = False
SEEDS        = [0, 1]
PAIRS        = [("qwen05", "qwen15"), ("qwen15", "qwen05"),        # same family (low mismatch)
                ("smol360", "qwen05"), ("qwen05", "tiny11"),       # cross-family
                ("tiny11", "pythia410"), ("pythia410", "smol360")]
HF = {"qwen05": "Qwen/Qwen2.5-0.5B", "qwen15": "Qwen/Qwen2.5-1.5B", "smol360": "HuggingFaceTB/SmolLM2-360M",
      "tiny11": "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T", "pythia410": "EleutherAI/pythia-410m"}
N_TRAIN, N_TEST_MCQ, N_TEST_GEN = 1500, 500, 300
K_PREFIX, EPOCHS, BS, MAXNEW = 4, 2, 8, 6
LAYER_FRAC, TIME_BUDGET_H, N_BOOT = 0.75, 8.5, 2000
LR_PROMPT, LR_PROJ = 5e-3, 1e-3
if QUICK_TEST:
    SEEDS, PAIRS, N_TRAIN, N_TEST_MCQ, N_TEST_GEN, EPOCHS = [0], [("smol360", "qwen05")], 64, 32, 24, 1

W = "/kaggle/working" if os.path.exists("/kaggle") else os.path.abspath(".")
E = f"{W}/exp3{'_quick' if QUICK_TEST else ''}"; os.makedirs(f"{E}/features", exist_ok=True); os.makedirs(f"{E}/report3", exist_ok=True)
dev = "cuda" if (HAVE_TORCH and torch.cuda.is_available()) else "cpu"
T0 = time.time()

def restore_previous():
    n = 0
    for root, _, files in os.walk("/kaggle/input"):
        if "exp3" in root.split(os.sep):
            rel = root.split("exp3", 1)[1].lstrip(os.sep)
            for f in files:
                d = os.path.join(E, rel, f)
                if not os.path.exists(d): os.makedirs(os.path.dirname(d), exist_ok=True); shutil.copy2(os.path.join(root, f), d); n += 1
    if n: print(f"[resume] restored {n} files")
if os.path.exists("/kaggle/input"): restore_previous()

# ----------------------------------- data ---------------------------------------------
def build_data():
    p = f"{E}/data.json"
    if os.path.exists(p): return json.load(open(p))
    from datasets import load_dataset
    rng = np.random.RandomState(0)
    def arc(cfg, split):
        try: ds = load_dataset("allenai/ai2_arc", cfg, split=split)
        except Exception: ds = load_dataset("ai2_arc", cfg, split=split)
        out = []
        for r in ds:
            t, l = r["choices"]["text"], r["choices"]["label"]
            if len(t) == 4 and r["answerKey"] in l: out.append(dict(q=r["question"][:1200], c=[x[:300] for x in t], y=l.index(r["answerKey"])))
        return out
    def sciq(split):
        try: ds = load_dataset("allenai/sciq", split=split)
        except Exception: ds = load_dataset("sciq", split=split)
        mcq, gen = [], []
        for r in ds:
            c = [r["correct_answer"], r["distractor1"], r["distractor2"], r["distractor3"]]; perm = rng.permutation(4)
            mcq.append(dict(q=r["question"][:1200], c=[c[i][:300] for i in perm], y=int(np.where(perm == 0)[0][0])))
            gen.append(dict(q=r["question"][:1200], a=r["correct_answer"][:100]))
        return mcq, gen
    sub = lambda items, n: [items[i] for i in rng.permutation(len(items))[:n]]
    sq_tr_m, sq_tr_g = sciq("train"); sq_te_m, sq_te_g = sciq("test")
    data = {"mcq_train": sub(arc("ARC-Easy", "train"), N_TRAIN), "arce": sub(arc("ARC-Easy", "test"), N_TEST_MCQ),
            "arcc": sub(arc("ARC-Challenge", "test"), N_TEST_MCQ), "sciq_mcq": sub(sq_te_m, N_TEST_MCQ),
            "gen_train": sub(sq_tr_g, N_TRAIN), "gen_test": sub(sq_te_g, N_TEST_GEN)}
    json.dump(data, open(p, "w")); return data

def mcq_prompt(it, hint=None):
    s = f"Question: {it['q']}\n" + "".join(f"{l}. {c}\n" for l, c in zip("ABCD", it["c"]))
    return s + (f"A peer model suggests: {hint}\n" if hint else "") + "Answer:"
def gen_prompt(it, hint=None): return f"Question: {it['q']}\n" + (f"A peer model suggests: {hint}\n" if hint else "") + "Answer:"
def norm(s): return " ".join(s.lower().translate(str.maketrans("", "", string.punctuation)).split())

# ----------------------------------- model utilities ----------------------------------
def load_lm(key):
    tok = AutoTokenizer.from_pretrained(HF[key]); tok.pad_token = tok.pad_token or tok.eos_token
    lm = AutoModelForCausalLM.from_pretrained(HF[key], torch_dtype=torch.float16).to(dev).eval(); lm.config.use_cache = False
    for p in lm.parameters(): p.requires_grad_(False)
    return tok, lm
def free(*objs):
    for o in objs: del o
    gc.collect(); torch.cuda.empty_cache()
def encode(tok, text): return tok(text, add_special_tokens=True)["input_ids"][-400:]
def letter_ids(tok): return [tok("Answer: " + l, add_special_tokens=False)["input_ids"][-1] for l in "ABCD"]
def embed_from_ids(lm, tok, ids_list, prefix):
    B, T = len(ids_list), max(len(x) for x in ids_list); pad = tok.pad_token_id
    ids = torch.full((B, T), pad, dtype=torch.long); mask = torch.zeros((B, T), dtype=torch.long)
    for i, x in enumerate(ids_list): ids[i, :len(x)] = torch.tensor(x); mask[i, :len(x)] = 1
    ids, mask = ids.to(dev), mask.to(dev); emb = lm.get_input_embeddings()(ids)
    if prefix is not None:
        emb = torch.cat([prefix.to(emb.dtype), emb], 1); mask = torch.cat([torch.ones(B, prefix.shape[1], dtype=torch.long, device=dev), mask], 1)
    return emb, mask
def hidden(lm, emb, mask): return lm.base_model(inputs_embeds=emb, attention_mask=mask).last_hidden_state
def mcq_logits(lm, tok, ids_list, prefix, lids):
    emb, mask = embed_from_ids(lm, tok, ids_list, prefix); h = hidden(lm, emb, mask); K = 0 if prefix is None else prefix.shape[1]
    idx = torch.tensor([K + len(x) - 1 for x in ids_list], device=dev); hs = h[torch.arange(len(ids_list), device=dev), idx]
    return lm.get_output_embeddings()(hs)[:, lids].float()
@torch.no_grad()
def greedy(lm, tok, ids, prefix):
    cur, out = list(ids), []
    for _ in range(MAXNEW):
        emb, mask = embed_from_ids(lm, tok, [cur], prefix); h = hidden(lm, emb, mask)
        nxt = int(lm.get_output_embeddings()(h[0, -1]).argmax())
        if nxt == tok.eos_token_id: break
        out.append(nxt); cur.append(nxt)
        if "\n" in tok.decode(out): break
    return tok.decode(out).split("\n")[0].strip()
def gen_correct(pred, gold): g, p = norm(gold), norm(pred); return float(len(p) > 0 and (g in p or p == g))

class PromptOnly(nn.Module):
    def __init__(self, Dr, K, std): super().__init__(); self.p = nn.Parameter(torch.randn(K, Dr) * std)
    def forward(self, f, B): return self.p.unsqueeze(0).expand(B, -1, -1)
class Projector(nn.Module):
    def __init__(self, Ds, Dr, K, std):
        super().__init__(); self.K, self.Dr, self.std = K, Dr, std; self.ln = nn.LayerNorm(Ds, elementwise_affine=False)
        self.net = nn.Sequential(nn.Linear(Ds, 1024), nn.GELU(), nn.Linear(1024, K * Dr))
    def forward(self, f, B): return self.net(self.ln(f)).view(f.shape[0], self.K, self.Dr) * (self.std * 3)

@torch.no_grad()
def source_features(lm, tok, texts, bs=16):
    L = int(round(LAYER_FRAC * lm.config.num_hidden_layers)); out = []
    for i in range(0, len(texts), bs):
        ids = [encode(tok, t) for t in texts[i:i + bs]]; emb, mask = embed_from_ids(lm, tok, ids, None)
        hs = lm.base_model(inputs_embeds=emb, attention_mask=mask, output_hidden_states=True).hidden_states[L].float(); m = mask.unsqueeze(-1).float()
        out.append(((hs * m).sum(1) / m.sum(1)).cpu().numpy())
    return np.concatenate(out).astype(np.float32)

# ------------------------- zero-shot predictions, hints and features (per model, cached) ----
def model_cache(key, data):
    p = f"{E}/features/{key}_cache.npz"
    if os.path.exists(p): z = np.load(p, allow_pickle=True); return {k: z[k] for k in z.files}
    print(f"[cache] {key}"); tok, lm = load_lm(key); lids = letter_ids(tok); C = {}
    for sp in ["mcq_train", "arce", "arcc", "sciq_mcq"]:
        items = data[sp]; texts = [mcq_prompt(it) for it in items]; C[f"feat_{sp}"] = source_features(lm, tok, texts)
        preds = []
        with torch.no_grad():
            for i in range(0, len(texts), 16): preds += mcq_logits(lm, tok, [encode(tok, t) for t in texts[i:i + 16]], None, lids).argmax(1).tolist()
        C[f"pred_{sp}"] = np.array(preds)
    for sp in ["gen_train", "gen_test"]:
        items = data[sp]; texts = [gen_prompt(it) for it in items]; C[f"feat_{sp}"] = source_features(lm, tok, texts)
        C[f"pred_{sp}"] = np.array([greedy(lm, tok, encode(tok, t), None) for t in texts], dtype=object)
    np.savez(p, **C); free(lm); return C

# ----------------------------------- training / evaluation ----------------------------
def train_arm(lm, tok, kind, family, data, Cs, Cr, seed, hint_arm):
    """kind in prompt_only | proj_shuf | proj ; hint_arm adds the source's zero-shot answer to the receiver's prompt."""
    torch.manual_seed(seed); rng = np.random.RandomState(seed)
    Dr = lm.config.hidden_size; std = lm.get_input_embeddings().weight.float().std().item(); lids = letter_ids(tok)
    sp = "mcq_train" if family == "mcq" else "gen_train"; items = data[sp]; n = len(items)
    pool = rng.permutation(n)[:N_TRAIN] if n > N_TRAIN else np.arange(n)
    feats = torch.tensor(Cs[f"feat_{sp}"]) if kind != "prompt_only" else None
    mod = (Projector(feats.shape[1], Dr, K_PREFIX, std) if feats is not None else PromptOnly(Dr, K_PREFIX, std)).to(dev)
    opt = torch.optim.AdamW(mod.parameters(), lr=LR_PROJ if feats is not None else LR_PROMPT, weight_decay=0.0)
    scaler = torch.cuda.amp.GradScaler(); shuf = rng.permutation(n); total = EPOCHS * math.ceil(len(pool) / BS); step = 0
    def prompt_ids(it, hint): return encode(tok, (mcq_prompt if family == "mcq" else gen_prompt)(it, hint))
    for ep in range(EPOCHS):
        order = rng.permutation(pool)
        for i in range(0, len(order), BS):
            idx = order[i:i + BS]; B = len(idx)
            f = None
            if feats is not None: f = feats[idx if kind == "proj" else shuf[idx]].to(dev)
            with torch.autocast("cuda", dtype=torch.float16):
                prefix = mod(f, B)
                if family == "mcq":
                    ids = [prompt_ids(items[j], (None if not hint_arm else "ABCD"[int(Cs["pred_" + sp][j])] ) ) for j in idx]
                    # the hint must be the SOURCE's prediction; Cs holds the source model's zero-shot predictions
                    logits = mcq_logits(lm, tok, ids, prefix, lids); y = torch.tensor([items[j]["y"] for j in idx], device=dev)
                    loss = F.cross_entropy(logits, y)
                else:
                    losses = []
                    for b, j in enumerate(idx):
                        hint = str(Cs["pred_" + sp][j]) if hint_arm else None; pid = prompt_ids(items[j], hint)
                        aid = tok(" " + items[j]["a"], add_special_tokens=False)["input_ids"][:MAXNEW]; full = pid + aid
                        emb, mask = embed_from_ids(lm, tok, [full], prefix[b:b + 1]); h = hidden(lm, emb, mask)
                        pos = torch.arange(K_PREFIX + len(pid) - 1, K_PREFIX + len(full) - 1, device=dev)
                        lg = lm.get_output_embeddings()(h[0, pos]).float(); losses.append(F.cross_entropy(lg, torch.tensor(aid, device=dev)))
                    loss = torch.stack(losses).mean()
            scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(mod.parameters(), 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); step += 1
    return mod
@torch.no_grad()
def evaluate(lm, tok, mod, kind, family, data, Cs, hint_arm, seed):
    lids = letter_ids(tok); res = {}; mod.eval()
    for sp in (["arce", "arcc", "sciq_mcq"] if family == "mcq" else ["gen_test"]):
        items = data[sp]; feats = torch.tensor(Cs[f"feat_{sp}"]) if kind != "prompt_only" else None
        perm = np.random.RandomState(seed + 99).permutation(len(items)); correct = []; nll = []
        for i in range(0, len(items), 8 if family == "mcq" else 1):
            idx = list(range(i, min(i + (8 if family == "mcq" else 1), len(items))))
            f = None if feats is None else feats[idx if kind == "proj" else perm[idx]].to(dev)
            with torch.autocast("cuda", dtype=torch.float16):
                prefix = mod(f, len(idx))
                if family == "mcq":
                    ids = [encode(tok, mcq_prompt(items[j], "ABCD"[int(Cs["pred_" + sp][j])] if hint_arm else None)) for j in idx]
                    lg = mcq_logits(lm, tok, ids, prefix, lids); pr = lg.argmax(1).tolist()
                    correct += [float(p == items[j]["y"]) for p, j in zip(pr, idx)]
                    nll += (-F.log_softmax(lg, 1)[torch.arange(len(idx)), torch.tensor([items[j]["y"] for j in idx])]).tolist()
                else:
                    j = idx[0]; hint = str(Cs["pred_" + sp][j]) if hint_arm else None
                    correct.append(gen_correct(greedy(lm, tok, encode(tok, gen_prompt(items[j], hint)), prefix), items[j]["a"]))
        res[sp if sp != "gen_test" else "sciq_gen"] = dict(acc=float(np.mean(correct)), nll=float(np.mean(nll)) if nll else np.nan, n=len(correct))
    return res
@torch.no_grad()
def zero_shot(lm, tok, family, data, C):
    lids = letter_ids(tok); out = {}
    if family == "mcq":
        for sp in ["arce", "arcc", "sciq_mcq"]: out[sp] = dict(acc=float(np.mean(C[f"pred_{sp}"] == np.array([it["y"] for it in data[sp]]))), nll=np.nan, n=len(data[sp]))
    else: out["sciq_gen"] = dict(acc=float(np.mean([gen_correct(str(p), it["a"]) for p, it in zip(C["pred_gen_test"], data["gen_test"])])), nll=np.nan, n=len(data["gen_test"]))
    return out

def linear_cka(X, Y):
    X = X - X.mean(0); Y = Y - Y.mean(0); return float(np.linalg.norm(Y.T @ X) ** 2 / (np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y) + 1e-12))
def vocab_jaccard(a, b):
    nm = lambda t: {k.replace("Ġ", "").replace("▁", "").lower() for k in AutoTokenizer.from_pretrained(HF[t]).get_vocab()} - {""}
    A, B = nm(a), nm(b); return len(A & B) / len(A | B)

def run(data):
    rp = f"{E}/results3.csv"; rows = pd.read_csv(rp).to_dict("records") if os.path.exists(rp) else []
    done = {(r["src"], r["rcv"], r["seed"], r["arm"], r["task"]) for r in rows}
    keys = sorted({k for p in PAIRS for k in p}); caches = {k: model_cache(k, data) for k in keys}
    # pair statistics
    pp = f"{E}/pair_stats3.csv"
    if not os.path.exists(pp):
        ps = []
        for a, b in PAIRS:
            try: vj = vocab_jaccard(a, b)
            except Exception: vj = np.nan
            ck = linear_cka(caches[a]["feat_arce"], caches[b]["feat_arce"])
            ps.append(dict(src=a, rcv=b, cka=ck, mismatch=1 - ck, vocab_jaccard=vj, dim_ratio=caches[b]["feat_arce"].shape[1] / caches[a]["feat_arce"].shape[1]))
        pd.DataFrame(ps).to_csv(pp, index=False)
    for seed in SEEDS:
        for a, b in PAIRS:
            if (time.time() - T0) / 3600 > TIME_BUDGET_H: print("[budget] time budget reached; re-run to resume"); return
            tok, lm = load_lm(b); Cs, Cr = caches[a], caches[b]
            for family in ["mcq", "gen"]:
                arms = [("prompt_only", "prompt_only", False), ("proj_shuf", "proj_shuf", False), ("proj", "proj", False), ("text_hint", "prompt_only", True)]
                for arm, kind, hint in arms:
                    tasks = ["arce", "arcc", "sciq_mcq"] if family == "mcq" else ["sciq_gen"]
                    if all((a, b, seed, arm, t) in done for t in tasks): continue
                    t1 = time.time(); mod = train_arm(lm, tok, kind, family, data, Cs, Cr, seed, hint); res = evaluate(lm, tok, mod, kind, family, data, Cs, hint, seed)
                    for t, r in res.items(): rows.append(dict(src=a, rcv=b, seed=seed, arm=arm, task=t, params=sum(p.numel() for p in mod.parameters()), train_s=time.time() - t1, **r)); done.add((a, b, seed, arm, t))
                    pd.DataFrame(rows).to_csv(rp + ".tmp", index=False); os.replace(rp + ".tmp", rp); print(f"[run] {a}->{b} seed{seed} {family} {arm}: {res}  ({time.time()-t1:.0f}s)")
                for who, key_, C_ in (("zs_receiver", b, Cr), ("zs_source", a, Cs)):
                    for t, r in zero_shot(lm, tok, family, data, C_).items():
                        k_ = (a, b, seed, who, t)
                        if k_ not in done: rows.append(dict(src=a, rcv=b, seed=seed, arm=who, task=t, params=0, train_s=0, **r)); done.add(k_)
            pd.DataFrame(rows).to_csv(rp, index=False); free(lm)

# ----------------------------------- analysis -----------------------------------------
def cboot(clusters, n=N_BOOT, seed=0):
    rs = np.random.RandomState(seed); arr = [np.asarray(c) for c in clusters]; k = len(arr)
    m = [np.mean(np.concatenate([arr[i] for i in rs.randint(0, k, k)])) for _ in range(n)]
    return float(np.mean(np.concatenate(arr))), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))
def analyse():
    R = pd.read_csv(f"{E}/results3.csv"); PS = pd.read_csv(f"{E}/pair_stats3.csv"); R["pair"] = R.src + ">" + R.rcv; rep = f"{E}/report3"; L = []
    def save(df, name):
        df.to_csv(f"{rep}/{name}.csv")
        try: t = df.round(4).to_markdown()
        except Exception: t = df.round(4).to_string()
        L.append(f"\n### {name}\n\n{t}")
    tasks = ["arce", "arcc", "sciq_mcq", "sciq_gen"]; arms = ["zs_source", "zs_receiver", "prompt_only", "proj_shuf", "proj", "text_hint"]; rows = []
    for ar in arms:
        r = {"arm": ar}
        for t in tasks:
            g = R[(R.arm == ar) & (R.task == t)].groupby("pair").acc.apply(list)
            if len(g): mu, lo, hi = cboot(g.values); r[t] = f"{mu:.3f} [{lo:.3f},{hi:.3f}]"
        rows.append(r)
    save(pd.DataFrame(rows).set_index("arm"), "T6_accuracy_by_arm")
    P = R.pivot_table(index=["pair", "seed", "task"], columns="arm", values="acc").reset_index()
    def diff(a, b, t):
        d = P[P.task == t].assign(d=lambda x: x[a] - x[b]).dropna(subset=["d"]).groupby("pair").d.apply(list)
        mu, lo, hi = cboot(d.values); return dict(contrast=f"{a} - {b}", task=t, diff=mu, ci_lo=lo, ci_hi=hi, pairs_positive=int((d.apply(np.mean) > 0).sum()), n_pairs=len(d))
    rows = [diff(a, b, t) for t in tasks for a, b in [("proj", "prompt_only"), ("proj", "proj_shuf"), ("proj", "text_hint"), ("text_hint", "prompt_only"), ("proj_shuf", "prompt_only")] if a in P and b in P]
    save(pd.DataFrame(rows).set_index("contrast"), "T7_contrasts_G1_G2")
    # G3 / G4
    D = P[P.task.isin(["arce", "sciq_gen"])].copy(); D["gain"] = D["proj"] - D["prompt_only"]; D["tgain"] = D["text_hint"] - D["prompt_only"]; D["gap"] = D["zs_source"] - D["zs_receiver"]
    PS["pair"] = PS.src + ">" + PS.rcv; D = D.merge(PS[["pair", "mismatch", "vocab_jaccard", "dim_ratio"]], on="pair", how="left"); out = []
    for t in ["arce", "sciq_gen"]:
        d = D[D.task == t]
        for col in ["mismatch", "vocab_jaccard", "gap"]:
            for g in ["gain", "tgain"]:
                ok = d[col].notna() & d[g].notna()
                if ok.sum() > 5 and d[col][ok].nunique() > 1: rho, p = sps.spearmanr(d[col][ok], d[g][ok]); out.append(dict(task=t, outcome=g, predictor=col, spearman=rho, p=p, n=int(ok.sum())))
    save(pd.DataFrame(out).set_index("task") if out else pd.DataFrame({"note": ["too few points"]}), "T8_G3_G4_spearman")
    L.append(f"\nMismatch (1 - CKA) range across the {len(PS)} pairs: {PS.mismatch.min():.3f} to {PS.mismatch.max():.3f}. (Experiment 1: 0.011 to 0.049.)")
    try:
        import statsmodels.formula.api as smf
        Z = D[D.task == "arce"].dropna(subset=["gain", "mismatch", "gap"]); fit = smf.mixedlm("gain ~ mismatch + gap", Z, groups=Z["pair"]).fit()
        L.append("\n### Mixed model (ARC-Easy): gain(proj - prompt_only) ~ mismatch + source-receiver gap + (1|pair)\n```\n" + str(fit.summary()) + "\n```")
    except Exception as e: L.append(f"\n(mixed model skipped: {e})")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for t, c in [("arce", "#1d3b5f"), ("sciq_gen", "#c0504d")]:
        d = D[D.task == t].groupby("pair").agg(x=("mismatch", "mean"), y=("gain", "mean"), g=("gap", "mean")); ax[0].scatter(d.x, d.y, label=t, c=c); ax[1].scatter(d.g, d.y, c=c, label=t)
    ax[0].set_xlabel("mismatch (1 - CKA)"); ax[1].set_xlabel("source - receiver zero-shot accuracy gap"); ax[0].set_ylabel("gain: proj - prompt_only"); ax[0].legend()
    plt.tight_layout(); plt.savefig(f"{rep}/fig4_gain.png", dpi=200); plt.close()
    open(f"{rep}/REPORT3.md", "w").write("# Experiment 3: trained projector with controls\n" + "\n".join(L)); print("\n".join(L))
    with zipfile.ZipFile(f"{W}/exp3_results.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in ["results3.csv", "pair_stats3.csv"]: z.write(f"{E}/{f}", f)
        for f in glob.glob(f"{rep}/*"): z.write(f, "report3/" + os.path.basename(f))

if __name__ == "__main__" or "ipykernel" in sys.modules:
    if not HAVE_TORCH: sys.exit("This script needs torch + transformers (Kaggle has them).")
    data = build_data(); print({k: len(v) for k, v in data.items()}); run(data); analyse(); print(f"DONE in {(time.time()-T0)/3600:.2f} h -> {E}")
