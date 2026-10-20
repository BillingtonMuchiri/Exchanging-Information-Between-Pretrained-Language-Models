# Code and results: When Does Exchanging Information Between Pretrained Language Models Help?

Archive accompanying the paper *"When Does Exchanging Information Between Pretrained Language Models Help? A Controlled Study of Text, Distillation, and Latent Channels"* (IJCDS submission).

- **DOI:** [AUTHORS: Zenodo DOI after deposit]
- **Authors:**
  - Billington Muchiri Muria (corresponding author), Department of Informatics and Computing, Taita Taveta University, Voi, Kenya
  - Solomon Mwanjele, Department of Informatics and Computing, Taita Taveta University, Voi, Kenya
  - Joash Kiprotich Bii, Department of Computer Science, Maasai Mara University, Narok, Kenya
- **Licence:** code under the MIT licence, see `LICENSE`. The results files (CSVs, reports and figures produced by the scripts) are released under CC BY 4.0. Datasets and model weights are **not** redistributed and keep their own licences, with one exception: the item lists `data_full.json` and `data.json` contain the ARC and SciQ question text of the items used, and remain under those datasets' own licences (see their Hugging Face pages), not CC BY 4.0. The published journal article has its own licence, set by the journal.
- **How to cite:** see `CITATION.cff`.

> Everything marked `[AUTHORS: ...]` must be filled in or deleted before deposit. Section 9 discloses a known issue in the text-hint code.

## 1. What is in this archive

| Experiment | Question | Script | Main outputs |
|---|---|---|---|
| 1 | Which mechanism family performs best? | `kaggle_experiment.py` | `exp/results.csv`, `exp/pair_stats.csv`, `exp/data_full.json`, `exp/report/`, `exp_results.zip` |
| 2 | Do better label-free bridges close the gap? | `followup_experiment.py` (see Section 4) | `exp2/results2.csv`, `exp2/report2/`, `exp2_results.zip` |
| 3 | Does a trained projector use source information? | `exp3_trained_projector.py` | `exp3/results3.csv`, `exp3/pair_stats3.csv`, `exp3/data.json`, `exp3/report3/`, `exp3_results.zip` |
| 3b | Is the projector result sensitive to its settings? | `exp3b_projector_sensitivity.py` | `exp3b_results.zip` (`projector_sensitivity.csv` and five `<variant>_results3.csv` files) |
| 4 | What predicts the benefit of a text hint? | `exp4_predictor_study.py` | `exp4/results4.csv`, `exp4/predictors4.csv`, `exp4/pair_stats4.csv`, `exp4/data_full.json`, `exp4/report/`, `exp4_results.zip` |
| 5 | Does the ordering survive matched supervision? | `exp5_matched_supervision.py` | `exp5/results5.csv`, `exp5/data_full.json`, `exp5/report/`, `exp5_results.zip` |
| Dependence checks and appendix tables | | `analysis_sensitivity.py` | `analysis_out/` (Section 6) |
| Environment record | | `env_report.py` | `env_report.json` (also printed; saved as `libraries.txt`) |

Experiment 3b reruns the Experiment 3 training code with one setting changed per variant. Experiment 2 reads the data and base features of Experiment 1. Experiment 5 reuses the cached features of Experiment 4 when they are available.

Row counts of the per-run CSVs (from the paper's reproducibility table): Experiment 1: 972; Experiment 2: 1,728; Experiment 3: 288; Experiment 3b: 225 in five files; Experiment 4: 2,016; Experiment 5: 2,688.

## 2. Setup

The runs were made on Kaggle notebooks with one Tesla T4 GPU (CUDA 12.8), Linux, October 2026 (exact dates were not logged). No container or environment file was produced. `env_report.py` prints and writes (as `env_report.json`) the Python and package versions (torch, transformers, datasets, huggingface_hub, numpy, scipy, pandas, scikit-learn, statsmodels, matplotlib, tokenizers), the CUDA device, and the Hugging Face commit hashes of the models and datasets below. It needs Internet access for the hub queries.

```
python env_report.py | tee libraries.txt
```

Models (Hugging Face Hub, all ungated):

| Experiments | Models |
|---|---|
| 1, 2 | `Qwen/Qwen2.5-0.5B`, `HuggingFaceTB/SmolLM2-360M`, `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T`, `EleutherAI/pythia-410m` |
| 3, 3b | the four above plus `Qwen/Qwen2.5-1.5B` |
| 4, 5 | the five above plus `HuggingFaceTB/SmolLM2-135M`, `HuggingFaceTB/SmolLM2-1.7B`, `facebook/opt-350m` |

Data: `allenai/ai2_arc` (ARC-Easy and ARC-Challenge) and `allenai/sciq`, downloaded by the scripts through the Hugging Face `datasets` library (with the older dataset names as a fallback). Items without exactly four choices are removed. SciQ choices are shuffled with a fixed seed. The exact items used are saved in `data_full.json` (Experiments 1, 4 and 5) and `data.json` (Experiments 3 and 3b).

Credentials: none are needed (all models and datasets are ungated). No script contains a token; keep it that way.

## 3. Run order

Each script is stand-alone and resumable: completed (pair, seed, mechanism or arm) rows are written to a CSV, and a re-run skips them. Features are extracted once per model and cached as float16 `.npy` files. In a new Kaggle session, add the previous version's output as an Input; the scripts copy the earlier files back.

1. `env_report.py`
2. `kaggle_experiment.py` (Experiment 1)
3. `followup_experiment.py` (Experiment 2), after Experiment 1
4. `exp3_trained_projector.py` (Experiment 3), then `exp3b_projector_sensitivity.py` (Experiment 3b; it reuses Experiment 3's cached source features and `data.json` when they are provided as Input, and rebuilds them otherwise)
5. `exp4_predictor_study.py` (Experiment 4), then `exp5_matched_supervision.py` (Experiment 5; it reuses Experiment 4's features when provided as Input, and rebuilds them otherwise, which needs a GPU and Internet)
6. `analysis_sensitivity.py` on the final CSVs (Section 6)

Every script has `QUICK_TEST` (and Experiments 1, 4, 5 also `DRY_RUN`) at the top. Keep both False to match the paper. `QUICK_TEST = True` is a short smoke test; `DRY_RUN = True` uses synthetic features and only tests the code logic.

### Kaggle settings and approximate run times

The times below are the estimates in the script headers, not measured times. The runs were spread over several sessions.

| Experiment | Accelerator | Internet | Approximate time |
|---|---|---|---|
| 1 | GPU (T4 used) | On | features 30–50 min (once); experiments and analysis under 30 min |
| 2 | CPU with `REBUILD_FEATURES = False`; GPU with the default `True` | Off with `False`; on with `True` | 20–40 min (about 35 min more if features are rebuilt) |
| 3 | GPU required | On | 3–5 h on one T4 (full run); `QUICK_TEST` about 10 min |
| 3b | GPU required | On | 1.5–2.5 h on one T4 |
| 4 | GPU for feature extraction | On | 2.5–3.5 h features plus about 30 min |
| 5 | CPU with Experiment 4's cached features | Off with cached features; on and GPU otherwise | about 30 min |

The headers of Experiments 1, 4 and 5 list "T4 x2 or P100" as acceptable accelerators; the paper's runs used one T4, and the scripts use one device.

## 4. Experiment 2: script version

`followup_experiment.py` in this archive is the third of three versions that were produced (the one with the `FRESH_START` and `REBUILD_FEATURES` flags). The versions differ only in how they find or rebuild the Experiment 1 data and features; the arms, fitting and analysis code are identical, so the other two are not deposited.

The flags at the top of the script:

| Flag | Default | Effect |
|---|---|---|
| `FRESH_START` | `True` | Ignores earlier files: wipes `exp2/` and its `results2.csv` and starts from zero. |
| `REBUILD_FEATURES` | `True` | Also deletes `exp/features` and `exp/data_full.json` and rebuilds the data and base features from the models with the Experiment 1 recipe (needs a GPU and Internet, about 35 min). |

With `REBUILD_FEATURES = False` the script reuses the data and features found in `/kaggle/input` or `/kaggle/working/exp` and runs on CPU in about 30 min. **Run with the defaults, the script deletes the Experiment 1 features and `data_full.json` in the same working directory**, so run Experiment 2 in a separate session or set the flag to `False`.

The Experiment 2 results in the paper were produced from a fresh start (`FRESH_START = True`), with the script's default `REBUILD_FEATURES = True`, so the data and base features were rebuilt with the Experiment 1 recipe. [AUTHORS: confirm that `REBUILD_FEATURES` was left at its default `True`. The paper's Methods say Experiment 2 reuses Experiment 1's features; if the features were rebuilt, change that sentence to say they were regenerated with the same recipe, models and seeds.]

Arms: ridge maps fitted on 300, 600, 1,200 and 1,600 unlabeled items; reduced-rank maps (ranks 16, 64, 256); CCA (32, 128 components); relative representations (128, 512 anchors); and refitting the receiver's probe on 20, 100 or 400 labels. Three seeds, 12 ordered pairs.

## 5. Settings of each experiment

- **Experiments 1 and 2:** 1,600 ARC-Easy training items, 1,200 per seed for training, 800 test items per task (ARC-Easy, ARC-Challenge, SciQ), mean-pooled hidden state at 75% of depth, prompts truncated to 160 tokens, probe L2 chosen from {1e-4, 1e-3, 1e-2, 1e-1} by an 80/20 split, L-BFGS with at most 150 iterations, 2,000 bootstrap resamples, seeds 0, 1, 2.
- **Experiment 3:** six ordered pairs (Qwen2.5-0.5B ↔ Qwen2.5-1.5B, SmolLM2-360M → Qwen2.5-0.5B, Qwen2.5-0.5B → TinyLlama, TinyLlama → Pythia-410M, Pythia-410M → SmolLM2-360M), 1,500 training items, 500 multiple-choice and 300 generation test items, prefix length 4, 2 epochs, batch size 8, learning rates 5e-3 (soft prompt) and 1e-3 (projector), seeds 0, 1. The text hint is written into the receiver's prompt as a letter.
- **Experiment 3b:** three pairs (Qwen2.5-0.5B → Qwen2.5-1.5B, SmolLM2-360M → Qwen2.5-0.5B, TinyLlama → Pythia-410M), seed 0, five variants: `base_h1024_lr1e-3_ep2`, `small_h128`, `lowlr_3e-4`, `long_ep4`, `prefix8`.
- **Experiments 4 and 5:** eight models, 56 ordered pairs, ARC-Easy pool of 1,600 items split into 1,200 training and 400 validation items, 600 test items per task, prompts truncated to 160 tokens, seeds 0, 1.

## 6. Which script produces which table and figure

Script outputs use their own numbering (T1, T2, …, `fig1_…`); the paper numbers tables and figures differently. The mapping:

| Paper | Produced by | File |
|---|---|---|
| Tables IV, V; H1–H3 statistics | `kaggle_experiment.py` | `exp/report/T1_…`, `T2_…`, `T3_…`, `H1_…`, `H2_…`, `H3_…`, `REPORT.md` |
| Figure 1 (retention against mismatch) | `kaggle_experiment.py` | `exp/report/fig2_mismatch_vs_retention.png` (`fig1_accuracy.png` is not in the paper) |
| Table VI | `followup_experiment.py` | `exp2/report2/T4_accuracy_by_arm.csv`, `T5_fidelity_vs_ridge_baseline.csv` |
| Figure 2 | `followup_experiment.py` | `exp2/report2/fig3_scaling_and_arms.png` |
| Table VII | `exp3_trained_projector.py` | `exp3/report3/T6_accuracy_by_arm.csv`, `T7_contrasts_G1_G2.csv`, `T8_G3_G4_spearman.csv` |
| Figure 3 | `exp3_trained_projector.py` | `exp3/report3/fig4_gain.png` |
| Table VIII | `exp4_predictor_study.py` | `exp4/report/T11_leave_two_models_out.csv` (also `T9`, `T10`, `T12`); baselines regenerated by `analysis_sensitivity.py` as `table8_predictors.csv` |
| Figure 4 | `exp4_predictor_study.py` | `exp4/report/fig5_prediction.png` |
| Tables IX, X | `analysis_sensitivity.py` | `sens_exp1.csv`, `sens_exp3.csv`; Kendall's W variants in `kendall_w.json` |
| Table XI | `exp3b_projector_sensitivity.py` | `projector_sensitivity.csv` |
| Tables XII, XIII | `exp5_matched_supervision.py` | `exp5/report/T13_mean_accuracy_by_arm_and_task.csv`, `T14_contrasts_M1_M2_M3.csv` |
| Figure 5 | `exp5_matched_supervision.py` | `exp5/report/fig6_matched_supervision.png` |
| Tables B1–B3 | `analysis_sensitivity.py` | `tableB1_exp1_per_pair.csv`, `tableB2_exp3_per_pair.csv`, `tableB3_exp4_per_pair.csv` |
| Tables I, II, III, XIV, R1, R2 | not produced by code | R2 values come from `env_report.json` |

`analysis_sensitivity.py` usage:

```
python analysis_sensitivity.py --exp1 exp --exp3 exp3 --exp4 exp4 --out analysis_out
```

It needs `exp/results.csv`, `exp3/results3.csv`, and `exp4/results4.csv`, `predictors4.csv` and `pair_stats4.csv`. (Experiment 3 is skipped if `results3.csv` is missing.) The comments at the top of the script refer to "Section 5.9" and "Table 8"; these are the paper's Section 5-I and Table VIII.

## 7. Checking the reproduction

1. Re-run `analysis_sensitivity.py` on the final CSVs and confirm that the Section 5-I and appendix tables regenerate.
2. Check that the CSV row counts match Section 1.
3. Small numerical differences are expected across library versions and GPUs; compare against `env_report.json`.

## 8. Not included

- Model weights and the full ARC and SciQ datasets (public; the commit hashes in `env_report.json` and Table R2 pin them). Only the item lists `data_full.json` and `data.json` are included, as noted in the licence line above.
- Cached feature files (the `features/` folders). They are large and are rebuilt by the scripts, so they are omitted.

## 9. Known issues

**Text-hint feature indexing (Experiments 1, 4 and 5).** The hint features are saved with axes `[item, hint, choice]` (`prompts()` loops over hints, then choices, and the array is reshaped to `(n, 4, 4, D)`). The text-hint arm reads them as `[np.arange(n), :, hy, :]`, which fixes the *choice* at the source's predicted answer `hy` and varies the *hint*. The intended read is `[np.arange(n), hy, :, :]` (fixed hint, all four choices). The affected lines are:

- `kaggle_experiment.py`: 301 (training) and 304 (test)
- `exp4_predictor_study.py`: 295 and 298
- `exp5_matched_supervision.py`: 276, 277 and 283

**Status.** The scripts in this archive are the original versions that produced every result in the paper, unchanged. The results were **not** re-run with the corrected index, so the effect on the text-hint results is unknown. This affects the text-hint rows of Experiments 1, 4 and 5 and the analyses built on them (the paper's Tables IV, V, VIII, IX, XII and XIII and Figures 1, 4 and 5). The other mechanisms (knowledge distillation and the three bridges) and Experiments 2, 3 and 3b do not use these arrays. One result is consistent with the problem: in Experiment 5's no-label regime the text-hint arm matches the source's accuracy in every pair (Section 5-K of the paper).

**Corrected scripts.** The `patched/` folder holds copies of the three scripts with the index corrected (seven lines) and with the `DRY_RUN` synthetic hint features changed to the same axis order. Nothing else differs. They were compiled and smoke-tested in dry-run mode only and have not been used to produce any reported result. `patched/compare_text_hint.py` compares results before and after the fix. To re-run, rename the existing results file (for example `exp/results.csv` to `exp/results_old.csv`), run the patched script on the cached features, and compare the two files.

## 10. Known limitations

These are described in the paper and are not repeated in full here: small models (0.135–1.7B), mean-pooled features at one fixed depth, four-choice question answering, limited seeds, and non-independent pairs that share models. See the paper's limitations section.
