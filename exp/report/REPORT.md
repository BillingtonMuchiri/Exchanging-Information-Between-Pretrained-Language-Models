# Cross-model compatibility benchmark: auto-generated report

### T1_accuracy_by_mechanism_task

| mech              | arce                | arcc                | sciq                |
|:------------------|:--------------------|:--------------------|:--------------------|
| source_alone      | 0.642 [0.591,0.690] | 0.368 [0.332,0.402] | 0.677 [0.619,0.730] |
| receiver_alone    | 0.642 [0.589,0.690] | 0.368 [0.331,0.401] | 0.677 [0.618,0.730] |
| ensemble          | 0.704 [0.676,0.732] | 0.393 [0.374,0.412] | 0.730 [0.701,0.759] |
| fusion_concat     | 0.708 [0.683,0.731] | 0.400 [0.384,0.416] | 0.732 [0.709,0.753] |
| text_hint         | 0.692 [0.667,0.718] | 0.407 [0.391,0.421] | 0.733 [0.706,0.757] |
| kd                | 0.586 [0.548,0.627] | 0.332 [0.308,0.358] | 0.630 [0.587,0.675] |
| bridge_ridge      | 0.515 [0.469,0.561] | 0.313 [0.293,0.335] | 0.548 [0.501,0.599] |
| bridge_procrustes | 0.473 [0.444,0.502] | 0.315 [0.298,0.333] | 0.518 [0.477,0.561] |
| bridge_nonlinear  | 0.396 [0.370,0.424] | 0.289 [0.278,0.300] | 0.421 [0.387,0.459] |

### T2_fidelity_arce

|                   |   retention |   gain_over_receiver |
|:------------------|------------:|---------------------:|
| text_hint         |      1.0949 |               0.0508 |
| kd                |      0.9296 |              -0.0552 |
| bridge_ridge      |      0.8044 |              -0.127  |
| bridge_procrustes |      0.7418 |              -0.169  |
| bridge_nonlinear  |      0.6214 |              -0.2452 |

### T3_calibration_robustness_cost

| mech              |    ece |    nll |   fit_s |          params |   shift_drop_arcc |   shift_drop_sciq |
|:------------------|-------:|-------:|--------:|----------------:|------------------:|------------------:|
| source_alone      | 0.1268 | 0.977  |  0      |  1232           |            0.2734 |           -0.0353 |
| receiver_alone    | 0.1268 | 0.977  |  0      |  1232           |            0.2734 |           -0.0353 |
| ensemble          | 0.1985 | 0.915  |  0      |  2464           |            0.3109 |           -0.0256 |
| fusion_concat     | 0.1213 | 0.852  |  1.5119 |  2464           |            0.3089 |           -0.0235 |
| text_hint         | 0.0861 | 0.8799 |  1.2337 |  1232           |            0.2855 |           -0.0406 |
| kd                | 0.1923 | 1.1374 |  0.11   |  1232           |            0.2541 |           -0.0439 |
| bridge_ridge      | 0.1125 | 1.2139 |  1.3768 |     1.44316e+06 |            0.2013 |           -0.0338 |
| bridge_procrustes | 0.1028 | 1.2468 |  4.4454 | 65536           |            0.1574 |           -0.0453 |
| bridge_nonlinear  | 0.1157 | 1.3999 |  3.2706 |     1.26157e+06 |            0.1073 |           -0.0245 |

### H1_winner_counts

|           |   n_pairs_won |
|:----------|--------------:|
| text_hint |            12 |

**H1** Kendall's W across 12 ordered pairs = 0.943 (chi-square p = 3.499e-09); distinct winners = 1. W near 1 = identical rankings in every pair; low W / several winners supports H1 (rankings change across pairs).

### H2_spearman

| mech              | predictor     |   spearman |      p |
|:------------------|:--------------|-----------:|-------:|
| text_hint         | mismatch      |    -0.2    | 0.2423 |
| text_hint         | vocab_jaccard |     0.036  | 0.8347 |
| text_hint         | dim_ratio     |    -0.0426 | 0.805  |
| kd                | mismatch      |    -0.0225 | 0.8965 |
| kd                | vocab_jaccard |    -0.0454 | 0.7926 |
| kd                | dim_ratio     |    -0.0062 | 0.9714 |
| bridge_ridge      | mismatch      |     0.3974 | 0.0164 |
| bridge_ridge      | vocab_jaccard |    -0.4643 | 0.0043 |
| bridge_ridge      | dim_ratio     |     0.2789 | 0.0995 |
| bridge_procrustes | mismatch      |     0.0755 | 0.6615 |
| bridge_procrustes | vocab_jaccard |     0.148  | 0.3891 |
| bridge_procrustes | dim_ratio     |    -0.1685 | 0.326  |
| bridge_nonlinear  | mismatch      |     0.1479 | 0.3892 |
| bridge_nonlinear  | vocab_jaccard |     0.0297 | 0.8633 |
| bridge_nonlinear  | dim_ratio     |     0.055  | 0.75   |

### H2_mixed_model (retention ~ mech*mismatch + (1|pair))
```
                     Mixed Linear Model Regression Results
================================================================================
Model:                     MixedLM         Dependent Variable:         retention
No. Observations:          180             Method:                     REML     
No. Groups:                12              Scale:                      0.0041   
Min. group size:           15              Log-Likelihood:             212.2563 
Max. group size:           15              Converged:                  Yes      
Mean group size:           15.0                                                 
--------------------------------------------------------------------------------
                                      Coef.  Std.Err.   z    P>|z| [0.025 0.975]
--------------------------------------------------------------------------------
Intercept                              0.606    0.057 10.539 0.000  0.493  0.718
C(mech)[T.bridge_procrustes]           0.128    0.032  4.060 0.000  0.066  0.190
C(mech)[T.bridge_ridge]                0.145    0.032  4.573 0.000  0.083  0.207
C(mech)[T.kd]                          0.313    0.032  9.906 0.000  0.251  0.375
C(mech)[T.text_hint]                   0.570    0.032 18.037 0.000  0.508  0.632
mismatch                               0.570    1.818  0.313 0.754 -2.993  4.133
C(mech)[T.bridge_procrustes]:mismatch -0.287    1.000 -0.287 0.774 -2.247  1.673
C(mech)[T.bridge_ridge]:mismatch       1.385    1.000  1.385 0.166 -0.575  3.345
C(mech)[T.kd]:mismatch                -0.176    1.000 -0.176 0.860 -2.136  1.784
C(mech)[T.text_hint]:mismatch         -3.478    1.000 -3.478 0.001 -5.438 -1.518
Group Var                              0.008    0.057                           
================================================================================

```

### H3_bridge_vs_text

| mech              |   acc_diff_vs_text |   ci_lo |   ci_hi |   fit_seconds_diff |
|:------------------|-------------------:|--------:|--------:|-------------------:|
| bridge_ridge      |            -0.1778 | -0.2152 | -0.1412 |             0.1431 |
| bridge_procrustes |            -0.2198 | -0.2374 | -0.204  |             3.2117 |
| bridge_nonlinear  |            -0.296  | -0.3152 | -0.2797 |             2.0369 |
| kd                |            -0.106  | -0.1423 | -0.0698 |            -1.1237 |

**Alignment vs fidelity**: Spearman(R2 of bridge, retention) = 0.400 (p = 1.78e-05).