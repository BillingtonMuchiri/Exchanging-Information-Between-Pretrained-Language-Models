# Cross-model compatibility benchmark: auto-generated report

### T1_accuracy_by_mechanism_task

| mech              | arce                | arcc                | sciq                |
|:------------------|:--------------------|:--------------------|:--------------------|
| source_alone      | 0.642 [0.591,0.690] | 0.368 [0.332,0.402] | 0.677 [0.619,0.730] |
| receiver_alone    | 0.642 [0.589,0.690] | 0.368 [0.331,0.401] | 0.677 [0.618,0.730] |
| ensemble          | 0.704 [0.676,0.732] | 0.393 [0.374,0.412] | 0.730 [0.701,0.759] |
| fusion_concat     | 0.708 [0.683,0.731] | 0.400 [0.384,0.416] | 0.732 [0.709,0.753] |
| text_hint         | 0.666 [0.632,0.699] | 0.388 [0.366,0.408] | 0.705 [0.669,0.739] |
| kd                | 0.586 [0.548,0.627] | 0.332 [0.308,0.358] | 0.630 [0.587,0.675] |
| bridge_ridge      | 0.515 [0.469,0.561] | 0.313 [0.293,0.335] | 0.548 [0.501,0.599] |
| bridge_procrustes | 0.473 [0.444,0.502] | 0.315 [0.298,0.333] | 0.518 [0.477,0.561] |
| bridge_nonlinear  | 0.396 [0.370,0.424] | 0.289 [0.278,0.300] | 0.421 [0.387,0.459] |

### T2_fidelity_arce

|                   |   retention |   gain_over_receiver |
|:------------------|------------:|---------------------:|
| text_hint         |      1.0477 |               0.0243 |
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
| fusion_concat     | 0.1213 | 0.852  |  1.3466 |  2464           |            0.3089 |           -0.0235 |
| text_hint         | 0.0886 | 0.9541 |  1.1705 |  1232           |            0.2783 |           -0.0385 |
| kd                | 0.1923 | 1.1374 |  0.1017 |  1232           |            0.2541 |           -0.0439 |
| bridge_ridge      | 0.1125 | 1.2139 |  1.2171 |     1.44316e+06 |            0.2013 |           -0.0338 |
| bridge_procrustes | 0.1028 | 1.2468 |  3.9462 | 65536           |            0.1574 |           -0.0453 |
| bridge_nonlinear  | 0.1157 | 1.3999 |  2.9092 |     1.26157e+06 |            0.1073 |           -0.0245 |

### H1_winner_counts

|           |   n_pairs_won |
|:----------|--------------:|
| text_hint |            10 |
| kd        |             2 |

**H1** Kendall's W across 12 ordered pairs = 0.912 (chi-square p = 7.06e-09); distinct winners = 2. W near 1 = identical rankings in every pair; low W / several winners supports H1 (rankings change across pairs).

### H2_spearman

| mech              | predictor     |   spearman |      p |
|:------------------|:--------------|-----------:|-------:|
| text_hint         | mismatch      |    -0.2986 | 0.0769 |
| text_hint         | vocab_jaccard |     0.1843 | 0.282  |
| text_hint         | dim_ratio     |     0.1415 | 0.4106 |
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
No. Groups:                12              Scale:                      0.0037   
Min. group size:           15              Log-Likelihood:             221.4530 
Max. group size:           15              Converged:                  Yes      
Mean group size:           15.0                                                 
--------------------------------------------------------------------------------
                                      Coef.  Std.Err.   z    P>|z| [0.025 0.975]
--------------------------------------------------------------------------------
Intercept                              0.606    0.052 11.688 0.000  0.504  0.707
C(mech)[T.bridge_procrustes]           0.128    0.030  4.271 0.000  0.069  0.187
C(mech)[T.bridge_ridge]                0.145    0.030  4.810 0.000  0.086  0.203
C(mech)[T.kd]                          0.313    0.030 10.420 0.000  0.254  0.372
C(mech)[T.text_hint]                   0.498    0.030 16.569 0.000  0.439  0.557
mismatch                               0.570    1.639  0.348 0.728 -2.643  3.783
C(mech)[T.bridge_procrustes]:mismatch -0.287    0.951 -0.302 0.763 -2.150  1.576
C(mech)[T.bridge_ridge]:mismatch       1.385    0.951  1.457 0.145 -0.478  3.248
C(mech)[T.kd]:mismatch                -0.176    0.951 -0.185 0.853 -2.040  1.687
C(mech)[T.text_hint]:mismatch         -2.577    0.951 -2.711 0.007 -4.441 -0.714
Group Var                              0.006    0.048                           
================================================================================

```

### H3_bridge_vs_text

| mech              |   acc_diff_vs_text |   ci_lo |   ci_hi |   fit_seconds_diff |
|:------------------|-------------------:|--------:|--------:|-------------------:|
| bridge_ridge      |            -0.1514 | -0.1868 | -0.1161 |             0.0465 |
| bridge_procrustes |            -0.1933 | -0.2091 | -0.1792 |             2.7757 |
| bridge_nonlinear  |            -0.2696 | -0.2867 | -0.252  |             1.7387 |
| kd                |            -0.0795 | -0.1261 | -0.0351 |            -1.0688 |

**Alignment vs fidelity**: Spearman(R2 of bridge, retention) = 0.400 (p = 1.78e-05).