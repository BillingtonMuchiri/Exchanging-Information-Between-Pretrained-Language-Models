# Experiment 5: matched supervision

### T13_mean_accuracy_by_arm_and_task

| arm               |   arcc |   arce |   sciq |
|:------------------|-------:|-------:|-------:|
| receiver_gold     | 0.4038 | 0.6739 | 0.6944 |
| text_hint_gold    | 0.4534 | 0.7251 | 0.7475 |
| kd_gold           | 0.3868 | 0.6664 | 0.6917 |
| bridge_gold       | 0.328  | 0.5453 | 0.5855 |
| receiver_nolabel  | 0.3488 | 0.6076 | 0.6406 |
| text_hint_nolabel | 0.4036 | 0.6739 | 0.694  |
| kd_nolabel        | 0.3432 | 0.6071 | 0.6401 |
| bridge_nolabel    | 0.3217 | 0.5327 | 0.568  |

### T14_contrasts_M1_M2_M3

| contrast                  |    mean |   ordered_lo |   ordered_hi |   unordered_lo |   unordered_hi |   model_lo |   model_hi |   pairs_positive |   n_pairs |
|:--------------------------|--------:|-------------:|-------------:|---------------:|---------------:|-----------:|-----------:|-----------------:|----------:|
| gain_text_hint_gold       |  0.0513 |       0.0193 |       0.0881 |         0.0349 |         0.0698 |     0.0176 |     0.087  |               28 |        56 |
| gain_kd_gold              | -0.0074 |      -0.0112 |      -0.0035 |        -0.0106 |        -0.0044 |    -0.0139 |    -0.0008 |               18 |        56 |
| gain_bridge_gold          | -0.1285 |      -0.1659 |      -0.0902 |        -0.1448 |        -0.1135 |    -0.1595 |    -0.0989 |               11 |        56 |
| text_minus_kd_gold        |  0.0587 |       0.0276 |       0.0936 |         0.04   |         0.08   |     0.02   |     0.1004 |               29 |        56 |
| text_minus_bridge_gold    |  0.1798 |       0.1559 |       0.2057 |         0.1496 |         0.2144 |     0.1218 |     0.2447 |               56 |        56 |
| gain_text_hint_nolabel    |  0.0663 |       0.0327 |       0.1042 |         0.0557 |         0.0788 |     0.0427 |     0.0926 |               32 |        56 |
| gain_kd_nolabel           | -0.0005 |      -0.0072 |       0.0057 |        -0.0067 |         0.0054 |    -0.0115 |     0.0116 |               34 |        56 |
| gain_bridge_nolabel       | -0.0749 |      -0.0988 |      -0.0501 |        -0.085  |        -0.0654 |    -0.094  |    -0.0551 |               15 |        56 |
| text_minus_kd_nolabel     |  0.0668 |       0.0356 |       0.102  |         0.0535 |         0.0821 |     0.0401 |     0.0983 |               31 |        56 |
| text_minus_bridge_nolabel |  0.1411 |       0.1244 |       0.1603 |         0.1278 |         0.1556 |     0.1171 |     0.1674 |               56 |        56 |

Mean-accuracy ranking of the three channels (1 = best): gold {'text_hint_gold': 1.0, 'kd_gold': 2.0, 'bridge_gold': 3.0}; no-label {'text_hint_nolabel': 1.0, 'kd_nolabel': 2.0, 'bridge_nolabel': 3.0}.