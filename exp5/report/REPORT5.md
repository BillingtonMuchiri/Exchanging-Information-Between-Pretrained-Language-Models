# Experiment 5: matched supervision

### T13_mean_accuracy_by_arm_and_task

| arm               |   arcc |   arce |   sciq |
|:------------------|-------:|-------:|-------:|
| receiver_gold     | 0.4038 | 0.6739 | 0.6944 |
| text_hint_gold    | 0.473  | 0.7461 | 0.7672 |
| kd_gold           | 0.3868 | 0.6664 | 0.6917 |
| bridge_gold       | 0.328  | 0.5453 | 0.5855 |
| receiver_nolabel  | 0.3488 | 0.6076 | 0.6406 |
| text_hint_nolabel | 0.4037 | 0.6739 | 0.6943 |
| kd_nolabel        | 0.3432 | 0.6071 | 0.6401 |
| bridge_nolabel    | 0.3217 | 0.5327 | 0.568  |

### T14_contrasts_M1_M2_M3

| contrast                  |    mean |   ordered_lo |   ordered_hi |   unordered_lo |   unordered_hi |   model_lo |   model_hi |   pairs_positive |   n_pairs |
|:--------------------------|--------:|-------------:|-------------:|---------------:|---------------:|-----------:|-----------:|-----------------:|----------:|
| gain_text_hint_gold       |  0.0722 |       0.0445 |       0.1043 |         0.0557 |         0.0905 |     0.0401 |     0.1058 |               35 |        56 |
| gain_kd_gold              | -0.0074 |      -0.0112 |      -0.0035 |        -0.0106 |        -0.0044 |    -0.0139 |    -0.0008 |               18 |        56 |
| gain_bridge_gold          | -0.1285 |      -0.1659 |      -0.0902 |        -0.1448 |        -0.1135 |    -0.1595 |    -0.0989 |               11 |        56 |
| text_minus_kd_gold        |  0.0796 |       0.0528 |       0.1112 |         0.0607 |         0.1005 |     0.042  |     0.1186 |               41 |        56 |
| text_minus_bridge_gold    |  0.2007 |       0.1759 |       0.2268 |         0.1704 |         0.2344 |     0.1431 |     0.2623 |               56 |        56 |
| gain_text_hint_nolabel    |  0.0663 |       0.0327 |       0.1043 |         0.0557 |         0.0788 |     0.0427 |     0.0925 |               32 |        56 |
| gain_kd_nolabel           | -0.0005 |      -0.0072 |       0.0057 |        -0.0067 |         0.0054 |    -0.0115 |     0.0116 |               34 |        56 |
| gain_bridge_nolabel       | -0.0749 |      -0.0988 |      -0.0501 |        -0.085  |        -0.0654 |    -0.094  |    -0.0551 |               15 |        56 |
| text_minus_kd_nolabel     |  0.0668 |       0.0355 |       0.1021 |         0.0535 |         0.082  |     0.0401 |     0.0982 |               31 |        56 |
| text_minus_bridge_nolabel |  0.1411 |       0.1244 |       0.1603 |         0.1277 |         0.1556 |     0.1168 |     0.1675 |               56 |        56 |

Mean-accuracy ranking of the three channels (1 = best): gold {'text_hint_gold': 1.0, 'kd_gold': 2.0, 'bridge_gold': 3.0}; no-label {'text_hint_nolabel': 1.0, 'kd_nolabel': 2.0, 'bridge_nolabel': 3.0}.