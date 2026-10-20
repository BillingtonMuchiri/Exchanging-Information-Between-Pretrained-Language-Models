# Experiment 3: trained projector with controls

### T6_accuracy_by_arm

| arm         | arce                | arcc                | sciq_mcq            | sciq_gen            |
|:------------|:--------------------|:--------------------|:--------------------|:--------------------|
| zs_source   | 0.344 [0.344,0.344] | 0.344 [0.344,0.344] | 0.281 [0.281,0.281] | 0.333 [0.333,0.333] |
| zs_receiver | 0.719 [0.719,0.719] | 0.688 [0.688,0.688] | 0.656 [0.656,0.656] | 0.042 [0.042,0.042] |
| prompt_only | 0.719 [0.719,0.719] | 0.719 [0.719,0.719] | 0.781 [0.781,0.781] | 0.417 [0.417,0.417] |
| proj_shuf   | 0.688 [0.688,0.688] | 0.688 [0.688,0.688] | 0.750 [0.750,0.750] | 0.333 [0.333,0.333] |
| proj        | 0.750 [0.750,0.750] | 0.719 [0.719,0.719] | 0.688 [0.688,0.688] | 0.083 [0.083,0.083] |
| text_hint   | 0.656 [0.656,0.656] | 0.500 [0.500,0.500] | 0.750 [0.750,0.750] | 0.292 [0.292,0.292] |

### T7_contrasts_G1_G2

| contrast                | task     |    diff |   ci_lo |   ci_hi |   pairs_positive |   n_pairs |
|:------------------------|:---------|--------:|--------:|--------:|-----------------:|----------:|
| proj - prompt_only      | arce     |  0.0312 |  0.0312 |  0.0312 |                1 |         1 |
| proj - proj_shuf        | arce     |  0.0625 |  0.0625 |  0.0625 |                1 |         1 |
| proj - text_hint        | arce     |  0.0938 |  0.0938 |  0.0938 |                1 |         1 |
| text_hint - prompt_only | arce     | -0.0625 | -0.0625 | -0.0625 |                0 |         1 |
| proj_shuf - prompt_only | arce     | -0.0312 | -0.0312 | -0.0312 |                0 |         1 |
| proj - prompt_only      | arcc     |  0      |  0      |  0      |                0 |         1 |
| proj - proj_shuf        | arcc     |  0.0312 |  0.0312 |  0.0312 |                1 |         1 |
| proj - text_hint        | arcc     |  0.2188 |  0.2188 |  0.2188 |                1 |         1 |
| text_hint - prompt_only | arcc     | -0.2188 | -0.2188 | -0.2188 |                0 |         1 |
| proj_shuf - prompt_only | arcc     | -0.0312 | -0.0312 | -0.0312 |                0 |         1 |
| proj - prompt_only      | sciq_mcq | -0.0938 | -0.0938 | -0.0938 |                0 |         1 |
| proj - proj_shuf        | sciq_mcq | -0.0625 | -0.0625 | -0.0625 |                0 |         1 |
| proj - text_hint        | sciq_mcq | -0.0625 | -0.0625 | -0.0625 |                0 |         1 |
| text_hint - prompt_only | sciq_mcq | -0.0312 | -0.0312 | -0.0312 |                0 |         1 |
| proj_shuf - prompt_only | sciq_mcq | -0.0312 | -0.0312 | -0.0312 |                0 |         1 |
| proj - prompt_only      | sciq_gen | -0.3333 | -0.3333 | -0.3333 |                0 |         1 |
| proj - proj_shuf        | sciq_gen | -0.25   | -0.25   | -0.25   |                0 |         1 |
| proj - text_hint        | sciq_gen | -0.2083 | -0.2083 | -0.2083 |                0 |         1 |
| text_hint - prompt_only | sciq_gen | -0.125  | -0.125  | -0.125  |                0 |         1 |
| proj_shuf - prompt_only | sciq_gen | -0.0833 | -0.0833 | -0.0833 |                0 |         1 |

### T8_G3_G4_spearman

|    | note           |
|---:|:---------------|
|  0 | too few points |

Mismatch (1 - CKA) range across the 1 pairs: 0.020 to 0.020. (Experiment 1: 0.011 to 0.049.)

(mixed model skipped: Singular matrix)