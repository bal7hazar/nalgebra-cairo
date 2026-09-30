# Packages

Generated file, do not edit by hand. Run: 2026-09-30, runner: ubuntu-latest, shards of 4 vCPU / 15 GB, 5 cold build round(s) per consumer (medians of the per-round differences). Regenerate: run `python3 scripts/consumer_cost.py --repeat 5 --interleave --json consumer_cost.json` (CI: the `Consumer cost` job uploads `consumer_cost.json` and this table as the artifact `consumer-cost`), then `python3 scripts/packages_table.py consumer_cost.json --runner RUNNER --output docs/PACKAGES.md`.

Gates: at most 40,000 library lines; marginal cost at most 5 s / 1 GB (gate 2); closures 15 s / 3 GB unless they declare a budget (gate 3). Each figure is `value / limit (margin)`; the margin is the room left below the limit, negative when over it.

## Published packages

| package | version | lines | marginal time | marginal memory | verdict |
|---|---|---:|---:|---:|---|
| nalgebra (reported) | 0.1.0 | 13,191 (gate 3 only) | closure 29.8 s / 15 s (-99 %) | closure 9.96 GB / 3 GB (-232 %) | reported (FAIL: closure time > 15 s, closure memory > 3 GB) |
| nalgebra_blas | 0.1.0 | 12,105 / 40,000 (+70 %) | 1.4 s / 5 s (+72 %) | 0.36 GB / 1 GB (+64 %) | ok |
| nalgebra_blocks | 0.1.0 | 29,129 / 40,000 (+27 %) | 1.2 s / 5 s (+76 %) | 0.43 GB / 1 GB (+57 %) | ok |
| nalgebra_core | 0.1.0 | 2,371 / 40,000 (+94 %) | 0.5 s / 5 s (+91 %) | 0.04 GB / 1 GB (+96 %) | ok |
| nalgebra_dynamic | 0.1.0 | 10,442 / 40,000 (+74 %) | 2.6 s / 5 s (+47 %) | 0.32 GB / 1 GB (+68 %) | ok |
| nalgebra_geometry2 | 0.1.0 | 6,350 / 40,000 (+84 %) | 0.6 s / 5 s (+89 %) | 0.13 GB / 1 GB (+87 %) | ok |
| nalgebra_geometry3 | 0.1.0 | 10,400 / 40,000 (+74 %) | 0.8 s / 5 s (+85 %) | 0.23 GB / 1 GB (+77 %) | ok |
| nalgebra_geometry4 | 0.1.0 | 1,419 / 40,000 (+96 %) | 0.2 s / 5 s (+96 %) | 0.03 GB / 1 GB (+97 %) | ok |
| nalgebra_geometry5 | 0.1.0 | 3,191 / 40,000 (+92 %) | 0.3 s / 5 s (+94 %) | 0.08 GB / 1 GB (+92 %) | ok |
| nalgebra_geometry6 | 0.1.0 | 2,750 / 40,000 (+93 %) | 0.2 s / 5 s (+96 %) | 0.08 GB / 1 GB (+92 %) | ok |
| nalgebra_glam | 0.1.0 | 1,304 / 40,000 (+97 %) | 0.2 s / 5 s (+96 %) | 0.02 GB / 1 GB (+98 %) | ok |
| nalgebra_linalg2 | 0.1.0 | 1,996 / 40,000 (+95 %) | 0.2 s / 5 s (+96 %) | 0.04 GB / 1 GB (+96 %) | ok |
| nalgebra_linalg3 | 0.1.0 | 2,872 / 40,000 (+93 %) | 0.3 s / 5 s (+94 %) | 0.10 GB / 1 GB (+90 %) | ok |
| nalgebra_linalg4 | 0.1.0 | 4,549 / 40,000 (+89 %) | 0.7 s / 5 s (+86 %) | 0.16 GB / 1 GB (+84 %) | ok |
| nalgebra_linalg5 | 0.1.0 | 3,901 / 40,000 (+90 %) | 0.4 s / 5 s (+93 %) | 0.09 GB / 1 GB (+91 %) | ok |
| nalgebra_linalg6 | 0.1.0 | 8,735 / 40,000 (+78 %) | 0.6 s / 5 s (+87 %) | 0.18 GB / 1 GB (+82 %) | ok |
| nalgebra_linalg_core | 0.1.0 | 828 / 40,000 (+98 %) | 0.2 s / 5 s (+96 %) | 0.01 GB / 1 GB (+99 %) | ok |
| nalgebra_linalg_pivot2 | 0.1.0 | 2,249 / 40,000 (+94 %) | 0.4 s / 5 s (+92 %) | 0.05 GB / 1 GB (+95 %) | ok |
| nalgebra_linalg_pivot3 | 0.1.0 | 3,513 / 40,000 (+91 %) | 0.3 s / 5 s (+94 %) | 0.09 GB / 1 GB (+91 %) | ok |
| nalgebra_linalg_pivot4 | 0.1.0 | 7,000 / 40,000 (+82 %) | 0.5 s / 5 s (+90 %) | 0.13 GB / 1 GB (+87 %) | ok |
| nalgebra_linalg_pivot5 | 0.1.0 | 13,529 / 40,000 (+66 %) | 0.7 s / 5 s (+87 %) | 0.22 GB / 1 GB (+78 %) | ok |
| nalgebra_linalg_pivot6 | 0.1.0 | 24,674 / 40,000 (+38 %) | 1.3 s / 5 s (+74 %) | 0.37 GB / 1 GB (+63 %) | ok |
| nalgebra_linalg_spectral2 | 0.1.0 | 2,215 / 40,000 (+94 %) | 0.3 s / 5 s (+95 %) | 0.06 GB / 1 GB (+94 %) | ok |
| nalgebra_linalg_spectral3 | 0.1.0 | 3,378 / 40,000 (+92 %) | 0.5 s / 5 s (+90 %) | 0.10 GB / 1 GB (+90 %) | ok |
| nalgebra_linalg_spectral4 | 0.1.0 | 6,553 / 40,000 (+84 %) | 0.6 s / 5 s (+88 %) | 0.15 GB / 1 GB (+85 %) | ok |
| nalgebra_linalg_spectral5 | 0.1.0 | 13,293 / 40,000 (+67 %) | 0.9 s / 5 s (+81 %) | 0.29 GB / 1 GB (+71 %) | ok |
| nalgebra_linalg_spectral6 | 0.1.0 | 23,843 / 40,000 (+40 %) | 1.9 s / 5 s (+63 %) | 0.47 GB / 1 GB (+53 %) | ok |
| nalgebra_linalg_svd_eigen2 | 0.1.0 | 2,195 / 40,000 (+95 %) | 0.2 s / 5 s (+96 %) | 0.05 GB / 1 GB (+95 %) | ok |
| nalgebra_linalg_svd_eigen3 | 0.1.0 | 3,304 / 40,000 (+92 %) | 0.6 s / 5 s (+88 %) | 0.16 GB / 1 GB (+84 %) | ok |
| nalgebra_linalg_svd_eigen4 | 0.1.0 | 5,738 / 40,000 (+86 %) | 0.8 s / 5 s (+85 %) | 0.28 GB / 1 GB (+72 %) | ok |
| nalgebra_linalg_svd_eigen5 | 0.1.0 | 10,672 / 40,000 (+73 %) | 1.3 s / 5 s (+74 %) | 0.45 GB / 1 GB (+55 %) | ok |
| nalgebra_linalg_svd_eigen6 | 0.1.0 | 17,390 / 40,000 (+57 %) | 2.3 s / 5 s (+54 %) | 0.77 GB / 1 GB (+23 %) | ok |
| nalgebra_norm | 0.1.0 | 3,744 / 40,000 (+91 %) | 0.7 s / 5 s (+85 %) | 0.10 GB / 1 GB (+90 %) | ok |
| nalgebra_sparse | 0.1.0 | 3,485 / 40,000 (+91 %) | 0.6 s / 5 s (+88 %) | 0.10 GB / 1 GB (+90 %) | ok |
| nalgebra_static2 | 0.1.0 | 4,570 / 40,000 (+89 %) | 0.4 s / 5 s (+92 %) | 0.10 GB / 1 GB (+90 %) | ok |
| nalgebra_static3 | 0.1.0 | 9,267 / 40,000 (+77 %) | 0.8 s / 5 s (+83 %) | 0.20 GB / 1 GB (+80 %) | ok |
| nalgebra_static4 | 0.1.0 | 16,175 / 40,000 (+60 %) | 1.0 s / 5 s (+80 %) | 0.33 GB / 1 GB (+67 %) | ok |
| nalgebra_static5 | 0.1.0 | 26,820 / 40,000 (+33 %) | 2.6 s / 5 s (+49 %) | 0.51 GB / 1 GB (+49 %) | ok |
| nalgebra_static6_tall | 0.1.0 | 24,451 / 40,000 (+39 %) | 1.4 s / 5 s (+71 %) | 0.39 GB / 1 GB (+61 %) | ok |
| nalgebra_static6_wide | 0.1.0 | 31,481 / 40,000 (+21 %) | 1.4 s / 5 s (+71 %) | 0.50 GB / 1 GB (+50 %) | ok |
| nalgebra_static_core | 0.1.0 | 1,441 / 40,000 (+96 %) | 0.1 s / 5 s (+98 %) | 0.05 GB / 1 GB (+95 %) | ok |
| nalgebra_statistics2 | 0.1.0 | 605 / 40,000 (+98 %) | 0.2 s / 5 s (+95 %) | 0.01 GB / 1 GB (+99 %) | ok |
| nalgebra_statistics3 | 0.1.0 | 1,139 / 40,000 (+97 %) | 0.0 s / 5 s (+100 %) | 0.02 GB / 1 GB (+98 %) | ok |
| nalgebra_statistics4 | 0.1.0 | 2,228 / 40,000 (+94 %) | 0.2 s / 5 s (+96 %) | 0.03 GB / 1 GB (+97 %) | ok |
| nalgebra_statistics5 | 0.1.0 | 5,270 / 40,000 (+87 %) | 0.3 s / 5 s (+93 %) | 0.07 GB / 1 GB (+93 %) | ok |
| nalgebra_statistics6 | 0.1.0 | 9,201 / 40,000 (+77 %) | 0.3 s / 5 s (+94 %) | 0.11 GB / 1 GB (+89 %) | ok |
| nalgebra_transform2 | 0.1.0 | 3,495 / 40,000 (+91 %) | 0.4 s / 5 s (+93 %) | 0.08 GB / 1 GB (+92 %) | ok |
| nalgebra_transform3 | 0.1.0 | 3,945 / 40,000 (+90 %) | 0.5 s / 5 s (+90 %) | 0.09 GB / 1 GB (+91 %) | ok |
| nalgebra_types2 | 0.1.0 | 3,261 / 40,000 (+92 %) | 0.3 s / 5 s (+95 %) | 0.09 GB / 1 GB (+91 %) | ok |
| nalgebra_types3 | 0.1.0 | 6,477 / 40,000 (+84 %) | 0.4 s / 5 s (+92 %) | 0.14 GB / 1 GB (+86 %) | ok |
| nalgebra_types4 | 0.1.0 | 12,374 / 40,000 (+69 %) | 0.9 s / 5 s (+82 %) | 0.26 GB / 1 GB (+74 %) | ok |
| nalgebra_types5 | 0.1.0 | 24,933 / 40,000 (+38 %) | 1.0 s / 5 s (+79 %) | 0.47 GB / 1 GB (+53 %) | ok |
| nalgebra_types6 | 0.1.0 | 37,436 / 40,000 (+6 %) | 2.5 s / 5 s (+50 %) | 0.67 GB / 1 GB (+33 %) | ok |
| nalgebra_views | 0.1.0 | 23,233 / 40,000 (+42 %) | 0.7 s / 5 s (+85 %) | 0.31 GB / 1 GB (+69 %) | ok |

## Declared closures

| closure | members | time | memory | budget | verdict |
|---|---|---:|---:|---|---|
| core_pivot | nalgebra_linalg_pivot2, nalgebra_linalg_pivot3, nalgebra_linalg_pivot4 | 3.5 s / 15 s (+77 %) | 1.05 GB / 3 GB (+65 %) | 15 s / 3 GB | ok |
| facade_no_default_features (reported) | nalgebra | 26.7 s / 15 s (-78 %) | 6.29 GB / 3 GB (-110 %) | 15 s / 3 GB | reported (FAIL: time > 15 s, memory > 3 GB) |
| glam_0_4_1_alone (reported) | glam_core@0.4.1 | 1.8 s / 15 s (+88 %) | 0.50 GB / 3 GB (+83 %) | 15 s / 3 GB | reported (ok) |
| nalgebra_glam | nalgebra_glam, nalgebra_core, nalgebra_types2, nalgebra_types3, nalgebra_types4, nalgebra_static2, nalgebra_static3, nalgebra_geometry2, nalgebra_geometry3, glam_core@0.4.1 | 7.6 s / 15 s (+49 %) | 1.69 GB / 3 GB (+44 %) | 15 s / 3 GB | ok |
| nalgebra_with_glam (reported) | nalgebra, nalgebra_glam | 35.3 s / 15 s (-135 %) | 10.20 GB / 3 GB (-240 %) | 15 s / 3 GB | reported (FAIL: time > 15 s, memory > 3 GB) |
| pivot6 | nalgebra_linalg_pivot6 | 10.7 s / 20 s (+47 %) | 3.90 GB / 4.5 GB (+13 %) | 20 s / 4.5 GB | ok |
| static3_geometry | nalgebra_static3, nalgebra_geometry2, nalgebra_geometry3 | 6.1 s / 15 s (+59 %) | 1.43 GB / 3 GB (+52 %) | 15 s / 3 GB | ok |
| static3_svd | nalgebra_static3, nalgebra_linalg_svd_eigen2, nalgebra_linalg_svd_eigen3 | 5.5 s / 15 s (+63 %) | 1.21 GB / 3 GB (+60 %) | 15 s / 3 GB | ok |
| static4_blas | nalgebra_static4, nalgebra_blas | 11.7 s / 15 s (+22 %) | 2.83 GB / 3 GB (+6 %) | 15 s / 3 GB | ok |
| static4_blocks (reported) | nalgebra_static4, nalgebra_blocks | 13.8 s / 15 s (+8 %) | 3.63 GB / 3 GB (-21 %) | 15 s / 3 GB | reported (FAIL: memory > 3 GB) |
| static4_factor | nalgebra_static4, nalgebra_linalg2, nalgebra_linalg3, nalgebra_linalg4 | 8.3 s / 15 s (+44 %) | 2.00 GB / 3 GB (+33 %) | 15 s / 3 GB | ok |
| static4_geometry | nalgebra_static4, nalgebra_geometry2, nalgebra_geometry3, nalgebra_geometry4, nalgebra_transform2, nalgebra_transform3 | 10.4 s / 15 s (+31 %) | 2.36 GB / 3 GB (+21 %) | 15 s / 3 GB | ok |
| static4_norm (reported) | nalgebra_static4, nalgebra_norm | 9.5 s / 15 s (+36 %) | 3.72 GB / 3 GB (-24 %) | 15 s / 3 GB | reported (FAIL: memory > 3 GB) |
| static4_statistics | nalgebra_static4, nalgebra_statistics2, nalgebra_statistics3, nalgebra_statistics4 | 7.7 s / 15 s (+48 %) | 1.87 GB / 3 GB (+38 %) | 15 s / 3 GB | ok |
| static4_statistics_all | nalgebra_static4, nalgebra_statistics2, nalgebra_statistics3, nalgebra_statistics4, nalgebra_statistics5, nalgebra_statistics6 | 10.9 s / 15 s (+28 %) | 2.75 GB / 3 GB (+8 %) | 15 s / 3 GB | ok |
| static4_svd | nalgebra_static4, nalgebra_linalg_svd_eigen2, nalgebra_linalg_svd_eigen3, nalgebra_linalg_svd_eigen4 | 8.9 s / 15 s (+41 %) | 2.07 GB / 3 GB (+31 %) | 15 s / 3 GB | ok |
| static4_views (reported) | nalgebra_static4, nalgebra_views | 14.5 s / 15 s (+4 %) | 3.95 GB / 3 GB (-32 %) | 15 s / 3 GB | reported (FAIL: memory > 3 GB) |
| static5 | nalgebra_static5 | 11.8 s / 20 s (+41 %) | 2.94 GB / 4.5 GB (+35 %) | 20 s / 4.5 GB | ok |
| static5_factor | nalgebra_static5, nalgebra_linalg5 | 9.1 s / 20 s (+54 %) | 3.01 GB / 4.5 GB (+33 %) | 20 s / 4.5 GB | ok |
| static5_geometry | nalgebra_static5, nalgebra_geometry5 | 11.9 s / 20 s (+40 %) | 3.02 GB / 4.5 GB (+33 %) | 20 s / 4.5 GB | ok |
| static5_pivot | nalgebra_static5, nalgebra_linalg_pivot5 | 12.3 s / 20 s (+39 %) | 3.16 GB / 4.5 GB (+30 %) | 20 s / 4.5 GB | ok |
| static5_spectral | nalgebra_static5, nalgebra_linalg_spectral5 | 12.8 s / 20 s (+36 %) | 3.22 GB / 4.5 GB (+28 %) | 20 s / 4.5 GB | ok |
| static5_svd | nalgebra_static5, nalgebra_linalg_svd_eigen5 | 13.8 s / 20 s (+31 %) | 3.40 GB / 4.5 GB (+24 %) | 20 s / 4.5 GB | ok |
| static6 | nalgebra_static6_wide | 13.3 s / 20 s (+33 %) | 3.63 GB / 4.5 GB (+19 %) | 20 s / 4.5 GB | ok |
| static6_factor | nalgebra_static6_wide, nalgebra_linalg6 | 14.2 s / 20 s (+29 %) | 3.80 GB / 4.5 GB (+15 %) | 20 s / 4.5 GB | ok |
| static6_geometry | nalgebra_static6_wide, nalgebra_geometry6 | 13.9 s / 20 s (+31 %) | 3.70 GB / 4.5 GB (+18 %) | 20 s / 4.5 GB | ok |
| static6_pivot | nalgebra_static6_wide, nalgebra_linalg_pivot6 | 14.8 s / 20 s (+26 %) | 3.99 GB / 4.5 GB (+11 %) | 20 s / 4.5 GB | ok |
| static6_spectral (reported) | nalgebra_static6_wide, nalgebra_linalg_spectral6 | 15.2 s / 20 s (+24 %) | 4.13 GB / 4.5 GB (+8 %) | 20 s / 4.5 GB | reported (ok) |
| static6_svd (reported) | nalgebra_static6_wide, nalgebra_linalg_svd_eigen6 | 11.6 s / 20 s (+42 %) | 4.38 GB / 4.5 GB (+3 %) | 20 s / 4.5 GB | reported (ok) |
| svd_eigen6 | nalgebra_linalg_svd_eigen6 | 8.8 s / 20 s (+56 %) | 2.92 GB / 4.5 GB (+35 %) | 20 s / 4.5 GB | ok |
