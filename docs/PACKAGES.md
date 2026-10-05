# Packages

Generated file, do not edit by hand. Run: 2026-10-05, runner: ubuntu-latest, shards of 4 vCPU / 15 GB, 5 cold build round(s) per consumer (medians of the per-round differences). Regenerate: run `python3 scripts/consumer_cost.py --repeat 5 --interleave --json consumer_cost.json` (CI: the `Consumer cost` job uploads `consumer_cost.json` and this table as the artifact `consumer-cost`), then `python3 scripts/packages_table.py consumer_cost.json --runner RUNNER --output docs/PACKAGES.md`.

Gates: at most 40,000 library lines; marginal cost at most 5 s / 1 GB (gate 2); closures 15 s / 3 GB unless they declare a budget (gate 3). Each figure is `value / limit (margin)`; the margin is the room left below the limit, negative when over it.

## Published packages

| package | version | lines | marginal time | marginal memory | verdict |
|---|---|---:|---:|---:|---|
| nalgebra (reported) | 0.3.0 | 13,192 (gate 3 only) | closure 30.6 s / 15 s (-104 %) | closure 8.90 GB / 3 GB (-197 %) | reported (FAIL: closure time > 15 s, closure memory > 3 GB) |
| nalgebra_blas | 0.3.0 | 12,105 / 40,000 (+70 %) | 1.4 s / 5 s (+72 %) | 0.31 GB / 1 GB (+69 %) | ok |
| nalgebra_blocks | 0.3.0 | 29,129 / 40,000 (+27 %) | 1.2 s / 5 s (+77 %) | 0.39 GB / 1 GB (+61 %) | ok |
| nalgebra_core | 0.3.0 | 2,371 / 40,000 (+94 %) | 0.7 s / 5 s (+86 %) | 0.04 GB / 1 GB (+96 %) | ok |
| nalgebra_dynamic | 0.3.0 | 10,442 / 40,000 (+74 %) | 1.6 s / 5 s (+68 %) | 0.28 GB / 1 GB (+72 %) | ok |
| nalgebra_geometry2 | 0.3.0 | 6,368 / 40,000 (+84 %) | 0.4 s / 5 s (+92 %) | 0.12 GB / 1 GB (+88 %) | ok |
| nalgebra_geometry3 | 0.3.0 | 10,448 / 40,000 (+74 %) | 0.7 s / 5 s (+86 %) | 0.20 GB / 1 GB (+80 %) | ok |
| nalgebra_geometry4 | 0.3.0 | 1,419 / 40,000 (+96 %) | 0.1 s / 5 s (+97 %) | 0.04 GB / 1 GB (+96 %) | ok |
| nalgebra_geometry5 | 0.3.0 | 3,191 / 40,000 (+92 %) | 0.3 s / 5 s (+93 %) | 0.09 GB / 1 GB (+91 %) | ok |
| nalgebra_geometry6 | 0.3.0 | 2,750 / 40,000 (+93 %) | 0.3 s / 5 s (+95 %) | 0.08 GB / 1 GB (+92 %) | ok |
| nalgebra_glam | 0.3.0 | 1,304 / 40,000 (+97 %) | -0.0 s / 5 s (+101 %) | 0.02 GB / 1 GB (+98 %) | ok |
| nalgebra_linalg2 | 0.3.0 | 1,996 / 40,000 (+95 %) | 0.1 s / 5 s (+98 %) | 0.04 GB / 1 GB (+96 %) | ok |
| nalgebra_linalg3 | 0.3.0 | 2,877 / 40,000 (+93 %) | 0.4 s / 5 s (+93 %) | 0.09 GB / 1 GB (+91 %) | ok |
| nalgebra_linalg4 | 0.3.0 | 4,549 / 40,000 (+89 %) | 0.6 s / 5 s (+88 %) | 0.14 GB / 1 GB (+86 %) | ok |
| nalgebra_linalg5 | 0.3.0 | 3,901 / 40,000 (+90 %) | 0.1 s / 5 s (+98 %) | 0.07 GB / 1 GB (+93 %) | ok |
| nalgebra_linalg6 | 0.3.0 | 8,730 / 40,000 (+78 %) | 0.5 s / 5 s (+90 %) | 0.17 GB / 1 GB (+83 %) | ok |
| nalgebra_linalg_core | 0.3.0 | 828 / 40,000 (+98 %) | 0.3 s / 5 s (+94 %) | 0.02 GB / 1 GB (+98 %) | ok |
| nalgebra_linalg_pivot2 | 0.3.0 | 2,249 / 40,000 (+94 %) | 0.3 s / 5 s (+95 %) | 0.04 GB / 1 GB (+96 %) | ok |
| nalgebra_linalg_pivot3 | 0.3.0 | 3,513 / 40,000 (+91 %) | 0.2 s / 5 s (+96 %) | 0.08 GB / 1 GB (+92 %) | ok |
| nalgebra_linalg_pivot4 | 0.3.0 | 7,000 / 40,000 (+82 %) | 0.2 s / 5 s (+96 %) | 0.10 GB / 1 GB (+90 %) | ok |
| nalgebra_linalg_pivot5 | 0.3.0 | 13,529 / 40,000 (+66 %) | 0.7 s / 5 s (+87 %) | 0.19 GB / 1 GB (+81 %) | ok |
| nalgebra_linalg_pivot6 | 0.3.0 | 24,674 / 40,000 (+38 %) | 0.8 s / 5 s (+85 %) | 0.32 GB / 1 GB (+68 %) | ok |
| nalgebra_linalg_spectral2 | 0.3.0 | 2,215 / 40,000 (+94 %) | 0.1 s / 5 s (+97 %) | 0.06 GB / 1 GB (+94 %) | ok |
| nalgebra_linalg_spectral3 | 0.3.0 | 3,378 / 40,000 (+92 %) | 0.4 s / 5 s (+92 %) | 0.08 GB / 1 GB (+92 %) | ok |
| nalgebra_linalg_spectral4 | 0.3.0 | 6,553 / 40,000 (+84 %) | 0.6 s / 5 s (+88 %) | 0.14 GB / 1 GB (+86 %) | ok |
| nalgebra_linalg_spectral5 | 0.3.0 | 13,293 / 40,000 (+67 %) | 0.6 s / 5 s (+88 %) | 0.25 GB / 1 GB (+75 %) | ok |
| nalgebra_linalg_spectral6 | 0.3.0 | 23,843 / 40,000 (+40 %) | 1.2 s / 5 s (+76 %) | 0.44 GB / 1 GB (+56 %) | ok |
| nalgebra_linalg_svd_eigen2 | 0.3.0 | 2,195 / 40,000 (+95 %) | 0.2 s / 5 s (+96 %) | 0.06 GB / 1 GB (+94 %) | ok |
| nalgebra_linalg_svd_eigen3 | 0.3.0 | 3,316 / 40,000 (+92 %) | 0.3 s / 5 s (+94 %) | 0.13 GB / 1 GB (+87 %) | ok |
| nalgebra_linalg_svd_eigen4 | 0.3.0 | 5,738 / 40,000 (+86 %) | 0.9 s / 5 s (+83 %) | 0.23 GB / 1 GB (+77 %) | ok |
| nalgebra_linalg_svd_eigen5 | 0.3.0 | 10,672 / 40,000 (+73 %) | 1.3 s / 5 s (+74 %) | 0.40 GB / 1 GB (+60 %) | ok |
| nalgebra_linalg_svd_eigen6 | 0.3.0 | 17,390 / 40,000 (+57 %) | 2.1 s / 5 s (+57 %) | 0.67 GB / 1 GB (+33 %) | ok |
| nalgebra_norm | 0.3.0 | 3,744 / 40,000 (+91 %) | 0.7 s / 5 s (+87 %) | 0.09 GB / 1 GB (+91 %) | ok |
| nalgebra_sparse | 0.3.0 | 3,485 / 40,000 (+91 %) | 0.6 s / 5 s (+88 %) | 0.06 GB / 1 GB (+94 %) | ok |
| nalgebra_static2 | 0.3.0 | 4,570 / 40,000 (+89 %) | 0.3 s / 5 s (+94 %) | 0.10 GB / 1 GB (+90 %) | ok |
| nalgebra_static3 | 0.3.0 | 9,297 / 40,000 (+77 %) | 0.5 s / 5 s (+91 %) | 0.18 GB / 1 GB (+82 %) | ok |
| nalgebra_static4 | 0.3.0 | 16,175 / 40,000 (+60 %) | 1.0 s / 5 s (+81 %) | 0.27 GB / 1 GB (+73 %) | ok |
| nalgebra_static5 | 0.3.0 | 26,820 / 40,000 (+33 %) | 1.9 s / 5 s (+62 %) | 0.44 GB / 1 GB (+56 %) | ok |
| nalgebra_static6_tall | 0.3.0 | 24,451 / 40,000 (+39 %) | 1.3 s / 5 s (+75 %) | 0.35 GB / 1 GB (+65 %) | ok |
| nalgebra_static6_wide | 0.3.0 | 31,497 / 40,000 (+21 %) | 1.1 s / 5 s (+78 %) | 0.44 GB / 1 GB (+56 %) | ok |
| nalgebra_static_core | 0.3.0 | 1,441 / 40,000 (+96 %) | 0.1 s / 5 s (+98 %) | 0.03 GB / 1 GB (+97 %) | ok |
| nalgebra_statistics2 | 0.3.0 | 605 / 40,000 (+98 %) | 0.2 s / 5 s (+97 %) | 0.01 GB / 1 GB (+99 %) | ok |
| nalgebra_statistics3 | 0.3.0 | 1,139 / 40,000 (+97 %) | 0.0 s / 5 s (+99 %) | 0.03 GB / 1 GB (+97 %) | ok |
| nalgebra_statistics4 | 0.3.0 | 2,228 / 40,000 (+94 %) | -0.1 s / 5 s (+103 %) | 0.03 GB / 1 GB (+97 %) | ok |
| nalgebra_statistics5 | 0.3.0 | 5,270 / 40,000 (+87 %) | 0.2 s / 5 s (+96 %) | 0.06 GB / 1 GB (+94 %) | ok |
| nalgebra_statistics6 | 0.3.0 | 9,201 / 40,000 (+77 %) | -0.0 s / 5 s (+100 %) | 0.10 GB / 1 GB (+90 %) | ok |
| nalgebra_transform2 | 0.3.0 | 3,495 / 40,000 (+91 %) | 0.1 s / 5 s (+98 %) | 0.07 GB / 1 GB (+93 %) | ok |
| nalgebra_transform3 | 0.3.0 | 3,945 / 40,000 (+90 %) | 0.4 s / 5 s (+92 %) | 0.07 GB / 1 GB (+93 %) | ok |
| nalgebra_types2 | 0.3.0 | 3,262 / 40,000 (+92 %) | 0.2 s / 5 s (+97 %) | 0.06 GB / 1 GB (+94 %) | ok |
| nalgebra_types3 | 0.3.0 | 6,478 / 40,000 (+84 %) | 0.5 s / 5 s (+91 %) | 0.13 GB / 1 GB (+87 %) | ok |
| nalgebra_types4 | 0.3.0 | 12,374 / 40,000 (+69 %) | 0.7 s / 5 s (+86 %) | 0.23 GB / 1 GB (+77 %) | ok |
| nalgebra_types5 | 0.3.0 | 24,938 / 40,000 (+38 %) | 1.1 s / 5 s (+78 %) | 0.43 GB / 1 GB (+57 %) | ok |
| nalgebra_types6 | 0.3.0 | 37,443 / 40,000 (+6 %) | 1.9 s / 5 s (+61 %) | 0.61 GB / 1 GB (+39 %) | ok |
| nalgebra_views | 0.3.0 | 23,233 / 40,000 (+42 %) | 0.6 s / 5 s (+88 %) | 0.30 GB / 1 GB (+70 %) | ok |

## Declared closures

| closure | members | time | memory | budget | verdict |
|---|---|---:|---:|---|---|
| core_pivot | nalgebra_linalg_pivot2, nalgebra_linalg_pivot3, nalgebra_linalg_pivot4 | 2.8 s / 15 s (+81 %) | 0.95 GB / 3 GB (+68 %) | 15 s / 3 GB | ok |
| facade_no_default_features (reported) | nalgebra | 21.3 s / 15 s (-42 %) | 5.59 GB / 3 GB (-86 %) | 15 s / 3 GB | reported (FAIL: time > 15 s, memory > 3 GB) |
| glam_0_5_0_alone (reported) | glam_core@0.5.0 | 1.4 s / 15 s (+91 %) | 0.44 GB / 3 GB (+85 %) | 15 s / 3 GB | reported (ok) |
| nalgebra_glam | nalgebra_glam, nalgebra_core, nalgebra_types2, nalgebra_types3, nalgebra_types4, nalgebra_static2, nalgebra_static3, nalgebra_geometry2, nalgebra_geometry3, glam_core@0.5.0 | 6.5 s / 15 s (+57 %) | 1.55 GB / 3 GB (+48 %) | 15 s / 3 GB | ok |
| nalgebra_with_glam (reported) | nalgebra, nalgebra_glam | 26.8 s / 15 s (-78 %) | 9.14 GB / 3 GB (-205 %) | 15 s / 3 GB | reported (FAIL: time > 15 s, memory > 3 GB) |
| pivot6 | nalgebra_linalg_pivot6 | 8.2 s / 20 s (+59 %) | 3.44 GB / 4.5 GB (+24 %) | 20 s / 4.5 GB | ok |
| static3_geometry | nalgebra_static3, nalgebra_geometry2, nalgebra_geometry3 | 5.0 s / 15 s (+67 %) | 1.27 GB / 3 GB (+58 %) | 15 s / 3 GB | ok |
| static3_svd | nalgebra_static3, nalgebra_linalg_svd_eigen2, nalgebra_linalg_svd_eigen3 | 4.9 s / 15 s (+67 %) | 1.07 GB / 3 GB (+64 %) | 15 s / 3 GB | ok |
| static4_blas | nalgebra_static4, nalgebra_blas | 10.3 s / 15 s (+31 %) | 2.53 GB / 3 GB (+16 %) | 15 s / 3 GB | ok |
| static4_blocks (reported) | nalgebra_static4, nalgebra_blocks | 10.9 s / 15 s (+28 %) | 3.24 GB / 3 GB (-8 %) | 15 s / 3 GB | reported (FAIL: memory > 3 GB) |
| static4_factor | nalgebra_static4, nalgebra_linalg2, nalgebra_linalg3, nalgebra_linalg4 | 6.9 s / 15 s (+54 %) | 1.80 GB / 3 GB (+40 %) | 15 s / 3 GB | ok |
| static4_geometry | nalgebra_static4, nalgebra_geometry2, nalgebra_geometry3, nalgebra_geometry4, nalgebra_transform2, nalgebra_transform3 | 8.2 s / 15 s (+46 %) | 2.10 GB / 3 GB (+30 %) | 15 s / 3 GB | ok |
| static4_norm (reported) | nalgebra_static4, nalgebra_norm | 9.6 s / 15 s (+36 %) | 3.29 GB / 3 GB (-10 %) | 15 s / 3 GB | reported (FAIL: memory > 3 GB) |
| static4_statistics | nalgebra_static4, nalgebra_statistics2, nalgebra_statistics3, nalgebra_statistics4 | 6.1 s / 15 s (+59 %) | 1.67 GB / 3 GB (+44 %) | 15 s / 3 GB | ok |
| static4_statistics_all | nalgebra_static4, nalgebra_statistics2, nalgebra_statistics3, nalgebra_statistics4, nalgebra_statistics5, nalgebra_statistics6 | 9.0 s / 15 s (+40 %) | 2.46 GB / 3 GB (+18 %) | 15 s / 3 GB | ok |
| static4_svd | nalgebra_static4, nalgebra_linalg_svd_eigen2, nalgebra_linalg_svd_eigen3, nalgebra_linalg_svd_eigen4 | 8.0 s / 15 s (+47 %) | 1.82 GB / 3 GB (+39 %) | 15 s / 3 GB | ok |
| static4_views (reported) | nalgebra_static4, nalgebra_views | 11.5 s / 15 s (+24 %) | 3.54 GB / 3 GB (-18 %) | 15 s / 3 GB | reported (FAIL: memory > 3 GB) |
| static5 | nalgebra_static5 | 9.3 s / 20 s (+54 %) | 2.60 GB / 4.5 GB (+42 %) | 20 s / 4.5 GB | ok |
| static5_factor | nalgebra_static5, nalgebra_linalg5 | 7.1 s / 20 s (+65 %) | 2.68 GB / 4.5 GB (+40 %) | 20 s / 4.5 GB | ok |
| static5_geometry | nalgebra_static5, nalgebra_geometry5 | 10.8 s / 20 s (+46 %) | 2.68 GB / 4.5 GB (+40 %) | 20 s / 4.5 GB | ok |
| static5_pivot | nalgebra_static5, nalgebra_linalg_pivot5 | 10.2 s / 20 s (+49 %) | 2.81 GB / 4.5 GB (+38 %) | 20 s / 4.5 GB | ok |
| static5_spectral | nalgebra_static5, nalgebra_linalg_spectral5 | 10.2 s / 20 s (+49 %) | 2.87 GB / 4.5 GB (+36 %) | 20 s / 4.5 GB | ok |
| static5_svd | nalgebra_static5, nalgebra_linalg_svd_eigen5 | 12.1 s / 20 s (+40 %) | 3.01 GB / 4.5 GB (+33 %) | 20 s / 4.5 GB | ok |
| static6 | nalgebra_static6_wide | 11.0 s / 20 s (+45 %) | 3.22 GB / 4.5 GB (+28 %) | 20 s / 4.5 GB | ok |
| static6_factor | nalgebra_static6_wide, nalgebra_linalg6 | 12.8 s / 20 s (+36 %) | 3.37 GB / 4.5 GB (+25 %) | 20 s / 4.5 GB | ok |
| static6_geometry | nalgebra_static6_wide, nalgebra_geometry6 | 11.1 s / 20 s (+44 %) | 3.27 GB / 4.5 GB (+27 %) | 20 s / 4.5 GB | ok |
| static6_pivot | nalgebra_static6_wide, nalgebra_linalg_pivot6 | 11.8 s / 20 s (+41 %) | 3.52 GB / 4.5 GB (+22 %) | 20 s / 4.5 GB | ok |
| static6_spectral (reported) | nalgebra_static6_wide, nalgebra_linalg_spectral6 | 12.5 s / 20 s (+37 %) | 3.68 GB / 4.5 GB (+18 %) | 20 s / 4.5 GB | reported (ok) |
| static6_svd (reported) | nalgebra_static6_wide, nalgebra_linalg_svd_eigen6 | 11.8 s / 20 s (+41 %) | 3.87 GB / 4.5 GB (+14 %) | 20 s / 4.5 GB | reported (ok) |
| svd_eigen6 | nalgebra_linalg_svd_eigen6 | 7.7 s / 20 s (+61 %) | 2.61 GB / 4.5 GB (+42 %) | 20 s / 4.5 GB | ok |
