# Packages

Generated file, do not edit by hand. Run: 2026-10-01, runner: ubuntu-latest, shards of 4 vCPU / 15 GB, 5 cold build round(s) per consumer (medians of the per-round differences). Regenerate: run `python3 scripts/consumer_cost.py --repeat 5 --interleave --json consumer_cost.json` (CI: the `Consumer cost` job uploads `consumer_cost.json` and this table as the artifact `consumer-cost`), then `python3 scripts/packages_table.py consumer_cost.json --runner RUNNER --output docs/PACKAGES.md`.

Gates: at most 40,000 library lines; marginal cost at most 5 s / 1 GB (gate 2); closures 15 s / 3 GB unless they declare a budget (gate 3). Each figure is `value / limit (margin)`; the margin is the room left below the limit, negative when over it.

## Published packages

| package | version | lines | marginal time | marginal memory | verdict |
|---|---|---:|---:|---:|---|
| nalgebra (reported) | 0.1.1 | 13,191 (gate 3 only) | closure 46.0 s / 15 s (-206 %) | closure 9.94 GB / 3 GB (-231 %) | reported (FAIL: closure time > 15 s, closure memory > 3 GB) |
| nalgebra_blas | 0.1.1 | 12,105 / 40,000 (+70 %) | 2.3 s / 5 s (+55 %) | 0.36 GB / 1 GB (+64 %) | ok |
| nalgebra_blocks | 0.1.1 | 29,129 / 40,000 (+27 %) | 1.3 s / 5 s (+74 %) | 0.45 GB / 1 GB (+55 %) | ok |
| nalgebra_core | 0.1.1 | 2,371 / 40,000 (+94 %) | 0.7 s / 5 s (+86 %) | 0.03 GB / 1 GB (+97 %) | ok |
| nalgebra_dynamic | 0.1.1 | 10,442 / 40,000 (+74 %) | 2.2 s / 5 s (+56 %) | 0.33 GB / 1 GB (+67 %) | ok |
| nalgebra_geometry2 | 0.1.1 | 6,350 / 40,000 (+84 %) | 0.4 s / 5 s (+92 %) | 0.13 GB / 1 GB (+87 %) | ok |
| nalgebra_geometry3 | 0.1.1 | 10,400 / 40,000 (+74 %) | 1.0 s / 5 s (+80 %) | 0.23 GB / 1 GB (+77 %) | ok |
| nalgebra_geometry4 | 0.1.1 | 1,419 / 40,000 (+96 %) | 0.2 s / 5 s (+97 %) | 0.03 GB / 1 GB (+97 %) | ok |
| nalgebra_geometry5 | 0.1.1 | 3,191 / 40,000 (+92 %) | 0.2 s / 5 s (+95 %) | 0.08 GB / 1 GB (+92 %) | ok |
| nalgebra_geometry6 | 0.1.1 | 2,750 / 40,000 (+93 %) | 0.3 s / 5 s (+95 %) | 0.08 GB / 1 GB (+92 %) | ok |
| nalgebra_glam | 0.1.1 | 1,304 / 40,000 (+97 %) | -0.1 s / 5 s (+102 %) | 0.02 GB / 1 GB (+98 %) | ok |
| nalgebra_linalg2 | 0.1.1 | 1,996 / 40,000 (+95 %) | 0.1 s / 5 s (+99 %) | 0.05 GB / 1 GB (+95 %) | ok |
| nalgebra_linalg3 | 0.1.1 | 2,872 / 40,000 (+93 %) | 0.6 s / 5 s (+89 %) | 0.09 GB / 1 GB (+91 %) | ok |
| nalgebra_linalg4 | 0.1.1 | 4,549 / 40,000 (+89 %) | 0.7 s / 5 s (+85 %) | 0.16 GB / 1 GB (+84 %) | ok |
| nalgebra_linalg5 | 0.1.1 | 3,901 / 40,000 (+90 %) | 0.4 s / 5 s (+91 %) | 0.08 GB / 1 GB (+92 %) | ok |
| nalgebra_linalg6 | 0.1.1 | 8,735 / 40,000 (+78 %) | 0.7 s / 5 s (+85 %) | 0.18 GB / 1 GB (+82 %) | ok |
| nalgebra_linalg_core | 0.1.1 | 828 / 40,000 (+98 %) | 0.3 s / 5 s (+93 %) | 0.02 GB / 1 GB (+98 %) | ok |
| nalgebra_linalg_pivot2 | 0.1.1 | 2,249 / 40,000 (+94 %) | 0.5 s / 5 s (+90 %) | 0.05 GB / 1 GB (+95 %) | ok |
| nalgebra_linalg_pivot3 | 0.1.1 | 3,513 / 40,000 (+91 %) | 0.2 s / 5 s (+95 %) | 0.07 GB / 1 GB (+93 %) | ok |
| nalgebra_linalg_pivot4 | 0.1.1 | 7,000 / 40,000 (+82 %) | 0.5 s / 5 s (+90 %) | 0.13 GB / 1 GB (+87 %) | ok |
| nalgebra_linalg_pivot5 | 0.1.1 | 13,529 / 40,000 (+66 %) | 0.8 s / 5 s (+85 %) | 0.22 GB / 1 GB (+78 %) | ok |
| nalgebra_linalg_pivot6 | 0.1.1 | 24,674 / 40,000 (+38 %) | 1.0 s / 5 s (+79 %) | 0.37 GB / 1 GB (+63 %) | ok |
| nalgebra_linalg_spectral2 | 0.1.1 | 2,215 / 40,000 (+94 %) | 0.4 s / 5 s (+92 %) | 0.07 GB / 1 GB (+93 %) | ok |
| nalgebra_linalg_spectral3 | 0.1.1 | 3,378 / 40,000 (+92 %) | 0.5 s / 5 s (+90 %) | 0.09 GB / 1 GB (+91 %) | ok |
| nalgebra_linalg_spectral4 | 0.1.1 | 6,553 / 40,000 (+84 %) | 0.6 s / 5 s (+87 %) | 0.16 GB / 1 GB (+84 %) | ok |
| nalgebra_linalg_spectral5 | 0.1.1 | 13,293 / 40,000 (+67 %) | 0.6 s / 5 s (+89 %) | 0.28 GB / 1 GB (+72 %) | ok |
| nalgebra_linalg_spectral6 | 0.1.1 | 23,843 / 40,000 (+40 %) | 1.7 s / 5 s (+67 %) | 0.47 GB / 1 GB (+53 %) | ok |
| nalgebra_linalg_svd_eigen2 | 0.1.1 | 2,195 / 40,000 (+95 %) | 0.3 s / 5 s (+94 %) | 0.06 GB / 1 GB (+94 %) | ok |
| nalgebra_linalg_svd_eigen3 | 0.1.1 | 3,304 / 40,000 (+92 %) | 0.7 s / 5 s (+86 %) | 0.15 GB / 1 GB (+85 %) | ok |
| nalgebra_linalg_svd_eigen4 | 0.1.1 | 5,738 / 40,000 (+86 %) | 1.3 s / 5 s (+75 %) | 0.26 GB / 1 GB (+74 %) | ok |
| nalgebra_linalg_svd_eigen5 | 0.1.1 | 10,672 / 40,000 (+73 %) | 2.3 s / 5 s (+54 %) | 0.44 GB / 1 GB (+56 %) | ok |
| nalgebra_linalg_svd_eigen6 | 0.1.1 | 17,390 / 40,000 (+57 %) | 3.5 s / 5 s (+31 %) | 0.77 GB / 1 GB (+23 %) | ok |
| nalgebra_norm | 0.1.1 | 3,744 / 40,000 (+91 %) | 0.5 s / 5 s (+90 %) | 0.10 GB / 1 GB (+90 %) | ok |
| nalgebra_sparse | 0.1.1 | 3,485 / 40,000 (+91 %) | 0.7 s / 5 s (+86 %) | 0.10 GB / 1 GB (+90 %) | ok |
| nalgebra_static2 | 0.1.1 | 4,570 / 40,000 (+89 %) | 0.5 s / 5 s (+89 %) | 0.11 GB / 1 GB (+89 %) | ok |
| nalgebra_static3 | 0.1.1 | 9,267 / 40,000 (+77 %) | 1.2 s / 5 s (+75 %) | 0.20 GB / 1 GB (+80 %) | ok |
| nalgebra_static4 | 0.1.1 | 16,175 / 40,000 (+60 %) | 1.7 s / 5 s (+66 %) | 0.34 GB / 1 GB (+66 %) | ok |
| nalgebra_static5 | 0.1.1 | 26,820 / 40,000 (+33 %) | 2.5 s / 5 s (+50 %) | 0.49 GB / 1 GB (+51 %) | ok |
| nalgebra_static6_tall | 0.1.1 | 24,451 / 40,000 (+39 %) | 1.5 s / 5 s (+69 %) | 0.37 GB / 1 GB (+63 %) | ok |
| nalgebra_static6_wide | 0.1.1 | 31,481 / 40,000 (+21 %) | 2.2 s / 5 s (+56 %) | 0.49 GB / 1 GB (+51 %) | ok |
| nalgebra_static_core | 0.1.1 | 1,441 / 40,000 (+96 %) | 0.2 s / 5 s (+97 %) | 0.03 GB / 1 GB (+97 %) | ok |
| nalgebra_statistics2 | 0.1.1 | 605 / 40,000 (+98 %) | 0.1 s / 5 s (+98 %) | 0.00 GB / 1 GB (+100 %) | ok |
| nalgebra_statistics3 | 0.1.1 | 1,139 / 40,000 (+97 %) | 0.1 s / 5 s (+99 %) | 0.03 GB / 1 GB (+97 %) | ok |
| nalgebra_statistics4 | 0.1.1 | 2,228 / 40,000 (+94 %) | 0.0 s / 5 s (+100 %) | 0.04 GB / 1 GB (+96 %) | ok |
| nalgebra_statistics5 | 0.1.1 | 5,270 / 40,000 (+87 %) | 0.2 s / 5 s (+96 %) | 0.07 GB / 1 GB (+93 %) | ok |
| nalgebra_statistics6 | 0.1.1 | 9,201 / 40,000 (+77 %) | 0.6 s / 5 s (+89 %) | 0.10 GB / 1 GB (+90 %) | ok |
| nalgebra_transform2 | 0.1.1 | 3,495 / 40,000 (+91 %) | 0.4 s / 5 s (+92 %) | 0.07 GB / 1 GB (+93 %) | ok |
| nalgebra_transform3 | 0.1.1 | 3,945 / 40,000 (+90 %) | 0.9 s / 5 s (+83 %) | 0.09 GB / 1 GB (+91 %) | ok |
| nalgebra_types2 | 0.1.1 | 3,261 / 40,000 (+92 %) | 0.2 s / 5 s (+95 %) | 0.07 GB / 1 GB (+93 %) | ok |
| nalgebra_types3 | 0.1.1 | 6,477 / 40,000 (+84 %) | 0.6 s / 5 s (+88 %) | 0.15 GB / 1 GB (+85 %) | ok |
| nalgebra_types4 | 0.1.1 | 12,374 / 40,000 (+69 %) | 0.8 s / 5 s (+84 %) | 0.27 GB / 1 GB (+73 %) | ok |
| nalgebra_types5 | 0.1.1 | 24,933 / 40,000 (+38 %) | 1.6 s / 5 s (+69 %) | 0.46 GB / 1 GB (+54 %) | ok |
| nalgebra_types6 | 0.1.1 | 37,436 / 40,000 (+6 %) | 1.7 s / 5 s (+65 %) | 0.68 GB / 1 GB (+32 %) | ok |
| nalgebra_views | 0.1.1 | 23,233 / 40,000 (+42 %) | 0.4 s / 5 s (+92 %) | 0.32 GB / 1 GB (+68 %) | ok |

## Declared closures

| closure | members | time | memory | budget | verdict |
|---|---|---:|---:|---|---|
| core_pivot | nalgebra_linalg_pivot2, nalgebra_linalg_pivot3, nalgebra_linalg_pivot4 | 4.8 s / 15 s (+68 %) | 1.06 GB / 3 GB (+65 %) | 15 s / 3 GB | ok |
| facade_no_default_features (reported) | nalgebra | 26.7 s / 15 s (-78 %) | 6.30 GB / 3 GB (-110 %) | 15 s / 3 GB | reported (FAIL: time > 15 s, memory > 3 GB) |
| glam_0_4_1_alone (reported) | glam_core@0.4.1 | 1.8 s / 15 s (+88 %) | 0.48 GB / 3 GB (+84 %) | 15 s / 3 GB | reported (ok) |
| nalgebra_glam | nalgebra_glam, nalgebra_core, nalgebra_types2, nalgebra_types3, nalgebra_types4, nalgebra_static2, nalgebra_static3, nalgebra_geometry2, nalgebra_geometry3, glam_core@0.4.1 | 5.5 s / 15 s (+64 %) | 1.70 GB / 3 GB (+43 %) | 15 s / 3 GB | ok |
| nalgebra_with_glam (reported) | nalgebra, nalgebra_glam | 46.5 s / 15 s (-210 %) | 10.19 GB / 3 GB (-240 %) | 15 s / 3 GB | reported (FAIL: time > 15 s, memory > 3 GB) |
| pivot6 | nalgebra_linalg_pivot6 | 14.0 s / 20 s (+30 %) | 3.90 GB / 4.5 GB (+13 %) | 20 s / 4.5 GB | ok |
| static3_geometry | nalgebra_static3, nalgebra_geometry2, nalgebra_geometry3 | 6.3 s / 15 s (+58 %) | 1.42 GB / 3 GB (+53 %) | 15 s / 3 GB | ok |
| static3_svd | nalgebra_static3, nalgebra_linalg_svd_eigen2, nalgebra_linalg_svd_eigen3 | 5.5 s / 15 s (+63 %) | 1.20 GB / 3 GB (+60 %) | 15 s / 3 GB | ok |
| static4_blas | nalgebra_static4, nalgebra_blas | 12.0 s / 15 s (+20 %) | 2.83 GB / 3 GB (+6 %) | 15 s / 3 GB | ok |
| static4_blocks (reported) | nalgebra_static4, nalgebra_blocks | 13.7 s / 15 s (+9 %) | 3.63 GB / 3 GB (-21 %) | 15 s / 3 GB | reported (FAIL: memory > 3 GB) |
| static4_factor | nalgebra_static4, nalgebra_linalg2, nalgebra_linalg3, nalgebra_linalg4 | 6.0 s / 15 s (+60 %) | 2.00 GB / 3 GB (+33 %) | 15 s / 3 GB | ok |
| static4_geometry | nalgebra_static4, nalgebra_geometry2, nalgebra_geometry3, nalgebra_geometry4, nalgebra_transform2, nalgebra_transform3 | 10.4 s / 15 s (+31 %) | 2.37 GB / 3 GB (+21 %) | 15 s / 3 GB | ok |
| static4_norm (reported) | nalgebra_static4, nalgebra_norm | 14.3 s / 15 s (+5 %) | 3.72 GB / 3 GB (-24 %) | 15 s / 3 GB | reported (FAIL: memory > 3 GB) |
| static4_statistics | nalgebra_static4, nalgebra_statistics2, nalgebra_statistics3, nalgebra_statistics4 | 7.7 s / 15 s (+49 %) | 1.86 GB / 3 GB (+38 %) | 15 s / 3 GB | ok |
| static4_statistics_all | nalgebra_static4, nalgebra_statistics2, nalgebra_statistics3, nalgebra_statistics4, nalgebra_statistics5, nalgebra_statistics6 | 10.9 s / 15 s (+27 %) | 2.74 GB / 3 GB (+9 %) | 15 s / 3 GB | ok |
| static4_svd | nalgebra_static4, nalgebra_linalg_svd_eigen2, nalgebra_linalg_svd_eigen3, nalgebra_linalg_svd_eigen4 | 9.0 s / 15 s (+40 %) | 2.06 GB / 3 GB (+31 %) | 15 s / 3 GB | ok |
| static4_views (reported) | nalgebra_static4, nalgebra_views | 14.3 s / 15 s (+4 %) | 3.94 GB / 3 GB (-31 %) | 15 s / 3 GB | reported (FAIL: memory > 3 GB) |
| static5 | nalgebra_static5 | 11.8 s / 20 s (+41 %) | 2.92 GB / 4.5 GB (+35 %) | 20 s / 4.5 GB | ok |
| static5_factor | nalgebra_static5, nalgebra_linalg5 | 12.1 s / 20 s (+39 %) | 3.02 GB / 4.5 GB (+33 %) | 20 s / 4.5 GB | ok |
| static5_geometry | nalgebra_static5, nalgebra_geometry5 | 12.0 s / 20 s (+40 %) | 3.01 GB / 4.5 GB (+33 %) | 20 s / 4.5 GB | ok |
| static5_pivot | nalgebra_static5, nalgebra_linalg_pivot5 | 12.5 s / 20 s (+38 %) | 3.16 GB / 4.5 GB (+30 %) | 20 s / 4.5 GB | ok |
| static5_spectral | nalgebra_static5, nalgebra_linalg_spectral5 | 9.2 s / 20 s (+54 %) | 3.22 GB / 4.5 GB (+28 %) | 20 s / 4.5 GB | ok |
| static5_svd | nalgebra_static5, nalgebra_linalg_svd_eigen5 | 14.1 s / 20 s (+29 %) | 3.40 GB / 4.5 GB (+24 %) | 20 s / 4.5 GB | ok |
| static6 | nalgebra_static6_wide | 13.4 s / 20 s (+33 %) | 3.62 GB / 4.5 GB (+20 %) | 20 s / 4.5 GB | ok |
| static6_factor | nalgebra_static6_wide, nalgebra_linalg6 | 14.5 s / 20 s (+28 %) | 3.79 GB / 4.5 GB (+16 %) | 20 s / 4.5 GB | ok |
| static6_geometry | nalgebra_static6_wide, nalgebra_geometry6 | 9.8 s / 20 s (+51 %) | 3.69 GB / 4.5 GB (+18 %) | 20 s / 4.5 GB | ok |
| static6_pivot | nalgebra_static6_wide, nalgebra_linalg_pivot6 | 10.7 s / 20 s (+47 %) | 3.99 GB / 4.5 GB (+11 %) | 20 s / 4.5 GB | ok |
| static6_spectral (reported) | nalgebra_static6_wide, nalgebra_linalg_spectral6 | 15.5 s / 20 s (+23 %) | 4.13 GB / 4.5 GB (+8 %) | 20 s / 4.5 GB | reported (ok) |
| static6_svd (reported) | nalgebra_static6_wide, nalgebra_linalg_svd_eigen6 | 17.4 s / 20 s (+13 %) | 4.37 GB / 4.5 GB (+3 %) | 20 s / 4.5 GB | reported (ok) |
| svd_eigen6 | nalgebra_linalg_svd_eigen6 | 12.0 s / 20 s (+40 %) | 2.93 GB / 4.5 GB (+35 %) | 20 s / 4.5 GB | ok |
