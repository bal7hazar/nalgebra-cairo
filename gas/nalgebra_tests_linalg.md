# Gas report

Sierra gas (`l2_gas`) per benchmark; `net` = raw - group baseline.

## nalgebra_tests_linalg::cholesky::benches

### cholesky2_determinant

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `diagonal_product` | 11770 | 3260 | x1.00 |

### cholesky2_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 28360 | 18350 | x1.00 |
| `triangular` | 31020 | 21010 | x1.14 |

### cholesky2_l

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `expand` | 10110 | 100 | x1.00 |

### cholesky2_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `factorize` | 23410 | 13700 | x1.00 |

### cholesky2_solve

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 29840 | 20230 | x1.00 |
| `substitution` | 32830 | 23220 | x1.15 |

### cholesky3_determinant

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `diagonal_product` | 14050 | 4940 | x1.00 |

### cholesky3_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 48680 | 35570 | x1.00 |
| `triangular` | 56660 | 43550 | x1.22 |

### cholesky3_l

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `expand` | 13110 | 0 | - |

### cholesky3_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `factorize` | 35710 | 23500 | x1.00 |

### cholesky3_solve

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `substitution` | 38350 | 27440 | x1.00 |
| `alt_recip` | 42920 | 32010 | x1.17 |

### cholesky4_determinant

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `diagonal_product` | 16530 | 6620 | x1.00 |

### cholesky4_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 80240 | 60830 | x1.00 |
| `triangular` | 95890 | 76480 | x1.26 |

### cholesky4_l

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `expand` | 21010 | 1600 | x1.00 |

### cholesky4_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `factorize` | 68080 | 52470 | x1.00 |

### cholesky4_solve

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 56700 | 44290 | x1.00 |
| `substitution` | 62810 | 50400 | x1.14 |

### cholesky6_determinant

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `diagonal_product` | 22090 | 9980 | x1.00 |

### cholesky6_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 165050 | 131440 | x1.00 |
| `triangular` | 202640 | 169030 | x1.29 |

### cholesky6_l

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `expand` | 37210 | 3600 | x1.00 |

### cholesky6_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `factorize` | 121810 | 94200 | x1.00 |

### cholesky6_solve

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `substitution` | 78490 | 62480 | x1.00 |
| `alt_recip` | 86360 | 70350 | x1.13 |

## nalgebra_tests_linalg::lu::lu2::tests

### lu2_determinant

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `pivots` | 11690 | 2780 | x1.00 |

### lu2_factors

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `l` | 10510 | 100 | x1.00 |
| `u` | 10510 | 100 | x1.00 |

### lu2_is_invertible

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `pivots` | 9620 | 700 | x1.00 |

### lu2_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_no_pivot` | 19060 | 8850 | x1.00 |
| `pivot` | 22680 | 12470 | x1.41 |

### lu2_p

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `field` | 8910 | 0 | - |

### lu2_solve

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `substitution` | 27050 | 16740 | x1.00 |

### lu2_solve_singular

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `none` | 26560 | 17040 | x1.00 |

### lu2_try_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 32540 | 21830 | x1.00 |
| `columns` | 34950 | 24240 | x1.11 |
| `alt_solve_columns` | 47790 | 37080 | x1.70 |

## nalgebra_tests_linalg::lu::lu3::tests

### lu3_determinant

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `pivots` | 18020 | 7910 | x1.00 |

### lu3_factors

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `l` | 14110 | 0 | - |
| `u` | 14110 | 0 | - |

### lu3_is_invertible

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `pivots` | 10920 | 800 | x1.00 |

### lu3_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_no_pivot` | 41060 | 27350 | x1.00 |
| `pivot` | 51550 | 37840 | x1.38 |

### lu3_p

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `field` | 10810 | 0 | - |

### lu3_solve

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `substitution` | 32430 | 20220 | x1.00 |

### lu3_solve_singular

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `none` | 13920 | 3000 | x1.00 |

### lu3_try_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 60570 | 46160 | x1.00 |
| `columns` | 72230 | 57820 | x1.25 |
| `alt_solve_columns` | 88930 | 74520 | x1.61 |

### lu3_vs_matrix3_determinant

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `cofactors` | 20060 | 10350 | x1.00 |
| `lu` | 55560 | 45850 | x4.43 |

### lu3_vs_matrix3_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `cofactors` | 102370 | 88360 | x1.00 |
| `lu` | 109770 | 95760 | x1.08 |

## nalgebra_tests_linalg::lu::lu4::tests

### lu4_determinant

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `pivots` | 22710 | 11000 | x1.00 |

### lu4_factors

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `l` | 22810 | 1600 | x1.00 |
| `u` | 22810 | 1600 | x1.00 |

### lu4_is_invertible

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `pivots` | 12620 | 900 | x1.00 |

### lu4_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_no_pivot` | 82400 | 61790 | x1.00 |
| `pivot` | 104300 | 83690 | x1.35 |

### lu4_p

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `field` | 13010 | 0 | - |

### lu4_solve

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `substitution` | 53550 | 39040 | x1.00 |

### lu4_solve_singular

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `none` | 52060 | 39340 | x1.00 |

### lu4_try_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 107330 | 84220 | x1.00 |
| `columns` | 131950 | 108840 | x1.29 |
| `alt_solve_columns` | 185870 | 162760 | x1.93 |

## nalgebra_tests_linalg::lu::lu6::benches

### lu6_determinant

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `pivots` | 33890 | 17780 | x1.00 |

### lu6_factors

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `l` | 41210 | 3600 | x1.00 |
| `u` | 41210 | 3600 | x1.00 |

### lu6_is_invertible

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `pivots` | 17220 | 1100 | x1.00 |

### lu6_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_no_pivot` | 224740 | 188130 | x1.00 |
| `pivot` | 248120 | 211510 | x1.12 |

### lu6_p

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `field` | 18610 | 0 | - |

### lu6_solve

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `substitution` | 79560 | 59250 | x1.00 |

### lu6_solve_singular

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `none` | 24120 | 6600 | x1.00 |

### lu6_try_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 249750 | 208240 | x1.00 |
| `columns` | 313190 | 271680 | x1.30 |
| `alt_solve_columns` | 439150 | 397640 | x1.91 |

### matrix6_lu

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `determinant` | 268150 | 253040 | x1.00 |
| `try_inverse` | 547450 | 532340 | x2.10 |

## nalgebra_tests_linalg::qr::qr2::tests

### qr2_factors

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `unpack` | 39210 | 700 | x1.00 |

### qr2_is_invertible

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `pivots` | 39320 | 700 | x1.00 |

### qr2_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `gram_schmidt` | 41090 | 32380 | x1.00 |

### qr2_solve

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `substitution` | 57740 | 18520 | x1.00 |

### qr2_solve_singular

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `none` | 57520 | 18420 | x1.00 |

### qr2_try_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `substitution` | 64240 | 25620 | x1.00 |

## nalgebra_tests_linalg::qr::qr3::tests

### qr3_determinant_closed_form

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `matrix3_cofactors` | 20670 | 10950 | x1.00 |

### qr3_factors

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `unpack` | 84530 | 700 | x1.00 |

### qr3_is_invertible

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `pivots` | 84630 | 800 | x1.00 |

### qr3_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_classical_gram_schmidt` | 80600 | 70880 | x1.00 |
| `gram_schmidt` | 87610 | 77890 | x1.10 |

### qr3_solve

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `substitution` | 113350 | 28720 | x1.00 |

### qr3_solve_singular

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `none` | 113050 | 28620 | x1.00 |

### qr3_try_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `substitution` | 141170 | 57340 | x1.00 |

## nalgebra_tests_linalg::qr::qr4::tests

### qr4_factors

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `unpack` | 158940 | 700 | x1.00 |

### qr4_is_invertible

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `pivots` | 159140 | 900 | x1.00 |

### qr4_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `gram_schmidt` | 162800 | 151680 | x1.00 |

### qr4_solve

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `substitution` | 203360 | 44120 | x1.00 |

### qr4_try_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `substitution` | 282640 | 124400 | x1.00 |

## nalgebra_tests_linalg::svd2::tests

### svd2_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 81580 | 72860 | x1.00 |

### svd2_pseudo_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `reciprocals` | 110700 | 28400 | x1.00 |

### svd2_rank

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `comparisons` | 84940 | 2640 | x1.00 |

### svd2_recompose

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `scaled_product` | 101770 | 19470 | x1.00 |

### svd2_singular_values

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `from_left_vectors` | 81580 | 72860 | x1.00 |

### svd2_solve

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `divisions` | 104540 | 21640 | x1.00 |

### svd2_to_polar

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `quadform` | 107940 | 25640 | x1.00 |

## nalgebra_tests_linalg::svd3::tests

### svd3_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 650930 | 641210 | x1.00 |

### svd3_pseudo_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `reciprocals` | 727780 | 54430 | x1.00 |

### svd3_rank

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `comparisons` | 677160 | 3810 | x1.00 |

### svd3_recompose

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `scaled_product` | 714520 | 41170 | x1.00 |

### svd3_singular_values

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `from_left_vectors` | 650930 | 641210 | x1.00 |

### svd3_solve

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `divisions` | 706860 | 32710 | x1.00 |

### svd3_to_polar

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `quadform` | 727030 | 53680 | x1.00 |

## nalgebra_tests_linalg::udu::benches

### udu2_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `reversed_ldlt` | 19270 | 9850 | x1.00 |

### udu3_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `reversed_ldlt` | 34410 | 23990 | x1.00 |

### udu4_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `reversed_ldlt` | 65340 | 47550 | x1.00 |

### udu6_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `reversed_ldlt` | 140220 | 114230 | x1.00 |

