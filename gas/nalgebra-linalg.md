# Gas report

Sierra gas (`l2_gas`) per benchmark; `net` = raw - group baseline.

## nalgebra::linalg::cholesky::benches

### cholesky2_determinant

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `diagonal_product` | 19600 | 3260 | x1.00 |

### cholesky2_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 35590 | 18250 | x1.00 |
| `triangular` | 38250 | 20910 | x1.15 |

### cholesky2_l

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `expand` | 17940 | 100 | x1.00 |

### cholesky2_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `factorize` | 31040 | 13700 | x1.00 |

### cholesky2_solve

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 37670 | 20230 | x1.00 |
| `substitution` | 40660 | 23220 | x1.15 |

### cholesky3_determinant

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `diagonal_product` | 21880 | 4940 | x1.00 |

### cholesky3_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 54710 | 35270 | x1.00 |
| `triangular` | 62690 | 43250 | x1.23 |

### cholesky3_l

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `expand` | 20940 | 0 | - |

### cholesky3_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `factorize` | 48970 | 29530 | x1.00 |

### cholesky3_solve

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 50750 | 32010 | x1.00 |
| `substitution` | 55300 | 36560 | x1.14 |

### cholesky4_determinant

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `diagonal_product` | 24360 | 6620 | x1.00 |

### cholesky4_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 88070 | 60830 | x1.00 |
| `triangular` | 103720 | 76480 | x1.26 |

### cholesky4_l

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `expand` | 28840 | 1600 | x1.00 |

### cholesky4_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `factorize` | 75910 | 52470 | x1.00 |

### cholesky4_solve

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 64530 | 44290 | x1.00 |
| `substitution` | 70640 | 50400 | x1.14 |

### cholesky6_determinant

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `diagonal_product` | 29920 | 9980 | x1.00 |

### cholesky6_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 172880 | 131440 | x1.00 |
| `triangular` | 210470 | 169030 | x1.29 |

### cholesky6_l

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `expand` | 45940 | 3600 | x1.00 |

### cholesky6_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `factorize` | 156550 | 121110 | x1.00 |

### cholesky6_solve

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 94190 | 70350 | x1.00 |
| `substitution` | 103420 | 79580 | x1.13 |

## nalgebra::linalg::ldlt::benches

### ldlt2_d

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `accessor` | 17040 | 0 | - |

### ldlt2_determinant

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `diagonal_product` | 17920 | 1580 | x1.00 |

### ldlt2_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 32430 | 15090 | x1.00 |
| `triangular` | 35090 | 17750 | x1.18 |

### ldlt2_l

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `expand` | 17940 | 100 | x1.00 |

### ldlt2_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `factorize` | 25790 | 8450 | x1.00 |
| `alt_products` | 27670 | 10330 | x1.22 |

### ldlt2_solve

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `substitution` | 32110 | 14670 | x1.00 |
| `alt_recip` | 34310 | 16870 | x1.15 |

### ldlt3_d

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `accessor` | 18240 | 0 | - |

### ldlt3_determinant

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `diagonal_product` | 20200 | 3260 | x1.00 |

### ldlt3_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 49970 | 30530 | x1.00 |
| `triangular` | 57950 | 38510 | x1.26 |

### ldlt3_l

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `expand` | 20940 | 0 | - |

### ldlt3_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `factorize` | 41430 | 21990 | x1.00 |
| `alt_products` | 46470 | 27030 | x1.23 |

### ldlt3_solve

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `substitution` | 42410 | 23670 | x1.00 |
| `alt_recip` | 45710 | 26970 | x1.14 |

### ldlt4_d

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `accessor` | 19640 | 0 | - |

### ldlt4_determinant

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `diagonal_product` | 22680 | 4940 | x1.00 |

### ldlt4_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 81750 | 54510 | x1.00 |
| `triangular` | 97400 | 70160 | x1.29 |

### ldlt4_l

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `expand` | 28840 | 1600 | x1.00 |

### ldlt4_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `factorize` | 66190 | 42750 | x1.00 |
| `alt_products` | 76270 | 52830 | x1.24 |

### ldlt4_solve

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `substitution` | 53410 | 33170 | x1.00 |
| `alt_recip` | 57810 | 37570 | x1.13 |

### ldlt6_d

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `accessor` | 23340 | 0 | - |

### ldlt6_determinant

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `diagonal_product` | 28540 | 8300 | x1.00 |

### ldlt6_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 163700 | 121960 | x1.00 |
| `triangular` | 201290 | 159550 | x1.31 |

### ldlt6_l

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `expand` | 46240 | 3600 | x1.00 |

### ldlt6_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `factorize` | 143070 | 107330 | x1.00 |
| `alt_products` | 168270 | 132530 | x1.23 |

### ldlt6_solve

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `substitution` | 77810 | 53670 | x1.00 |
| `alt_recip` | 84410 | 60270 | x1.12 |

## nalgebra::linalg::lu::lu2::tests

### lu2_determinant

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `pivots` | 19520 | 2780 | x1.00 |

### lu2_factors

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `l` | 18340 | 100 | x1.00 |
| `u` | 18340 | 100 | x1.00 |

### lu2_is_invertible

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `pivots` | 17450 | 700 | x1.00 |

### lu2_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_no_pivot` | 26890 | 8850 | x1.00 |
| `pivot` | 30510 | 12470 | x1.41 |

### lu2_p

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `field` | 16740 | 0 | - |

### lu2_permute

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `transpositions` | 18640 | 800 | x1.00 |

### lu2_permute_rows

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `transpositions` | 20440 | 1200 | x1.00 |

### lu2_solve

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `substitution` | 34880 | 16740 | x1.00 |
| `alt_recip` | 37210 | 19070 | x1.14 |

### lu2_solve_singular

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `none` | 34390 | 17040 | x1.00 |

### lu2_try_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 40370 | 21830 | x1.00 |
| `columns` | 42780 | 24240 | x1.11 |
| `alt_solve_columns` | 55620 | 37080 | x1.70 |

## nalgebra::linalg::lu::lu3::tests

### lu3_determinant

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `pivots` | 25850 | 7910 | x1.00 |

### lu3_factors

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `l` | 21940 | 0 | - |
| `u` | 21940 | 0 | - |

### lu3_is_invertible

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `pivots` | 18750 | 800 | x1.00 |

### lu3_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_no_pivot` | 48890 | 27350 | x1.00 |
| `pivot` | 59380 | 37840 | x1.38 |

### lu3_p

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `field` | 18640 | 0 | - |

### lu3_permute

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `transpositions` | 21480 | 1740 | x1.00 |

### lu3_permute_rows

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `transpositions` | 27320 | 3380 | x1.00 |

### lu3_solve

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `substitution` | 47280 | 27240 | x1.00 |
| `alt_recip` | 50710 | 30670 | x1.13 |

### lu3_solve_singular

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `none` | 46290 | 27540 | x1.00 |

### lu3_try_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 68400 | 46160 | x1.00 |
| `columns` | 80060 | 57820 | x1.25 |
| `alt_solve_columns` | 108860 | 86620 | x1.88 |

### lu3_vs_matrix3_determinant

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `cofactors` | 27890 | 10350 | x1.00 |
| `lu` | 63390 | 45850 | x4.43 |

### lu3_vs_matrix3_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `cofactors` | 110200 | 88360 | x1.00 |
| `lu` | 117600 | 95760 | x1.08 |

## nalgebra::linalg::lu::lu4::tests

### lu4_determinant

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `pivots` | 30540 | 11000 | x1.00 |

### lu4_factors

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `l` | 30640 | 1600 | x1.00 |
| `u` | 30640 | 1600 | x1.00 |

### lu4_is_invertible

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `pivots` | 20450 | 900 | x1.00 |

### lu4_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_no_pivot` | 90230 | 61790 | x1.00 |
| `pivot` | 112130 | 83690 | x1.35 |

### lu4_p

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `field` | 20840 | 0 | - |

### lu4_permute

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `transpositions` | 25330 | 3290 | x1.00 |

### lu4_permute_rows

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `transpositions` | 45140 | 12700 | x1.00 |

### lu4_solve

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `substitution` | 61380 | 39040 | x1.00 |
| `alt_recip` | 65910 | 43570 | x1.12 |

### lu4_solve_singular

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `none` | 59890 | 39340 | x1.00 |

### lu4_try_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 115160 | 84220 | x1.00 |
| `columns` | 139780 | 108840 | x1.29 |
| `alt_solve_columns` | 193700 | 162760 | x1.93 |

## nalgebra::linalg::lu::lu6::benches

### lu6_determinant

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `pivots` | 42320 | 18380 | x1.00 |

### lu6_factors

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `l` | 49940 | 3600 | x1.00 |
| `u` | 49940 | 3600 | x1.00 |

### lu6_is_invertible

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `pivots` | 25050 | 1100 | x1.00 |

### lu6_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_no_pivot` | 232570 | 188130 | x1.00 |
| `pivot` | 297360 | 252920 | x1.34 |

### lu6_p

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `field` | 26440 | 0 | - |

### lu6_permute

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `transpositions` | 42640 | 14800 | x1.00 |

### lu6_permute_rows

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `transpositions` | 97640 | 44800 | x1.00 |

### lu6_solve

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `substitution` | 100380 | 72240 | x1.00 |
| `alt_recip` | 107110 | 78970 | x1.09 |

### lu6_solve_singular

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `none` | 97890 | 72540 | x1.00 |

### lu6_try_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 257580 | 208240 | x1.00 |
| `columns` | 321020 | 271680 | x1.30 |
| `alt_solve_columns` | 495180 | 445840 | x2.14 |

### matrix6_lu

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `determinant` | 290740 | 267800 | x1.00 |
| `solve` | 348800 | 325860 | x1.22 |
| `try_inverse` | 573340 | 550400 | x2.06 |

## nalgebra::linalg::qr::qr2::tests

### qr2_determinant

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `diagonal_product` | 54110 | 7660 | x1.00 |

### qr2_factors

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `unpack` | 47040 | 700 | x1.00 |

### qr2_is_invertible

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `pivots` | 47150 | 700 | x1.00 |

### qr2_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `gram_schmidt` | 48920 | 32380 | x1.00 |

### qr2_solve

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `substitution` | 65570 | 18520 | x1.00 |

### qr2_solve_singular

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `none` | 65350 | 18420 | x1.00 |

### qr2_try_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `substitution` | 72070 | 25620 | x1.00 |

## nalgebra::linalg::qr::qr3::tests

### qr3_determinant

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `diagonal_product` | 110670 | 19010 | x1.00 |

### qr3_determinant_closed_form

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `matrix3_cofactors` | 28500 | 10950 | x1.00 |

### qr3_factors

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `unpack` | 92360 | 700 | x1.00 |

### qr3_is_invertible

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `pivots` | 92460 | 800 | x1.00 |

### qr3_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_classical_gram_schmidt` | 88430 | 70880 | x1.00 |
| `gram_schmidt` | 95440 | 77890 | x1.10 |
| `alt_completed_basis` | 123720 | 106170 | x1.50 |
| `alt_householder` | 199220 | 181670 | x2.56 |

### qr3_solve

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `substitution` | 121180 | 28720 | x1.00 |

### qr3_solve_singular

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `none` | 120880 | 28620 | x1.00 |

### qr3_try_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `substitution` | 149000 | 57340 | x1.00 |

## nalgebra::linalg::qr::qr4::tests

### qr4_determinant

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `diagonal_product` | 206480 | 40410 | x1.00 |

### qr4_factors

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `unpack` | 166770 | 700 | x1.00 |

### qr4_is_invertible

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `pivots` | 166970 | 900 | x1.00 |

### qr4_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `gram_schmidt` | 170630 | 151680 | x1.00 |

### qr4_solve

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `substitution` | 211190 | 44120 | x1.00 |

### qr4_try_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `substitution` | 290470 | 124400 | x1.00 |

## nalgebra::linalg::svd2::tests

### svd2_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 88910 | 72360 | x1.00 |
| `alt_normalised_columns` | 116210 | 99660 | x1.38 |

### svd2_pseudo_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `reciprocals` | 117230 | 28000 | x1.00 |

### svd2_rank

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `comparisons` | 91870 | 2640 | x1.00 |

### svd2_recompose

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `scaled_product` | 107400 | 18170 | x1.00 |

### svd2_singular_values

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_sqrt_eigenvalues` | 36470 | 19920 | x1.00 |
| `from_left_vectors` | 88910 | 72360 | x3.63 |

### svd2_solve

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `divisions` | 111070 | 21240 | x1.00 |

### svd2_to_polar

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `quadform` | 114670 | 25440 | x1.00 |

## nalgebra::linalg::svd3::tests

### svd3_gram

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 30030 | 12480 | x1.00 |
| `transpose_mul_transpose` | 33160 | 15610 | x1.25 |

### svd3_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 743440 | 725890 | x1.00 |
| `alt_one_sided_jacobi` | 779160 | 761610 | x1.05 |
| `alt_normalised_columns` | 808610 | 791060 | x1.09 |

### svd3_pseudo_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `reciprocals` | 798990 | 54030 | x1.00 |

### svd3_rank

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `comparisons` | 748770 | 3810 | x1.00 |

### svd3_recompose

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `scaled_product` | 784330 | 39370 | x1.00 |

### svd3_singular_values

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_sqrt_eigenvalues` | 380560 | 363010 | x1.00 |
| `from_left_vectors` | 743440 | 725890 | x2.00 |

### svd3_solve

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `divisions` | 778070 | 32310 | x1.00 |

### svd3_to_polar

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `quadform` | 798440 | 53480 | x1.00 |

## nalgebra::linalg::symmetric_eigen2::tests

### symmetric_eigen2_eigenvalues

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `closed_form` | 23470 | 7130 | x1.00 |

### symmetric_eigen2_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `closed_form` | 43670 | 27330 | x1.00 |

### symmetric_eigen2_recompose

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `quadform` | 58390 | 14590 | x1.00 |

## nalgebra::linalg::symmetric_eigen3::tests

### symmetric_eigen3_eigenvalues

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `without_eigenvectors` | 358570 | 341630 | x1.00 |
| `via_new` | 607030 | 590090 | x1.73 |

### symmetric_eigen3_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `jacobi_3_sweeps` | 473860 | 456920 | x1.00 |
| `no_renormalisation` | 576070 | 559130 | x1.22 |
| `jacobi_4_sweeps` | 607030 | 590090 | x1.29 |
| `diagonal_input` | 607490 | 590550 | x1.29 |
| `jacobi_5_sweeps` | 740200 | 723260 | x1.58 |
| `jacobi_6_sweeps` | 873370 | 856430 | x1.87 |

### symmetric_eigen3_recompose

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `quadform` | 638620 | 30430 | x1.00 |

### symmetric_eigen3_sweep

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `three_rotations_without_eigenvectors` | 94990 | 76250 | x1.00 |
| `three_rotations` | 151710 | 132970 | x1.74 |

