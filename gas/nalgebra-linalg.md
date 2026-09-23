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
| `alt_recip` | 32770 | 15430 | x1.00 |
| `triangular` | 33920 | 16580 | x1.07 |

### cholesky2_l

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `expand` | 17940 | 100 | x1.00 |

### cholesky2_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `factorize` | 29530 | 12190 | x1.00 |

### cholesky2_solve

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `substitution` | 34620 | 17180 | x1.00 |
| `alt_recip` | 34850 | 17410 | x1.01 |

### cholesky3_determinant

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `diagonal_product` | 21880 | 4940 | x1.00 |

### cholesky3_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 50480 | 31040 | x1.00 |
| `triangular` | 53930 | 34490 | x1.11 |

### cholesky3_l

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `expand` | 20940 | 0 | - |

### cholesky3_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `factorize` | 44440 | 25000 | x1.00 |

### cholesky3_solve

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `substitution` | 46240 | 27500 | x1.00 |
| `alt_recip` | 46520 | 27780 | x1.01 |

### cholesky4_determinant

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `diagonal_product` | 24360 | 6620 | x1.00 |

### cholesky4_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 82430 | 55190 | x1.00 |
| `triangular` | 89330 | 62090 | x1.13 |

### cholesky4_l

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `expand` | 28840 | 1600 | x1.00 |

### cholesky4_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `factorize` | 67160 | 43720 | x1.00 |

### cholesky4_solve

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `substitution` | 58560 | 38320 | x1.00 |
| `alt_recip` | 58890 | 38650 | x1.01 |

### cholesky6_determinant

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `diagonal_product` | 29920 | 9980 | x1.00 |

### cholesky6_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 164420 | 122980 | x1.00 |
| `triangular` | 181670 | 140230 | x1.14 |

### cholesky6_l

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `expand` | 45940 | 3600 | x1.00 |

### cholesky6_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `factorize` | 135130 | 99690 | x1.00 |

### cholesky6_solve

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `substitution` | 85300 | 61460 | x1.00 |
| `alt_recip` | 85730 | 61890 | x1.01 |

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
| `alt_recip` | 29610 | 12270 | x1.00 |
| `triangular` | 30760 | 13420 | x1.09 |

### ldlt2_l

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `expand` | 17940 | 100 | x1.00 |

### ldlt2_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `factorize` | 24280 | 6940 | x1.00 |
| `alt_products` | 26160 | 8820 | x1.27 |

### ldlt2_solve

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `substitution` | 29090 | 11650 | x1.00 |
| `alt_recip` | 31490 | 14050 | x1.21 |

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
| `alt_recip` | 45740 | 26300 | x1.00 |
| `triangular` | 49190 | 29750 | x1.13 |

### ldlt3_l

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `expand` | 20940 | 0 | - |

### ldlt3_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `factorize` | 36900 | 17460 | x1.00 |
| `alt_products` | 41940 | 22500 | x1.29 |

### ldlt3_solve

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `substitution` | 37880 | 19140 | x1.00 |
| `alt_recip` | 41480 | 22740 | x1.19 |

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
| `alt_recip` | 76110 | 48870 | x1.00 |
| `triangular` | 83010 | 55770 | x1.14 |

### ldlt4_l

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `expand` | 28840 | 1600 | x1.00 |

### ldlt4_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `factorize` | 57130 | 33690 | x1.00 |
| `alt_products` | 67210 | 43770 | x1.30 |

### ldlt4_solve

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `substitution` | 47370 | 27130 | x1.00 |
| `alt_recip` | 52170 | 31930 | x1.18 |

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
| `alt_recip` | 155240 | 113500 | x1.00 |
| `triangular` | 172490 | 130750 | x1.15 |

### ldlt6_l

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `expand` | 46240 | 3600 | x1.00 |

### ldlt6_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `factorize` | 120420 | 84680 | x1.00 |
| `alt_products` | 145620 | 109880 | x1.30 |

### ldlt6_solve

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `substitution` | 68750 | 44610 | x1.00 |
| `alt_recip` | 75950 | 51810 | x1.16 |

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
| `alt_no_pivot` | 25380 | 7340 | x1.00 |
| `pivot` | 29000 | 10960 | x1.49 |

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
| `substitution` | 31860 | 13720 | x1.00 |
| `alt_recip` | 34390 | 16250 | x1.18 |

### lu2_solve_singular

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `none` | 31370 | 14020 | x1.00 |

### lu2_try_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `columns` | 36840 | 18300 | x1.00 |
| `alt_recip` | 37550 | 19010 | x1.04 |
| `alt_solve_columns` | 49580 | 31040 | x1.70 |

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
| `alt_no_pivot` | 44360 | 22820 | x1.00 |
| `pivot` | 54850 | 33310 | x1.46 |

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
| `substitution` | 42750 | 22710 | x1.00 |
| `alt_recip` | 46480 | 26440 | x1.16 |

### lu3_solve_singular

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `none` | 41760 | 23010 | x1.00 |

### lu3_try_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 64170 | 41930 | x1.00 |
| `columns` | 67190 | 44950 | x1.07 |
| `alt_solve_columns` | 95270 | 73030 | x1.74 |

### lu3_vs_matrix3_determinant

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `cofactors` | 27890 | 10350 | x1.00 |
| `lu` | 58860 | 41320 | x3.99 |

### lu3_vs_matrix3_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `lu` | 100200 | 78360 | x1.00 |
| `cofactors` | 107280 | 85440 | x1.09 |

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
| `alt_no_pivot` | 81170 | 52730 | x1.00 |
| `pivot` | 103380 | 74940 | x1.42 |

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
| `substitution` | 55340 | 33000 | x1.00 |
| `alt_recip` | 60270 | 37930 | x1.15 |

### lu4_solve_singular

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `none` | 53850 | 33300 | x1.00 |

### lu4_try_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 109520 | 78580 | x1.00 |
| `columns` | 118340 | 87400 | x1.11 |
| `alt_solve_columns` | 169540 | 138600 | x1.76 |

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
| `alt_no_pivot` | 209920 | 165480 | x1.00 |
| `pivot` | 277020 | 232580 | x1.41 |

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
| `substitution` | 91320 | 63180 | x1.00 |
| `alt_recip` | 98650 | 70510 | x1.12 |

### lu6_solve_singular

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `none` | 88830 | 63480 | x1.00 |

### lu6_try_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 249120 | 199780 | x1.00 |
| `columns` | 276440 | 227100 | x1.14 |
| `alt_solve_columns` | 440820 | 391480 | x1.96 |

### matrix6_lu

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `determinant` | 270400 | 247460 | x1.00 |
| `solve` | 319400 | 296460 | x1.20 |
| `try_inverse` | 508420 | 485480 | x1.96 |

## nalgebra::linalg::qr::qr2::tests

### qr2_determinant

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `diagonal_product` | 48070 | 7660 | x1.00 |

### qr2_factors

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `unpack` | 41000 | 700 | x1.00 |

### qr2_is_invertible

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `pivots` | 41110 | 700 | x1.00 |

### qr2_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `gram_schmidt` | 42880 | 26340 | x1.00 |

### qr2_solve

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `substitution` | 56510 | 15500 | x1.00 |

### qr2_solve_singular

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `none` | 54920 | 15400 | x1.00 |

### qr2_try_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `substitution` | 59990 | 19580 | x1.00 |

## nalgebra::linalg::qr::qr3::tests

### qr3_determinant

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `diagonal_product` | 98340 | 19010 | x1.00 |

### qr3_determinant_closed_form

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `matrix3_cofactors` | 28500 | 10950 | x1.00 |

### qr3_factors

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `unpack` | 80030 | 700 | x1.00 |

### qr3_is_invertible

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `pivots` | 80130 | 800 | x1.00 |

### qr3_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_classical_gram_schmidt` | 74840 | 57290 | x1.00 |
| `gram_schmidt` | 82780 | 65230 | x1.14 |
| `alt_completed_basis` | 108040 | 90490 | x1.58 |
| `alt_householder` | 191670 | 174120 | x3.04 |

### qr3_solve

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `substitution` | 104320 | 24190 | x1.00 |

### qr3_solve_singular

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `none` | 101150 | 24090 | x1.00 |

### qr3_try_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `substitution` | 123080 | 43750 | x1.00 |

## nalgebra::linalg::qr::qr4::tests

### qr4_determinant

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `diagonal_product` | 186020 | 40410 | x1.00 |

### qr4_factors

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `unpack` | 146310 | 700 | x1.00 |

### qr4_is_invertible

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `pivots` | 146510 | 900 | x1.00 |

### qr4_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `gram_schmidt` | 149550 | 130600 | x1.00 |

### qr4_solve

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `substitution` | 184690 | 38080 | x1.00 |

### qr4_try_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `substitution` | 245850 | 100240 | x1.00 |

## nalgebra::linalg::svd2::tests

### svd2_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 79850 | 63300 | x1.00 |
| `alt_normalised_columns` | 101110 | 84560 | x1.34 |

### svd2_pseudo_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `reciprocals` | 105350 | 25180 | x1.00 |

### svd2_rank

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `comparisons` | 82810 | 2640 | x1.00 |

### svd2_recompose

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `scaled_product` | 98340 | 18170 | x1.00 |

### svd2_singular_values

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_sqrt_eigenvalues` | 36470 | 19920 | x1.00 |
| `from_left_vectors` | 79850 | 63300 | x3.18 |

### svd2_solve

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `divisions` | 98990 | 18220 | x1.00 |

### svd2_to_polar

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `quadform` | 105610 | 25440 | x1.00 |

## nalgebra::linalg::svd3::tests

### svd3_gram

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 30030 | 12480 | x1.00 |
| `transpose_mul_transpose` | 33160 | 15610 | x1.25 |

### svd3_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 664960 | 647410 | x1.00 |
| `alt_one_sided_jacobi` | 712140 | 694590 | x1.07 |
| `alt_normalised_columns` | 717470 | 699920 | x1.08 |

### svd3_pseudo_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `reciprocals` | 714480 | 49800 | x1.00 |

### svd3_rank

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `comparisons` | 668490 | 3810 | x1.00 |

### svd3_recompose

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `scaled_product` | 704050 | 39370 | x1.00 |

### svd3_singular_values

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_sqrt_eigenvalues` | 326200 | 308650 | x1.00 |
| `from_left_vectors` | 664960 | 647410 | x2.10 |

### svd3_solve

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `divisions` | 693260 | 27780 | x1.00 |

### svd3_to_polar

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `quadform` | 718160 | 53480 | x1.00 |

## nalgebra::linalg::symmetric_eigen2::tests

### symmetric_eigen2_eigenvalues

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `closed_form` | 23470 | 7130 | x1.00 |

### symmetric_eigen2_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `closed_form` | 40650 | 24310 | x1.00 |

### symmetric_eigen2_recompose

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `quadform` | 55370 | 14590 | x1.00 |

## nalgebra::linalg::symmetric_eigen3::tests

### symmetric_eigen3_eigenvalues

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `without_eigenvectors` | 304210 | 287270 | x1.00 |
| `via_new` | 544230 | 527290 | x1.84 |

### symmetric_eigen3_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `jacobi_3_sweeps` | 424650 | 407710 | x1.00 |
| `no_renormalisation` | 521710 | 504770 | x1.24 |
| `jacobi_4_sweeps` | 544230 | 527290 | x1.29 |
| `diagonal_input` | 544690 | 527750 | x1.29 |
| `jacobi_5_sweeps` | 663810 | 646870 | x1.59 |
| `jacobi_6_sweeps` | 783390 | 766450 | x1.88 |

### symmetric_eigen3_recompose

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `quadform` | 575820 | 30430 | x1.00 |

### symmetric_eigen3_sweep

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `three_rotations_without_eigenvectors` | 87850 | 69110 | x1.00 |
| `three_rotations` | 138120 | 119380 | x1.73 |

