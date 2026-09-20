# Gas report

Sierra gas (`l2_gas`) per benchmark; `net` = raw - group baseline.

## nalgebra::linalg::cholesky::benches

### cholesky2_determinant

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `diagonal_product` | 19940 | 3600 | x1.00 |

### cholesky2_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 33450 | 16110 | x1.00 |
| `triangular` | 34420 | 17080 | x1.06 |

### cholesky2_l

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `expand` | 17940 | 100 | x1.00 |

### cholesky2_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `factorize` | 29690 | 12350 | x1.00 |

### cholesky2_solve

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `substitution` | 35120 | 17680 | x1.00 |
| `alt_recip` | 35700 | 18260 | x1.03 |

### cholesky3_determinant

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `diagonal_product` | 22390 | 5450 | x1.00 |

### cholesky3_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 52450 | 33010 | x1.00 |
| `triangular` | 55360 | 35920 | x1.09 |

### cholesky3_l

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `expand` | 20940 | 0 | - |

### cholesky3_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `factorize` | 44920 | 25480 | x1.00 |

### cholesky3_solve

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `substitution` | 47060 | 28320 | x1.00 |
| `alt_recip` | 48150 | 29410 | x1.04 |

### cholesky4_determinant

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `diagonal_product` | 25040 | 7300 | x1.00 |

### cholesky4_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 86200 | 58960 | x1.00 |
| `triangular` | 92020 | 64780 | x1.10 |

### cholesky4_l

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `expand` | 28840 | 1600 | x1.00 |

### cholesky4_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `factorize` | 68120 | 44680 | x1.00 |

### cholesky4_solve

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `substitution` | 59700 | 39460 | x1.00 |
| `alt_recip` | 61300 | 41060 | x1.04 |

### cholesky6_determinant

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `diagonal_product` | 30940 | 11000 | x1.00 |

### cholesky6_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 173690 | 132250 | x1.00 |
| `triangular` | 188240 | 146800 | x1.11 |

### cholesky6_l

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `expand` | 45940 | 3600 | x1.00 |

### cholesky6_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `factorize` | 137530 | 102090 | x1.00 |

### cholesky6_solve

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `substitution` | 87080 | 63240 | x1.00 |
| `alt_recip` | 89700 | 65860 | x1.04 |

## nalgebra::linalg::ldlt::benches

### ldlt2_d

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `accessor` | 17040 | 0 | - |

### ldlt2_determinant

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `diagonal_product` | 18090 | 1750 | x1.00 |

### ldlt2_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 29950 | 12610 | x1.00 |
| `triangular` | 30920 | 13580 | x1.08 |

### ldlt2_l

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `expand` | 17940 | 100 | x1.00 |

### ldlt2_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `factorize` | 24070 | 6730 | x1.00 |
| `alt_products` | 26120 | 8780 | x1.30 |

### ldlt2_solve

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `substitution` | 29040 | 11600 | x1.00 |
| `alt_recip` | 32000 | 14560 | x1.26 |

### ldlt3_d

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `accessor` | 18240 | 0 | - |

### ldlt3_determinant

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `diagonal_product` | 20540 | 3600 | x1.00 |

### ldlt3_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 47200 | 27760 | x1.00 |
| `triangular` | 50110 | 30670 | x1.10 |

### ldlt3_l

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `expand` | 20940 | 0 | - |

### ldlt3_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `factorize` | 37010 | 17570 | x1.00 |
| `alt_products` | 42560 | 23120 | x1.32 |

### ldlt3_solve

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `substitution` | 38160 | 19420 | x1.00 |
| `alt_recip` | 42600 | 23860 | x1.23 |

### ldlt4_d

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `accessor` | 19640 | 0 | - |

### ldlt4_determinant

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `diagonal_product` | 23190 | 5450 | x1.00 |

### ldlt4_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 79200 | 51960 | x1.00 |
| `triangular` | 85020 | 57780 | x1.11 |

### ldlt4_l

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `expand` | 28840 | 1600 | x1.00 |

### ldlt4_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `factorize` | 57720 | 34280 | x1.00 |
| `alt_products` | 68820 | 45380 | x1.32 |

### ldlt4_solve

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `substitution` | 47980 | 27740 | x1.00 |
| `alt_recip` | 53900 | 33660 | x1.21 |

### ldlt6_d

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `accessor` | 23340 | 0 | - |

### ldlt6_determinant

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `diagonal_product` | 29390 | 9150 | x1.00 |

### ldlt6_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 163490 | 121750 | x1.00 |
| `triangular` | 178040 | 136300 | x1.12 |

### ldlt6_l

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `expand` | 46240 | 3600 | x1.00 |

### ldlt6_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `factorize` | 122450 | 86710 | x1.00 |
| `alt_products` | 150200 | 114460 | x1.32 |

### ldlt6_solve

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `substitution` | 70020 | 45880 | x1.00 |
| `alt_recip` | 78900 | 54760 | x1.19 |

## nalgebra::linalg::lu::lu2::tests

### lu2_determinant

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `pivots` | 19690 | 2950 | x1.00 |

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
| `alt_no_pivot` | 25270 | 7230 | x1.00 |
| `pivot` | 28590 | 10550 | x1.46 |

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
| `substitution` | 32380 | 14240 | x1.00 |
| `alt_recip` | 34900 | 16760 | x1.18 |

### lu2_solve_singular

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `none` | 31890 | 14540 | x1.00 |

### lu2_try_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `columns` | 37720 | 19180 | x1.00 |
| `alt_recip` | 38400 | 19860 | x1.04 |
| `alt_solve_columns` | 50620 | 32080 | x1.67 |

## nalgebra::linalg::lu::lu3::tests

### lu3_determinant

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `pivots` | 25820 | 7880 | x1.00 |

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
| `alt_no_pivot` | 44810 | 23270 | x1.00 |
| `pivot` | 54800 | 33260 | x1.43 |

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
| `substitution` | 43600 | 23560 | x1.00 |
| `alt_recip` | 47600 | 27560 | x1.17 |

### lu3_solve_singular

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `none` | 42610 | 23860 | x1.00 |

### lu3_try_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 67190 | 44950 | x1.00 |
| `columns` | 68470 | 46230 | x1.03 |
| `alt_solve_columns` | 97820 | 75580 | x1.68 |

### lu3_vs_matrix3_determinant

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `cofactors` | 28200 | 10660 | x1.00 |
| `lu` | 58780 | 41240 | x3.87 |

### lu3_vs_matrix3_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `lu` | 101430 | 79590 | x1.00 |
| `cofactors` | 110780 | 88940 | x1.12 |

## nalgebra::linalg::lu::lu4::tests

### lu4_determinant

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `pivots` | 30680 | 11140 | x1.00 |

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
| `alt_no_pivot` | 83120 | 54680 | x1.00 |
| `pivot` | 104430 | 75990 | x1.39 |

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
| `substitution` | 56520 | 34180 | x1.00 |
| `alt_recip` | 62000 | 39660 | x1.16 |

### lu4_solve_singular

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `none` | 55030 | 34480 | x1.00 |

### lu4_try_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 115190 | 84250 | x1.00 |
| `columns` | 120810 | 89870 | x1.07 |
| `alt_solve_columns` | 174260 | 143320 | x1.70 |

## nalgebra::linalg::lu::lu6::benches

### lu6_determinant

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `pivots` | 42800 | 18860 | x1.00 |

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
| `alt_no_pivot` | 218750 | 174310 | x1.00 |
| `pivot` | 283850 | 239410 | x1.37 |

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
| `substitution` | 93160 | 65020 | x1.00 |
| `alt_recip` | 101600 | 73460 | x1.13 |

### lu6_solve_singular

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `none` | 90670 | 65320 | x1.00 |

### lu6_try_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 262640 | 213300 | x1.00 |
| `columns` | 282760 | 233420 | x1.09 |
| `alt_solve_columns` | 451860 | 402520 | x1.89 |

### matrix6_lu

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `determinant` | 277710 | 254770 | x1.00 |
| `solve` | 328070 | 305130 | x1.20 |
| `try_inverse` | 521570 | 498630 | x1.96 |

## nalgebra::linalg::qr::qr2::tests

### qr2_determinant

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `diagonal_product` | 50350 | 7630 | x1.00 |

### qr2_factors

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `unpack` | 43310 | 700 | x1.00 |

### qr2_is_invertible

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `pivots` | 43420 | 700 | x1.00 |

### qr2_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `gram_schmidt` | 43250 | 26710 | x1.00 |

### qr2_solve

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `substitution` | 59510 | 16190 | x1.00 |

### qr2_solve_singular

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `none` | 59410 | 16090 | x1.00 |

### qr2_try_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `substitution` | 62700 | 19980 | x1.00 |

## nalgebra::linalg::qr::qr3::tests

### qr3_determinant

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `diagonal_product` | 101680 | 19290 | x1.00 |

### qr3_determinant_closed_form

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `matrix3_cofactors` | 28810 | 11260 | x1.00 |

### qr3_factors

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `unpack` | 83090 | 700 | x1.00 |

### qr3_is_invertible

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `pivots` | 83190 | 800 | x1.00 |

### qr3_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_classical_gram_schmidt` | 76280 | 58730 | x1.00 |
| `gram_schmidt` | 84130 | 66580 | x1.13 |
| `alt_completed_basis` | 109920 | 92370 | x1.57 |
| `alt_householder` | 195130 | 177580 | x3.02 |

### qr3_solve

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `substitution` | 108400 | 25210 | x1.00 |

### qr3_solve_singular

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `none` | 108100 | 25110 | x1.00 |

### qr3_try_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `substitution` | 127070 | 44680 | x1.00 |

## nalgebra::linalg::qr::qr4::tests

### qr4_determinant

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `diagonal_product` | 194340 | 42390 | x1.00 |

### qr4_factors

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `unpack` | 152650 | 700 | x1.00 |

### qr4_is_invertible

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `pivots` | 152850 | 900 | x1.00 |

### qr4_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `gram_schmidt` | 153790 | 134840 | x1.00 |

### qr4_solve

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `substitution` | 192380 | 39430 | x1.00 |

### qr4_try_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `substitution` | 254870 | 102920 | x1.00 |

## nalgebra::linalg::svd2::tests

### svd2_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 81290 | 64740 | x1.00 |
| `alt_normalised_columns` | 103090 | 86540 | x1.34 |

### svd2_pseudo_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `reciprocals` | 96760 | 26370 | x1.00 |

### svd2_rank

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `comparisons` | 73030 | 2640 | x1.00 |

### svd2_recompose

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `scaled_product` | 89550 | 19160 | x1.00 |

### svd2_singular_values

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_sqrt_eigenvalues` | 37320 | 20770 | x1.00 |
| `from_left_vectors` | 81290 | 64740 | x3.12 |

### svd2_solve

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `divisions` | 89700 | 18710 | x1.00 |

### svd2_to_polar

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `quadform` | 96960 | 26570 | x1.00 |

## nalgebra::linalg::svd3::tests

### svd3_gram

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 31050 | 13500 | x1.00 |
| `transpose_mul_transpose` | 33810 | 16260 | x1.20 |

### svd3_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 692950 | 675400 | x1.00 |
| `alt_one_sided_jacobi` | 737370 | 719820 | x1.07 |
| `alt_normalised_columns` | 746900 | 729350 | x1.08 |

### svd3_pseudo_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `reciprocals` | 736160 | 52790 | x1.00 |

### svd3_rank

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `comparisons` | 687180 | 3810 | x1.00 |

### svd3_recompose

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `scaled_product` | 725430 | 42060 | x1.00 |

### svd3_singular_values

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_sqrt_eigenvalues` | 343420 | 325870 | x1.00 |
| `from_left_vectors` | 692950 | 675400 | x2.07 |

### svd3_solve

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `divisions` | 712870 | 28700 | x1.00 |

### svd3_to_polar

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `quadform` | 740190 | 56820 | x1.00 |

## nalgebra::linalg::symmetric_eigen2::tests

### symmetric_eigen2_eigenvalues

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `closed_form` | 23810 | 7470 | x1.00 |

### symmetric_eigen2_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `closed_form` | 40870 | 24530 | x1.00 |

### symmetric_eigen2_recompose

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `quadform` | 50640 | 15410 | x1.00 |

## nalgebra::linalg::symmetric_eigen3::tests

### symmetric_eigen3_eigenvalues

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `without_eigenvectors` | 320410 | 303470 | x1.00 |
| `via_new` | 567710 | 550770 | x1.81 |

### symmetric_eigen3_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `jacobi_3_sweeps` | 442130 | 425190 | x1.00 |
| `no_renormalisation` | 545250 | 528310 | x1.24 |
| `jacobi_4_sweeps` | 567710 | 550770 | x1.30 |
| `diagonal_input` | 568170 | 551230 | x1.30 |
| `jacobi_5_sweeps` | 693290 | 676350 | x1.59 |
| `jacobi_6_sweeps` | 818870 | 801930 | x1.89 |

### symmetric_eigen3_recompose

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `quadform` | 601480 | 32610 | x1.00 |

### symmetric_eigen3_sweep

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `three_rotations_without_eigenvectors` | 92200 | 73460 | x1.00 |
| `three_rotations` | 144120 | 125380 | x1.71 |

