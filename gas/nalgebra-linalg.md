# Gas report

Sierra gas (`l2_gas`) per benchmark; `net` = raw - group baseline.

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
| `accessor` | 23040 | 0 | - |

### ldlt6_determinant

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `diagonal_product` | 28240 | 8300 | x1.00 |

### ldlt6_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 163400 | 121960 | x1.00 |
| `triangular` | 200990 | 159550 | x1.31 |

### ldlt6_l

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `expand` | 45040 | 3600 | x1.00 |

### ldlt6_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `factorize` | 142770 | 107330 | x1.00 |
| `alt_products` | 167970 | 132530 | x1.23 |

### ldlt6_solve

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `substitution` | 77510 | 53670 | x1.00 |
| `alt_recip` | 84110 | 60270 | x1.12 |

## nalgebra::linalg::lu::lu2::tests

### lu2_permute

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `transpositions` | 18640 | - | x1.00 |

### lu2_permute_rows

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `transpositions` | 20440 | - | x1.00 |

### lu2_solve

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 37210 | - | x1.00 |

## nalgebra::linalg::lu::lu3::tests

### lu3_permute

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `transpositions` | 21480 | - | x1.00 |

### lu3_permute_rows

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `transpositions` | 27320 | - | x1.00 |

### lu3_solve

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 50710 | - | x1.00 |

## nalgebra::linalg::lu::lu4::tests

### lu4_permute

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `transpositions` | 25330 | - | x1.00 |

### lu4_permute_rows

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `transpositions` | 45140 | - | x1.00 |

### lu4_solve

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 65910 | - | x1.00 |

## nalgebra::linalg::lu::lu6::benches

### lu6_permute

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `transpositions` | 42640 | - | x1.00 |

### lu6_permute_rows

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `transpositions` | 97640 | - | x1.00 |

### lu6_solve

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 107110 | - | x1.00 |

## nalgebra::linalg::qr::qr2::tests

### qr2_determinant

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `diagonal_product` | 54110 | - | x1.00 |

## nalgebra::linalg::qr::qr3::tests

### qr3_determinant

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `diagonal_product` | 110670 | - | x1.00 |

### qr3_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_completed_basis` | 123720 | - | x1.00 |
| `alt_householder` | 199220 | - | x1.61 |

## nalgebra::linalg::qr::qr4::tests

### qr4_determinant

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `diagonal_product` | 206480 | - | x1.00 |

## nalgebra::linalg::svd2::tests

### svd2_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_normalised_columns` | 116210 | - | x1.00 |

### svd2_singular_values

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_sqrt_eigenvalues` | 36470 | - | x1.00 |

## nalgebra::linalg::svd3::tests

### svd3_gram

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 30030 | - | x1.00 |
| `transpose_mul_transpose` | 33160 | - | x1.10 |

### svd3_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_one_sided_jacobi` | 779160 | - | x1.00 |
| `alt_normalised_columns` | 808100 | - | x1.04 |

### svd3_singular_values

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_sqrt_eigenvalues` | 380740 | - | x1.00 |

## nalgebra::linalg::symmetric_eigen2::tests

### symmetric_eigen2_eigenvalues

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `closed_form` | 23470 | 7130 | x1.00 |

### symmetric_eigen2_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `closed_form` | 43670 | 27330 | x1.00 |

### symmetric_eigen2_new_matrix

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `public` | 43870 | 27330 | x1.00 |

### symmetric_eigen2_recompose

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `quadform` | 58390 | 14590 | x1.00 |

## nalgebra::linalg::symmetric_eigen3::tests

### symmetric_eigen3_eigenvalues

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `without_eigenvectors` | 358660 | 341720 | x1.00 |
| `via_new` | 607230 | 590290 | x1.73 |

### symmetric_eigen3_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `jacobi_3_sweeps` | 473860 | 456920 | x1.00 |
| `no_renormalisation` | 576070 | 559130 | x1.22 |
| `jacobi_4_sweeps` | 607230 | 590290 | x1.29 |
| `diagonal_input` | 607990 | 591050 | x1.29 |
| `jacobi_5_sweeps` | 740200 | 723260 | x1.58 |
| `jacobi_6_sweeps` | 873370 | 856430 | x1.87 |

### symmetric_eigen3_new_matrix

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `public` | 607830 | 590290 | x1.00 |

### symmetric_eigen3_recompose

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `quadform` | 638820 | 30430 | x1.00 |

### symmetric_eigen3_sweep

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `three_rotations_without_eigenvectors` | 94990 | 76250 | x1.00 |
| `three_rotations` | 151710 | 132970 | x1.74 |

