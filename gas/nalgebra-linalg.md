# Gas report

Sierra gas (`l2_gas`) per benchmark; `net` = raw - group baseline.

## nalgebra::linalg::ldlt::benches

### ldlt2_d

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `accessor` | 9210 | 0 | - |

### ldlt2_determinant

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `diagonal_product` | 10090 | 1580 | x1.00 |

### ldlt2_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 24600 | 15090 | x1.00 |
| `triangular` | 27260 | 17750 | x1.18 |

### ldlt2_l

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `expand` | 10110 | 100 | x1.00 |

### ldlt2_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `factorize` | 17960 | 8450 | x1.00 |
| `alt_products` | 19840 | 10330 | x1.22 |

### ldlt2_solve

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `substitution` | 24280 | 14670 | x1.00 |
| `alt_recip` | 26480 | 16870 | x1.15 |

### ldlt3_d

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `accessor` | 10410 | 0 | - |

### ldlt3_determinant

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `diagonal_product` | 12370 | 3260 | x1.00 |

### ldlt3_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 42140 | 30530 | x1.00 |
| `triangular` | 50120 | 38510 | x1.26 |

### ldlt3_l

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `expand` | 13110 | 0 | - |

### ldlt3_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `factorize` | 33600 | 21990 | x1.00 |
| `alt_products` | 38640 | 27030 | x1.23 |

### ldlt3_solve

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `substitution` | 34580 | 23670 | x1.00 |
| `alt_recip` | 37880 | 26970 | x1.14 |

### ldlt4_d

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `accessor` | 11810 | 0 | - |

### ldlt4_determinant

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `diagonal_product` | 14850 | 4940 | x1.00 |

### ldlt4_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 73920 | 54510 | x1.00 |
| `triangular` | 89570 | 70160 | x1.29 |

### ldlt4_l

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `expand` | 21010 | 1600 | x1.00 |

### ldlt4_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `factorize` | 58360 | 42750 | x1.00 |
| `alt_products` | 68440 | 52830 | x1.24 |

### ldlt4_solve

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `substitution` | 45580 | 33170 | x1.00 |
| `alt_recip` | 49980 | 37570 | x1.13 |

### ldlt6_d

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `accessor` | 15210 | 0 | - |

### ldlt6_determinant

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `diagonal_product` | 20410 | 8300 | x1.00 |

### ldlt6_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 155570 | 121960 | x1.00 |
| `triangular` | 193160 | 159550 | x1.31 |

### ldlt6_l

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `expand` | 37210 | 3600 | x1.00 |

### ldlt6_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `factorize` | 134940 | 107330 | x1.00 |
| `alt_products` | 160140 | 132530 | x1.23 |

### ldlt6_solve

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `substitution` | 69680 | 53670 | x1.00 |
| `alt_recip` | 76280 | 60270 | x1.12 |

## nalgebra::linalg::lu::lu2::tests

### lu2_permute

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `transpositions` | 10810 | - | x1.00 |

### lu2_permute_rows

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `transpositions` | 12610 | - | x1.00 |

### lu2_solve

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 29380 | - | x1.00 |

## nalgebra::linalg::lu::lu3::tests

### lu3_permute

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `transpositions` | 13650 | - | x1.00 |

### lu3_permute_rows

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `transpositions` | 19490 | - | x1.00 |

### lu3_solve

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 42880 | - | x1.00 |

## nalgebra::linalg::lu::lu4::tests

### lu4_permute

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `transpositions` | 17500 | - | x1.00 |

### lu4_permute_rows

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `transpositions` | 37310 | - | x1.00 |

### lu4_solve

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 58080 | - | x1.00 |

## nalgebra::linalg::lu::lu6::benches

### lu6_permute

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `transpositions` | 34810 | - | x1.00 |

### lu6_permute_rows

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `transpositions` | 89810 | - | x1.00 |

### lu6_solve

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 99280 | - | x1.00 |

## nalgebra::linalg::qr::qr2::tests

### qr2_determinant

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `diagonal_product` | 46280 | - | x1.00 |

## nalgebra::linalg::qr::qr3::tests

### qr3_determinant

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `diagonal_product` | 100110 | - | x1.00 |

### qr3_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_completed_basis` | 115890 | - | x1.00 |
| `alt_householder` | 191390 | - | x1.65 |

## nalgebra::linalg::qr::qr4::tests

### qr4_determinant

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `diagonal_product` | 198650 | - | x1.00 |

## nalgebra::linalg::svd2::tests

### svd2_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_normalised_columns` | 108980 | - | x1.00 |

### svd2_singular_values

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_sqrt_eigenvalues` | 28640 | - | x1.00 |

## nalgebra::linalg::svd3::tests

### svd3_gram

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 22200 | - | x1.00 |
| `transpose_mul_transpose` | 25330 | - | x1.14 |

### svd3_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_normalised_columns` | 734000 | - | x1.00 |
| `alt_one_sided_jacobi` | 771530 | - | x1.05 |

### svd3_singular_values

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_sqrt_eigenvalues` | 372910 | - | x1.00 |

## nalgebra::linalg::symmetric_eigen2::tests

### symmetric_eigen2_eigenvalues

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `closed_form` | 15640 | 7130 | x1.00 |

### symmetric_eigen2_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `closed_form` | 35840 | 27330 | x1.00 |

### symmetric_eigen2_new_matrix

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `public` | 36040 | 27330 | x1.00 |

### symmetric_eigen2_recompose

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `quadform` | 50560 | 14590 | x1.00 |

## nalgebra::linalg::symmetric_eigen3::tests

### symmetric_eigen3_eigenvalues

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `without_eigenvectors` | 350830 | 341720 | x1.00 |
| `via_new` | 537530 | 528420 | x1.55 |

### symmetric_eigen3_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `no_renormalisation` | 392420 | 383310 | x1.00 |
| `jacobi_3_sweeps` | 419060 | 409950 | x1.07 |
| `jacobi_4_sweeps` | 537530 | 528420 | x1.38 |
| `diagonal_input` | 538290 | 529180 | x1.38 |
| `jacobi_5_sweeps` | 657020 | 647910 | x1.69 |
| `jacobi_6_sweeps` | 776000 | 766890 | x2.00 |

### symmetric_eigen3_new_matrix

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `public` | 538130 | 528420 | x1.00 |

### symmetric_eigen3_recompose

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `quadform` | 569120 | 30430 | x1.00 |

### symmetric_eigen3_sweep

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `three_rotations_without_eigenvectors` | 87160 | 76250 | x1.00 |
| `three_rotations` | 122500 | 111590 | x1.46 |

