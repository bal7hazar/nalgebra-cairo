# Gas report

Sierra gas (`l2_gas`) per benchmark; `net` = raw - group baseline.

## nalgebra_tests_linalg_svd5::svd5

### svd5_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 4561550 | 4547260 | x1.00 |
| `without_u` | 4562450 | 4548160 | x1.00 |
| `without_v` | 4562450 | 4548160 | x1.00 |

### svd5_pseudo_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `scaled_product` | 4707230 | 140150 | x1.00 |

### svd5_singular_values

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 4009260 | 3994970 | x1.00 |
| `svd_without_factors` | 4562450 | 4548160 | x1.14 |

### svd5_to_polar

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `products` | 4726810 | 159730 | x1.00 |

