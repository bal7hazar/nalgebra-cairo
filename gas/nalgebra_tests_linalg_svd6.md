# Gas report

Sierra gas (`l2_gas`) per benchmark; `net` = raw - group baseline.

## nalgebra_tests_linalg_svd6::svd6

### svd6_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 7530230 | 7500910 | x1.00 |
| `without_u` | 7531130 | 7501810 | x1.00 |
| `without_v` | 7531130 | 7501810 | x1.00 |

### svd6_pseudo_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `scaled_product` | 7740230 | 202070 | x1.00 |

### svd6_singular_values

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 6703950 | 6674630 | x1.00 |
| `svd_without_factors` | 7531130 | 7501810 | x1.12 |

### svd6_to_polar

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `products` | 7776530 | 238370 | x1.00 |

