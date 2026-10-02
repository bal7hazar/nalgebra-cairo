# Gas report

Sierra gas (`l2_gas`) per benchmark; `net` = raw - group baseline.

## nalgebra_tests_linalg_svd6::svd6

### svd6_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 7527300 | 7500910 | x1.00 |
| `without_u` | 7528200 | 7501810 | x1.00 |
| `without_v` | 7528200 | 7501810 | x1.00 |

### svd6_pseudo_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `scaled_product` | 7737300 | 202070 | x1.00 |

### svd6_singular_values

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 6701020 | 6674630 | x1.00 |
| `svd_without_factors` | 7528200 | 7501810 | x1.12 |

### svd6_to_polar

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `products` | 7773600 | 238370 | x1.00 |

