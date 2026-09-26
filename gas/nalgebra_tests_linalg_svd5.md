# Gas report

Sierra gas (`l2_gas`) per benchmark; `net` = raw - group baseline.

## nalgebra_tests_linalg_svd5::svd5

### svd5_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 4569380 | 4547260 | x1.00 |
| `without_u` | 4570280 | 4548160 | x1.00 |
| `without_v` | 4570280 | 4548160 | x1.00 |

### svd5_pseudo_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `scaled_product` | 4715060 | 140150 | x1.00 |

### svd5_singular_values

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 4017090 | 3994970 | x1.00 |
| `svd_without_factors` | 4570280 | 4548160 | x1.14 |

### svd5_to_polar

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `products` | 4734640 | 159730 | x1.00 |

