# Gas report

Sierra gas (`l2_gas`) per benchmark; `net` = raw - group baseline.

## nalgebra_tests_linalg_svd_tall6b::svd6x4

### svd6x4_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 2394610 | 2380520 | x1.00 |
| `without_u` | 2395510 | 2381420 | x1.00 |
| `without_v` | 2395510 | 2381420 | x1.00 |

### svd6x4_pseudo_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `scaled_product` | 2510260 | 112120 | x1.00 |

### svd6x4_singular_values

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 1925110 | 1911020 | x1.00 |
| `svd_without_factors` | 2395510 | 2381420 | x1.25 |

### svd6x4_to_polar

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `products` | 2558390 | 160250 | x1.00 |

## nalgebra_tests_linalg_svd_tall6b::svd6x5

### svd6x5_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 4710230 | 4689940 | x1.00 |
| `without_u` | 4711130 | 4690840 | x1.00 |
| `without_v` | 4711130 | 4690840 | x1.00 |

### svd6x5_pseudo_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `scaled_product` | 4869840 | 154080 | x1.00 |

### svd6x5_singular_values

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 4069020 | 4048730 | x1.00 |
| `svd_without_factors` | 4711130 | 4690840 | x1.16 |

### svd6x5_to_polar

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `products` | 4913670 | 197910 | x1.00 |

