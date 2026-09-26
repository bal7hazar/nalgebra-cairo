# Gas report

Sierra gas (`l2_gas`) per benchmark; `net` = raw - group baseline.

## nalgebra_tests_linalg_svd_tall6b::svd6x4

### svd6x4_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 2402440 | 2380520 | x1.00 |
| `without_u` | 2403340 | 2381420 | x1.00 |
| `without_v` | 2403340 | 2381420 | x1.00 |

### svd6x4_pseudo_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `scaled_product` | 2518090 | 112120 | x1.00 |

### svd6x4_singular_values

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 1932940 | 1911020 | x1.00 |
| `svd_without_factors` | 2403340 | 2381420 | x1.25 |

### svd6x4_to_polar

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `products` | 2566220 | 160250 | x1.00 |

## nalgebra_tests_linalg_svd_tall6b::svd6x5

### svd6x5_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 4717460 | 4689940 | x1.00 |
| `without_u` | 4718360 | 4690840 | x1.00 |
| `without_v` | 4718360 | 4690840 | x1.00 |

### svd6x5_pseudo_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `scaled_product` | 4877070 | 154080 | x1.00 |

### svd6x5_singular_values

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 4076250 | 4048730 | x1.00 |
| `svd_without_factors` | 4718360 | 4690840 | x1.16 |

### svd6x5_to_polar

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `products` | 4920900 | 197910 | x1.00 |

