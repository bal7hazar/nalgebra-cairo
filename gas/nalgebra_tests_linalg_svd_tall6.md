# Gas report

Sierra gas (`l2_gas`) per benchmark; `net` = raw - group baseline.

## nalgebra_tests_linalg_svd_tall6::svd6x1

### svd6x1_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 61290 | 50800 | x1.00 |
| `without_u` | 62190 | 51700 | x1.02 |
| `without_v` | 62190 | 51700 | x1.02 |

### svd6x1_pseudo_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `scaled_product` | 83310 | 22090 | x1.00 |

### svd6x1_singular_values

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 33440 | 22950 | x1.00 |
| `svd_without_factors` | 62190 | 51700 | x2.25 |

### svd6x1_to_polar

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `products` | 125290 | 64070 | x1.00 |

## nalgebra_tests_linalg_svd_tall6::svd6x2

### svd6x2_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 352630 | 340940 | x1.00 |
| `without_u` | 353530 | 341840 | x1.00 |
| `without_v` | 353530 | 341840 | x1.00 |

### svd6x2_pseudo_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `scaled_product` | 399500 | 46140 | x1.00 |

### svd6x2_singular_values

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 187070 | 175380 | x1.00 |
| `svd_without_factors` | 353530 | 341840 | x1.95 |

### svd6x2_to_polar

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `products` | 446690 | 93330 | x1.00 |

## nalgebra_tests_linalg_svd_tall6::svd6x3

### svd6x3_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 1103600 | 1090710 | x1.00 |
| `without_u` | 1104500 | 1091610 | x1.00 |
| `without_v` | 1104500 | 1091610 | x1.00 |

### svd6x3_pseudo_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `scaled_product` | 1181680 | 76150 | x1.00 |

### svd6x3_singular_values

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 792450 | 779560 | x1.00 |
| `svd_without_factors` | 1104500 | 1091610 | x1.40 |

### svd6x3_to_polar

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `products` | 1230920 | 125390 | x1.00 |

