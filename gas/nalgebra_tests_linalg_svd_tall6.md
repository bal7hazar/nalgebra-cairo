# Gas report

Sierra gas (`l2_gas`) per benchmark; `net` = raw - group baseline.

## nalgebra_tests_linalg_svd_tall6::svd6x1

### svd6x1_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 69120 | 50800 | x1.00 |
| `without_u` | 70020 | 51700 | x1.02 |
| `without_v` | 70020 | 51700 | x1.02 |

### svd6x1_pseudo_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `scaled_product` | 91140 | 22090 | x1.00 |

### svd6x1_singular_values

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 41270 | 22950 | x1.00 |
| `svd_without_factors` | 70020 | 51700 | x2.25 |

### svd6x1_to_polar

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `products` | 133120 | 64070 | x1.00 |

## nalgebra_tests_linalg_svd_tall6::svd6x2

### svd6x2_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 360460 | 340940 | x1.00 |
| `without_u` | 361360 | 341840 | x1.00 |
| `without_v` | 361360 | 341840 | x1.00 |

### svd6x2_pseudo_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `scaled_product` | 407330 | 46140 | x1.00 |

### svd6x2_singular_values

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 194900 | 175380 | x1.00 |
| `svd_without_factors` | 361360 | 341840 | x1.95 |

### svd6x2_to_polar

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `products` | 454520 | 93330 | x1.00 |

## nalgebra_tests_linalg_svd_tall6::svd6x3

### svd6x3_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 1173300 | 1152580 | x1.00 |
| `without_u` | 1174200 | 1153480 | x1.00 |
| `without_v` | 1174200 | 1153480 | x1.00 |

### svd6x3_pseudo_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `scaled_product` | 1251380 | 76150 | x1.00 |

### svd6x3_singular_values

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 862150 | 841430 | x1.00 |
| `svd_without_factors` | 1174200 | 1153480 | x1.37 |

### svd6x3_to_polar

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `products` | 1300620 | 125390 | x1.00 |

