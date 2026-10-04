# Gas report

Sierra gas (`l2_gas`) per benchmark; `net` = raw - group baseline.

## nalgebra_tests_linalg_svd_wide5::svd1x5

### svd1x5_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 53000 | 42710 | x1.00 |
| `without_u` | 53800 | 43510 | x1.02 |
| `without_v` | 53800 | 43510 | x1.02 |

### svd1x5_pseudo_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `scaled_product` | 79060 | 25330 | x1.00 |

### svd1x5_singular_values

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 28730 | 18440 | x1.00 |
| `svd_without_factors` | 53700 | 43410 | x2.35 |

### svd1x5_to_polar

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `products` | 70920 | 17190 | x1.00 |

## nalgebra_tests_linalg_svd_wide5::svd2x5

### svd2x5_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 314350 | 303060 | x1.00 |
| `without_u` | 315250 | 303960 | x1.00 |
| `without_v` | 315250 | 303960 | x1.00 |

### svd2x5_pseudo_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `scaled_product` | 366340 | 51260 | x1.00 |

### svd2x5_singular_values

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 167930 | 156640 | x1.00 |
| `svd_without_factors` | 315250 | 303960 | x1.94 |

### svd2x5_to_polar

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `products` | 353970 | 38890 | x1.00 |

## nalgebra_tests_linalg_svd_wide5::svd3x5

### svd3x5_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 1036390 | 1024100 | x1.00 |
| `without_u` | 1037290 | 1025000 | x1.00 |
| `without_v` | 1037290 | 1025000 | x1.00 |

### svd3x5_pseudo_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `scaled_product` | 1116810 | 78490 | x1.00 |

### svd3x5_singular_values

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 764040 | 751750 | x1.00 |
| `svd_without_factors` | 1037290 | 1025000 | x1.36 |

### svd3x5_to_polar

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `products` | 1107850 | 69530 | x1.00 |

## nalgebra_tests_linalg_svd_wide5::svd4x5

### svd4x5_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 2296320 | 2283030 | x1.00 |
| `without_u` | 2297220 | 2283930 | x1.00 |
| `without_v` | 2297220 | 2283930 | x1.00 |

### svd4x5_pseudo_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `scaled_product` | 2407970 | 108120 | x1.00 |

### svd4x5_singular_values

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 1885380 | 1872090 | x1.00 |
| `svd_without_factors` | 2297220 | 2283930 | x1.22 |

### svd4x5_to_polar

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `products` | 2409460 | 109610 | x1.00 |

