# Gas report

Sierra gas (`l2_gas`) per benchmark; `net` = raw - group baseline.

## nalgebra_tests_linalg_svd_wide5::svd1x5

### svd1x5_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 55330 | 45040 | x1.00 |
| `without_u` | 56130 | 45840 | x1.02 |
| `without_v` | 56130 | 45840 | x1.02 |

### svd1x5_pseudo_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `scaled_product` | 82590 | 26530 | x1.00 |

### svd1x5_singular_values

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 31060 | 20770 | x1.00 |
| `svd_without_factors` | 56030 | 45740 | x2.20 |

### svd1x5_to_polar

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `products` | 73250 | 17190 | x1.00 |

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
| `eigen_of_gram` | 1098260 | 1085970 | x1.00 |
| `without_u` | 1099160 | 1086870 | x1.00 |
| `without_v` | 1099160 | 1086870 | x1.00 |

### svd3x5_pseudo_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `scaled_product` | 1178680 | 78490 | x1.00 |

### svd3x5_singular_values

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 825910 | 813620 | x1.00 |
| `svd_without_factors` | 1099160 | 1086870 | x1.34 |

### svd3x5_to_polar

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `products` | 1169720 | 69530 | x1.00 |

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

