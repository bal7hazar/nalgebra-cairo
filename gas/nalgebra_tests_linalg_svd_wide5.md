# Gas report

Sierra gas (`l2_gas`) per benchmark; `net` = raw - group baseline.

## nalgebra_tests_linalg_svd_wide5::svd1x5

### svd1x5_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 63160 | 45040 | x1.00 |
| `without_u` | 63960 | 45840 | x1.02 |
| `without_v` | 63960 | 45840 | x1.02 |

### svd1x5_pseudo_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `scaled_product` | 90420 | 26530 | x1.00 |

### svd1x5_singular_values

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 38890 | 20770 | x1.00 |
| `svd_without_factors` | 63860 | 45740 | x2.20 |

### svd1x5_to_polar

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `products` | 81080 | 17190 | x1.00 |

## nalgebra_tests_linalg_svd_wide5::svd2x5

### svd2x5_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 322180 | 303060 | x1.00 |
| `without_u` | 323080 | 303960 | x1.00 |
| `without_v` | 323080 | 303960 | x1.00 |

### svd2x5_pseudo_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `scaled_product` | 374170 | 51260 | x1.00 |

### svd2x5_singular_values

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 175760 | 156640 | x1.00 |
| `svd_without_factors` | 323080 | 303960 | x1.94 |

### svd2x5_to_polar

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `products` | 361800 | 38890 | x1.00 |

## nalgebra_tests_linalg_svd_wide5::svd3x5

### svd3x5_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 1106090 | 1085970 | x1.00 |
| `without_u` | 1106990 | 1086870 | x1.00 |
| `without_v` | 1106990 | 1086870 | x1.00 |

### svd3x5_pseudo_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `scaled_product` | 1186510 | 78490 | x1.00 |

### svd3x5_singular_values

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 833740 | 813620 | x1.00 |
| `svd_without_factors` | 1106990 | 1086870 | x1.34 |

### svd3x5_to_polar

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `products` | 1177550 | 69530 | x1.00 |

## nalgebra_tests_linalg_svd_wide5::svd4x5

### svd4x5_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 2304150 | 2283030 | x1.00 |
| `without_u` | 2305050 | 2283930 | x1.00 |
| `without_v` | 2305050 | 2283930 | x1.00 |

### svd4x5_pseudo_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `scaled_product` | 2415800 | 108120 | x1.00 |

### svd4x5_singular_values

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 1893210 | 1872090 | x1.00 |
| `svd_without_factors` | 2305050 | 2283930 | x1.22 |

### svd4x5_to_polar

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `products` | 2417290 | 109610 | x1.00 |

