# Gas report

Sierra gas (`l2_gas`) per benchmark; `net` = raw - group baseline.

## nalgebra_tests_linalg_svd_tall5::svd5x1

### svd5x1_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 54730 | 44440 | x1.00 |
| `without_u` | 55630 | 45340 | x1.02 |
| `without_v` | 55630 | 45340 | x1.02 |

### svd5x1_pseudo_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `scaled_product` | 74770 | 20110 | x1.00 |

### svd5x1_singular_values

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 31060 | 20770 | x1.00 |
| `svd_without_factors` | 55630 | 45340 | x2.18 |

### svd5x1_to_polar

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `products` | 103990 | 49330 | x1.00 |

## nalgebra_tests_linalg_svd_tall5::svd5x2

### svd5x2_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 309950 | 298660 | x1.00 |
| `without_u` | 310850 | 299560 | x1.00 |
| `without_v` | 310850 | 299560 | x1.00 |

### svd5x2_pseudo_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `scaled_product` | 352460 | 41780 | x1.00 |

### svd5x2_singular_values

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 167930 | 156640 | x1.00 |
| `svd_without_factors` | 310850 | 299560 | x1.91 |

### svd5x2_to_polar

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `products` | 384010 | 73330 | x1.00 |

## nalgebra_tests_linalg_svd_tall5::svd5x3

### svd5x3_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 1029890 | 1017600 | x1.00 |
| `without_u` | 1030790 | 1018500 | x1.00 |
| `without_v` | 1030790 | 1018500 | x1.00 |

### svd5x3_pseudo_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `scaled_product` | 1100830 | 69010 | x1.00 |

### svd5x3_singular_values

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 762440 | 750150 | x1.00 |
| `svd_without_factors` | 1030790 | 1018500 | x1.36 |

### svd5x3_to_polar

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `products` | 1131550 | 99730 | x1.00 |

## nalgebra_tests_linalg_svd_tall5::svd5x4

### svd5x4_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 2287320 | 2274030 | x1.00 |
| `without_u` | 2288220 | 2274930 | x1.00 |
| `without_v` | 2288220 | 2274930 | x1.00 |

### svd5x4_pseudo_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `scaled_product` | 2392650 | 101800 | x1.00 |

### svd5x4_singular_values

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 1883280 | 1869990 | x1.00 |
| `svd_without_factors` | 2288220 | 2274930 | x1.22 |

### svd5x4_to_polar

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `products` | 2419380 | 128530 | x1.00 |

