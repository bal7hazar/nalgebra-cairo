# Gas report

Sierra gas (`l2_gas`) per benchmark; `net` = raw - group baseline.

## nalgebra_tests_linalg_svd_tall5::svd5x1

### svd5x1_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 62560 | 44440 | x1.00 |
| `without_u` | 63460 | 45340 | x1.02 |
| `without_v` | 63460 | 45340 | x1.02 |

### svd5x1_pseudo_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `scaled_product` | 82600 | 20110 | x1.00 |

### svd5x1_singular_values

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 38890 | 20770 | x1.00 |
| `svd_without_factors` | 63460 | 45340 | x2.18 |

### svd5x1_to_polar

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `products` | 111820 | 49330 | x1.00 |

## nalgebra_tests_linalg_svd_tall5::svd5x2

### svd5x2_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 317780 | 298660 | x1.00 |
| `without_u` | 318680 | 299560 | x1.00 |
| `without_v` | 318680 | 299560 | x1.00 |

### svd5x2_pseudo_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `scaled_product` | 360290 | 41780 | x1.00 |

### svd5x2_singular_values

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 175760 | 156640 | x1.00 |
| `svd_without_factors` | 318680 | 299560 | x1.91 |

### svd5x2_to_polar

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `products` | 391840 | 73330 | x1.00 |

## nalgebra_tests_linalg_svd_tall5::svd5x3

### svd5x3_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 1099590 | 1079470 | x1.00 |
| `without_u` | 1100490 | 1080370 | x1.00 |
| `without_v` | 1100490 | 1080370 | x1.00 |

### svd5x3_pseudo_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `scaled_product` | 1170530 | 69010 | x1.00 |

### svd5x3_singular_values

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 832140 | 812020 | x1.00 |
| `svd_without_factors` | 1100490 | 1080370 | x1.33 |

### svd5x3_to_polar

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `products` | 1201250 | 99730 | x1.00 |

## nalgebra_tests_linalg_svd_tall5::svd5x4

### svd5x4_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 2295150 | 2274030 | x1.00 |
| `without_u` | 2296050 | 2274930 | x1.00 |
| `without_v` | 2296050 | 2274930 | x1.00 |

### svd5x4_pseudo_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `scaled_product` | 2400480 | 101800 | x1.00 |

### svd5x4_singular_values

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 1891110 | 1869990 | x1.00 |
| `svd_without_factors` | 2296050 | 2274930 | x1.22 |

### svd5x4_to_polar

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `products` | 2427210 | 128530 | x1.00 |

