# Gas report

Sierra gas (`l2_gas`) per benchmark; `net` = raw - group baseline.

## nalgebra_tests_linalg_svd_tall::svd2x1

### svd2x1_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 37840 | 20320 | x1.00 |
| `without_u` | 38740 | 21220 | x1.04 |
| `without_v` | 38740 | 21220 | x1.04 |

### svd2x1_pseudo_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `scaled_product` | 50940 | 13170 | x1.00 |

### svd2x1_singular_values

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 26470 | 8950 | x1.00 |
| `svd_without_factors` | 38740 | 21220 | x2.37 |

### svd2x1_to_polar

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `products` | 52960 | 15190 | x1.00 |

## nalgebra_tests_linalg_svd_tall::svd3x1

### svd3x1_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 44140 | 26420 | x1.00 |
| `without_u` | 45040 | 27320 | x1.03 |
| `without_v` | 45040 | 27320 | x1.03 |

### svd3x1_pseudo_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `scaled_product` | 59020 | 14950 | x1.00 |

### svd3x1_singular_values

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 28750 | 11030 | x1.00 |
| `svd_without_factors` | 45040 | 27320 | x2.48 |

### svd3x1_to_polar

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `products` | 67960 | 23890 | x1.00 |

## nalgebra_tests_linalg_svd_tall::svd3x2

### svd3x2_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 215900 | 197580 | x1.00 |
| `without_u` | 216800 | 198480 | x1.00 |
| `without_v` | 216800 | 198480 | x1.00 |

### svd3x2_pseudo_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `scaled_product` | 249690 | 33060 | x1.00 |

### svd3x2_singular_values

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 130110 | 111790 | x1.00 |
| `svd_without_factors` | 216800 | 198480 | x1.78 |

### svd3x2_to_polar

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `products` | 256200 | 39570 | x1.00 |

## nalgebra_tests_linalg_svd_tall::svd4x1

### svd4x1_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 50720 | 32800 | x1.00 |
| `without_u` | 51620 | 33700 | x1.03 |
| `without_v` | 51620 | 33700 | x1.03 |

### svd4x1_pseudo_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `scaled_product` | 67380 | 16730 | x1.00 |

### svd4x1_singular_values

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 31230 | 13310 | x1.00 |
| `svd_without_factors` | 51620 | 33700 | x2.53 |

### svd4x1_to_polar

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `products` | 85020 | 34370 | x1.00 |

## nalgebra_tests_linalg_svd_tall::svd4x2

### svd4x2_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 259700 | 240980 | x1.00 |
| `without_u` | 260600 | 241880 | x1.00 |
| `without_v` | 260600 | 241880 | x1.00 |

### svd4x2_pseudo_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `scaled_product` | 297850 | 37420 | x1.00 |

### svd4x2_singular_values

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 150370 | 131650 | x1.00 |
| `svd_without_factors` | 260600 | 241880 | x1.84 |

### svd4x2_to_polar

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `products` | 315840 | 55410 | x1.00 |

## nalgebra_tests_linalg_svd_tall::svd4x3

### svd4x3_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 997360 | 977840 | x1.00 |
| `without_u` | 998260 | 978740 | x1.00 |
| `without_v` | 998260 | 978740 | x1.00 |

### svd4x3_pseudo_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `scaled_product` | 1061160 | 61870 | x1.00 |

### svd4x3_singular_values

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 791910 | 772390 | x1.00 |
| `svd_without_factors` | 998260 | 978740 | x1.27 |

### svd4x3_to_polar

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `products` | 1075640 | 76350 | x1.00 |

