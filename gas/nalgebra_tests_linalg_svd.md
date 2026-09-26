# Gas report

Sierra gas (`l2_gas`) per benchmark; `net` = raw - group baseline.

## nalgebra_tests_linalg_svd::alt

### svd4_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_one_sided_jacobi` | 1924670 | 1904350 | x1.00 |

## nalgebra_tests_linalg_svd::svd1

### svd1_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 27970 | 10650 | x1.00 |
| `without_u` | 28870 | 11550 | x1.08 |
| `without_v` | 28870 | 11550 | x1.08 |

### svd1_pseudo_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `scaled_product` | 39290 | 11390 | x1.00 |

### svd1_singular_values

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 21040 | 3720 | x1.00 |
| `svd_without_factors` | 28870 | 11550 | x3.10 |

### svd1_to_polar

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `products` | 36170 | 8270 | x1.00 |

## nalgebra_tests_linalg_svd::svd2

### svd2_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 90780 | 72860 | x1.00 |
| `without_u` | 91580 | 73660 | x1.01 |
| `without_v` | 91580 | 73660 | x1.01 |

### svd2_pseudo_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `scaled_product` | 120100 | 28600 | x1.00 |

### svd2_singular_values

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 77540 | 59620 | x1.00 |
| `svd_without_factors` | 91580 | 73660 | x1.24 |

### svd2_to_polar

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `products` | 116940 | 25440 | x1.00 |

## nalgebra_tests_linalg_svd::svd3

### svd3_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 745390 | 726470 | x1.00 |
| `without_u` | 746290 | 727370 | x1.00 |
| `without_v` | 746290 | 727370 | x1.00 |

### svd3_pseudo_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `scaled_product` | 801950 | 54630 | x1.00 |

### svd3_singular_values

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 690340 | 671420 | x1.00 |
| `svd_without_factors` | 746290 | 727370 | x1.08 |

### svd3_to_polar

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `products` | 800800 | 53480 | x1.00 |

## nalgebra_tests_linalg_svd::svd4

### svd4_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 2147440 | 2127120 | x1.00 |
| `without_u` | 2148340 | 2128020 | x1.00 |
| `without_v` | 2148340 | 2128020 | x1.00 |

### svd4_pseudo_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `scaled_product` | 2242450 | 91480 | x1.00 |

### svd4_singular_values

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 1836310 | 1815990 | x1.00 |
| `svd_without_factors` | 2148340 | 2128020 | x1.17 |

### svd4_to_polar

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `products` | 2250260 | 99290 | x1.00 |

