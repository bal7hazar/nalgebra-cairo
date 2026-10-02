# Gas report

Sierra gas (`l2_gas`) per benchmark; `net` = raw - group baseline.

## nalgebra_tests_linalg_svd::alt

### svd4_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_one_sided_jacobi` | 1916840 | 1904350 | x1.00 |

## nalgebra_tests_linalg_svd::svd1

### svd1_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 20140 | 10650 | x1.00 |
| `without_u` | 21040 | 11550 | x1.08 |
| `without_v` | 21040 | 11550 | x1.08 |

### svd1_pseudo_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `scaled_product` | 31460 | 11390 | x1.00 |

### svd1_singular_values

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 13210 | 3720 | x1.00 |
| `svd_without_factors` | 21040 | 11550 | x3.10 |

### svd1_to_polar

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `products` | 28340 | 8270 | x1.00 |

## nalgebra_tests_linalg_svd::svd2

### svd2_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 82950 | 72860 | x1.00 |
| `without_u` | 83750 | 73660 | x1.01 |
| `without_v` | 83750 | 73660 | x1.01 |

### svd2_pseudo_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `scaled_product` | 112270 | 28600 | x1.00 |

### svd2_singular_values

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 69710 | 59620 | x1.00 |
| `svd_without_factors` | 83750 | 73660 | x1.24 |

### svd2_to_polar

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `products` | 109110 | 25440 | x1.00 |

## nalgebra_tests_linalg_svd::svd3

### svd3_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 737560 | 726470 | x1.00 |
| `without_u` | 738460 | 727370 | x1.00 |
| `without_v` | 738460 | 727370 | x1.00 |

### svd3_pseudo_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `scaled_product` | 794120 | 54630 | x1.00 |

### svd3_singular_values

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 682510 | 671420 | x1.00 |
| `svd_without_factors` | 738460 | 727370 | x1.08 |

### svd3_to_polar

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `products` | 792970 | 53480 | x1.00 |

## nalgebra_tests_linalg_svd::svd4

### svd4_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 2139610 | 2127120 | x1.00 |
| `without_u` | 2140510 | 2128020 | x1.00 |
| `without_v` | 2140510 | 2128020 | x1.00 |

### svd4_pseudo_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `scaled_product` | 2234620 | 91480 | x1.00 |

### svd4_singular_values

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 1828480 | 1815990 | x1.00 |
| `svd_without_factors` | 2140510 | 2128020 | x1.17 |

### svd4_to_polar

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `products` | 2242430 | 99290 | x1.00 |

