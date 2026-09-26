# Gas report

Sierra gas (`l2_gas`) per benchmark; `net` = raw - group baseline.

## nalgebra_tests_linalg_svd_wide6::svd1x6

### svd1x6_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 69720 | 51400 | x1.00 |
| `without_u` | 70520 | 52200 | x1.02 |
| `without_v` | 70520 | 52200 | x1.02 |

### svd1x6_pseudo_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `scaled_product` | 100640 | 30090 | x1.00 |

### svd1x6_singular_values

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 41270 | 22950 | x1.00 |
| `svd_without_factors` | 70420 | 52100 | x2.27 |

### svd1x6_to_polar

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `products` | 89720 | 19170 | x1.00 |

## nalgebra_tests_linalg_svd_wide6::svd2x6

### svd2x6_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 365260 | 345740 | x1.00 |
| `without_u` | 366160 | 346640 | x1.00 |
| `without_v` | 366160 | 346640 | x1.00 |

### svd2x6_pseudo_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `scaled_product` | 424770 | 58780 | x1.00 |

### svd2x6_singular_values

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 194900 | 175380 | x1.00 |
| `svd_without_factors` | 366160 | 346640 | x1.98 |

### svd2x6_to_polar

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `products` | 409240 | 43250 | x1.00 |

## nalgebra_tests_linalg_svd_wide6::svd3x6

### svd3x6_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 1180400 | 1159680 | x1.00 |
| `without_u` | 1181300 | 1160580 | x1.00 |
| `without_v` | 1181300 | 1160580 | x1.00 |

### svd3x6_pseudo_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `scaled_product` | 1272700 | 90370 | x1.00 |

### svd3x6_singular_values

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 864050 | 843330 | x1.00 |
| `svd_without_factors` | 1181300 | 1160580 | x1.38 |

### svd3x6_to_polar

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `products` | 1259000 | 76670 | x1.00 |

