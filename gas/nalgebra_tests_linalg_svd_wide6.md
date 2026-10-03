# Gas report

Sierra gas (`l2_gas`) per benchmark; `net` = raw - group baseline.

## nalgebra_tests_linalg_svd_wide6::svd1x6

### svd1x6_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 61890 | 51400 | x1.00 |
| `without_u` | 62690 | 52200 | x1.02 |
| `without_v` | 62690 | 52200 | x1.02 |

### svd1x6_pseudo_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `scaled_product` | 92810 | 30090 | x1.00 |

### svd1x6_singular_values

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 33440 | 22950 | x1.00 |
| `svd_without_factors` | 62590 | 52100 | x2.27 |

### svd1x6_to_polar

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `products` | 81890 | 19170 | x1.00 |

## nalgebra_tests_linalg_svd_wide6::svd2x6

### svd2x6_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 357430 | 345740 | x1.00 |
| `without_u` | 358330 | 346640 | x1.00 |
| `without_v` | 358330 | 346640 | x1.00 |

### svd2x6_pseudo_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `scaled_product` | 416940 | 58780 | x1.00 |

### svd2x6_singular_values

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 187070 | 175380 | x1.00 |
| `svd_without_factors` | 358330 | 346640 | x1.98 |

### svd2x6_to_polar

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `products` | 401410 | 43250 | x1.00 |

## nalgebra_tests_linalg_svd_wide6::svd3x6

### svd3x6_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 1110700 | 1097810 | x1.00 |
| `without_u` | 1111600 | 1098710 | x1.00 |
| `without_v` | 1111600 | 1098710 | x1.00 |

### svd3x6_pseudo_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `scaled_product` | 1203000 | 90370 | x1.00 |

### svd3x6_singular_values

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 794350 | 781460 | x1.00 |
| `svd_without_factors` | 1111600 | 1098710 | x1.41 |

### svd3x6_to_polar

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `products` | 1189300 | 76670 | x1.00 |

