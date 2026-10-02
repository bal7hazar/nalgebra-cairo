# Gas report

Sierra gas (`l2_gas`) per benchmark; `net` = raw - group baseline.

## nalgebra_tests_linalg_svd_wide6b::svd4x6

### svd4x6_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 2404410 | 2390320 | x1.00 |
| `without_u` | 2405310 | 2391220 | x1.00 |
| `without_v` | 2405310 | 2391220 | x1.00 |

### svd4x6_pseudo_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `scaled_product` | 2532700 | 124760 | x1.00 |

### svd4x6_singular_values

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 1927610 | 1913520 | x1.00 |
| `svd_without_factors` | 2405310 | 2391220 | x1.25 |

### svd4x6_to_polar

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `products` | 2527870 | 119930 | x1.00 |

## nalgebra_tests_linalg_svd_wide6b::svd5x6

### svd5x6_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 4723130 | 4702840 | x1.00 |
| `without_u` | 4724030 | 4703740 | x1.00 |
| `without_v` | 4724030 | 4703740 | x1.00 |

### svd5x6_pseudo_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `scaled_product` | 4890640 | 161980 | x1.00 |

### svd5x6_singular_values

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eigen_of_gram` | 4072120 | 4051830 | x1.00 |
| `svd_without_factors` | 4724030 | 4703740 | x1.16 |

### svd5x6_to_polar

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `products` | 4902290 | 173630 | x1.00 |

