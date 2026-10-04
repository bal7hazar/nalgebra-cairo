# Gas report

Sierra gas (`l2_gas`) per benchmark; `net` = raw - group baseline.

## nalgebra_tests_geometry_transform::affine::benches

### affine2_transform_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `affine` | 17630 | 4160 | x1.00 |

### affine2_try_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_block` | 52170 | 42460 | x1.00 |
| `refined_block` | 60890 | 51180 | x1.21 |
| `alt_full_matrix` | 87390 | 77680 | x1.83 |

### affine3_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `full_product` | 314730 | 43010 | x1.00 |

### affine3_mul_isometry

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `structured` | 139250 | 63080 | x1.00 |
| `alt_full_product` | 141650 | 65480 | x1.04 |

### affine3_mul_rotation

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `structured` | 113390 | 31190 | x1.00 |
| `alt_full_product` | 125210 | 43010 | x1.38 |

### affine3_mul_translation

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `structured` | 37150 | 10320 | x1.00 |
| `alt_full_product` | 69840 | 43010 | x4.17 |

### affine3_transform_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `affine` | 33750 | 10470 | x1.00 |
| `alt_homogeneous` | 48910 | 25630 | x2.45 |

### affine3_try_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_block` | 99930 | 88820 | x1.00 |
| `refined_block` | 114210 | 103100 | x1.16 |
| `alt_full_matrix` | 206330 | 195220 | x2.20 |

### isometry3_mul_affine3

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `structured` | 126610 | 56760 | x1.00 |
| `alt_full_product` | 135330 | 65480 | x1.15 |

## nalgebra_tests_geometry_transform::projective::benches

### affine3_div_projective3

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `inverse_then_product` | 498970 | 238330 | x1.00 |

### projective3_inverse_transform_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `inverse_then_transform` | 454710 | 220950 | x1.00 |

### projective3_mul_affine3

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `full_product` | 108330 | 43010 | x1.00 |

### projective3_transform_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `homogeneous` | 64070 | 25630 | x1.00 |

### projective3_try_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `matrix4` | 412750 | 196920 | x1.00 |

