# Gas report

Sierra gas (`l2_gas`) per benchmark; `net` = raw - group baseline.

## nalgebra_tests_geometry_transform::affine::benches

### affine2_transform_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `affine` | 25460 | 4160 | x1.00 |

### affine2_try_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_block` | 60000 | 42460 | x1.00 |
| `refined_block` | 68720 | 51180 | x1.21 |
| `alt_full_matrix` | 105900 | 88360 | x2.08 |

### affine3_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `full_product` | 349180 | 43010 | x1.00 |

### affine3_mul_isometry

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `structured` | 147080 | 63080 | x1.00 |
| `alt_full_product` | 149480 | 65480 | x1.04 |

### affine3_mul_rotation

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `structured` | 121220 | 31190 | x1.00 |
| `alt_full_product` | 133040 | 43010 | x1.38 |

### affine3_mul_translation

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `structured` | 44980 | 10320 | x1.00 |
| `alt_full_product` | 77670 | 43010 | x4.17 |

### affine3_transform_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `affine` | 41580 | 10470 | x1.00 |
| `alt_homogeneous` | 56740 | 25630 | x2.45 |

### affine3_try_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_block` | 121070 | 102130 | x1.00 |
| `refined_block` | 135350 | 116410 | x1.14 |
| `alt_full_matrix` | 214160 | 195220 | x1.91 |

### isometry3_mul_affine3

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `structured` | 134440 | 56760 | x1.00 |
| `alt_full_product` | 143160 | 65480 | x1.15 |

## nalgebra_tests_geometry_transform::projective::benches

### affine3_div_projective3

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `inverse_then_product` | 506800 | 238330 | x1.00 |

### projective3_inverse_transform_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `inverse_then_transform` | 462540 | 220950 | x1.00 |

### projective3_mul_affine3

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `full_product` | 116160 | 43010 | x1.00 |

### projective3_transform_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `homogeneous` | 71900 | 25630 | x1.00 |

### projective3_try_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `matrix4` | 420580 | 196920 | x1.00 |

