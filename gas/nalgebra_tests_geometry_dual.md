# Gas report

Sierra gas (`l2_gas`) per benchmark; `net` = raw - group baseline.

## nalgebra_tests_geometry_dual::dual_quaternion::benches

### dual_quaternion_lerp

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `lerp` | 38780 | 15740 | x1.00 |

### dual_quaternion_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 50640 | 28000 | x1.00 |
| `alt_two_products` | 61250 | 38610 | x1.38 |

### dual_quaternion_normalize

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `normalize` | 48250 | 27410 | x1.00 |

### dual_quaternion_try_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `try_inverse` | 67410 | 46570 | x1.00 |

## nalgebra_tests_geometry_dual::unit_dual_quaternion::benches

### unit_dual_quaternion_div

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `div` | 75340 | 52700 | x1.00 |

### unit_dual_quaternion_from_parts

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 32690 | 11850 | x1.00 |
| `alt_upstream` | 47450 | 26610 | x2.25 |

### unit_dual_quaternion_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_conjugate` | 22840 | 2000 | x1.00 |
| `literal` | 46240 | 25400 | x12.70 |

### unit_dual_quaternion_inverse_transform_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_to_isometry` | 55300 | 36160 | x1.00 |
| `literal` | 70880 | 51740 | x1.43 |

### unit_dual_quaternion_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `mul` | 50640 | 28000 | x1.00 |

### unit_dual_quaternion_mul_isometry

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `mul_isometry` | 67760 | 39250 | x1.00 |

### unit_dual_quaternion_mul_translation

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 36650 | 14610 | x1.00 |
| `alt_from_parts` | 61190 | 39150 | x2.68 |

### unit_dual_quaternion_mul_unit_quaternion

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 45840 | 24000 | x1.00 |
| `alt_from_rotation` | 49840 | 28000 | x1.17 |

### unit_dual_quaternion_nlerp

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `nlerp` | 66290 | 43250 | x1.00 |

### unit_dual_quaternion_sclerp

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `sclerp` | 208450 | 185410 | x1.00 |

### unit_dual_quaternion_to_isometry

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `to_isometry` | 31310 | 10570 | x1.00 |

### unit_dual_quaternion_transform_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `literal` | 46180 | 27040 | x1.00 |
| `alt_to_isometry` | 52380 | 33240 | x1.23 |

### unit_dual_quaternion_translation

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 28210 | 9870 | x1.00 |
| `alt_upstream` | 33010 | 14670 | x1.49 |

