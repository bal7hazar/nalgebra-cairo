# Gas report

Sierra gas (`l2_gas`) per benchmark; `net` = raw - group baseline.

## nalgebra_tests_geometry_dual::dual_quaternion::benches

### dual_quaternion_lerp

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `lerp` | 30950 | 15740 | x1.00 |

### dual_quaternion_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 42810 | 28000 | x1.00 |
| `alt_two_products` | 53420 | 38610 | x1.38 |

### dual_quaternion_normalize

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `normalize` | 40420 | 27410 | x1.00 |

### dual_quaternion_try_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `try_inverse` | 59580 | 46570 | x1.00 |

## nalgebra_tests_geometry_dual::unit_dual_quaternion::benches

### unit_dual_quaternion_div

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `div` | 67510 | 52700 | x1.00 |

### unit_dual_quaternion_from_parts

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 24860 | 11850 | x1.00 |
| `alt_upstream` | 39620 | 26610 | x2.25 |

### unit_dual_quaternion_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_conjugate` | 15010 | 2000 | x1.00 |
| `literal` | 38410 | 25400 | x12.70 |

### unit_dual_quaternion_inverse_transform_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_to_isometry` | 47470 | 36160 | x1.00 |
| `literal` | 63050 | 51740 | x1.43 |

### unit_dual_quaternion_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `mul` | 42810 | 28000 | x1.00 |

### unit_dual_quaternion_mul_isometry

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `mul_isometry` | 59930 | 39250 | x1.00 |

### unit_dual_quaternion_mul_translation

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 28820 | 14610 | x1.00 |
| `alt_from_parts` | 53360 | 39150 | x2.68 |

### unit_dual_quaternion_mul_unit_quaternion

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 38010 | 24000 | x1.00 |
| `alt_from_rotation` | 42010 | 28000 | x1.17 |

### unit_dual_quaternion_nlerp

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `nlerp` | 58460 | 43250 | x1.00 |

### unit_dual_quaternion_sclerp

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `sclerp` | 200620 | 185410 | x1.00 |

### unit_dual_quaternion_to_isometry

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `to_isometry` | 23480 | 10570 | x1.00 |

### unit_dual_quaternion_transform_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `literal` | 38350 | 27040 | x1.00 |
| `alt_to_isometry` | 44550 | 33240 | x1.23 |

### unit_dual_quaternion_translation

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 20380 | 9870 | x1.00 |
| `alt_upstream` | 25180 | 14670 | x1.49 |

