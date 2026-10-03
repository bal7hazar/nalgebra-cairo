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
| `fused` | 42080 | 27270 | x1.00 |
| `alt_two_products` | 45430 | 30620 | x1.12 |

### dual_quaternion_normalize

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `normalize` | 40420 | 27410 | x1.00 |

### dual_quaternion_try_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `try_inverse` | 55650 | 42640 | x1.00 |

## nalgebra_tests_geometry_dual::unit_dual_quaternion::benches

### unit_dual_quaternion_div

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `div` | 63950 | 49140 | x1.00 |

### unit_dual_quaternion_from_parts

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 24860 | 11850 | x1.00 |
| `alt_upstream` | 36690 | 23680 | x2.00 |

### unit_dual_quaternion_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_conjugate` | 15010 | 2000 | x1.00 |
| `literal` | 35580 | 22570 | x11.29 |

### unit_dual_quaternion_inverse_transform_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_to_isometry` | 43140 | 31830 | x1.00 |
| `literal` | 60220 | 48910 | x1.54 |

### unit_dual_quaternion_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `mul` | 42080 | 27270 | x1.00 |

### unit_dual_quaternion_mul_isometry

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `mul_isometry` | 59200 | 38520 | x1.00 |

### unit_dual_quaternion_mul_translation

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 28820 | 14610 | x1.00 |
| `alt_from_parts` | 52630 | 38420 | x2.63 |

### unit_dual_quaternion_mul_unit_quaternion

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 32150 | 18140 | x1.00 |
| `alt_from_rotation` | 41280 | 27270 | x1.50 |

### unit_dual_quaternion_nlerp

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `nlerp` | 58460 | 43250 | x1.00 |

### unit_dual_quaternion_sclerp

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `sclerp` | 200760 | 185550 | x1.00 |

### unit_dual_quaternion_to_isometry

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `to_isometry` | 23480 | 10570 | x1.00 |

### unit_dual_quaternion_transform_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `literal` | 38350 | 27040 | x1.00 |
| `alt_to_isometry` | 41520 | 30210 | x1.12 |

### unit_dual_quaternion_translation

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 20380 | 9870 | x1.00 |
| `alt_upstream` | 22850 | 12340 | x1.25 |

