# Gas report

Sierra gas (`l2_gas`) per benchmark; `net` = raw - group baseline.

## nalgebra_tests_geometry_scale::reflection::benches

### reflection2_reflect

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `vector2` | 25410 | 7970 | x1.00 |

### reflection3_new_containing_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `exact_dot` | 19120 | 1980 | x1.00 |

### reflection3_reflect

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `vector3` | 28590 | 10250 | x1.00 |
| `alt_sum_prod` | 30170 | 11830 | x1.15 |
| `matrix3x2` | 38910 | 20570 | x2.01 |

### reflection3_reflect_rows

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `matrix2x3` | 58170 | 37730 | x1.00 |

### reflection3_reflect_with_sign

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `vector3` | 28890 | - | x1.00 |

### reflection6_reflect

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `vector6` | 38130 | 17090 | x1.00 |

## nalgebra_tests_geometry_scale::scale::benches

### scale2_transform_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `product` | 20500 | 3260 | x1.00 |

### scale3_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `component_product` | 23080 | 4940 | x1.00 |

### scale3_mul_vector

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `component_product` | 23080 | 4940 | x1.00 |

### scale3_to_homogeneous

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `diagonal` | 16540 | 200 | x1.00 |

### scale3_transform_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `product` | 23080 | 4940 | x1.00 |

### scale3_try_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `unchecked` | 25760 | 8420 | x1.00 |
| `reciprocals` | 25960 | 8620 | x1.02 |
| `pseudo` | 26060 | 8720 | x1.04 |
| `in_place` | 26160 | 8820 | x1.05 |

### scale3_try_inverse_transform_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `division` | 28460 | 10320 | x1.00 |
| `alt_reciprocal_product` | 37190 | 19050 | x1.85 |

### scale6_transform_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `product` | 30820 | 9980 | x1.00 |

