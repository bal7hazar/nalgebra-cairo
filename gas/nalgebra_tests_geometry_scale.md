# Gas report

Sierra gas (`l2_gas`) per benchmark; `net` = raw - group baseline.

## nalgebra_tests_geometry_scale::reflection::benches

### reflection2_reflect

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `vector2` | 17580 | 7970 | x1.00 |

### reflection3_new_containing_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `exact_dot` | 11290 | 1980 | x1.00 |

### reflection3_reflect

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `vector3` | 20760 | 10250 | x1.00 |
| `alt_sum_prod` | 22340 | 11830 | x1.15 |
| `matrix3x2` | 31080 | 20570 | x2.01 |

### reflection3_reflect_rows

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `matrix2x3` | 50340 | 37730 | x1.00 |

### reflection3_reflect_with_sign

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `vector3` | 21060 | - | x1.00 |

### reflection6_reflect

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `vector6` | 30300 | 17090 | x1.00 |

## nalgebra_tests_geometry_scale::scale::benches

### scale2_transform_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `product` | 12670 | 3260 | x1.00 |

### scale3_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `component_product` | 15250 | 4940 | x1.00 |

### scale3_mul_vector

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `component_product` | 15250 | 4940 | x1.00 |

### scale3_to_homogeneous

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `diagonal` | 8710 | 200 | x1.00 |

### scale3_transform_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `product` | 15250 | 4940 | x1.00 |

### scale3_try_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `unchecked` | 17930 | 8420 | x1.00 |
| `reciprocals` | 18130 | 8620 | x1.02 |
| `pseudo` | 18230 | 8720 | x1.04 |
| `in_place` | 18330 | 8820 | x1.05 |

### scale3_try_inverse_transform_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `division` | 20630 | 10320 | x1.00 |
| `alt_reciprocal_product` | 29360 | 19050 | x1.85 |

### scale6_transform_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `product` | 22990 | 9980 | x1.00 |

