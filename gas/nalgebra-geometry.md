# Gas report

Sierra gas (`l2_gas`) per benchmark; `net` = raw - group baseline.

## nalgebra::geometry::isometry2::benches

### isometry2_lerp_nlerp

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `lerp_normalize` | 40890 | - | x1.00 |

### isometry2_renormalize

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `exact` | 26920 | - | x1.00 |

### isometry2_renormalize_fast

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `newton` | 25360 | - | x1.00 |

## nalgebra::geometry::isometry3::benches

### isometry3_lerp_nlerp

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `lerp_normalize` | 58640 | - | x1.00 |

### isometry3_renormalize

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `exact` | 35220 | - | x1.00 |

### isometry3_renormalize_fast

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `newton` | 31120 | - | x1.00 |

## nalgebra::geometry::quaternion::benches

### quaternion_conj_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 30590 | - | x1.00 |

## nalgebra::geometry::rotation3::benches

### rotation3_renormalize

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_quaternion` | 94820 | - | x1.00 |

## nalgebra::geometry::unit_complex::benches

### unit_complex_rotation_between

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_normalized_inputs` | 45820 | - | x1.00 |

## nalgebra::geometry::unit_quaternion::benches

### unit_quaternion_conj_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 30590 | - | x1.00 |

