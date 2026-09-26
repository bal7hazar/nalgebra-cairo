# Gas report

Sierra gas (`l2_gas`) per benchmark; `net` = raw - group baseline.

## nalgebra_glam::benches

### isometry2_to_mat3

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `into` | 16340 | 200 | x1.00 |

### isometry2_to_vec2_angle

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `into` | 45940 | 29800 | x1.00 |

### isometry3_to_mat4

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `into` | 39110 | 22370 | x1.00 |

### isometry3_to_vec3_quat

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `into` | 16740 | 0 | - |

### mat2_to_rotation2

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `into` | 24310 | 8170 | x1.00 |

### mat3_to_isometry2

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `try_into` | 39200 | 22060 | x1.00 |

### mat3_to_similarity2

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `try_into` | 51130 | 33990 | x1.00 |

### mat4_to_isometry3

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `try_into` | 97440 | 78900 | x1.00 |

### mat4_to_isometry3_reject

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `bottom_first` | 23150 | 1210 | x1.00 |
| `orthogonal_first` | 101140 | 79200 | x65.45 |
| `single_function` | 101140 | 79200 | x65.45 |

### mat4_to_matrix4

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `into` | 18540 | 0 | - |

### mat4_to_similarity3

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `one_pass` | 117880 | 99340 | x1.00 |
| `two_pass` | 163170 | 144630 | x1.46 |

### matrix4_to_mat4

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `into` | 21940 | 0 | - |

### quat_to_rotation3

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `into` | 53090 | 36680 | x1.00 |

### quat_to_unit_quaternion

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `into` | 30120 | 13980 | x1.00 |

### rotation3_to_quat

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `into` | 83170 | 28080 | x1.00 |

### similarity3_to_mat4

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `into` | 59260 | 40720 | x1.00 |

### unit_quaternion_to_quat

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `into` | 16140 | 0 | - |

### vec2_angle_to_isometry2

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `into` | 47140 | 30430 | x1.00 |

### vec2_to_isometry2

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `identity` | 15240 | -500 | - |
| `angle_zero` | 33050 | 17310 | - |

### vec3_quat_to_isometry3

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `into` | 30650 | 13140 | x1.00 |

### vec3_to_unit_vector3

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `try_into` | 27770 | 11830 | x1.00 |

### vec3_to_vector3

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `into` | 15940 | 0 | - |

### vector3_to_vec3

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `into` | 15940 | 0 | - |

