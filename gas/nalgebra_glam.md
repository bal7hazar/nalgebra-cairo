# Gas report

Sierra gas (`l2_gas`) per benchmark; `net` = raw - group baseline.

## nalgebra_glam::benches

### isometry2_to_mat3

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `into` | 8510 | 200 | x1.00 |

### isometry2_to_vec2_angle

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `into` | 38110 | 29800 | x1.00 |

### isometry3_to_mat4

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `into` | 31280 | 22370 | x1.00 |

### isometry3_to_vec3_quat

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `into` | 8910 | 0 | - |

### mat2_to_rotation2

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `into` | 16480 | 8170 | x1.00 |

### mat3_to_isometry2

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `try_into` | 28640 | 19330 | x1.00 |

### mat3_to_similarity2

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `try_into` | 43300 | 33990 | x1.00 |

### mat4_to_isometry3

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `try_into` | 82450 | 71740 | x1.00 |

### mat4_to_isometry3_reject

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `bottom_first` | 15320 | 1210 | x1.00 |
| `orthogonal_first` | 86150 | 72040 | x59.54 |
| `single_function` | 86150 | 72040 | x59.54 |

### mat4_to_matrix4

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `into` | 10710 | 0 | - |

### mat4_to_similarity3

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `one_pass` | 107320 | 96610 | x1.00 |
| `two_pass` | 152610 | 141900 | x1.47 |

### matrix4_to_mat4

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `into` | 14110 | 0 | - |

### quat_to_rotation3

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `into` | 45260 | 36680 | x1.00 |

### quat_to_unit_quaternion

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `into` | 22290 | 13980 | x1.00 |

### rotation3_to_quat

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `into` | 75340 | 28080 | x1.00 |

### similarity3_to_mat4

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `into` | 51430 | 40720 | x1.00 |

### unit_quaternion_to_quat

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `into` | 8310 | 0 | - |

### vec2_angle_to_isometry2

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `into` | 39310 | 30430 | x1.00 |

### vec2_to_isometry2

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `identity` | 7410 | -500 | - |
| `angle_zero` | 25220 | 17310 | - |

### vec3_quat_to_isometry3

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `into` | 22820 | 13140 | x1.00 |

### vec3_to_unit_vector3

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `try_into` | 19940 | 11830 | x1.00 |

### vec3_to_vector3

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `into` | 8110 | 0 | - |

### vector3_to_vec3

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `into` | 8110 | 0 | - |

