# Gas report

Sierra gas (`l2_gas`) per benchmark; `net` = raw - group baseline.

## nalgebra_tests_geometry_projections::orthographic3::benches

### orthographic3_from_fov

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `from_fov` | 122840 | 102430 | x1.00 |

### orthographic3_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip_mul` | 35970 | 15360 | x1.00 |
| `div` | 41160 | 20550 | x1.34 |

### orthographic3_left

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `left` | 15250 | 4140 | x1.00 |

### orthographic3_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `new` | 55250 | 34040 | x1.00 |

### orthographic3_project_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `mul_add` | 18450 | 5540 | x1.00 |
| `alt_split` | 20370 | 7460 | x1.35 |

### orthographic3_set_left

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `set_left` | 23910 | 11900 | x1.00 |

### orthographic3_unproject_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `unproject` | 25250 | 12340 | x1.00 |

## nalgebra_tests_geometry_projections::perspective3::benches

### perspective3_fovy

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fovy` | 37860 | 26750 | x1.00 |

### perspective3_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `inverse` | 29260 | 16150 | x1.00 |

### perspective3_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `new` | 98940 | 88130 | x1.00 |

### perspective3_project_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip_mul` | 26450 | 13540 | x1.00 |
| `div3` | 27960 | 15050 | x1.11 |
| `alt_three_div` | 28310 | 15400 | x1.14 |

### perspective3_project_vector

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `two_div` | 23300 | 10390 | x1.00 |
| `alt_div3` | 25490 | 12580 | x1.21 |

### perspective3_set_fovy

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `set_fovy` | 71540 | 59530 | x1.00 |

### perspective3_set_znear_and_zfar

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `set` | 23010 | 10600 | x1.00 |

### perspective3_unproject_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `unproject` | 27330 | 14420 | x1.00 |

### perspective3_znear

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `one_div` | 15240 | 4130 | x1.00 |
| `alt_upstream` | 24960 | 13850 | x3.35 |

