# Gas report

Sierra gas (`l2_gas`) per benchmark; `net` = raw - group baseline.

## nalgebra_tests_geometry_projections::orthographic3::benches

### orthographic3_from_fov

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `from_fov` | 130670 | 102430 | x1.00 |

### orthographic3_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip_mul` | 43800 | 15360 | x1.00 |
| `div` | 48990 | 20550 | x1.34 |

### orthographic3_left

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `left` | 23080 | 4140 | x1.00 |

### orthographic3_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `new` | 63080 | 34040 | x1.00 |

### orthographic3_project_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `mul_add` | 26280 | 5540 | x1.00 |
| `alt_split` | 28200 | 7460 | x1.35 |

### orthographic3_set_left

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `set_left` | 31740 | 11900 | x1.00 |

### orthographic3_unproject_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `unproject` | 33080 | 12340 | x1.00 |

## nalgebra_tests_geometry_projections::perspective3::benches

### perspective3_fovy

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fovy` | 45690 | 26750 | x1.00 |

### perspective3_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `inverse` | 37090 | 16150 | x1.00 |

### perspective3_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `new` | 106770 | 88130 | x1.00 |

### perspective3_project_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip_mul` | 34280 | 13540 | x1.00 |
| `div3` | 35790 | 15050 | x1.11 |
| `alt_three_div` | 36140 | 15400 | x1.14 |

### perspective3_project_vector

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `two_div` | 31130 | 10390 | x1.00 |
| `alt_div3` | 33320 | 12580 | x1.21 |

### perspective3_set_fovy

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `set_fovy` | 79370 | 59530 | x1.00 |

### perspective3_set_znear_and_zfar

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `set` | 30840 | 10600 | x1.00 |

### perspective3_unproject_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `unproject` | 35160 | 14420 | x1.00 |

### perspective3_znear

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `one_div` | 23070 | 4130 | x1.00 |
| `alt_upstream` | 32790 | 13850 | x3.35 |

