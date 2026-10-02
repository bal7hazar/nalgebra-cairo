# Gas report

Sierra gas (`l2_gas`) per benchmark; `net` = raw - group baseline.

## nalgebra_tests_sparse::benches

### axpy_cs

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 327910 | 104970 | x1.00 |

### convolve

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `valid` | 242410 | 135150 | x1.00 |
| `same` | 349560 | 242300 | x1.79 |
| `full` | 410880 | 303620 | x2.25 |

### convolve_vector6

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `same` | 102190 | 93000 | x1.00 |
| `full` | 130010 | 120820 | x1.30 |

### cs_add

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 7129300 | 660010 | x1.00 |

### cs_cholesky

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `symbolic` | 8399460 | 3578550 | x1.00 |
| `library` | 10391950 | 5571040 | x1.56 |
| `alt_left_looking` | 12895790 | 8074880 | x2.26 |

### cs_cholesky_lap1d

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 4268810 | 2614590 | x1.00 |

### cs_cholesky_numeric

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 10387850 | 1988390 | x1.00 |

### cs_from_dmatrix

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 6735060 | 1002450 | x1.00 |

### cs_from_triplet

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 4820910 | 2161100 | x1.00 |

### cs_into_dmatrix

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 5732610 | 911700 | x1.00 |

### cs_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 7706160 | 4403560 | x1.00 |

### cs_mul_lap2d

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 12366820 | 7545910 | x1.00 |

### cs_scale

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 5039080 | 217470 | x1.00 |

### cs_solve_lower

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 12867250 | 2398010 | x1.00 |

### cs_solve_lower_cs

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 12893090 | 2464380 | x1.00 |

### cs_tr_solve_lower

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 11096540 | 627300 | x1.00 |

### cs_transpose

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 6638750 | 1817840 | x1.00 |
| `alt_packed_keys` | 6964990 | 2144080 | x1.18 |
| `alt_dict` | 8863190 | 4042280 | x2.22 |

### matrix_market

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `probe_iter` | 1077720 | 1068340 | x1.00 |
| `probe_bytes` | 1248610 | 1239230 | x1.16 |
| `library` | 1855230 | 1845850 | x1.73 |

