# Gas report

Sierra gas (`l2_gas`) per benchmark; `net` = raw - group baseline.

## nalgebra_tests_dynamic::benches

### dmatrix_add16

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 2898780 | 419490 | x1.00 |

### dmatrix_add6

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 417730 | 63640 | x1.00 |

### dmatrix_index

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 184050 | 2180 | x1.00 |

### dmatrix_insert_columns6

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 231150 | 49750 | x1.00 |

### dmatrix_insert_rows6

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 349180 | 167780 | x1.00 |

### dmatrix_mul16

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 10181940 | 7702650 | x1.00 |

### dmatrix_mul3

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 116690 | 23420 | x1.00 |

### dmatrix_mul6

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 475300 | 121210 | x1.00 |

### dmatrix_mul_vec16

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 2495090 | 1175700 | x1.00 |

### dmatrix_mul_vec3

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 74360 | 10770 | x1.00 |

### dmatrix_mul_vec6

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 233580 | 25090 | x1.00 |

### dmatrix_remove_rows16

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 1710660 | 466390 | x1.00 |

### dmatrix_resize16

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 1594150 | 349880 | x1.00 |

### dmatrix_resize6

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 323130 | 141460 | x1.00 |

### dmatrix_transpose16

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 1980230 | 737130 | x1.00 |

### dmatrix_transpose6

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 322080 | 141580 | x1.00 |

### dvector_dot16

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 184190 | 24100 | x1.00 |

### dvector_dot3

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 40660 | 6150 | x1.00 |

### dvector_dot6

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 70840 | 7350 | x1.00 |

### dvector_norm16

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 103880 | 20380 | x1.00 |

### matrix3x4_insert_columns

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 104750 | 39100 | x1.00 |

### matrix3x4_insert_fixed_columns

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 66280 | 1100 | x1.00 |
| `alt_direct` | 70380 | 5200 | x4.73 |

### matrix6_into_dmatrix

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 184900 | -300 | - |

## nalgebra_tests_dynamic::layout

### dyn_layout_column16

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `col` | 1271810 | 28410 | x1.00 |
| `row` | 1299800 | 56400 | x1.99 |

### dyn_layout_dot16

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `chunk8` | 180040 | 19950 | x1.00 |
| `chunk4` | 181520 | 21430 | x1.07 |
| `pop` | 195460 | 35370 | x1.77 |
| `index` | 218460 | 58370 | x2.93 |

### dyn_layout_dot6

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `dispatch` | 70340 | 6850 | x1.00 |
| `chunk4` | 77730 | 14240 | x2.08 |
| `pop` | 79160 | 15670 | x2.29 |
| `chunk8` | 80470 | 16980 | x2.48 |

### dyn_layout_index

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `col` | 183650 | 2180 | x1.00 |
| `dict_build_only` | 482750 | 301280 | x138.20 |
| `dict` | 483630 | 302160 | x138.61 |

### dyn_layout_mul16

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `colseq4` | 10038870 | 7560380 | x1.00 |
| `colseq` | 13607510 | 11129020 | x1.47 |
| `row` | 16920050 | 14441560 | x1.91 |
| `col` | 25639640 | 23161150 | x3.06 |
| `dict` | 38300730 | 35822240 | x4.74 |

### dyn_layout_mul6

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `colseqd` | 937790 | 584500 | x1.00 |
| `colseq4` | 1203830 | 850540 | x1.46 |
| `colseq` | 1255310 | 902020 | x1.54 |
| `row` | 1292450 | 939160 | x1.61 |
| `col` | 1648840 | 1295550 | x2.22 |
| `dict` | 2860130 | 2506840 | x4.29 |

### dyn_layout_resize6

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `col` | 327090 | 146290 | x1.00 |
| `dict` | 804760 | 623960 | x4.27 |

### dyn_layout_transpose16

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `strided4` | 1979830 | 737130 | x1.00 |
| `runs` | 2039570 | 796870 | x1.08 |
| `strided` | 2086380 | 843680 | x1.14 |

