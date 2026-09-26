# Gas report

Sierra gas (`l2_gas`) per benchmark; `net` = raw - group baseline.

## nalgebra_tests_dynamic::benches

### dmatrix_add16

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 2906610 | 419490 | x1.00 |

### dmatrix_add6

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 425560 | 63640 | x1.00 |

### dmatrix_index

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 191880 | 2180 | x1.00 |

### dmatrix_insert_columns6

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 238980 | 49750 | x1.00 |

### dmatrix_insert_rows6

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 357010 | 167780 | x1.00 |

### dmatrix_mul16

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 10189770 | 7702650 | x1.00 |

### dmatrix_mul3

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 126250 | 25150 | x1.00 |

### dmatrix_mul6

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 483130 | 121210 | x1.00 |

### dmatrix_mul_vec16

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 2502920 | 1175700 | x1.00 |

### dmatrix_mul_vec3

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 82190 | 10770 | x1.00 |

### dmatrix_mul_vec6

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 246630 | 30310 | x1.00 |

### dmatrix_remove_rows16

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 1718490 | 466390 | x1.00 |

### dmatrix_resize16

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 1601980 | 349880 | x1.00 |

### dmatrix_resize6

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 330960 | 141460 | x1.00 |

### dmatrix_transpose16

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 1988060 | 737130 | x1.00 |

### dmatrix_transpose6

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 329910 | 141580 | x1.00 |

### dvector_dot16

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 192020 | 24100 | x1.00 |

### dvector_dot3

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 48490 | 6150 | x1.00 |

### dvector_dot6

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 78670 | 7350 | x1.00 |

### dvector_norm16

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 111710 | 20380 | x1.00 |

### matrix3x4_insert_columns

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 112580 | 39100 | x1.00 |

### matrix3x4_insert_fixed_columns

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 74110 | 1100 | x1.00 |
| `alt_direct` | 78210 | 5200 | x4.73 |

### matrix6_into_dmatrix

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 192730 | -300 | - |

## nalgebra_tests_dynamic::layout

### dyn_layout_column16

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `col` | 1279640 | 28410 | x1.00 |
| `row` | 1307630 | 56400 | x1.99 |

### dyn_layout_dot16

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `chunk8` | 187870 | 19950 | x1.00 |
| `chunk4` | 189350 | 21430 | x1.07 |
| `pop` | 203290 | 35370 | x1.77 |
| `index` | 226290 | 58370 | x2.93 |

### dyn_layout_dot6

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `dispatch` | 78170 | 6850 | x1.00 |
| `chunk4` | 85560 | 14240 | x2.08 |
| `pop` | 86990 | 15670 | x2.29 |
| `chunk8` | 88300 | 16980 | x2.48 |

### dyn_layout_index

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `col` | 191480 | 2180 | x1.00 |
| `dict_build_only` | 490480 | 301180 | x138.16 |
| `dict` | 491360 | 302060 | x138.56 |

### dyn_layout_mul16

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `colseq4` | 10046700 | 7560380 | x1.00 |
| `colseq` | 13615340 | 11129020 | x1.47 |
| `row` | 16927880 | 14441560 | x1.91 |
| `col` | 25647470 | 23161150 | x3.06 |
| `dict` | 38308460 | 35822140 | x4.74 |

### dyn_layout_mul6

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `colseqd` | 945620 | 584500 | x1.00 |
| `colseq4` | 1211660 | 850540 | x1.46 |
| `colseq` | 1263140 | 902020 | x1.54 |
| `row` | 1300280 | 939160 | x1.61 |
| `col` | 1656670 | 1295550 | x2.22 |
| `dict` | 2867860 | 2506740 | x4.29 |

### dyn_layout_resize6

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `col` | 334920 | 146290 | x1.00 |
| `dict` | 812490 | 623860 | x4.26 |

### dyn_layout_transpose16

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `strided4` | 1987660 | 737130 | x1.00 |
| `runs` | 2047400 | 796870 | x1.08 |
| `strided` | 2094210 | 843680 | x1.14 |

