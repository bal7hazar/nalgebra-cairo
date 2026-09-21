# Gas report

Sierra gas (`l2_gas`) per benchmark; `net` = raw - group baseline.

## simba_fixed::benches

### div

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `real_div` | 19080 | 2740 | x1.00 |
| `alt_glam_operator` | 19170 | 2830 | x1.03 |

### simba_fixed_convert

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `roundtrip` | 15540 | 0 | - |

### simba_fixed_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `via_fixed` | 17920 | 1580 | x1.00 |
| `via_simba` | 18090 | 1750 | x1.11 |

### simba_fixed_norm3

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `via_fixed` | 18560 | 2220 | x1.00 |
| `via_glam_native` | 18560 | 2220 | x1.00 |
| `via_simba` | 18560 | 2220 | x1.00 |

### simba_fixed_sin_cos

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `via_simba` | 32760 | 17220 | x1.00 |
| `via_fixed` | 46940 | 31400 | x1.82 |
| `via_glam_native` | 46940 | 31400 | x1.82 |

### simba_fixed_sum_prod3

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `via_glam_native` | 19120 | 1980 | x1.00 |
| `via_fixed` | 19290 | 2150 | x1.09 |
| `via_simba` | 19290 | 2150 | x1.09 |

