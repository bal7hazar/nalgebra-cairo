# Gas report

Sierra gas (`l2_gas`) per benchmark; `net` = raw - group baseline.

## simba::benches

### real_cross3

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `direct` | 22480 | 5340 | x1.00 |
| `generic` | 22480 | 5340 | x1.00 |

### real_diff_prod

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `acc` | 18920 | 1780 | x1.00 |
| `named` | 18920 | 1780 | x1.00 |

### real_dot3

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `direct` | 19120 | 1980 | x1.00 |
| `generic` | 19120 | 1980 | x1.00 |

### real_mul_add

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `acc` | 18920 | 1780 | x1.00 |
| `named` | 18920 | 1780 | x1.00 |

### real_norm16

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `acc` | 23880 | 6740 | x1.00 |
| `typed` | 23880 | 6740 | x1.00 |

### real_norm3

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `direct` | 18560 | 2220 | x1.00 |
| `generic` | 18560 | 2220 | x1.00 |

### real_norm6

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `acc` | 22080 | 4940 | x1.00 |
| `typed` | 22080 | 4940 | x1.00 |

### real_norm9

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `acc` | 22480 | 5340 | x1.00 |
| `typed` | 22580 | 5440 | x1.02 |

### real_normalize3

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `recip` | 23840 | 7500 | x1.00 |
| `div` | 26610 | 10270 | x1.37 |

### real_scalar

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `add` | 16980 | 640 | x1.00 |
| `mul` | 17920 | 1580 | x2.47 |
| `sqrt` | 18160 | 1820 | x2.84 |
| `recip` | 18690 | 2350 | x3.67 |
| `div` | 19170 | 2830 | x4.42 |
| `inv_norm2` | 20680 | 4340 | x6.78 |
| `atan2` | 44760 | 28420 | x44.41 |
| `sin_cos` | 47840 | 31500 | x49.22 |

### real_sum_prod2

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `acc` | 18920 | 1780 | x1.00 |
| `named` | 18920 | 1780 | x1.00 |

### real_sum_prod3

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `acc` | 19120 | 1980 | x1.00 |
| `named` | 19120 | 1980 | x1.00 |

### real_sum_prod4

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `acc` | 19320 | 2180 | x1.00 |
| `named` | 19320 | 2180 | x1.00 |

