# Gas report

Sierra gas (`l2_gas`) per benchmark; `net` = raw - group baseline.

## primitives

### abs_i64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `neg_operator` | 15840 | 300 | x1.00 |
| `abs_bounded_constrain_to_u64` | 16210 | 670 | x2.23 |
| `neg_zero_minus_x` | 16280 | 740 | x2.47 |
| `sign_branches` | 16410 | 870 | x2.90 |
| `abs_if_lt_zero_neg` | 16610 | 1070 | x3.57 |
| `abs_via_felt252_to_u64` | 17050 | 1510 | x5.03 |

### arith_felt252

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `mul` | 15940 | 100 | x1.00 |
| `neg` | 15940 | 100 | x1.00 |
| `sub` | 15940 | 100 | x1.00 |
| `add` | 15940 | 100 | x1.00 |
| `felt252_div_const` | 16340 | 500 | x5.00 |
| `felt252_div` | 16440 | 600 | x6.00 |

### arith_i128

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `neg` | 16240 | 300 | x1.00 |
| `checked_add` | 16310 | 370 | x1.23 |
| `checked_sub` | 16310 | 370 | x1.23 |
| `add` | 16310 | 370 | x1.23 |
| `sub` | 16310 | 370 | x1.23 |
| `add_pos` | 16610 | 670 | x2.23 |
| `sub_pos` | 16610 | 670 | x2.23 |
| `overflowing_add` | 16710 | 770 | x2.57 |
| `saturating_sub` | 16710 | 770 | x2.57 |
| `overflowing_sub` | 16710 | 770 | x2.57 |
| `saturating_add` | 16710 | 770 | x2.57 |
| `wrapping_sub` | 16710 | 770 | x2.57 |
| `wrapping_add` | 16710 | 770 | x2.57 |
| `div_const` | 18190 | 2250 | x7.50 |
| `rem_const` | 18190 | 2250 | x7.50 |
| `div` | 20630 | 4690 | x15.63 |
| `rem` | 20630 | 4690 | x15.63 |
| `div_pos` | 20830 | 4890 | x16.30 |
| `rem_pos` | 20830 | 4890 | x16.30 |
| `mul` | 23290 | 7350 | x24.50 |
| `mul_const` | 23290 | 7350 | x24.50 |
| `mul_pos` | 23490 | 7550 | x25.17 |

### arith_i16

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `wide_mul` | 16040 | 100 | x1.00 |
| `neg` | 16240 | 300 | x3.00 |
| `mul` | 16580 | 640 | x6.40 |
| `mul_const` | 16580 | 640 | x6.40 |
| `add` | 16580 | 640 | x6.40 |
| `checked_add` | 16580 | 640 | x6.40 |
| `checked_sub` | 16580 | 640 | x6.40 |
| `sub` | 16580 | 640 | x6.40 |
| `add_pos` | 16880 | 940 | x9.40 |
| `sub_pos` | 16880 | 940 | x9.40 |
| `mul_pos` | 16880 | 940 | x9.40 |
| `overflowing_sub` | 16980 | 1040 | x10.40 |
| `saturating_add` | 16980 | 1040 | x10.40 |
| `overflowing_add` | 16980 | 1040 | x10.40 |
| `saturating_sub` | 16980 | 1040 | x10.40 |
| `wrapping_sub` | 16980 | 1040 | x10.40 |
| `wrapping_add` | 16980 | 1040 | x10.40 |
| `div_const` | 17720 | 1780 | x17.80 |
| `rem_const` | 17720 | 1780 | x17.80 |
| `div` | 20160 | 4220 | x42.20 |
| `rem` | 20160 | 4220 | x42.20 |
| `div_pos` | 20360 | 4420 | x44.20 |
| `rem_pos` | 20360 | 4420 | x44.20 |

### arith_i32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `wide_mul` | 16040 | 100 | x1.00 |
| `neg` | 16240 | 300 | x3.00 |
| `sub` | 16580 | 640 | x6.40 |
| `checked_add` | 16580 | 640 | x6.40 |
| `add` | 16580 | 640 | x6.40 |
| `checked_sub` | 16580 | 640 | x6.40 |
| `mul` | 16580 | 640 | x6.40 |
| `mul_const` | 16580 | 640 | x6.40 |
| `mul_pos` | 16880 | 940 | x9.40 |
| `sub_pos` | 16880 | 940 | x9.40 |
| `add_pos` | 16880 | 940 | x9.40 |
| `overflowing_sub` | 16980 | 1040 | x10.40 |
| `overflowing_add` | 16980 | 1040 | x10.40 |
| `wrapping_add` | 16980 | 1040 | x10.40 |
| `saturating_add` | 16980 | 1040 | x10.40 |
| `wrapping_sub` | 16980 | 1040 | x10.40 |
| `saturating_sub` | 16980 | 1040 | x10.40 |
| `rem_const` | 17720 | 1780 | x17.80 |
| `div_const` | 17720 | 1780 | x17.80 |
| `rem` | 20160 | 4220 | x42.20 |
| `div` | 20160 | 4220 | x42.20 |
| `rem_pos` | 20360 | 4420 | x44.20 |
| `div_pos` | 20360 | 4420 | x44.20 |

### arith_i64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `wide_mul` | 16040 | 100 | x1.00 |
| `neg` | 16240 | 300 | x3.00 |
| `sub` | 16580 | 640 | x6.40 |
| `checked_sub` | 16580 | 640 | x6.40 |
| `add` | 16580 | 640 | x6.40 |
| `checked_add` | 16580 | 640 | x6.40 |
| `mul` | 16580 | 640 | x6.40 |
| `mul_const` | 16580 | 640 | x6.40 |
| `sub_pos` | 16880 | 940 | x9.40 |
| `add_pos` | 16880 | 940 | x9.40 |
| `mul_pos` | 16880 | 940 | x9.40 |
| `overflowing_add` | 16980 | 1040 | x10.40 |
| `overflowing_sub` | 16980 | 1040 | x10.40 |
| `saturating_add` | 16980 | 1040 | x10.40 |
| `saturating_sub` | 16980 | 1040 | x10.40 |
| `wrapping_sub` | 16980 | 1040 | x10.40 |
| `wrapping_add` | 16980 | 1040 | x10.40 |
| `rem_const` | 17720 | 1780 | x17.80 |
| `div_const` | 17720 | 1780 | x17.80 |
| `rem` | 20160 | 4220 | x42.20 |
| `div` | 20160 | 4220 | x42.20 |
| `rem_pos` | 20360 | 4420 | x44.20 |
| `div_pos` | 20360 | 4420 | x44.20 |

### arith_i8

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `wide_mul` | 16040 | 100 | x1.00 |
| `neg` | 16240 | 300 | x3.00 |
| `sub` | 16580 | 640 | x6.40 |
| `checked_sub` | 16580 | 640 | x6.40 |
| `add` | 16580 | 640 | x6.40 |
| `checked_add` | 16580 | 640 | x6.40 |
| `mul_const` | 16580 | 640 | x6.40 |
| `mul` | 16580 | 640 | x6.40 |
| `mul_pos` | 16880 | 940 | x9.40 |
| `sub_pos` | 16880 | 940 | x9.40 |
| `add_pos` | 16880 | 940 | x9.40 |
| `overflowing_sub` | 16980 | 1040 | x10.40 |
| `overflowing_add` | 16980 | 1040 | x10.40 |
| `saturating_add` | 16980 | 1040 | x10.40 |
| `saturating_sub` | 16980 | 1040 | x10.40 |
| `wrapping_sub` | 16980 | 1040 | x10.40 |
| `wrapping_add` | 16980 | 1040 | x10.40 |
| `rem_const` | 17720 | 1780 | x17.80 |
| `div_const` | 17720 | 1780 | x17.80 |
| `rem` | 20160 | 4220 | x42.20 |
| `div` | 20160 | 4220 | x42.20 |
| `rem_pos` | 20360 | 4420 | x44.20 |
| `div_pos` | 20360 | 4420 | x44.20 |

### arith_u128

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `add` | 16210 | 270 | x1.00 |
| `sub` | 16210 | 270 | x1.00 |
| `checked_sub` | 16210 | 270 | x1.00 |
| `checked_add` | 16210 | 270 | x1.00 |
| `overflowing_add` | 16610 | 670 | x2.48 |
| `overflowing_sub` | 16610 | 670 | x2.48 |
| `saturating_add` | 16610 | 670 | x2.48 |
| `saturating_sub` | 16610 | 670 | x2.48 |
| `wrapping_add` | 16610 | 670 | x2.48 |
| `wrapping_sub` | 16610 | 670 | x2.48 |
| `rem` | 17320 | 1380 | x5.11 |
| `rem_const` | 17320 | 1380 | x5.11 |
| `div` | 17320 | 1380 | x5.11 |
| `div_const` | 17320 | 1380 | x5.11 |
| `mul` | 18970 | 3030 | x11.22 |
| `checked_mul` | 18970 | 3030 | x11.22 |
| `mul_const` | 19070 | 3130 | x11.59 |
| `overflowing_mul` | 19170 | 3230 | x11.96 |
| `wide_mul` | 19170 | 3230 | x11.96 |
| `wrapping_mul` | 19170 | 3230 | x11.96 |
| `saturating_mul` | 19270 | 3330 | x12.33 |

### arith_u16

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `wide_mul` | 16040 | 100 | x1.00 |
| `checked_sub` | 16210 | 270 | x2.70 |
| `sub` | 16210 | 270 | x2.70 |
| `add` | 16310 | 370 | x3.70 |
| `checked_add` | 16310 | 370 | x3.70 |
| `mul` | 16310 | 370 | x3.70 |
| `mul_const` | 16310 | 370 | x3.70 |
| `overflowing_sub` | 16610 | 670 | x6.70 |
| `saturating_sub` | 16610 | 670 | x6.70 |
| `wrapping_sub` | 16610 | 670 | x6.70 |
| `overflowing_add` | 16710 | 770 | x7.70 |
| `saturating_add` | 16710 | 770 | x7.70 |
| `wrapping_add` | 16710 | 770 | x7.70 |
| `div` | 16850 | 910 | x9.10 |
| `div_const` | 16850 | 910 | x9.10 |
| `rem` | 16850 | 910 | x9.10 |
| `rem_const` | 16850 | 910 | x9.10 |
| `checked_mul` | 17050 | 1110 | x11.10 |
| `overflowing_mul` | 17250 | 1310 | x13.10 |
| `wrapping_mul` | 17250 | 1310 | x13.10 |
| `saturating_mul` | 17350 | 1410 | x14.10 |

### arith_u256

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `overflowing_add` | 18490 | 1850 | x1.00 |
| `overflowing_sub` | 18490 | 1850 | x1.00 |
| `wrapping_sub` | 18490 | 1850 | x1.00 |
| `wrapping_add` | 18490 | 1850 | x1.00 |
| `checked_sub` | 18800 | 2160 | x1.17 |
| `sub` | 18800 | 2160 | x1.17 |
| `add` | 18810 | 2170 | x1.17 |
| `checked_add` | 18810 | 2170 | x1.17 |
| `saturating_sub` | 19300 | 2660 | x1.44 |
| `saturating_add` | 19310 | 2670 | x1.44 |
| `rem` | 22490 | 5850 | x3.16 |
| `rem_const` | 22490 | 5850 | x3.16 |
| `div` | 22690 | 6050 | x3.27 |
| `div_const` | 22690 | 6050 | x3.27 |
| `overflowing_mul` | 30420 | 13780 | x7.45 |
| `wrapping_mul` | 30420 | 13780 | x7.45 |
| `checked_mul` | 30520 | 13880 | x7.50 |
| `mul` | 30520 | 13880 | x7.50 |
| `mul_const` | 30520 | 13880 | x7.50 |
| `saturating_mul` | 30920 | 14280 | x7.72 |
| `wide_mul` | 35530 | 18890 | x10.21 |

### arith_u32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `wide_mul` | 16040 | 100 | x1.00 |
| `checked_sub` | 16210 | 270 | x2.70 |
| `sub` | 16210 | 270 | x2.70 |
| `add` | 16310 | 370 | x3.70 |
| `checked_add` | 16310 | 370 | x3.70 |
| `mul` | 16310 | 370 | x3.70 |
| `mul_const` | 16310 | 370 | x3.70 |
| `wrapping_sub` | 16610 | 670 | x6.70 |
| `overflowing_sub` | 16610 | 670 | x6.70 |
| `saturating_sub` | 16610 | 670 | x6.70 |
| `overflowing_add` | 16710 | 770 | x7.70 |
| `saturating_add` | 16710 | 770 | x7.70 |
| `wrapping_add` | 16710 | 770 | x7.70 |
| `div` | 16850 | 910 | x9.10 |
| `div_const` | 16850 | 910 | x9.10 |
| `rem` | 16850 | 910 | x9.10 |
| `rem_const` | 16850 | 910 | x9.10 |
| `checked_mul` | 17050 | 1110 | x11.10 |
| `overflowing_mul` | 17250 | 1310 | x13.10 |
| `wrapping_mul` | 17250 | 1310 | x13.10 |
| `saturating_mul` | 17350 | 1410 | x14.10 |

### arith_u512

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `u256_wide_mul` | 36130 | 19490 | x1.00 |
| `u512_safe_div_rem_by_u256` | 40330 | 23690 | x1.22 |
| `u256_wide_mul_then_div_rem_by_u256` | 58420 | 41780 | x2.14 |

### arith_u64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `wide_mul` | 16040 | 100 | x1.00 |
| `checked_sub` | 16210 | 270 | x2.70 |
| `sub` | 16210 | 270 | x2.70 |
| `add` | 16310 | 370 | x3.70 |
| `checked_add` | 16310 | 370 | x3.70 |
| `mul` | 16310 | 370 | x3.70 |
| `mul_const` | 16310 | 370 | x3.70 |
| `overflowing_sub` | 16610 | 670 | x6.70 |
| `saturating_sub` | 16610 | 670 | x6.70 |
| `wrapping_sub` | 16610 | 670 | x6.70 |
| `overflowing_add` | 16710 | 770 | x7.70 |
| `saturating_add` | 16710 | 770 | x7.70 |
| `wrapping_add` | 16710 | 770 | x7.70 |
| `div_const` | 16850 | 910 | x9.10 |
| `div` | 16850 | 910 | x9.10 |
| `rem_const` | 16850 | 910 | x9.10 |
| `rem` | 16850 | 910 | x9.10 |
| `checked_mul` | 17050 | 1110 | x11.10 |
| `overflowing_mul` | 17250 | 1310 | x13.10 |
| `wrapping_mul` | 17250 | 1310 | x13.10 |
| `saturating_mul` | 17350 | 1410 | x14.10 |

### arith_u8

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `wide_mul` | 16040 | 100 | x1.00 |
| `sub` | 16210 | 270 | x2.70 |
| `checked_sub` | 16210 | 270 | x2.70 |
| `mul` | 16310 | 370 | x3.70 |
| `mul_const` | 16310 | 370 | x3.70 |
| `add` | 16310 | 370 | x3.70 |
| `checked_add` | 16310 | 370 | x3.70 |
| `overflowing_sub` | 16610 | 670 | x6.70 |
| `saturating_sub` | 16610 | 670 | x6.70 |
| `wrapping_sub` | 16610 | 670 | x6.70 |
| `rem_const` | 16650 | 710 | x7.10 |
| `rem` | 16650 | 710 | x7.10 |
| `overflowing_add` | 16710 | 770 | x7.70 |
| `saturating_add` | 16710 | 770 | x7.70 |
| `wrapping_add` | 16710 | 770 | x7.70 |
| `div` | 16850 | 910 | x9.10 |
| `div_const` | 16850 | 910 | x9.10 |
| `checked_mul` | 17050 | 1110 | x11.10 |
| `overflowing_mul` | 17250 | 1310 | x13.10 |
| `wrapping_mul` | 17250 | 1310 | x13.10 |
| `saturating_mul` | 17350 | 1410 | x14.10 |

### array8

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `sum_of_8_locals_reference` | 21430 | 3890 | x1.00 |
| `span_try_into_fixed` | 24330 | 6790 | x1.75 |
| `span_multi_pop_front` | 24500 | 6960 | x1.79 |
| `span_pop_front` | 27030 | 9490 | x2.44 |
| `span_get_unwrap` | 29620 | 12080 | x3.11 |
| `array_at` | 29620 | 12080 | x3.11 |
| `span_index` | 29620 | 12080 | x3.11 |

### bittest_u64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `rem_then_compare` | 17720 | 2180 | x1.00 |
| `and_mask` | 18033 | 2493 | x1.14 |
| `div_then_rem` | 18260 | 2720 | x1.25 |

### bitwise_u128

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `not` | 16140 | 200 | x1.00 |
| `and` | 17523 | 1583 | x7.92 |
| `xor` | 17523 | 1583 | x7.92 |
| `or` | 17523 | 1583 | x7.92 |
| `and_or_xor_x3_assert` | 19989 | 4049 | x20.25 |

### bitwise_u16

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `not` | 16140 | 200 | x1.00 |
| `and` | 17523 | 1583 | x7.92 |
| `or` | 17523 | 1583 | x7.92 |
| `xor` | 17523 | 1583 | x7.92 |
| `and_or_xor_x3_assert` | 19989 | 4049 | x20.25 |

### bitwise_u256

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `not` | 17040 | 400 | x1.00 |
| `and` | 19006 | 2366 | x5.92 |
| `or` | 19206 | 2566 | x6.42 |
| `xor` | 19206 | 2566 | x6.42 |
| `and_or_xor_x3_assert` | 23738 | 7098 | x17.75 |

### bitwise_u32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `not` | 16140 | 200 | x1.00 |
| `xor` | 17523 | 1583 | x7.92 |
| `and` | 17523 | 1583 | x7.92 |
| `or` | 17523 | 1583 | x7.92 |
| `and_or_xor_x3_assert` | 19989 | 4049 | x20.25 |

### bitwise_u64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `not` | 16140 | 200 | x1.00 |
| `and` | 17523 | 1583 | x7.92 |
| `or` | 17523 | 1583 | x7.92 |
| `xor` | 17523 | 1583 | x7.92 |
| `and_or_xor_x3_assert` | 19989 | 4049 | x20.25 |

### bitwise_u8

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `not` | 16140 | 200 | x1.00 |
| `and` | 17523 | 1583 | x7.92 |
| `or` | 17523 | 1583 | x7.92 |
| `xor` | 17523 | 1583 | x7.92 |
| `and_or_xor_x3_assert` | 19989 | 4049 | x20.25 |

### box_mat4

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `box_black_box_unbox` | 17940 | -1000 | - |

### box_u64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `box_black_box_unbox` | 16040 | 200 | x1.00 |

### branch_u64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `early_return` | 16440 | 300 | x1.00 |
| `match_bool` | 16440 | 300 | x1.00 |
| `if_else` | 16440 | 300 | x1.00 |
| `if_else_not_taken` | 16540 | 400 | x1.33 |
| `branchless_felt_arith` | 16980 | 840 | x2.80 |
| `nested_4_conditions` | 17310 | 1170 | x3.90 |

### build8

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array_macro_through_black_box` | 17440 | 1900 | x1.00 |
| `array_append_through_black_box` | 17440 | 1900 | x1.00 |
| `fixed_array_through_black_box` | 17640 | 2100 | x1.11 |
| `tuple_through_black_box` | 17640 | 2100 | x1.11 |
| `array_append_loop_through_black_box` | 32150 | 16610 | x8.74 |

### cmp_felt252

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `ne` | 16340 | 900 | x1.00 |
| `eq` | 16340 | 900 | x1.00 |

### cmp_i128

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `le` | 15910 | 470 | x1.00 |
| `lt` | 16110 | 670 | x1.43 |
| `eq` | 16440 | 1000 | x2.13 |
| `ne` | 16440 | 1000 | x2.13 |
| `gt` | 16610 | 1170 | x2.49 |
| `ge` | 16710 | 1270 | x2.70 |

### cmp_i16

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `le` | 15910 | 470 | x1.00 |
| `lt` | 16110 | 670 | x1.43 |
| `eq` | 16440 | 1000 | x2.13 |
| `ne` | 16440 | 1000 | x2.13 |
| `gt` | 16610 | 1170 | x2.49 |
| `ge` | 16710 | 1270 | x2.70 |

### cmp_i32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `le` | 15910 | 470 | x1.00 |
| `lt` | 16110 | 670 | x1.43 |
| `eq` | 16440 | 1000 | x2.13 |
| `ne` | 16440 | 1000 | x2.13 |
| `gt` | 16610 | 1170 | x2.49 |
| `ge` | 16710 | 1270 | x2.70 |

### cmp_i64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `le` | 15910 | 470 | x1.00 |
| `lt` | 16110 | 670 | x1.43 |
| `eq` | 16440 | 1000 | x2.13 |
| `ne` | 16440 | 1000 | x2.13 |
| `gt` | 16610 | 1170 | x2.49 |
| `ge` | 16710 | 1270 | x2.70 |

### cmp_i8

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `le` | 15910 | 470 | x1.00 |
| `lt` | 16110 | 670 | x1.43 |
| `ne` | 16440 | 1000 | x2.13 |
| `eq` | 16440 | 1000 | x2.13 |
| `gt` | 16610 | 1170 | x2.49 |
| `ge` | 16710 | 1270 | x2.70 |

### cmp_u128

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `ge` | 15910 | 470 | x1.00 |
| `gt` | 16110 | 670 | x1.43 |
| `eq` | 16440 | 1000 | x2.13 |
| `ne` | 16440 | 1000 | x2.13 |
| `lt` | 16610 | 1170 | x2.49 |
| `le` | 16710 | 1270 | x2.70 |

### cmp_u16

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `ge` | 15910 | 470 | x1.00 |
| `gt` | 16110 | 670 | x1.43 |
| `eq` | 16440 | 1000 | x2.13 |
| `ne` | 16440 | 1000 | x2.13 |
| `lt` | 16610 | 1170 | x2.49 |
| `le` | 16710 | 1270 | x2.70 |

### cmp_u256

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `gt` | 16510 | 670 | x1.00 |
| `eq` | 17050 | 1210 | x1.81 |
| `ne` | 17050 | 1210 | x1.81 |
| `le` | 17310 | 1470 | x2.19 |
| `ge` | 17310 | 1470 | x2.19 |
| `lt` | 17310 | 1470 | x2.19 |

### cmp_u32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `ge` | 15910 | 470 | x1.00 |
| `gt` | 16110 | 670 | x1.43 |
| `eq` | 16440 | 1000 | x2.13 |
| `ne` | 16440 | 1000 | x2.13 |
| `lt` | 16610 | 1170 | x2.49 |
| `le` | 16710 | 1270 | x2.70 |

### cmp_u64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `ge` | 15910 | 470 | x1.00 |
| `gt` | 16110 | 670 | x1.43 |
| `eq` | 16440 | 1000 | x2.13 |
| `ne` | 16440 | 1000 | x2.13 |
| `lt` | 16610 | 1170 | x2.49 |
| `le` | 16710 | 1270 | x2.70 |

### cmp_u8

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `ge` | 15910 | 470 | x1.00 |
| `gt` | 16110 | 670 | x1.43 |
| `ne` | 16440 | 1000 | x2.13 |
| `eq` | 16440 | 1000 | x2.13 |
| `lt` | 16610 | 1170 | x2.49 |
| `le` | 16710 | 1270 | x2.70 |

### conv_from_bool

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `into_felt252` | 15240 | 100 | x1.00 |
| `if_else_u64` | 15640 | 500 | x5.00 |
| `into_felt252_try_into_u64` | 15880 | 740 | x7.40 |

### conv_from_felt252

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `try_into_u128` | 15710 | 270 | x1.00 |
| `try_into_i128` | 15810 | 370 | x1.37 |
| `try_into_u8` | 15980 | 540 | x2.00 |
| `try_into_u32` | 15980 | 540 | x2.00 |
| `try_into_u64` | 15980 | 540 | x2.00 |
| `try_into_i64` | 16080 | 640 | x2.37 |
| `into_u256` | 17650 | 2210 | x8.19 |

### conv_from_i128

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `into_felt252` | 15440 | -100 | - |
| `try_into_u128` | 15710 | 170 | - |
| `try_into_u64` | 15980 | 440 | - |
| `try_into_i64` | 16080 | 540 | - |

### conv_from_i64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `into_felt252` | 15440 | -100 | - |
| `into_i128` | 15540 | 0 | - |
| `try_into_u64` | 15710 | 170 | - |
| `try_into_u128` | 15710 | 170 | - |
| `try_into_i32` | 16080 | 540 | - |
| `try_into_i8` | 16080 | 540 | - |

### conv_from_u128

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `into_felt252` | 15440 | -100 | - |
| `into_u256` | 15540 | 0 | - |
| `try_into_i128` | 15810 | 270 | - |
| `try_into_u8` | 15810 | 270 | - |
| `try_into_u32` | 15810 | 270 | - |
| `try_into_u64` | 15810 | 270 | - |

### conv_from_u256

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `try_into_u128` | 15840 | -200 | - |
| `try_into_u64` | 16110 | 70 | - |
| `try_into_u8` | 16110 | 70 | - |
| `try_into_felt252` | 16610 | 570 | - |

### conv_from_u32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `into_felt252` | 15440 | -100 | - |
| `into_u256` | 15540 | 0 | - |
| `into_u128` | 15540 | 0 | - |
| `into_u64` | 15540 | 0 | - |
| `into_i64` | 15540 | 0 | - |
| `try_into_i32` | 15810 | 270 | - |
| `try_into_u16` | 15810 | 270 | - |
| `try_into_u8` | 15810 | 270 | - |

### conv_from_u64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `into_felt252` | 15440 | -100 | - |
| `into_i128` | 15540 | 0 | - |
| `into_u128` | 15540 | 0 | - |
| `into_u256` | 15540 | 0 | - |
| `try_into_u32` | 15810 | 270 | - |
| `try_into_i64` | 15810 | 270 | - |
| `try_into_u8` | 15810 | 270 | - |

### conv_from_u8

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `into_felt252` | 15440 | -100 | - |
| `into_i16` | 15540 | 0 | - |
| `into_u128` | 15540 | 0 | - |
| `into_i128` | 15540 | 0 | - |
| `into_u16` | 15540 | 0 | - |
| `into_u256` | 15540 | 0 | - |
| `into_u32` | 15540 | 0 | - |
| `into_u64` | 15540 | 0 | - |
| `try_into_i8` | 15810 | 270 | - |

### dict8

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array_8_appends_8_reads` | 28220 | 12310 | x1.00 |
| `dict_1_insert_1_get` | 31810 | 15900 | x1.29 |
| `dict_8_inserts` | 67780 | 51870 | x4.21 |
| `dict_8_inserts_8_gets` | 82980 | 67070 | x5.45 |

### divrem_i128

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `div_rem_const` | 18590 | 2350 | x1.00 |
| `div_rem` | 20930 | 4690 | x2.00 |
| `div_and_rem_separately` | 25720 | 9480 | x4.03 |

### divrem_i16

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `div_rem_const` | 18120 | 1880 | x1.00 |
| `div_rem` | 20460 | 4220 | x2.24 |
| `div_and_rem_separately` | 24780 | 8540 | x4.54 |

### divrem_i32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `div_rem_const` | 18120 | 1880 | x1.00 |
| `div_rem` | 20460 | 4220 | x2.24 |
| `div_and_rem_separately` | 24780 | 8540 | x4.54 |

### divrem_i64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `div_rem_const` | 18120 | 1880 | x1.00 |
| `div_rem` | 20460 | 4220 | x2.24 |
| `div_and_rem_separately` | 24780 | 8540 | x4.54 |

### divrem_i8

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `div_rem_const` | 18120 | 1880 | x1.00 |
| `div_rem` | 20460 | 4220 | x2.24 |
| `div_and_rem_separately` | 24780 | 8540 | x4.54 |

### divrem_u128

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `div_rem` | 17620 | 1380 | x1.00 |
| `div_rem_const` | 17620 | 1380 | x1.00 |
| `div_and_rem_separately` | 19100 | 2860 | x2.07 |

### divrem_u16

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `div_rem` | 17150 | 910 | x1.00 |
| `div_rem_const` | 17150 | 910 | x1.00 |
| `div_and_rem_separately` | 18160 | 1920 | x2.11 |

### divrem_u256

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `div_rem` | 23090 | 6050 | x1.00 |
| `div_rem_const` | 23090 | 6050 | x1.00 |
| `div_and_rem_separately` | 30640 | 13600 | x2.25 |

### divrem_u32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `div_rem` | 17150 | 910 | x1.00 |
| `div_rem_const` | 17150 | 910 | x1.00 |
| `div_and_rem_separately` | 18160 | 1920 | x2.11 |

### divrem_u64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `div_rem` | 17150 | 910 | x1.00 |
| `div_rem_const` | 17150 | 910 | x1.00 |
| `div_and_rem_separately` | 18160 | 1920 | x2.11 |

### divrem_u8

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `div_rem` | 16950 | 710 | x1.00 |
| `div_rem_const` | 16950 | 710 | x1.00 |
| `div_and_rem_separately` | 17960 | 1720 | x2.42 |

### dot3_i64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `bounded_floor` | 18990 | 2250 | x1.00 |
| `native_wide_acc` | 23000 | 6260 | x2.78 |
| `naive_3_fpmul` | 29320 | 12580 | x5.59 |

### dot3_u64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `bounded` | 18520 | 1780 | x1.00 |
| `native_wide_acc` | 19530 | 2790 | x1.57 |
| `naive_3_fpmul` | 26030 | 9290 | x5.22 |

### fixed8

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `destructure` | 22930 | 5590 | x1.00 |
| `tuple_destructure` | 22930 | 5590 | x1.00 |
| `span_index` | 29920 | 12580 | x2.25 |

### fpmul_i64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `bounded_floor` | 17790 | 1850 | x1.00 |
| `bounded_floor_pos` | 17790 | 1850 | x1.00 |
| `bounded_trunc_pos` | 17990 | 2050 | x1.11 |
| `bounded_trunc` | 18290 | 2350 | x1.27 |
| `native_widemul_i128_div_pos` | 18730 | 2790 | x1.51 |
| `native_widemul_i128_div` | 18930 | 2990 | x1.62 |

### fpmul_signmag

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `bounded` | 19040 | 2290 | x1.00 |
| `native` | 19510 | 2760 | x1.21 |

### fpmul_u128

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `limbs_bounded` | 20450 | 4510 | x1.00 |
| `limbs` | 26420 | 10480 | x2.32 |
| `naive_u256_div` | 27220 | 11280 | x2.50 |

### fpmul_u64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `bounded` | 17320 | 1380 | x1.00 |
| `native_widemul_div` | 17790 | 1850 | x1.34 |
| `felt_mul_div` | 18060 | 2120 | x1.54 |
| `naive_u128_mul_div` | 20820 | 4880 | x3.54 |
| `naive_u256_mul_div` | 36690 | 20750 | x15.04 |

### index1

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `span_index` | 18910 | 670 | x1.00 |
| `fixed_array_span_index` | 19410 | 1170 | x1.75 |

### inline_large_x1

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `inline_always` | 27800 | 11460 | x1.00 |
| `inline_default` | 29900 | 13560 | x1.18 |
| `inline_never` | 29900 | 13560 | x1.18 |

### inline_large_x4

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `inline_always` | 63890 | 47550 | x1.00 |
| `inline_default` | 72290 | 55950 | x1.18 |
| `inline_never` | 72290 | 55950 | x1.18 |

### inline_medium_x1

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `inline_always` | 18730 | 2790 | x1.00 |
| `inline_default` | 18730 | 2790 | x1.00 |
| `inline_never` | 21130 | 5190 | x1.86 |

### inline_medium_x8

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `inline_default` | 38960 | 23020 | x1.00 |
| `inline_always` | 38960 | 23020 | x1.00 |
| `inline_never` | 58160 | 42220 | x1.83 |

### inline_small_x1

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `hand_inlined` | 17180 | 840 | x1.00 |
| `inline_always` | 17180 | 840 | x1.00 |
| `inline_default` | 17180 | 840 | x1.00 |
| `inline_never` | 19280 | 2940 | x3.50 |

### inline_small_x8

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `inline_default` | 23760 | 7420 | x1.00 |
| `hand_inlined` | 23760 | 7420 | x1.00 |
| `inline_always` | 23760 | 7420 | x1.00 |
| `inline_never` | 40560 | 24220 | x3.26 |

### ispow2_u64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `math_divides_2_pow_63` | 16650 | 1110 | x1.00 |
| `bitwise_x_and_x_minus_1` | 17433 | 1893 | x1.71 |
| `loop_16_iterations` | 51800 | 36260 | x32.67 |

### lookup_pow2

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `shared_span_index` | 19990 | 2810 | x1.00 |
| `const_array_span_index` | 20390 | 3210 | x1.14 |
| `match_jump_table` | 23990 | 6810 | x2.42 |
| `if_tree_binary` | 30440 | 13260 | x4.72 |
| `bits_felt_mul` | 51890 | 34710 | x12.35 |
| `if_chain_linear` | 60080 | 42900 | x15.27 |
| `corelib_pow` | 62330 | 45150 | x16.07 |
| `loop_mul` | 180160 | 162980 | x58.00 |

### loop_fixed1

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `destructure_unrolled` | 16240 | 300 | x1.00 |
| `span_for_iter` | 19480 | 3540 | x11.80 |

### loop_fixed16

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `destructure_unrolled` | 29790 | 10850 | x1.00 |
| `span_for_iter` | 51980 | 33040 | x3.05 |

### loop_fixed4

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `destructure_unrolled` | 18150 | 1610 | x1.00 |
| `span_for_iter` | 26300 | 9760 | x6.06 |

### loop_sum1

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `unrolled_span_index` | 16340 | 200 | x1.00 |
| `unrolled_multi_pop_front` | 16510 | 370 | x1.85 |
| `recursion_tail_acc` | 19080 | 2940 | x14.70 |
| `loop_pop_front` | 19180 | 3040 | x15.20 |
| `for_span_iter` | 19180 | 3040 | x15.20 |
| `while_let_pop_front` | 19180 | 3040 | x15.20 |
| `for_span_iter_felt_acc` | 19920 | 3780 | x18.90 |
| `while_index` | 20120 | 3980 | x19.90 |
| `recursion` | 20150 | 4010 | x20.05 |
| `for_range_index` | 20930 | 4790 | x23.95 |

### loop_sum16

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `unrolled_multi_pop_front` | 30260 | 11120 | x1.00 |
| `unrolled_span_index` | 41540 | 22400 | x2.01 |
| `for_span_iter_felt_acc` | 43470 | 24330 | x2.19 |
| `recursion_tail_acc` | 49680 | 30540 | x2.75 |
| `loop_pop_front` | 49780 | 30640 | x2.76 |
| `for_span_iter` | 49780 | 30640 | x2.76 |
| `while_let_pop_front` | 49780 | 30640 | x2.76 |
| `recursion` | 58250 | 39110 | x3.52 |
| `for_range_index` | 64580 | 45440 | x4.09 |
| `while_index` | 72320 | 53180 | x4.78 |

### loop_sum4

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `unrolled_multi_pop_front` | 18820 | 2080 | x1.00 |
| `unrolled_span_index` | 23060 | 6320 | x3.04 |
| `for_span_iter_felt_acc` | 24630 | 7890 | x3.79 |
| `recursion_tail_acc` | 25200 | 8460 | x4.07 |
| `for_span_iter` | 25300 | 8560 | x4.12 |
| `loop_pop_front` | 25300 | 8560 | x4.12 |
| `while_let_pop_front` | 25300 | 8560 | x4.12 |
| `recursion` | 27770 | 11030 | x5.30 |
| `for_range_index` | 29660 | 12920 | x6.21 |
| `while_index` | 30560 | 13820 | x6.64 |

### lowbits_u128

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `low64_bounded_div_rem` | 16450 | 910 | x1.00 |
| `low64_rem_const` | 16920 | 1380 | x1.52 |
| `low64_and_mask` | 17223 | 1683 | x1.85 |

### lowbits_u256

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `low64_low_limb_bounded` | 16650 | 610 | x1.00 |
| `low64_and_mask` | 18606 | 2566 | x4.21 |
| `low64_rem_const` | 21890 | 5850 | x9.59 |

### lowbits_u64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `low32_bounded_div_rem` | 16450 | 910 | x1.00 |
| `low32_rem_const` | 16450 | 910 | x1.00 |
| `low8_bounded_div_rem` | 16450 | 910 | x1.00 |
| `low8_rem_const` | 16450 | 910 | x1.00 |
| `low32_and_mask` | 17223 | 1683 | x1.85 |
| `low8_and_mask` | 17223 | 1683 | x1.85 |

### marginal_felt252

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `add_x1` | 15940 | 100 | x1.00 |
| `mul_x1` | 15940 | 100 | x1.00 |
| `sub_x1` | 15940 | 100 | x1.00 |
| `eq_select_x1` | 16240 | 400 | x4.00 |
| `add_x9` | 16740 | 900 | x9.00 |
| `mul_x9` | 16740 | 900 | x9.00 |
| `sub_x9` | 16740 | 900 | x9.00 |
| `eq_select_x9` | 20240 | 4400 | x44.00 |

### marginal_i128

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `add_x1` | 16310 | 370 | x1.00 |
| `sub_x1` | 16310 | 370 | x1.00 |
| `eq_select_x1` | 16440 | 500 | x1.35 |
| `lt_select_x1` | 16710 | 770 | x2.08 |
| `eq_select_x9` | 20440 | 4500 | x12.16 |
| `div_x1` | 20630 | 4690 | x12.68 |
| `add_x9` | 22170 | 6230 | x16.84 |
| `sub_x9` | 22170 | 6230 | x16.84 |
| `mul_x1` | 23290 | 7350 | x19.86 |
| `lt_select_x9` | 23350 | 7410 | x20.03 |
| `div_x9` | 59950 | 44010 | x118.95 |
| `mul_x9` | 83390 | 67450 | x182.30 |

### marginal_i64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eq_select_x1` | 16440 | 500 | x1.00 |
| `add_x1` | 16580 | 640 | x1.28 |
| `sub_x1` | 16580 | 640 | x1.28 |
| `mul_x1` | 16580 | 640 | x1.28 |
| `lt_select_x1` | 16710 | 770 | x1.54 |
| `div_x1` | 20160 | 4220 | x8.44 |
| `eq_select_x9` | 20440 | 4500 | x9.00 |
| `lt_select_x9` | 23350 | 7410 | x14.82 |
| `add_x9` | 24330 | 8390 | x16.78 |
| `sub_x9` | 24330 | 8390 | x16.78 |
| `mul_x9` | 24530 | 8590 | x17.18 |
| `div_x9` | 55720 | 39780 | x79.56 |

### marginal_u128

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `add_x1` | 16210 | 270 | x1.00 |
| `sub_x1` | 16210 | 270 | x1.00 |
| `eq_select_x1` | 16440 | 500 | x1.85 |
| `lt_select_x1` | 16710 | 770 | x2.85 |
| `div_x1` | 17320 | 1380 | x5.11 |
| `rem_x1` | 17320 | 1380 | x5.11 |
| `and_x1` | 17523 | 1583 | x5.86 |
| `xor_x1` | 17523 | 1583 | x5.86 |
| `or_x1` | 17523 | 1583 | x5.86 |
| `mul_x1` | 18970 | 3030 | x11.22 |
| `eq_select_x9` | 20440 | 4500 | x16.67 |
| `add_x9` | 21470 | 5530 | x20.48 |
| `sub_x9` | 21470 | 5530 | x20.48 |
| `lt_select_x9` | 23350 | 7410 | x27.44 |
| `and_x9` | 24687 | 8747 | x32.40 |
| `xor_x9` | 24687 | 8747 | x32.40 |
| `or_x9` | 24687 | 8747 | x32.40 |
| `div_x9` | 30280 | 14340 | x53.11 |
| `rem_x9` | 30280 | 14340 | x53.11 |
| `mul_x9` | 46110 | 30170 | x111.74 |

### marginal_u256

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `lt_select_x1` | 17310 | 670 | x1.00 |
| `eq_select_x1` | 17350 | 710 | x1.06 |
| `sub_x1` | 18800 | 2160 | x3.22 |
| `add_x1` | 18810 | 2170 | x3.24 |
| `and_x1` | 19006 | 2366 | x3.53 |
| `or_x1` | 19206 | 2566 | x3.83 |
| `xor_x1` | 19206 | 2566 | x3.83 |
| `div_x1` | 22690 | 6050 | x9.03 |
| `rem_x1` | 22690 | 6050 | x9.03 |
| `eq_select_x9` | 24440 | 7800 | x11.64 |
| `mul_x1` | 30320 | 13680 | x20.42 |
| `lt_select_x9` | 31700 | 15060 | x22.48 |
| `and_x9` | 33134 | 16494 | x24.62 |
| `or_x9` | 33334 | 16694 | x24.92 |
| `xor_x9` | 33334 | 16694 | x24.92 |
| `sub_x9` | 36390 | 19750 | x29.48 |
| `add_x9` | 37290 | 20650 | x30.82 |
| `div_x9` | 73090 | 56450 | x84.25 |
| `rem_x9` | 73090 | 56450 | x84.25 |
| `mul_x9` | 144460 | 127820 | x190.78 |

### marginal_u32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `sub_x1` | 16210 | 270 | x1.00 |
| `add_x1` | 16310 | 370 | x1.37 |
| `mul_x1` | 16310 | 370 | x1.37 |
| `eq_select_x1` | 16440 | 500 | x1.85 |
| `lt_select_x1` | 16710 | 770 | x2.85 |
| `div_x1` | 16850 | 910 | x3.37 |
| `rem_x1` | 16850 | 910 | x3.37 |
| `and_x1` | 17523 | 1583 | x5.86 |
| `or_x1` | 17523 | 1583 | x5.86 |
| `xor_x1` | 17523 | 1583 | x5.86 |
| `eq_select_x9` | 20440 | 4500 | x16.67 |
| `sub_x9` | 21470 | 5530 | x20.48 |
| `add_x9` | 22270 | 6330 | x23.44 |
| `mul_x9` | 22270 | 6330 | x23.44 |
| `lt_select_x9` | 23350 | 7410 | x27.44 |
| `and_x9` | 24687 | 8747 | x32.40 |
| `or_x9` | 24687 | 8747 | x32.40 |
| `xor_x9` | 24687 | 8747 | x32.40 |
| `div_x9` | 26070 | 10130 | x37.52 |
| `rem_x9` | 26070 | 10130 | x37.52 |

### marginal_u64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `sub_x1` | 16210 | 270 | x1.00 |
| `add_x1` | 16310 | 370 | x1.37 |
| `mul_x1` | 16310 | 370 | x1.37 |
| `eq_select_x1` | 16440 | 500 | x1.85 |
| `lt_select_x1` | 16710 | 770 | x2.85 |
| `div_x1` | 16850 | 910 | x3.37 |
| `rem_x1` | 16850 | 910 | x3.37 |
| `and_x1` | 17523 | 1583 | x5.86 |
| `or_x1` | 17523 | 1583 | x5.86 |
| `xor_x1` | 17523 | 1583 | x5.86 |
| `eq_select_x9` | 20440 | 4500 | x16.67 |
| `sub_x9` | 21470 | 5530 | x20.48 |
| `add_x9` | 22270 | 6330 | x23.44 |
| `mul_x9` | 22270 | 6330 | x23.44 |
| `lt_select_x9` | 23350 | 7410 | x27.44 |
| `and_x9` | 24687 | 8747 | x32.40 |
| `or_x9` | 24687 | 8747 | x32.40 |
| `xor_x9` | 24687 | 8747 | x32.40 |
| `div_x9` | 26070 | 10130 | x37.52 |
| `rem_x9` | 26070 | 10130 | x37.52 |

### marginal_u8

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `sub_x1` | 16210 | 270 | x1.00 |
| `add_x1` | 16310 | 370 | x1.37 |
| `eq_select_x1` | 16440 | 500 | x1.85 |
| `lt_select_x1` | 16710 | 770 | x2.85 |
| `div_x1` | 16850 | 910 | x3.37 |
| `rem_x1` | 16850 | 910 | x3.37 |
| `and_x1` | 17523 | 1583 | x5.86 |
| `or_x1` | 17523 | 1583 | x5.86 |
| `xor_x1` | 17523 | 1583 | x5.86 |
| `eq_select_x9` | 20440 | 4500 | x16.67 |
| `sub_x9` | 21470 | 5530 | x20.48 |
| `add_x9` | 22270 | 6330 | x23.44 |
| `lt_select_x9` | 23350 | 7410 | x27.44 |
| `and_x9` | 24687 | 8747 | x32.40 |
| `or_x9` | 24687 | 8747 | x32.40 |
| `xor_x9` | 24687 | 8747 | x32.40 |
| `div_x9` | 25870 | 9930 | x36.78 |
| `rem_x9` | 26070 | 10130 | x37.52 |

### minmax_i64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `min_if` | 16710 | 770 | x1.00 |

### minmax_u64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `min_corelib` | 17010 | 670 | x1.00 |
| `min_overflowing_sub_match` | 17110 | 770 | x1.15 |
| `min_if` | 17110 | 770 | x1.15 |
| `clamp_if_chain` | 17380 | 1040 | x1.55 |
| `clamp_corelib_min_max` | 17880 | 1540 | x2.30 |
| `min_branchless_felt` | 17950 | 1610 | x2.40 |

### msb_u64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `comparison_tree` | 20360 | 4820 | x1.00 |
| `binary_search_cmp_div` | 29990 | 14450 | x3.00 |
| `binary_search_mask_div` | 34178 | 18638 | x3.87 |
| `loop_div2` | 118440 | 102900 | x21.35 |

### parity_u64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `rem_2` | 16750 | 1210 | x1.00 |
| `and_1` | 17433 | 1893 | x1.56 |

### pass_mat4

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `four_scalars_x4` | 33330 | 16950 | x1.00 |
| `boxed_x4` | 35630 | 19250 | x1.14 |
| `by_value_x4` | 38130 | 21750 | x1.28 |
| `fixed_array_x4` | 38130 | 21750 | x1.28 |
| `by_snapshot_x4` | 38130 | 21750 | x1.28 |
| `span_x4` | 45870 | 29490 | x1.74 |

### popcount_u64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `swar_bitwise` | 30152 | 14612 | x1.00 |
| `nibble_table_unrolled` | 58780 | 43240 | x2.96 |
| `nibble_table_loop` | 86230 | 70690 | x4.84 |
| `loop_divrem2` | 140440 | 124900 | x8.55 |

### pow_u64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `const_table_lookup` | 17110 | 1170 | x1.00 |
| `corelib_pow` | 29750 | 13810 | x11.80 |
| `square_and_multiply_loop` | 30950 | 15010 | x12.83 |
| `loop_15_mul` | 43710 | 27770 | x23.74 |

### shift_u128

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `shl64_bounded_downcast_mul` | 15910 | 370 | x1.00 |
| `shr64_bounded_div_rem` | 16450 | 910 | x2.46 |
| `shr64_div_const` | 16920 | 1380 | x3.73 |
| `shl64_mul_const_checked` | 18670 | 3130 | x8.46 |

### shift_u64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `shl8_mul_const_checked` | 15910 | 370 | x1.00 |
| `shr32_bounded_div_rem` | 16450 | 910 | x2.46 |
| `shr8_bounded_div_rem` | 16450 | 910 | x2.46 |
| `shr8_div_const` | 16450 | 910 | x2.46 |
| `shr32_div_const` | 16450 | 910 | x2.46 |
| `shl8_rem_then_mul_wrapping` | 16920 | 1380 | x3.73 |
| `shl8_widemul_rem_wrapping` | 17390 | 1850 | x5.00 |

### shift_var_u64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `shl_lookup_mul` | 18580 | 2640 | x1.00 |
| `shr_lookup_div` | 19120 | 3180 | x1.20 |

### split_u64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `bounded_div_rem` | 16750 | 610 | x1.00 |
| `divrem_const` | 16750 | 610 | x1.00 |
| `div_const_and_mask` | 18333 | 2193 | x3.60 |

### sqrt_u128

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `corelib` | 16620 | 1080 | x1.00 |
| `newton_loop` | 342240 | 326700 | x302.50 |

### sqrt_u64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `corelib` | 16620 | 1080 | x1.00 |
| `newton_unrolled_msb_seed` | 39970 | 24430 | x22.62 |
| `newton_loop` | 150880 | 135340 | x125.31 |

### sum4_u64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `bounded` | 17310 | 570 | x1.00 |
| `via_felt252` | 17480 | 740 | x1.30 |
| `native` | 18050 | 1310 | x2.30 |
| `via_u128` | 18120 | 1380 | x2.42 |

### vec3

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fixed_array_roundtrip_inlined` | 17180 | 840 | x1.00 |
| `plain_locals` | 17180 | 840 | x1.00 |
| `struct_roundtrip_inlined` | 17180 | 840 | x1.00 |
| `tuple_roundtrip_inlined` | 17180 | 840 | x1.00 |
| `call_with_struct_snapshot` | 19480 | 3140 | x3.74 |
| `call_with_fields` | 19480 | 3140 | x3.74 |
| `call_with_struct_value` | 19480 | 3140 | x3.74 |
| `call_make_then_sum` | 20280 | 3940 | x4.69 |

