# Gas report

Sierra gas (`l2_gas`) per benchmark; `net` = raw - group baseline.

## primitives::algo::tests

### abs_i64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `neg_operator` | 8010 | 300 | x1.00 |
| `abs_bounded_constrain_to_u64` | 8380 | 670 | x2.23 |
| `neg_zero_minus_x` | 8450 | 740 | x2.47 |
| `sign_branches` | 8580 | 870 | x2.90 |
| `abs_if_lt_zero_neg` | 8780 | 1070 | x3.57 |
| `abs_via_felt252_to_u64` | 9220 | 1510 | x5.03 |

### minmax_i64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `min_if` | 8880 | 770 | x1.00 |

### minmax_u64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `min_corelib` | 9180 | 670 | x1.00 |
| `min_if` | 9280 | 770 | x1.15 |
| `min_overflowing_sub_match` | 9280 | 770 | x1.15 |
| `clamp_if_chain` | 9550 | 1040 | x1.55 |
| `clamp_corelib_min_max` | 10050 | 1540 | x2.30 |
| `min_branchless_felt` | 10120 | 1610 | x2.40 |

### msb_u64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `comparison_tree` | 12530 | 4820 | x1.00 |
| `binary_search_cmp_div` | 22160 | 14450 | x3.00 |
| `binary_search_mask_div` | 26448 | 18738 | x3.89 |
| `loop_div2` | 110610 | 102900 | x21.35 |

### popcount_u64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `swar_bitwise` | 22422 | 14712 | x1.00 |
| `nibble_table_unrolled` | 50950 | 43240 | x2.94 |
| `nibble_table_loop` | 78400 | 70690 | x4.80 |
| `loop_divrem2` | 132610 | 124900 | x8.49 |

### pow_u64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `const_table_lookup` | 9280 | 1170 | x1.00 |
| `corelib_pow` | 21920 | 13810 | x11.80 |
| `square_and_multiply_loop` | 23120 | 15010 | x12.83 |
| `loop_15_mul` | 35880 | 27770 | x23.74 |

### sqrt_u128

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `corelib` | 8790 | 1080 | x1.00 |
| `newton_loop` | 334410 | 326700 | x302.50 |

### sqrt_u64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `corelib` | 8790 | 1080 | x1.00 |
| `newton_unrolled_msb_seed` | 32140 | 24430 | x22.62 |
| `newton_loop` | 143050 | 135340 | x125.31 |

## primitives::arith::tests

### arith_felt252

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `add` | 8110 | 100 | x1.00 |
| `mul` | 8110 | 100 | x1.00 |
| `neg` | 8110 | 100 | x1.00 |
| `sub` | 8110 | 100 | x1.00 |
| `felt252_div_const` | 8510 | 500 | x5.00 |
| `felt252_div` | 8610 | 600 | x6.00 |

### arith_i128

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `neg` | 8410 | 300 | x1.00 |
| `add` | 8480 | 370 | x1.23 |
| `checked_add` | 8480 | 370 | x1.23 |
| `checked_sub` | 8480 | 370 | x1.23 |
| `sub` | 8480 | 370 | x1.23 |
| `add_pos` | 8780 | 670 | x2.23 |
| `sub_pos` | 8780 | 670 | x2.23 |
| `overflowing_add` | 8880 | 770 | x2.57 |
| `overflowing_sub` | 8880 | 770 | x2.57 |
| `saturating_add` | 8880 | 770 | x2.57 |
| `saturating_sub` | 8880 | 770 | x2.57 |
| `wrapping_add` | 8880 | 770 | x2.57 |
| `wrapping_sub` | 8880 | 770 | x2.57 |
| `div_const` | 10360 | 2250 | x7.50 |
| `rem_const` | 10360 | 2250 | x7.50 |
| `div` | 12800 | 4690 | x15.63 |
| `rem` | 12800 | 4690 | x15.63 |
| `div_pos` | 13000 | 4890 | x16.30 |
| `rem_pos` | 13000 | 4890 | x16.30 |
| `mul` | 15460 | 7350 | x24.50 |
| `mul_const` | 15460 | 7350 | x24.50 |
| `mul_pos` | 15660 | 7550 | x25.17 |

### arith_i16

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `wide_mul` | 8210 | 100 | x1.00 |
| `neg` | 8410 | 300 | x3.00 |
| `add` | 8750 | 640 | x6.40 |
| `checked_add` | 8750 | 640 | x6.40 |
| `checked_sub` | 8750 | 640 | x6.40 |
| `mul` | 8750 | 640 | x6.40 |
| `mul_const` | 8750 | 640 | x6.40 |
| `sub` | 8750 | 640 | x6.40 |
| `add_pos` | 9050 | 940 | x9.40 |
| `mul_pos` | 9050 | 940 | x9.40 |
| `sub_pos` | 9050 | 940 | x9.40 |
| `overflowing_add` | 9150 | 1040 | x10.40 |
| `overflowing_sub` | 9150 | 1040 | x10.40 |
| `saturating_add` | 9150 | 1040 | x10.40 |
| `saturating_sub` | 9150 | 1040 | x10.40 |
| `wrapping_add` | 9150 | 1040 | x10.40 |
| `wrapping_sub` | 9150 | 1040 | x10.40 |
| `div_const` | 9890 | 1780 | x17.80 |
| `rem_const` | 9890 | 1780 | x17.80 |
| `div` | 12330 | 4220 | x42.20 |
| `rem` | 12330 | 4220 | x42.20 |
| `div_pos` | 12530 | 4420 | x44.20 |
| `rem_pos` | 12530 | 4420 | x44.20 |

### arith_i32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `wide_mul` | 8210 | 100 | x1.00 |
| `neg` | 8410 | 300 | x3.00 |
| `add` | 8750 | 640 | x6.40 |
| `checked_add` | 8750 | 640 | x6.40 |
| `checked_sub` | 8750 | 640 | x6.40 |
| `mul` | 8750 | 640 | x6.40 |
| `mul_const` | 8750 | 640 | x6.40 |
| `sub` | 8750 | 640 | x6.40 |
| `add_pos` | 9050 | 940 | x9.40 |
| `mul_pos` | 9050 | 940 | x9.40 |
| `sub_pos` | 9050 | 940 | x9.40 |
| `overflowing_add` | 9150 | 1040 | x10.40 |
| `overflowing_sub` | 9150 | 1040 | x10.40 |
| `saturating_add` | 9150 | 1040 | x10.40 |
| `saturating_sub` | 9150 | 1040 | x10.40 |
| `wrapping_add` | 9150 | 1040 | x10.40 |
| `wrapping_sub` | 9150 | 1040 | x10.40 |
| `div_const` | 9890 | 1780 | x17.80 |
| `rem_const` | 9890 | 1780 | x17.80 |
| `div` | 12330 | 4220 | x42.20 |
| `rem` | 12330 | 4220 | x42.20 |
| `div_pos` | 12530 | 4420 | x44.20 |
| `rem_pos` | 12530 | 4420 | x44.20 |

### arith_i64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `wide_mul` | 8210 | 100 | x1.00 |
| `neg` | 8410 | 300 | x3.00 |
| `add` | 8750 | 640 | x6.40 |
| `checked_add` | 8750 | 640 | x6.40 |
| `checked_sub` | 8750 | 640 | x6.40 |
| `mul` | 8750 | 640 | x6.40 |
| `mul_const` | 8750 | 640 | x6.40 |
| `sub` | 8750 | 640 | x6.40 |
| `add_pos` | 9050 | 940 | x9.40 |
| `mul_pos` | 9050 | 940 | x9.40 |
| `sub_pos` | 9050 | 940 | x9.40 |
| `overflowing_add` | 9150 | 1040 | x10.40 |
| `overflowing_sub` | 9150 | 1040 | x10.40 |
| `saturating_add` | 9150 | 1040 | x10.40 |
| `saturating_sub` | 9150 | 1040 | x10.40 |
| `wrapping_add` | 9150 | 1040 | x10.40 |
| `wrapping_sub` | 9150 | 1040 | x10.40 |
| `div_const` | 9890 | 1780 | x17.80 |
| `rem_const` | 9890 | 1780 | x17.80 |
| `div` | 12330 | 4220 | x42.20 |
| `rem` | 12330 | 4220 | x42.20 |
| `div_pos` | 12530 | 4420 | x44.20 |
| `rem_pos` | 12530 | 4420 | x44.20 |

### arith_i8

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `wide_mul` | 8210 | 100 | x1.00 |
| `neg` | 8410 | 300 | x3.00 |
| `add` | 8750 | 640 | x6.40 |
| `checked_add` | 8750 | 640 | x6.40 |
| `checked_sub` | 8750 | 640 | x6.40 |
| `mul` | 8750 | 640 | x6.40 |
| `mul_const` | 8750 | 640 | x6.40 |
| `sub` | 8750 | 640 | x6.40 |
| `add_pos` | 9050 | 940 | x9.40 |
| `mul_pos` | 9050 | 940 | x9.40 |
| `sub_pos` | 9050 | 940 | x9.40 |
| `overflowing_add` | 9150 | 1040 | x10.40 |
| `overflowing_sub` | 9150 | 1040 | x10.40 |
| `saturating_add` | 9150 | 1040 | x10.40 |
| `saturating_sub` | 9150 | 1040 | x10.40 |
| `wrapping_add` | 9150 | 1040 | x10.40 |
| `wrapping_sub` | 9150 | 1040 | x10.40 |
| `div_const` | 9890 | 1780 | x17.80 |
| `rem_const` | 9890 | 1780 | x17.80 |
| `div` | 12330 | 4220 | x42.20 |
| `rem` | 12330 | 4220 | x42.20 |
| `div_pos` | 12530 | 4420 | x44.20 |
| `rem_pos` | 12530 | 4420 | x44.20 |

### arith_u128

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `add` | 8380 | 270 | x1.00 |
| `checked_add` | 8380 | 270 | x1.00 |
| `checked_sub` | 8380 | 270 | x1.00 |
| `sub` | 8380 | 270 | x1.00 |
| `overflowing_add` | 8780 | 670 | x2.48 |
| `overflowing_sub` | 8780 | 670 | x2.48 |
| `saturating_add` | 8780 | 670 | x2.48 |
| `saturating_sub` | 8780 | 670 | x2.48 |
| `wrapping_add` | 8780 | 670 | x2.48 |
| `wrapping_sub` | 8780 | 670 | x2.48 |
| `div` | 9490 | 1380 | x5.11 |
| `div_const` | 9490 | 1380 | x5.11 |
| `rem` | 9490 | 1380 | x5.11 |
| `rem_const` | 9490 | 1380 | x5.11 |
| `checked_mul` | 11140 | 3030 | x11.22 |
| `mul` | 11140 | 3030 | x11.22 |
| `mul_const` | 11240 | 3130 | x11.59 |
| `overflowing_mul` | 11340 | 3230 | x11.96 |
| `wide_mul` | 11340 | 3230 | x11.96 |
| `wrapping_mul` | 11340 | 3230 | x11.96 |
| `saturating_mul` | 11440 | 3330 | x12.33 |

### arith_u16

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `wide_mul` | 8210 | 100 | x1.00 |
| `checked_sub` | 8380 | 270 | x2.70 |
| `sub` | 8380 | 270 | x2.70 |
| `add` | 8480 | 370 | x3.70 |
| `checked_add` | 8480 | 370 | x3.70 |
| `mul` | 8480 | 370 | x3.70 |
| `mul_const` | 8480 | 370 | x3.70 |
| `overflowing_sub` | 8780 | 670 | x6.70 |
| `saturating_sub` | 8780 | 670 | x6.70 |
| `wrapping_sub` | 8780 | 670 | x6.70 |
| `overflowing_add` | 8880 | 770 | x7.70 |
| `saturating_add` | 8880 | 770 | x7.70 |
| `wrapping_add` | 8880 | 770 | x7.70 |
| `div` | 9020 | 910 | x9.10 |
| `div_const` | 9020 | 910 | x9.10 |
| `rem` | 9020 | 910 | x9.10 |
| `rem_const` | 9020 | 910 | x9.10 |
| `checked_mul` | 9220 | 1110 | x11.10 |
| `overflowing_mul` | 9420 | 1310 | x13.10 |
| `wrapping_mul` | 9420 | 1310 | x13.10 |
| `saturating_mul` | 9520 | 1410 | x14.10 |

### arith_u256

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `overflowing_add` | 10660 | 1850 | x1.00 |
| `overflowing_sub` | 10660 | 1850 | x1.00 |
| `wrapping_add` | 10660 | 1850 | x1.00 |
| `wrapping_sub` | 10660 | 1850 | x1.00 |
| `checked_sub` | 10970 | 2160 | x1.17 |
| `sub` | 10970 | 2160 | x1.17 |
| `add` | 10980 | 2170 | x1.17 |
| `checked_add` | 10980 | 2170 | x1.17 |
| `saturating_sub` | 11470 | 2660 | x1.44 |
| `saturating_add` | 11480 | 2670 | x1.44 |
| `rem` | 14660 | 5850 | x3.16 |
| `rem_const` | 14660 | 5850 | x3.16 |
| `div` | 14860 | 6050 | x3.27 |
| `div_const` | 14860 | 6050 | x3.27 |
| `overflowing_mul` | 22590 | 13780 | x7.45 |
| `wrapping_mul` | 22590 | 13780 | x7.45 |
| `checked_mul` | 22690 | 13880 | x7.50 |
| `mul` | 22690 | 13880 | x7.50 |
| `mul_const` | 22690 | 13880 | x7.50 |
| `saturating_mul` | 23090 | 14280 | x7.72 |
| `wide_mul` | 27700 | 18890 | x10.21 |

### arith_u32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `wide_mul` | 8210 | 100 | x1.00 |
| `checked_sub` | 8380 | 270 | x2.70 |
| `sub` | 8380 | 270 | x2.70 |
| `add` | 8480 | 370 | x3.70 |
| `checked_add` | 8480 | 370 | x3.70 |
| `mul` | 8480 | 370 | x3.70 |
| `mul_const` | 8480 | 370 | x3.70 |
| `overflowing_sub` | 8780 | 670 | x6.70 |
| `saturating_sub` | 8780 | 670 | x6.70 |
| `wrapping_sub` | 8780 | 670 | x6.70 |
| `overflowing_add` | 8880 | 770 | x7.70 |
| `saturating_add` | 8880 | 770 | x7.70 |
| `wrapping_add` | 8880 | 770 | x7.70 |
| `div` | 9020 | 910 | x9.10 |
| `div_const` | 9020 | 910 | x9.10 |
| `rem` | 9020 | 910 | x9.10 |
| `rem_const` | 9020 | 910 | x9.10 |
| `checked_mul` | 9220 | 1110 | x11.10 |
| `overflowing_mul` | 9420 | 1310 | x13.10 |
| `wrapping_mul` | 9420 | 1310 | x13.10 |
| `saturating_mul` | 9520 | 1410 | x14.10 |

### arith_u512

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `u256_wide_mul` | 28300 | 19490 | x1.00 |
| `u512_safe_div_rem_by_u256` | 32500 | 23690 | x1.22 |
| `u256_wide_mul_then_div_rem_by_u256` | 50590 | 41780 | x2.14 |

### arith_u64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `wide_mul` | 8210 | 100 | x1.00 |
| `checked_sub` | 8380 | 270 | x2.70 |
| `sub` | 8380 | 270 | x2.70 |
| `add` | 8480 | 370 | x3.70 |
| `checked_add` | 8480 | 370 | x3.70 |
| `mul` | 8480 | 370 | x3.70 |
| `mul_const` | 8480 | 370 | x3.70 |
| `overflowing_sub` | 8780 | 670 | x6.70 |
| `saturating_sub` | 8780 | 670 | x6.70 |
| `wrapping_sub` | 8780 | 670 | x6.70 |
| `overflowing_add` | 8880 | 770 | x7.70 |
| `saturating_add` | 8880 | 770 | x7.70 |
| `wrapping_add` | 8880 | 770 | x7.70 |
| `div` | 9020 | 910 | x9.10 |
| `div_const` | 9020 | 910 | x9.10 |
| `rem` | 9020 | 910 | x9.10 |
| `rem_const` | 9020 | 910 | x9.10 |
| `checked_mul` | 9220 | 1110 | x11.10 |
| `overflowing_mul` | 9420 | 1310 | x13.10 |
| `wrapping_mul` | 9420 | 1310 | x13.10 |
| `saturating_mul` | 9520 | 1410 | x14.10 |

### arith_u8

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `wide_mul` | 8210 | 100 | x1.00 |
| `checked_sub` | 8380 | 270 | x2.70 |
| `sub` | 8380 | 270 | x2.70 |
| `add` | 8480 | 370 | x3.70 |
| `checked_add` | 8480 | 370 | x3.70 |
| `mul` | 8480 | 370 | x3.70 |
| `mul_const` | 8480 | 370 | x3.70 |
| `overflowing_sub` | 8780 | 670 | x6.70 |
| `saturating_sub` | 8780 | 670 | x6.70 |
| `wrapping_sub` | 8780 | 670 | x6.70 |
| `rem` | 8820 | 710 | x7.10 |
| `rem_const` | 8820 | 710 | x7.10 |
| `overflowing_add` | 8880 | 770 | x7.70 |
| `saturating_add` | 8880 | 770 | x7.70 |
| `wrapping_add` | 8880 | 770 | x7.70 |
| `div` | 9020 | 910 | x9.10 |
| `div_const` | 9020 | 910 | x9.10 |
| `checked_mul` | 9220 | 1110 | x11.10 |
| `overflowing_mul` | 9420 | 1310 | x13.10 |
| `wrapping_mul` | 9420 | 1310 | x13.10 |
| `saturating_mul` | 9520 | 1410 | x14.10 |

### cmp_felt252

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eq` | 8510 | 900 | x1.00 |
| `ne` | 8510 | 900 | x1.00 |

### cmp_i128

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `le` | 8080 | 470 | x1.00 |
| `lt` | 8280 | 670 | x1.43 |
| `eq` | 8610 | 1000 | x2.13 |
| `ne` | 8610 | 1000 | x2.13 |
| `gt` | 8780 | 1170 | x2.49 |
| `ge` | 8880 | 1270 | x2.70 |

### cmp_i16

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `le` | 8080 | 470 | x1.00 |
| `lt` | 8280 | 670 | x1.43 |
| `eq` | 8610 | 1000 | x2.13 |
| `ne` | 8610 | 1000 | x2.13 |
| `gt` | 8780 | 1170 | x2.49 |
| `ge` | 8880 | 1270 | x2.70 |

### cmp_i32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `le` | 8080 | 470 | x1.00 |
| `lt` | 8280 | 670 | x1.43 |
| `eq` | 8610 | 1000 | x2.13 |
| `ne` | 8610 | 1000 | x2.13 |
| `gt` | 8780 | 1170 | x2.49 |
| `ge` | 8880 | 1270 | x2.70 |

### cmp_i64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `le` | 8080 | 470 | x1.00 |
| `lt` | 8280 | 670 | x1.43 |
| `eq` | 8610 | 1000 | x2.13 |
| `ne` | 8610 | 1000 | x2.13 |
| `gt` | 8780 | 1170 | x2.49 |
| `ge` | 8880 | 1270 | x2.70 |

### cmp_i8

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `le` | 8080 | 470 | x1.00 |
| `lt` | 8280 | 670 | x1.43 |
| `eq` | 8610 | 1000 | x2.13 |
| `ne` | 8610 | 1000 | x2.13 |
| `gt` | 8780 | 1170 | x2.49 |
| `ge` | 8880 | 1270 | x2.70 |

### cmp_u128

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `ge` | 8080 | 470 | x1.00 |
| `gt` | 8280 | 670 | x1.43 |
| `eq` | 8610 | 1000 | x2.13 |
| `ne` | 8610 | 1000 | x2.13 |
| `lt` | 8780 | 1170 | x2.49 |
| `le` | 8880 | 1270 | x2.70 |

### cmp_u16

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `ge` | 8080 | 470 | x1.00 |
| `gt` | 8280 | 670 | x1.43 |
| `eq` | 8610 | 1000 | x2.13 |
| `ne` | 8610 | 1000 | x2.13 |
| `lt` | 8780 | 1170 | x2.49 |
| `le` | 8880 | 1270 | x2.70 |

### cmp_u256

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `gt` | 8680 | 670 | x1.00 |
| `eq` | 9220 | 1210 | x1.81 |
| `ne` | 9220 | 1210 | x1.81 |
| `ge` | 9480 | 1470 | x2.19 |
| `le` | 9480 | 1470 | x2.19 |
| `lt` | 9480 | 1470 | x2.19 |

### cmp_u32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `ge` | 8080 | 470 | x1.00 |
| `gt` | 8280 | 670 | x1.43 |
| `eq` | 8610 | 1000 | x2.13 |
| `ne` | 8610 | 1000 | x2.13 |
| `lt` | 8780 | 1170 | x2.49 |
| `le` | 8880 | 1270 | x2.70 |

### cmp_u64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `ge` | 8080 | 470 | x1.00 |
| `gt` | 8280 | 670 | x1.43 |
| `eq` | 8610 | 1000 | x2.13 |
| `ne` | 8610 | 1000 | x2.13 |
| `lt` | 8780 | 1170 | x2.49 |
| `le` | 8880 | 1270 | x2.70 |

### cmp_u8

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `ge` | 8080 | 470 | x1.00 |
| `gt` | 8280 | 670 | x1.43 |
| `eq` | 8610 | 1000 | x2.13 |
| `ne` | 8610 | 1000 | x2.13 |
| `lt` | 8780 | 1170 | x2.49 |
| `le` | 8880 | 1270 | x2.70 |

### divrem_i128

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `div_rem_const` | 10760 | 2350 | x1.00 |
| `div_rem` | 13100 | 4690 | x2.00 |
| `div_and_rem_separately` | 17890 | 9480 | x4.03 |

### divrem_i16

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `div_rem_const` | 10290 | 1880 | x1.00 |
| `div_rem` | 12630 | 4220 | x2.24 |
| `div_and_rem_separately` | 16950 | 8540 | x4.54 |

### divrem_i32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `div_rem_const` | 10290 | 1880 | x1.00 |
| `div_rem` | 12630 | 4220 | x2.24 |
| `div_and_rem_separately` | 16950 | 8540 | x4.54 |

### divrem_i64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `div_rem_const` | 10290 | 1880 | x1.00 |
| `div_rem` | 12630 | 4220 | x2.24 |
| `div_and_rem_separately` | 16950 | 8540 | x4.54 |

### divrem_i8

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `div_rem_const` | 10290 | 1880 | x1.00 |
| `div_rem` | 12630 | 4220 | x2.24 |
| `div_and_rem_separately` | 16950 | 8540 | x4.54 |

### divrem_u128

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `div_rem` | 9790 | 1380 | x1.00 |
| `div_rem_const` | 9790 | 1380 | x1.00 |
| `div_and_rem_separately` | 11270 | 2860 | x2.07 |

### divrem_u16

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `div_rem` | 9320 | 910 | x1.00 |
| `div_rem_const` | 9320 | 910 | x1.00 |
| `div_and_rem_separately` | 10330 | 1920 | x2.11 |

### divrem_u256

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `div_rem` | 15260 | 6050 | x1.00 |
| `div_rem_const` | 15260 | 6050 | x1.00 |
| `div_and_rem_separately` | 22810 | 13600 | x2.25 |

### divrem_u32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `div_rem` | 9320 | 910 | x1.00 |
| `div_rem_const` | 9320 | 910 | x1.00 |
| `div_and_rem_separately` | 10330 | 1920 | x2.11 |

### divrem_u64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `div_rem` | 9320 | 910 | x1.00 |
| `div_rem_const` | 9320 | 910 | x1.00 |
| `div_and_rem_separately` | 10330 | 1920 | x2.11 |

### divrem_u8

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `div_rem` | 9120 | 710 | x1.00 |
| `div_rem_const` | 9120 | 710 | x1.00 |
| `div_and_rem_separately` | 10130 | 1720 | x2.42 |

## primitives::bitops::tests

### bittest_u64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `rem_then_compare` | 9890 | 2180 | x1.00 |
| `and_mask` | 10303 | 2593 | x1.19 |
| `div_then_rem` | 10430 | 2720 | x1.25 |

### ispow2_u64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `math_divides_2_pow_63` | 8820 | 1110 | x1.00 |
| `bitwise_x_and_x_minus_1` | 9703 | 1993 | x1.80 |
| `loop_16_iterations` | 43970 | 36260 | x32.67 |

### lowbits_u128

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `low64_bounded_div_rem` | 8620 | 910 | x1.00 |
| `low64_rem_const` | 9090 | 1380 | x1.52 |
| `low64_and_mask` | 9493 | 1783 | x1.96 |

### lowbits_u256

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `low64_low_limb_bounded` | 8820 | 610 | x1.00 |
| `low64_and_mask` | 10876 | 2666 | x4.37 |
| `low64_rem_const` | 14060 | 5850 | x9.59 |

### lowbits_u64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `low32_bounded_div_rem` | 8620 | 910 | x1.00 |
| `low32_rem_const` | 8620 | 910 | x1.00 |
| `low8_bounded_div_rem` | 8620 | 910 | x1.00 |
| `low8_rem_const` | 8620 | 910 | x1.00 |
| `low32_and_mask` | 9493 | 1783 | x1.96 |
| `low8_and_mask` | 9493 | 1783 | x1.96 |

### parity_u64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `rem_2` | 8920 | 1210 | x1.00 |
| `and_1` | 9703 | 1993 | x1.65 |

### shift_u128

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `shl64_bounded_downcast_mul` | 8080 | 370 | x1.00 |
| `shr64_bounded_div_rem` | 8620 | 910 | x2.46 |
| `shr64_div_const` | 9090 | 1380 | x3.73 |
| `shl64_mul_const_checked` | 10840 | 3130 | x8.46 |

### shift_u64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `shl8_mul_const_checked` | 8080 | 370 | x1.00 |
| `shr32_bounded_div_rem` | 8620 | 910 | x2.46 |
| `shr32_div_const` | 8620 | 910 | x2.46 |
| `shr8_bounded_div_rem` | 8620 | 910 | x2.46 |
| `shr8_div_const` | 8620 | 910 | x2.46 |
| `shl8_rem_then_mul_wrapping` | 9090 | 1380 | x3.73 |
| `shl8_widemul_rem_wrapping` | 9560 | 1850 | x5.00 |

### shift_var_u64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `shl_lookup_mul` | 10750 | 2640 | x1.00 |
| `shr_lookup_div` | 11290 | 3180 | x1.20 |

### split_u64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `bounded_div_rem` | 8920 | 610 | x1.00 |
| `divrem_const` | 8920 | 610 | x1.00 |
| `div_const_and_mask` | 10603 | 2293 | x3.76 |

## primitives::bitwise::tests

### bitwise_u128

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `not` | 8310 | 200 | x1.00 |
| `and` | 9793 | 1683 | x8.41 |
| `or` | 9793 | 1683 | x8.41 |
| `xor` | 9793 | 1683 | x8.41 |
| `and_or_xor_x3_assert` | 12259 | 4149 | x20.75 |

### bitwise_u16

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `not` | 8310 | 200 | x1.00 |
| `and` | 9793 | 1683 | x8.41 |
| `or` | 9793 | 1683 | x8.41 |
| `xor` | 9793 | 1683 | x8.41 |
| `and_or_xor_x3_assert` | 12259 | 4149 | x20.75 |

### bitwise_u256

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `not` | 9210 | 400 | x1.00 |
| `and` | 11276 | 2466 | x6.17 |
| `or` | 11476 | 2666 | x6.67 |
| `xor` | 11476 | 2666 | x6.67 |
| `and_or_xor_x3_assert` | 16008 | 7198 | x18.00 |

### bitwise_u32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `not` | 8310 | 200 | x1.00 |
| `and` | 9793 | 1683 | x8.41 |
| `or` | 9793 | 1683 | x8.41 |
| `xor` | 9793 | 1683 | x8.41 |
| `and_or_xor_x3_assert` | 12259 | 4149 | x20.75 |

### bitwise_u64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `not` | 8310 | 200 | x1.00 |
| `and` | 9793 | 1683 | x8.41 |
| `or` | 9793 | 1683 | x8.41 |
| `xor` | 9793 | 1683 | x8.41 |
| `and_or_xor_x3_assert` | 12259 | 4149 | x20.75 |

### bitwise_u8

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `not` | 8310 | 200 | x1.00 |
| `and` | 9793 | 1683 | x8.41 |
| `or` | 9793 | 1683 | x8.41 |
| `xor` | 9793 | 1683 | x8.41 |
| `and_or_xor_x3_assert` | 12259 | 4149 | x20.75 |

## primitives::bounded::tests

### dot3_i64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `bounded_floor` | 11160 | 2250 | x1.00 |
| `native_wide_acc` | 15170 | 6260 | x2.78 |
| `naive_3_fpmul` | 21490 | 12580 | x5.59 |

### dot3_u64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `bounded` | 10690 | 1780 | x1.00 |
| `native_wide_acc` | 11700 | 2790 | x1.57 |
| `naive_3_fpmul` | 18200 | 9290 | x5.22 |

### fpmul_i64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `bounded_floor` | 9960 | 1850 | x1.00 |
| `bounded_floor_pos` | 9960 | 1850 | x1.00 |
| `bounded_trunc_pos` | 10160 | 2050 | x1.11 |
| `bounded_trunc` | 10460 | 2350 | x1.27 |
| `native_widemul_i128_div_pos` | 10900 | 2790 | x1.51 |
| `native_widemul_i128_div` | 11100 | 2990 | x1.62 |

### fpmul_signmag

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `bounded` | 11210 | 2290 | x1.00 |
| `native` | 11680 | 2760 | x1.21 |

### fpmul_u128

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `limbs_bounded` | 12620 | 4510 | x1.00 |
| `limbs` | 18590 | 10480 | x2.32 |
| `naive_u256_div` | 19390 | 11280 | x2.50 |

### fpmul_u64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `bounded` | 9490 | 1380 | x1.00 |
| `native_widemul_div` | 9960 | 1850 | x1.34 |
| `felt_mul_div` | 10230 | 2120 | x1.54 |
| `naive_u128_mul_div` | 12990 | 4880 | x3.54 |
| `naive_u256_mul_div` | 28860 | 20750 | x15.04 |

### sum4_u64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `bounded` | 9480 | 570 | x1.00 |
| `via_felt252` | 9650 | 740 | x1.30 |
| `native` | 10220 | 1310 | x2.30 |
| `via_u128` | 10290 | 1380 | x2.42 |

## primitives::control::tests

### branch_u64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `early_return` | 8610 | 300 | x1.00 |
| `if_else` | 8610 | 300 | x1.00 |
| `match_bool` | 8610 | 300 | x1.00 |
| `if_else_not_taken` | 8710 | 400 | x1.33 |
| `branchless_felt_arith` | 9150 | 840 | x2.80 |
| `nested_4_conditions` | 9480 | 1170 | x3.90 |

### loop_fixed1

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `destructure_unrolled` | 8410 | 300 | x1.00 |
| `span_for_iter` | 11650 | 3540 | x11.80 |

### loop_fixed16

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `destructure_unrolled` | 21960 | 10850 | x1.00 |
| `span_for_iter` | 44150 | 33040 | x3.05 |

### loop_fixed4

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `destructure_unrolled` | 10320 | 1610 | x1.00 |
| `span_for_iter` | 18470 | 9760 | x6.06 |

### loop_sum1

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `unrolled_span_index` | 8510 | 200 | x1.00 |
| `unrolled_multi_pop_front` | 8680 | 370 | x1.85 |
| `recursion_tail_acc` | 11250 | 2940 | x14.70 |
| `for_span_iter` | 11350 | 3040 | x15.20 |
| `loop_pop_front` | 11350 | 3040 | x15.20 |
| `while_let_pop_front` | 11350 | 3040 | x15.20 |
| `for_span_iter_felt_acc` | 12090 | 3780 | x18.90 |
| `while_index` | 12290 | 3980 | x19.90 |
| `recursion` | 12320 | 4010 | x20.05 |
| `for_range_index` | 13100 | 4790 | x23.95 |

### loop_sum16

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `unrolled_multi_pop_front` | 22430 | 11120 | x1.00 |
| `unrolled_span_index` | 33710 | 22400 | x2.01 |
| `for_span_iter_felt_acc` | 35640 | 24330 | x2.19 |
| `recursion_tail_acc` | 41850 | 30540 | x2.75 |
| `for_span_iter` | 41950 | 30640 | x2.76 |
| `loop_pop_front` | 41950 | 30640 | x2.76 |
| `while_let_pop_front` | 41950 | 30640 | x2.76 |
| `recursion` | 50420 | 39110 | x3.52 |
| `for_range_index` | 56750 | 45440 | x4.09 |
| `while_index` | 64490 | 53180 | x4.78 |

### loop_sum4

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `unrolled_multi_pop_front` | 10990 | 2080 | x1.00 |
| `unrolled_span_index` | 15230 | 6320 | x3.04 |
| `for_span_iter_felt_acc` | 16800 | 7890 | x3.79 |
| `recursion_tail_acc` | 17370 | 8460 | x4.07 |
| `for_span_iter` | 17470 | 8560 | x4.12 |
| `loop_pop_front` | 17470 | 8560 | x4.12 |
| `while_let_pop_front` | 17470 | 8560 | x4.12 |
| `recursion` | 19940 | 11030 | x5.30 |
| `for_range_index` | 21830 | 12920 | x6.21 |
| `while_index` | 22730 | 13820 | x6.64 |

## primitives::conversions::tests

### conv_from_bool

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `into_felt252` | 7410 | 100 | x1.00 |
| `if_else_u64` | 7810 | 500 | x5.00 |
| `into_felt252_try_into_u64` | 8050 | 740 | x7.40 |

### conv_from_felt252

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `try_into_u128` | 7880 | 270 | x1.00 |
| `try_into_i128` | 7980 | 370 | x1.37 |
| `try_into_u32` | 8150 | 540 | x2.00 |
| `try_into_u64` | 8150 | 540 | x2.00 |
| `try_into_u8` | 8150 | 540 | x2.00 |
| `try_into_i64` | 8250 | 640 | x2.37 |
| `into_u256` | 9820 | 2210 | x8.19 |

### conv_from_i128

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `into_felt252` | 7610 | -100 | - |
| `try_into_u128` | 7880 | 170 | - |
| `try_into_u64` | 8150 | 440 | - |
| `try_into_i64` | 8250 | 540 | - |

### conv_from_i64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `into_felt252` | 7610 | -100 | - |
| `into_i128` | 7710 | 0 | - |
| `try_into_u128` | 7880 | 170 | - |
| `try_into_u64` | 7880 | 170 | - |
| `try_into_i32` | 8250 | 540 | - |
| `try_into_i8` | 8250 | 540 | - |

### conv_from_u128

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `into_felt252` | 7610 | -100 | - |
| `into_u256` | 7710 | 0 | - |
| `try_into_i128` | 7980 | 270 | - |
| `try_into_u32` | 7980 | 270 | - |
| `try_into_u64` | 7980 | 270 | - |
| `try_into_u8` | 7980 | 270 | - |

### conv_from_u256

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `try_into_u128` | 8010 | -200 | - |
| `try_into_u64` | 8280 | 70 | - |
| `try_into_u8` | 8280 | 70 | - |
| `try_into_felt252` | 8780 | 570 | - |

### conv_from_u32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `into_felt252` | 7610 | -100 | - |
| `into_i64` | 7710 | 0 | - |
| `into_u128` | 7710 | 0 | - |
| `into_u256` | 7710 | 0 | - |
| `into_u64` | 7710 | 0 | - |
| `try_into_i32` | 7980 | 270 | - |
| `try_into_u16` | 7980 | 270 | - |
| `try_into_u8` | 7980 | 270 | - |

### conv_from_u64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `into_felt252` | 7610 | -100 | - |
| `into_i128` | 7710 | 0 | - |
| `into_u128` | 7710 | 0 | - |
| `into_u256` | 7710 | 0 | - |
| `try_into_i64` | 7980 | 270 | - |
| `try_into_u32` | 7980 | 270 | - |
| `try_into_u8` | 7980 | 270 | - |

### conv_from_u8

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `into_felt252` | 7610 | -100 | - |
| `into_i128` | 7710 | 0 | - |
| `into_i16` | 7710 | 0 | - |
| `into_u128` | 7710 | 0 | - |
| `into_u16` | 7710 | 0 | - |
| `into_u256` | 7710 | 0 | - |
| `into_u32` | 7710 | 0 | - |
| `into_u64` | 7710 | 0 | - |
| `try_into_i8` | 7980 | 270 | - |

## primitives::data::tests

### array8

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `sum_of_8_locals_reference` | 13600 | 3890 | x1.00 |
| `span_try_into_fixed` | 16500 | 6790 | x1.75 |
| `span_multi_pop_front` | 16670 | 6960 | x1.79 |
| `span_pop_front` | 19200 | 9490 | x2.44 |
| `array_at` | 21790 | 12080 | x3.11 |
| `span_get_unwrap` | 21790 | 12080 | x3.11 |
| `span_index` | 21790 | 12080 | x3.11 |

### box_mat4

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `box_black_box_unbox` | 10110 | -1000 | - |

### box_u64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `box_black_box_unbox` | 8210 | 200 | x1.00 |

### build8

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array_append_through_black_box` | 9610 | 1900 | x1.00 |
| `array_macro_through_black_box` | 9610 | 1900 | x1.00 |
| `fixed_array_through_black_box` | 9810 | 2100 | x1.11 |
| `tuple_through_black_box` | 9810 | 2100 | x1.11 |
| `array_append_loop_through_black_box` | 24320 | 16610 | x8.74 |

### dict8

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array_8_appends_8_reads` | 20390 | 12310 | x1.00 |
| `dict_1_insert_1_get` | 24080 | 16000 | x1.30 |
| `dict_8_inserts` | 60050 | 51970 | x4.22 |
| `dict_8_inserts_8_gets` | 75250 | 67170 | x5.46 |

### fixed8

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `destructure` | 15100 | 5590 | x1.00 |
| `tuple_destructure` | 15100 | 5590 | x1.00 |
| `span_index` | 22090 | 12580 | x2.25 |

### index1

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `span_index` | 11080 | 670 | x1.00 |
| `fixed_array_span_index` | 11580 | 1170 | x1.75 |

### pass_mat4

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `four_scalars_x4` | 25500 | 16950 | x1.00 |
| `boxed_x4` | 27800 | 19250 | x1.14 |
| `by_snapshot_x4` | 30300 | 21750 | x1.28 |
| `by_value_x4` | 30300 | 21750 | x1.28 |
| `fixed_array_x4` | 30300 | 21750 | x1.28 |
| `span_x4` | 38040 | 29490 | x1.74 |

### vec3

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fixed_array_roundtrip_inlined` | 9350 | 840 | x1.00 |
| `plain_locals` | 9350 | 840 | x1.00 |
| `struct_roundtrip_inlined` | 9350 | 840 | x1.00 |
| `tuple_roundtrip_inlined` | 9350 | 840 | x1.00 |
| `call_with_fields` | 11650 | 3140 | x3.74 |
| `call_with_struct_snapshot` | 11650 | 3140 | x3.74 |
| `call_with_struct_value` | 11650 | 3140 | x3.74 |
| `call_make_then_sum` | 12450 | 3940 | x4.69 |

## primitives::inlining::tests

### inline_large_x1

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `inline_always` | 19970 | 11460 | x1.00 |
| `inline_default` | 22070 | 13560 | x1.18 |
| `inline_never` | 22070 | 13560 | x1.18 |

### inline_large_x4

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `inline_always` | 56060 | 47550 | x1.00 |
| `inline_default` | 64460 | 55950 | x1.18 |
| `inline_never` | 64460 | 55950 | x1.18 |

### inline_medium_x1

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `inline_always` | 10900 | 2790 | x1.00 |
| `inline_default` | 10900 | 2790 | x1.00 |
| `inline_never` | 13300 | 5190 | x1.86 |

### inline_medium_x8

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `inline_always` | 31130 | 23020 | x1.00 |
| `inline_default` | 31130 | 23020 | x1.00 |
| `inline_never` | 50330 | 42220 | x1.83 |

### inline_small_x1

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `hand_inlined` | 9350 | 840 | x1.00 |
| `inline_always` | 9350 | 840 | x1.00 |
| `inline_default` | 9350 | 840 | x1.00 |
| `inline_never` | 11450 | 2940 | x3.50 |

### inline_small_x8

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `hand_inlined` | 15930 | 7420 | x1.00 |
| `inline_always` | 15930 | 7420 | x1.00 |
| `inline_default` | 15930 | 7420 | x1.00 |
| `inline_never` | 32730 | 24220 | x3.26 |

## primitives::lookup::tests

### lookup_pow2

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `shared_span_index` | 12160 | 2810 | x1.00 |
| `const_array_span_index` | 12560 | 3210 | x1.14 |
| `match_jump_table` | 16160 | 6810 | x2.42 |
| `if_tree_binary` | 22610 | 13260 | x4.72 |
| `bits_felt_mul` | 44060 | 34710 | x12.35 |
| `if_chain_linear` | 52250 | 42900 | x15.27 |
| `corelib_pow` | 54500 | 45150 | x16.07 |
| `loop_mul` | 172330 | 162980 | x58.00 |

## primitives::marginal::tests

### marginal_felt252

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `add_x1` | 8110 | 100 | x1.00 |
| `mul_x1` | 8110 | 100 | x1.00 |
| `sub_x1` | 8110 | 100 | x1.00 |
| `eq_select_x1` | 8410 | 400 | x4.00 |
| `add_x9` | 8910 | 900 | x9.00 |
| `mul_x9` | 8910 | 900 | x9.00 |
| `sub_x9` | 8910 | 900 | x9.00 |
| `eq_select_x9` | 12410 | 4400 | x44.00 |

### marginal_i128

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `add_x1` | 8480 | 370 | x1.00 |
| `sub_x1` | 8480 | 370 | x1.00 |
| `eq_select_x1` | 8610 | 500 | x1.35 |
| `lt_select_x1` | 8880 | 770 | x2.08 |
| `eq_select_x9` | 12610 | 4500 | x12.16 |
| `div_x1` | 12800 | 4690 | x12.68 |
| `add_x9` | 14340 | 6230 | x16.84 |
| `sub_x9` | 14340 | 6230 | x16.84 |
| `mul_x1` | 15460 | 7350 | x19.86 |
| `lt_select_x9` | 15520 | 7410 | x20.03 |
| `div_x9` | 52120 | 44010 | x118.95 |
| `mul_x9` | 75560 | 67450 | x182.30 |

### marginal_i64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `eq_select_x1` | 8610 | 500 | x1.00 |
| `add_x1` | 8750 | 640 | x1.28 |
| `mul_x1` | 8750 | 640 | x1.28 |
| `sub_x1` | 8750 | 640 | x1.28 |
| `lt_select_x1` | 8880 | 770 | x1.54 |
| `div_x1` | 12330 | 4220 | x8.44 |
| `eq_select_x9` | 12610 | 4500 | x9.00 |
| `lt_select_x9` | 15520 | 7410 | x14.82 |
| `add_x9` | 16500 | 8390 | x16.78 |
| `sub_x9` | 16500 | 8390 | x16.78 |
| `mul_x9` | 16700 | 8590 | x17.18 |
| `div_x9` | 47890 | 39780 | x79.56 |

### marginal_u128

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `add_x1` | 8380 | 270 | x1.00 |
| `sub_x1` | 8380 | 270 | x1.00 |
| `eq_select_x1` | 8610 | 500 | x1.85 |
| `lt_select_x1` | 8880 | 770 | x2.85 |
| `div_x1` | 9490 | 1380 | x5.11 |
| `rem_x1` | 9490 | 1380 | x5.11 |
| `and_x1` | 9793 | 1683 | x6.23 |
| `or_x1` | 9793 | 1683 | x6.23 |
| `xor_x1` | 9793 | 1683 | x6.23 |
| `mul_x1` | 11140 | 3030 | x11.22 |
| `eq_select_x9` | 12610 | 4500 | x16.67 |
| `add_x9` | 13640 | 5530 | x20.48 |
| `sub_x9` | 13640 | 5530 | x20.48 |
| `lt_select_x9` | 15520 | 7410 | x27.44 |
| `and_x9` | 16957 | 8847 | x32.77 |
| `or_x9` | 16957 | 8847 | x32.77 |
| `xor_x9` | 16957 | 8847 | x32.77 |
| `div_x9` | 22450 | 14340 | x53.11 |
| `rem_x9` | 22450 | 14340 | x53.11 |
| `mul_x9` | 38280 | 30170 | x111.74 |

### marginal_u256

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `lt_select_x1` | 9480 | 670 | x1.00 |
| `eq_select_x1` | 9520 | 710 | x1.06 |
| `sub_x1` | 10970 | 2160 | x3.22 |
| `add_x1` | 10980 | 2170 | x3.24 |
| `and_x1` | 11276 | 2466 | x3.68 |
| `or_x1` | 11476 | 2666 | x3.98 |
| `xor_x1` | 11476 | 2666 | x3.98 |
| `div_x1` | 14860 | 6050 | x9.03 |
| `rem_x1` | 14860 | 6050 | x9.03 |
| `eq_select_x9` | 16610 | 7800 | x11.64 |
| `mul_x1` | 22490 | 13680 | x20.42 |
| `lt_select_x9` | 23870 | 15060 | x22.48 |
| `and_x9` | 25404 | 16594 | x24.77 |
| `or_x9` | 25604 | 16794 | x25.07 |
| `xor_x9` | 25604 | 16794 | x25.07 |
| `sub_x9` | 28560 | 19750 | x29.48 |
| `add_x9` | 29460 | 20650 | x30.82 |
| `div_x9` | 65260 | 56450 | x84.25 |
| `rem_x9` | 65260 | 56450 | x84.25 |
| `mul_x9` | 136630 | 127820 | x190.78 |

### marginal_u32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `sub_x1` | 8380 | 270 | x1.00 |
| `add_x1` | 8480 | 370 | x1.37 |
| `mul_x1` | 8480 | 370 | x1.37 |
| `eq_select_x1` | 8610 | 500 | x1.85 |
| `lt_select_x1` | 8880 | 770 | x2.85 |
| `div_x1` | 9020 | 910 | x3.37 |
| `rem_x1` | 9020 | 910 | x3.37 |
| `and_x1` | 9793 | 1683 | x6.23 |
| `or_x1` | 9793 | 1683 | x6.23 |
| `xor_x1` | 9793 | 1683 | x6.23 |
| `eq_select_x9` | 12610 | 4500 | x16.67 |
| `sub_x9` | 13640 | 5530 | x20.48 |
| `add_x9` | 14440 | 6330 | x23.44 |
| `mul_x9` | 14440 | 6330 | x23.44 |
| `lt_select_x9` | 15520 | 7410 | x27.44 |
| `and_x9` | 16957 | 8847 | x32.77 |
| `or_x9` | 16957 | 8847 | x32.77 |
| `xor_x9` | 16957 | 8847 | x32.77 |
| `div_x9` | 18240 | 10130 | x37.52 |
| `rem_x9` | 18240 | 10130 | x37.52 |

### marginal_u64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `sub_x1` | 8380 | 270 | x1.00 |
| `add_x1` | 8480 | 370 | x1.37 |
| `mul_x1` | 8480 | 370 | x1.37 |
| `eq_select_x1` | 8610 | 500 | x1.85 |
| `lt_select_x1` | 8880 | 770 | x2.85 |
| `div_x1` | 9020 | 910 | x3.37 |
| `rem_x1` | 9020 | 910 | x3.37 |
| `and_x1` | 9793 | 1683 | x6.23 |
| `or_x1` | 9793 | 1683 | x6.23 |
| `xor_x1` | 9793 | 1683 | x6.23 |
| `eq_select_x9` | 12610 | 4500 | x16.67 |
| `sub_x9` | 13640 | 5530 | x20.48 |
| `add_x9` | 14440 | 6330 | x23.44 |
| `mul_x9` | 14440 | 6330 | x23.44 |
| `lt_select_x9` | 15520 | 7410 | x27.44 |
| `and_x9` | 16957 | 8847 | x32.77 |
| `or_x9` | 16957 | 8847 | x32.77 |
| `xor_x9` | 16957 | 8847 | x32.77 |
| `div_x9` | 18240 | 10130 | x37.52 |
| `rem_x9` | 18240 | 10130 | x37.52 |

### marginal_u8

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `sub_x1` | 8380 | 270 | x1.00 |
| `add_x1` | 8480 | 370 | x1.37 |
| `eq_select_x1` | 8610 | 500 | x1.85 |
| `lt_select_x1` | 8880 | 770 | x2.85 |
| `div_x1` | 9020 | 910 | x3.37 |
| `rem_x1` | 9020 | 910 | x3.37 |
| `and_x1` | 9793 | 1683 | x6.23 |
| `or_x1` | 9793 | 1683 | x6.23 |
| `xor_x1` | 9793 | 1683 | x6.23 |
| `eq_select_x9` | 12610 | 4500 | x16.67 |
| `sub_x9` | 13640 | 5530 | x20.48 |
| `add_x9` | 14440 | 6330 | x23.44 |
| `lt_select_x9` | 15520 | 7410 | x27.44 |
| `and_x9` | 16957 | 8847 | x32.77 |
| `or_x9` | 16957 | 8847 | x32.77 |
| `xor_x9` | 16957 | 8847 | x32.77 |
| `div_x9` | 18040 | 9930 | x36.78 |
| `rem_x9` | 18240 | 10130 | x37.52 |

