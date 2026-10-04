# Steps report

Cairo steps per probe (`snforge test --tracked-resource cairo-steps --detailed-resources`); `net` = raw - group baseline.

## nalgebra_probes_steps::cholesky

| probe | variant | baseline | raw | net steps |
|---|---|---:|---:|---:|
| `cholesky3_factor` | `op` | 121 | 326 | 205 |
| `cholesky3_solve` | `op` | 106 | 332 | 226 |
| `cholesky6_factor` | `op` | 275 | 1097 | 822 |
| `cholesky6_solve` | `op` | 157 | 686 | 529 |

## nalgebra_probes_steps::isometry3

| probe | variant | baseline | raw | net steps |
|---|---|---:|---:|---:|
| `isometry3_inv_mul` | `op` | 136 | 401 | 265 |
| `isometry3_inverse` | `op` | 122 | 299 | 177 |
| `isometry3_mul` | `op` | 136 | 389 | 253 |
| `isometry3_transform_point` | `op` | 108 | 281 | 173 |

## nalgebra_probes_steps::lu

| probe | variant | baseline | raw | net steps |
|---|---|---:|---:|---:|
| `lu3_factor` | `op` | 146 | 432 | 286 |
| `lu3_solve` | `op` | 116 | 289 | 173 |
| `lu6_factor` | `op` | 471 | 2328 | 1857 |
| `lu6_solve` | `op` | 197 | 680 | 483 |

## nalgebra_probes_steps::matrix3

| probe | variant | baseline | raw | net steps |
|---|---|---:|---:|---:|
| `matrix3_determinant` | `op` | 96 | 161 | 65 |
| `matrix3_mul` | `op` | 154 | 315 | 161 |
| `matrix3_mul_vec` | `op` | 112 | 165 | 53 |
| `matrix3_transpose` | `op` | 136 | 145 | 9 |
| `matrix3_try_inverse` | `op` | 136 | 548 | 412 |

## nalgebra_probes_steps::matrix6

| probe | variant | baseline | raw | net steps |
|---|---|---:|---:|---:|
| `matrix6_mul_vec` | `op` | 187 | 330 | 143 |

## nalgebra_probes_steps::scalar

| probe | variant | baseline | raw | net steps |
|---|---|---:|---:|---:|
| `scalar_mul_add` | `op` | 84 | 99 | 15 |
| `scalar_wide_dot3` | `op` | 90 | 107 | 17 |

## nalgebra_probes_steps::svd3

| probe | variant | baseline | raw | net steps |
|---|---|---:|---:|---:|
| `svd3` | `op` | 233 | 4946 | 4713 |

## nalgebra_probes_steps::symmetric_eigen3

| probe | variant | baseline | raw | net steps |
|---|---|---:|---:|---:|
| `symmetric_eigen3` | `op` | 151 | 3683 | 3532 |

## nalgebra_probes_steps::unit_quaternion

| probe | variant | baseline | raw | net steps |
|---|---|---:|---:|---:|
| `unit_quaternion_from_axis_angle` | `op` | 101 | 397 | 296 |
| `unit_quaternion_inverse` | `op` | 101 | 111 | 10 |
| `unit_quaternion_mul` | `op` | 109 | 188 | 79 |
| `unit_quaternion_slerp` | `op` | 111 | 1018 | 907 |
| `unit_quaternion_transform_vector` | `op` | 102 | 269 | 167 |

## nalgebra_probes_steps::vector3

| probe | variant | baseline | raw | net steps |
|---|---|---:|---:|---:|
| `vector3_cross` | `op` | 100 | 147 | 47 |
| `vector3_dot` | `op` | 90 | 107 | 17 |
| `vector3_norm` | `op` | 84 | 102 | 18 |
| `vector3_normalize` | `op` | 94 | 194 | 100 |

