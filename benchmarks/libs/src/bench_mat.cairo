//! 3x3 matrices in Q32.32 (plus one Q16.16 group for orion's native FP16x16 tensors).
//!
//!  - orion `Tensor<FP32x32>` of shape [3, 3] (`matmul`, broadcasting `add`) and orion
//!    `MutMatrix<FP32x32>` (a `Felt252Dict<Nullable<T>>` behind a matrix API);
//!  - origami `Matrix<T>` (`Span<T>` + `rows`/`cols: u8`, generic recursive `det`/`inv`),
//!    instantiated with cubit's `Fixed` and with the reference `Fix64`;
//!  - reference: `struct Mat3 { r0, r1, r2: Vec3 }`, unrolled.
//!
//! Containers are built once in `inputs()`, outside the measured operation. Every test of a group
//! pays the same overhead: `inputs()` + a `check*` that verifies one value per representation.
//!
//! A = [[2, 0, .5], [1, 3, 0], [0, -1, 4]], B = [[1, 2, 0], [0, .5, -1], [3, 0, 1]], det(A) = 23.5

use cubit::f64::types::fixed::{Fixed, FixedTrait};
use harness::black_box;
use origami_algebra::matrix::{Matrix, MatrixTrait};
use orion::numbers::{FP32x32, FP32x32Impl};
use orion::operators::tensor::{FP32x32Tensor, Tensor, TensorTrait};
use crate::reference::q32;
use crate::reference::q32::{Fix64, Mat3, Vec3, fix};

const A: [i64; 9] = [
    8589934592, 0, 2147483648, 4294967296, 12884901888, 0, 0, -4294967296, 17179869184,
];
const B: [i64; 9] = [
    4294967296, 8589934592, 0, 0, 2147483648, -4294967296, 12884901888, 0, 4294967296,
];
const AB: [i64; 9] = [
    15032385536, 17179869184, 2147483648, 4294967296, 15032385536, -12884901888, 51539607552,
    -2147483648, 21474836480,
];
const A_PLUS_B: [i64; 9] = [
    12884901888, 8589934592, 2147483648, 4294967296, 15032385536, -4294967296, 12884901888,
    -4294967296, 21474836480,
];
const A_T: [i64; 9] = [
    8589934592, 4294967296, 0, 0, 12884901888, -4294967296, 2147483648, 0, 17179869184,
];
const A_INV: [i64; 9] = [
    2193174789, -91382282, -274146848, -731058263, 1462116526, 91382282, -182764565, 365529131,
    1096587394,
];
const DET_A: i64 = 100931731456; // 23.5

fn to_cubit(v: i64) -> Fixed {
    if v < 0 {
        FixedTrait::new((-v).try_into().unwrap(), true)
    } else {
        FixedTrait::new(v.try_into().unwrap(), false)
    }
}

fn raw(f: Fixed) -> i64 {
    let mag: i64 = f.mag.try_into().unwrap();
    if f.sign {
        -mag
    } else {
        mag
    }
}

fn mat3(v: Span<i64>) -> Mat3 {
    Mat3 {
        r0: Vec3 { x: *v[0], y: *v[1], z: *v[2] },
        r1: Vec3 { x: *v[3], y: *v[4], z: *v[5] },
        r2: Vec3 { x: *v[6], y: *v[7], z: *v[8] },
    }
}

fn flat(m: Mat3) -> Span<i64> {
    array![m.r0.x, m.r0.y, m.r0.z, m.r1.x, m.r1.y, m.r1.z, m.r2.x, m.r2.y, m.r2.z].span()
}

#[derive(Copy, Drop)]
struct Inputs {
    ca: Span<Fixed>,
    cb: Span<Fixed>,
    fa: Span<Fix64>,
    fb: Span<Fix64>,
    ra: Mat3,
    rb: Mat3,
}

#[inline(never)]
fn inputs() -> Inputs {
    let a = black_box(A.span());
    let b = black_box(B.span());
    let mut ca = array![];
    let mut cb = array![];
    let mut fa = array![];
    let mut fb = array![];
    for k in 0..9_usize {
        ca.append(to_cubit(*a[k]));
        cb.append(to_cubit(*b[k]));
        fa.append(fix(*a[k]));
        fb.append(fix(*b[k]));
    }
    Inputs { ca: ca.span(), cb: cb.span(), fa: fa.span(), fb: fb.span(), ra: mat3(a), rb: mat3(b) }
}

/// Verifies the nine entries of the three representations.
#[inline(never)]
fn check9(
    c: Span<Fixed>, c_exp: Span<i64>, f: Span<Fix64>, f_exp: Span<i64>, r: Mat3, r_exp: Span<i64>,
) {
    let r = flat(r);
    for k in 0..9_usize {
        assert!(raw(*c[k]) == *c_exp[k]);
        assert!(*f[k].v == *f_exp[k]);
        assert!(*r[k] == *r_exp[k]);
    }
}

#[inline(never)]
fn check1(c: Fixed, c_exp: i64, f: Fix64, f_exp: i64, r: i64, r_exp: i64) {
    assert!(raw(c) == c_exp);
    assert!(f.v == f_exp);
    assert!(r == r_exp);
}

// ---- matmul -----------------------------------------------------------------------------------

#[test]
#[inline(never)]
fn bench_matmul3_q32__baseline() {
    let i = inputs();
    check9(i.ca, A.span(), i.fa, A.span(), i.ra, A.span());
}

#[test]
#[inline(never)]
fn bench_matmul3_q32__orion_tensor_fp32x32() {
    let i = inputs();
    let a: Tensor<FP32x32> = TensorTrait::new(array![3, 3].span(), i.ca);
    let b: Tensor<FP32x32> = TensorTrait::new(array![3, 3].span(), i.cb);
    check9(a.matmul(@b).data, AB.span(), i.fa, A.span(), i.ra, A.span());
}

#[test]
#[inline(never)]
fn bench_matmul3_q32__origami_matrix_of_cubit() {
    let i = inputs();
    let a: Matrix<Fixed> = MatrixTrait::new(i.ca, 3, 3);
    let b: Matrix<Fixed> = MatrixTrait::new(i.cb, 3, 3);
    check9((a * b).data, AB.span(), i.fa, A.span(), i.ra, A.span());
}

#[test]
#[inline(never)]
fn bench_matmul3_q32__origami_matrix_of_reference() {
    let i = inputs();
    let a: Matrix<Fix64> = MatrixTrait::new(i.fa, 3, 3);
    let b: Matrix<Fix64> = MatrixTrait::new(i.fb, 3, 3);
    check9(i.ca, A.span(), (a * b).data, AB.span(), i.ra, A.span());
}

#[test]
#[inline(never)]
fn bench_matmul3_q32__reference() {
    let i = inputs();
    check9(i.ca, A.span(), i.fa, A.span(), q32::matmul3(i.ra, i.rb), AB.span());
}

#[test]
#[inline(never)]
fn bench_matmul3_q32__reference_fused() {
    let i = inputs();
    check9(i.ca, A.span(), i.fa, A.span(), q32::matmul3_fused(i.ra, i.rb), AB.span());
}

// ---- element-wise add -------------------------------------------------------------------------

#[test]
#[inline(never)]
fn bench_matadd3_q32__baseline() {
    let i = inputs();
    check9(i.ca, A.span(), i.fa, A.span(), i.ra, A.span());
}

#[test]
#[inline(never)]
fn bench_matadd3_q32__orion_tensor_fp32x32() {
    let i = inputs();
    let a: Tensor<FP32x32> = TensorTrait::new(array![3, 3].span(), i.ca);
    let b: Tensor<FP32x32> = TensorTrait::new(array![3, 3].span(), i.cb);
    check9(TensorTrait::add(a, b).data, A_PLUS_B.span(), i.fa, A.span(), i.ra, A.span());
}

#[test]
#[inline(never)]
fn bench_matadd3_q32__origami_matrix_of_cubit() {
    let i = inputs();
    let a: Matrix<Fixed> = MatrixTrait::new(i.ca, 3, 3);
    let b: Matrix<Fixed> = MatrixTrait::new(i.cb, 3, 3);
    check9((a + b).data, A_PLUS_B.span(), i.fa, A.span(), i.ra, A.span());
}

#[test]
#[inline(never)]
fn bench_matadd3_q32__origami_matrix_of_reference() {
    let i = inputs();
    let a: Matrix<Fix64> = MatrixTrait::new(i.fa, 3, 3);
    let b: Matrix<Fix64> = MatrixTrait::new(i.fb, 3, 3);
    check9(i.ca, A.span(), (a + b).data, A_PLUS_B.span(), i.ra, A.span());
}

#[test]
#[inline(never)]
fn bench_matadd3_q32__reference() {
    let i = inputs();
    check9(i.ca, A.span(), i.fa, A.span(), q32::matadd3(i.ra, i.rb), A_PLUS_B.span());
}

// ---- transpose --------------------------------------------------------------------------------

#[test]
#[inline(never)]
fn bench_transpose3_q32__baseline() {
    let i = inputs();
    check9(i.ca, A.span(), i.fa, A.span(), i.ra, A.span());
}

#[test]
#[inline(never)]
fn bench_transpose3_q32__origami_matrix_of_cubit() {
    let i = inputs();
    let mut a: Matrix<Fixed> = MatrixTrait::new(i.ca, 3, 3);
    check9(a.transpose().data, A_T.span(), i.fa, A.span(), i.ra, A.span());
}

#[test]
#[inline(never)]
fn bench_transpose3_q32__reference() {
    let i = inputs();
    check9(i.ca, A.span(), i.fa, A.span(), q32::transpose3(i.ra), A_T.span());
}

// ---- determinant ------------------------------------------------------------------------------

#[test]
#[inline(never)]
fn bench_det3_q32__baseline() {
    let i = inputs();
    check1(*i.ca[0], 8589934592, *i.fa[0], 8589934592, i.ra.r0.x, 8589934592);
}

#[test]
#[inline(never)]
fn bench_det3_q32__origami_matrix_of_cubit() {
    let i = inputs();
    let mut a: Matrix<Fixed> = MatrixTrait::new(i.ca, 3, 3);
    check1(a.det(), DET_A, *i.fa[0], 8589934592, i.ra.r0.x, 8589934592);
}

#[test]
#[inline(never)]
fn bench_det3_q32__origami_matrix_of_reference() {
    let i = inputs();
    let mut a: Matrix<Fix64> = MatrixTrait::new(i.fa, 3, 3);
    check1(*i.ca[0], 8589934592, a.det(), DET_A, i.ra.r0.x, 8589934592);
}

#[test]
#[inline(never)]
fn bench_det3_q32__reference() {
    let i = inputs();
    check1(*i.ca[0], 8589934592, *i.fa[0], 8589934592, q32::det3(i.ra), DET_A);
}

// ---- inverse ----------------------------------------------------------------------------------

#[test]
#[inline(never)]
fn bench_inv3_q32__baseline() {
    let i = inputs();
    check9(i.ca, A.span(), i.fa, A.span(), i.ra, A.span());
}

#[test]
#[inline(never)]
fn bench_inv3_q32__origami_matrix_of_cubit() {
    let i = inputs();
    let mut a: Matrix<Fixed> = MatrixTrait::new(i.ca, 3, 3);
    check9(a.inv().data, A_INV.span(), i.fa, A.span(), i.ra, A.span());
}

#[test]
#[inline(never)]
fn bench_inv3_q32__origami_matrix_of_reference() {
    let i = inputs();
    let mut a: Matrix<Fix64> = MatrixTrait::new(i.fa, 3, 3);
    check9(i.ca, A.span(), a.inv().data, A_INV.span(), i.ra, A.span());
}

#[test]
#[inline(never)]
fn bench_inv3_q32__reference() {
    let i = inputs();
    check9(i.ca, A.span(), i.fa, A.span(), q32::inv3(i.ra), A_INV.span());
}

// ---- matrix x vector through orion's dictionary-backed MutMatrix
// ---------------------------------

mod matvec {
    use cubit::f64::types::fixed::Fixed;
    use harness::black_box;
    // NOTE: orion re-implements the core operator traits for cubit's type; importing its
    // `FP32x32PartialOrd` next to cubit's own impl is an "multiple implementations" error, so only
    // the impl cubit does not provide (`AddEq`) is imported.
    use orion::numbers::fixed_point::implementations::fp32x32::core::FP32x32AddEq;
    use orion::numbers::{FP32x32, FP32x32Impl, FP32x32Number};
    use orion::operators::matrix::{MutMatrix, MutMatrixImpl};
    use orion::operators::vec::{NullableVec, NullableVecImpl, VecTrait};
    use crate::reference::q32;
    use crate::reference::q32::{Mat3, Vec3};
    use super::{A, mat3, raw, to_cubit};

    const V: [i64; 3] = [6442450944, -8589934592, 2147483648]; // (1.5, -2, 0.5)
    const AV: [i64; 3] = [13958643712, -19327352832, 17179869184];

    /// The dictionaries are filled here, i.e. in the baseline too; what the variants pay is the
    /// product itself (9 + 3 dictionary reads, 3 dictionary writes) and the bigger squash.
    #[inline(never)]
    fn inputs() -> (MutMatrix<FP32x32>, NullableVec<FP32x32>, Mat3, Vec3) {
        let a = black_box(A.span());
        let v = black_box(V.span());
        let mut m: MutMatrix<FP32x32> = MutMatrixImpl::new(3, 3);
        for k in 0..9_usize {
            m.set(k / 3, k % 3, to_cubit(*a[k]));
        }
        let mut x: NullableVec<FP32x32> = VecTrait::new();
        for k in 0..3_usize {
            x.push(to_cubit(*v[k]));
        }
        (m, x, mat3(a), Vec3 { x: *v[0], y: *v[1], z: *v[2] })
    }

    #[inline(never)]
    fn check(ref c: NullableVec<FP32x32>, c_exp: Span<i64>, r: Vec3, r_exp: Span<i64>) {
        let c0: Fixed = c.at(0);
        let c1: Fixed = c.at(1);
        let c2: Fixed = c.at(2);
        assert!(raw(c0) == *c_exp[0] && raw(c1) == *c_exp[1] && raw(c2) == *c_exp[2]);
        assert!(r.x == *r_exp[0] && r.y == *r_exp[1] && r.z == *r_exp[2]);
    }

    #[test]
    #[inline(never)]
    fn bench_matvec3_q32__baseline() {
        let (_m, mut x, _a, v) = inputs();
        check(ref x, V.span(), v, V.span());
    }

    #[test]
    #[inline(never)]
    fn bench_matvec3_q32__orion_mutmatrix_fp32x32() {
        let (mut m, mut x, _a, v) = inputs();
        let mut y = m.matrix_vector_product(ref x);
        check(ref y, AV.span(), v, V.span());
    }

    #[test]
    #[inline(never)]
    fn bench_matvec3_q32__reference_fused() {
        let (_m, mut x, a, v) = inputs();
        check(ref x, V.span(), q32::matvec3_fused(a, v), AV.span());
    }
}

// ---- orion's native FP16x16 tensors
// --------------------------------------------------------------

mod q16 {
    use harness::black_box;
    use orion::numbers::fixed_point::core::FixedTrait;
    use orion::numbers::{FP16x16, FP16x16Impl};
    use orion::operators::tensor::{FP16x16Tensor, Tensor, TensorTrait};
    use crate::reference::q16;

    const A: [i32; 9] = [131072, 0, 32768, 65536, 196608, 0, 0, -65536, 262144];
    const B: [i32; 9] = [65536, 131072, 0, 0, 32768, -65536, 196608, 0, 65536];
    const AB: [i32; 9] = [229376, 262144, 32768, 65536, 229376, -196608, 786432, -32768, 327680];

    fn to_orion(v: i32) -> FP16x16 {
        if v < 0 {
            FixedTrait::new((-v).try_into().unwrap(), true)
        } else {
            FixedTrait::new(v.try_into().unwrap(), false)
        }
    }

    fn raw(f: FP16x16) -> i32 {
        let mag: i32 = f.mag.try_into().unwrap();
        if f.sign {
            -mag
        } else {
            mag
        }
    }

    #[inline(never)]
    fn inputs() -> (Span<FP16x16>, Span<FP16x16>, [i32; 9], [i32; 9]) {
        let a = black_box(A);
        let b = black_box(B);
        let mut oa = array![];
        let mut ob = array![];
        for k in 0..9_usize {
            oa.append(to_orion(*a.span()[k]));
            ob.append(to_orion(*b.span()[k]));
        }
        (oa.span(), ob.span(), a, b)
    }

    #[inline(never)]
    fn check9(o: Span<FP16x16>, o_exp: Span<i32>, r: [i32; 9], r_exp: Span<i32>) {
        let r = r.span();
        for k in 0..9_usize {
            assert!(raw(*o[k]) == *o_exp[k]);
            assert!(*r[k] == *r_exp[k]);
        }
    }

    #[test]
    #[inline(never)]
    fn bench_matmul3_q16__baseline() {
        let (oa, _ob, a, _b) = inputs();
        check9(oa, A.span(), a, A.span());
    }

    #[test]
    #[inline(never)]
    fn bench_matmul3_q16__orion_tensor_fp16x16() {
        let (oa, ob, a, _b) = inputs();
        let ta: Tensor<FP16x16> = TensorTrait::new(array![3, 3].span(), oa);
        let tb: Tensor<FP16x16> = TensorTrait::new(array![3, 3].span(), ob);
        check9(ta.matmul(@tb).data, AB.span(), a, A.span());
    }

    #[test]
    #[inline(never)]
    fn bench_matmul3_q16__reference_fused() {
        let (oa, _ob, a, b) = inputs();
        check9(oa, A.span(), q16::matmul3(a, b), AB.span());
    }
}
