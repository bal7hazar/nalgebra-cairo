//! 2D / 3D vectors in Q32.32.
//!
//!  - cubit `f64::Vec2` / `Vec3`: structs of sign-magnitude `Fixed`;
//!  - orion `Tensor<FP32x32>` of shape [3]: `{ shape: Span<usize>, data: Span<T> }`, dot = 1D
//!  matmul;
//!  - origami `Vector<T>` (`Span<T>`) and `Vec2<T>` (generic struct), instantiated with cubit's
//!    `Fixed` and with the reference `Fix64` to separate container cost from number cost;
//!  - alexandria `linalg::dot` over `Span<T>`, same two instantiations;
//!  - reference: `struct Vec3 { x, y, z: i64 }`, per-product rescale and fused (single rescale).
//!
//! All the containers are built once in `inputs()`, i.e. OUTSIDE the measured operation (the
//! `build_vec3` group prices the construction separately). Same fixed overhead in every test.
//!
//! a = (1.5, -2, 0.5), b = (0.25, 3, -4): a.b = -7.625, a x b = (6.5, 6.125, 5), |a| = sqrt(6.5).

use alexandria_linalg::dot::dot as alexandria_dot;
use cubit::f64::types::fixed::{Fixed, FixedTrait};
use cubit::f64::types::vec2::{Vec2 as CubitVec2, Vec2Trait as CubitVec2Trait};
use cubit::f64::types::vec3::{Vec3 as CubitVec3, Vec3Trait as CubitVec3Trait};
use harness::black_box;
use origami_algebra::vec2::{Vec2 as OrigamiVec2, Vec2Trait as OrigamiVec2Trait};
use origami_algebra::vector::{Vector, VectorTrait};
use orion::numbers::{FP32x32, FP32x32Impl};
use orion::operators::tensor::{FP32x32Tensor, Tensor, TensorTrait};
use crate::reference::q32::{Fix64, Vec2, Vec3, fix};
use crate::reference::{q32, q32_bounded};

const AX: u64 = 6442450944; // 1.5
const AY: u64 = 8589934592; // 2 (negative)
const AZ: u64 = 2147483648; // 0.5
const BX: u64 = 1073741824; // 0.25
const BY: u64 = 12884901888; // 3
const BZ: u64 = 17179869184; // 4 (negative)

#[derive(Copy, Drop)]
struct Inputs {
    ca: CubitVec3,
    cb: CubitVec3,
    ca2: CubitVec2,
    cb2: CubitVec2,
    ta: Tensor<FP32x32>,
    tb: Tensor<FP32x32>,
    oa: Vector<Fixed>,
    ob: Vector<Fixed>,
    oa2: OrigamiVec2<Fixed>,
    ob2: OrigamiVec2<Fixed>,
    sa: Span<Fix64>,
    sb: Span<Fix64>,
    ra: Vec3,
    rb: Vec3,
}

#[inline(never)]
fn inputs() -> Inputs {
    let ax = black_box(FixedTrait::new(AX, false));
    let ay = black_box(FixedTrait::new(AY, true));
    let az = black_box(FixedTrait::new(AZ, false));
    let bx = black_box(FixedTrait::new(BX, false));
    let by = black_box(FixedTrait::new(BY, false));
    let bz = black_box(FixedTrait::new(BZ, true));
    let ra = black_box(Vec3 { x: 6442450944, y: -8589934592, z: 2147483648 });
    let rb = black_box(Vec3 { x: 1073741824, y: 12884901888, z: -17179869184 });
    let da = array![ax, ay, az].span();
    let db = array![bx, by, bz].span();
    Inputs {
        ca: CubitVec3Trait::new(ax, ay, az),
        cb: CubitVec3Trait::new(bx, by, bz),
        ca2: CubitVec2Trait::new(ax, ay),
        cb2: CubitVec2Trait::new(bx, by),
        ta: TensorTrait::new(array![3].span(), da),
        tb: TensorTrait::new(array![3].span(), db),
        oa: VectorTrait::new(da),
        ob: VectorTrait::new(db),
        oa2: OrigamiVec2Trait::new(ax, ay),
        ob2: OrigamiVec2Trait::new(bx, by),
        sa: array![fix(ra.x), fix(ra.y), fix(ra.z)].span(),
        sb: array![fix(rb.x), fix(rb.y), fix(rb.z)].span(),
        ra,
        rb,
    }
}

#[inline(never)]
fn check(c: Fixed, c_mag: u64, c_sign: bool, r: i64, r_expected: i64) {
    assert!(c == FixedTrait::new(c_mag, c_sign));
    assert!(r == r_expected);
}

#[inline(never)]
fn check3(c: CubitVec3, cx: u64, cy: u64, cy_sign: bool, cz: u64, r: Vec3, r_expected: Vec3) {
    assert!(c.x == FixedTrait::new(cx, false));
    assert!(c.y == FixedTrait::new(cy, cy_sign));
    assert!(c.z == FixedTrait::new(cz, false));
    assert!(r == r_expected);
}

const DOT: u64 = 32749125632; // 7.625 (negative)
const NORM: u64 = 10950061026; // sqrt(6.5)

// ---- dot (3D) ---------------------------------------------------------------------------------

#[test]
#[inline(never)]
fn bench_dot3_q32__baseline() {
    let i = inputs();
    check(i.ca.x, AX, false, i.ra.x, 6442450944);
}

#[test]
#[inline(never)]
fn bench_dot3_q32__cubit_vec3() {
    let i = inputs();
    check(i.ca.dot(i.cb), DOT, true, i.ra.x, 6442450944);
}

#[test]
#[inline(never)]
fn bench_dot3_q32__orion_tensor_fp32x32() {
    let i = inputs();
    let r = i.ta.matmul(@i.tb);
    check(*r.data.at(0), DOT, true, i.ra.x, 6442450944);
}

#[test]
#[inline(never)]
fn bench_dot3_q32__origami_vector_of_cubit() {
    let i = inputs();
    check(i.oa.dot(i.ob), DOT, true, i.ra.x, 6442450944);
}

#[test]
#[inline(never)]
fn bench_dot3_q32__origami_vector_of_reference() {
    let i = inputs();
    let (a, b): (Vector<Fix64>, Vector<Fix64>) = (VectorTrait::new(i.sa), VectorTrait::new(i.sb));
    check(i.ca.x, AX, false, a.dot(b).v, -32749125632);
}

#[test]
#[inline(never)]
fn bench_dot3_q32__alexandria_dot_of_cubit() {
    let i = inputs();
    check(alexandria_dot(i.oa.data, i.ob.data), DOT, true, i.ra.x, 6442450944);
}

#[test]
#[inline(never)]
fn bench_dot3_q32__alexandria_dot_of_reference() {
    let i = inputs();
    check(i.ca.x, AX, false, alexandria_dot(i.sa, i.sb).v, -32749125632);
}

#[test]
#[inline(never)]
fn bench_dot3_q32__reference() {
    let i = inputs();
    check(i.ca.x, AX, false, q32::dot3(i.ra, i.rb), -32749125632);
}

#[test]
#[inline(never)]
fn bench_dot3_q32__reference_fused() {
    let i = inputs();
    check(i.ca.x, AX, false, q32::dot3_fused(i.ra, i.rb), -32749125632);
}

/// The fused BoundedInt biased-floor kernel from `benchmarks/primitives`.
#[test]
#[inline(never)]
fn bench_dot3_q32__reference_bounded() {
    let i = inputs();
    check(i.ca.x, AX, false, q32_bounded::dot3(i.ra, i.rb), -32749125632);
}

// ---- dot (2D) ---------------------------------------------------------------------------------

#[test]
#[inline(never)]
fn bench_dot2_q32__baseline() {
    let i = inputs();
    check(i.ca.x, AX, false, i.ra.x, 6442450944);
}

#[test]
#[inline(never)]
fn bench_dot2_q32__cubit_vec2() {
    let i = inputs();
    check(i.ca2.dot(i.cb2), 24159191040, true, i.ra.x, 6442450944);
}

#[test]
#[inline(never)]
fn bench_dot2_q32__origami_vec2_of_cubit() {
    let i = inputs();
    check(i.oa2.dot(i.ob2), 24159191040, true, i.ra.x, 6442450944);
}

#[test]
#[inline(never)]
fn bench_dot2_q32__reference_fused() {
    let i = inputs();
    let (a, b) = (Vec2 { x: i.ra.x, y: i.ra.y }, Vec2 { x: i.rb.x, y: i.rb.y });
    check(i.ca.x, AX, false, q32::dot2_fused(a, b), -24159191040);
}

// ---- cross ------------------------------------------------------------------------------------

const CROSS: Vec3 = Vec3 { x: 27917287424, y: 26306674688, z: 21474836480 };
const RA: Vec3 = Vec3 { x: 6442450944, y: -8589934592, z: 2147483648 };

#[test]
#[inline(never)]
fn bench_cross3_q32__baseline() {
    let i = inputs();
    check3(i.ca, AX, AY, true, AZ, i.ra, RA);
}

#[test]
#[inline(never)]
fn bench_cross3_q32__cubit_vec3() {
    let i = inputs();
    check3(i.ca.cross(i.cb), 27917287424, 26306674688, false, 21474836480, i.ra, RA);
}

#[test]
#[inline(never)]
fn bench_cross3_q32__reference() {
    let i = inputs();
    check3(i.ca, AX, AY, true, AZ, q32::cross3(i.ra, i.rb), CROSS);
}

#[test]
#[inline(never)]
fn bench_cross3_q32__reference_fused() {
    let i = inputs();
    check3(i.ca, AX, AY, true, AZ, q32::cross3_fused(i.ra, i.rb), CROSS);
}

// ---- add --------------------------------------------------------------------------------------

#[test]
#[inline(never)]
fn bench_add3_q32__baseline() {
    let i = inputs();
    check(i.ca.x, AX, false, i.ra.x, 6442450944);
}

#[test]
#[inline(never)]
fn bench_add3_q32__cubit_vec3() {
    let i = inputs();
    let r = i.ca + i.cb;
    check(r.x, 7516192768, false, i.ra.x, 6442450944);
}

#[test]
#[inline(never)]
fn bench_add3_q32__origami_vector_of_cubit() {
    let i = inputs();
    let mut r = i.oa + i.ob;
    check(r.get(0), 7516192768, false, i.ra.x, 6442450944);
}

#[test]
#[inline(never)]
fn bench_add3_q32__reference() {
    let i = inputs();
    check(i.ca.x, AX, false, q32::add3(i.ra, i.rb).x, 7516192768);
}

// ---- norm -------------------------------------------------------------------------------------

#[test]
#[inline(never)]
fn bench_norm3_q32__baseline() {
    let i = inputs();
    check(i.ca.x, AX, false, i.ra.x, 6442450944);
}

#[test]
#[inline(never)]
fn bench_norm3_q32__cubit_vec3() {
    let i = inputs();
    check(i.ca.norm(), NORM, false, i.ra.x, 6442450944);
}

#[test]
#[inline(never)]
fn bench_norm3_q32__reference() {
    let i = inputs();
    check(i.ca.x, AX, false, q32::norm3(i.ra), 10950061026);
}

#[test]
#[inline(never)]
fn bench_norm3_q32__reference_fused() {
    let i = inputs();
    check(i.ca.x, AX, false, q32::norm3_fused(i.ra), 10950061026);
}

// ---- construction -----------------------------------------------------------------------------

#[inline(never)]
fn scalars() -> (Fixed, Fixed, Fixed) {
    (
        black_box(FixedTrait::new(AX, false)),
        black_box(FixedTrait::new(AY, true)),
        black_box(FixedTrait::new(AZ, false)),
    )
}

#[test]
#[inline(never)]
fn bench_build_vec3__baseline() {
    let (x, _y, _z) = scalars();
    assert!(x.mag == AX);
}

#[test]
#[inline(never)]
fn bench_build_vec3__cubit_vec3() {
    let (x, y, z) = scalars();
    let v = CubitVec3Trait::new(x, y, z);
    assert!(v.x.mag == AX);
}

#[test]
#[inline(never)]
fn bench_build_vec3__orion_tensor() {
    let (x, y, z) = scalars();
    let v: Tensor<FP32x32> = TensorTrait::new(array![3].span(), array![x, y, z].span());
    assert!(v.data.at(0).mag == @AX);
}

#[test]
#[inline(never)]
fn bench_build_vec3__orion_tensor_build_and_read() {
    let (x, y, z) = scalars();
    let v: Tensor<FP32x32> = TensorTrait::new(array![3].span(), array![x, y, z].span());
    assert!(v.at(array![0].span()).mag == AX);
}

#[test]
#[inline(never)]
fn bench_build_vec3__origami_vector() {
    let (x, y, z) = scalars();
    let mut v: Vector<Fixed> = VectorTrait::new(array![x, y, z].span());
    assert!(v.get(0).mag == AX);
}
