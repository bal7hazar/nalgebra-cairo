//! Tests of the LU / QR / Cholesky / UDU API completion (WP 8.5-P14a): the unpacking accessors,
//! the generic `solve_mut` (bit-identical to `solve` on a vector, column by column on a matrix),
//! `try_inverse_to` / `try_invert_to` / `try_inverse_mut`, `Cholesky::{new_unchecked,
//! new_with_substitute, pack_dirty, ln_determinant}` and the `SquareMatrix` entry points
//! `cholesky()` / `udu()`, on the sizes 2, 3, 4, 6 (QR: 2, 3, 4). The operands go through
//! `black_box` (no per-test specialisation of the kernels).

use fixed::Fixed;
use nalgebra::linalg::{
    Cholesky2Trait, Cholesky3Trait, Cholesky4Trait, Cholesky6Trait, Lu2Trait, Lu3Trait, Lu4Trait,
    Lu6Trait, Matrix2CholeskyTrait, Matrix2InverseTrait, Matrix2UduTrait, Matrix3CholeskyTrait,
    Matrix3InverseTrait, Matrix3UduTrait, Matrix4CholeskyTrait, Matrix4InverseTrait,
    Matrix4UduTrait, Matrix6CholeskyTrait, Matrix6InverseTrait, Matrix6LuTrait, Matrix6UduTrait,
    Qr2Trait, Qr3Trait, Qr4Trait, Udu2Trait, Udu3Trait, Udu4Trait, Udu6Trait, try_invert_to,
};
use nalgebra::{
    Matrix2, Matrix2Trait, Matrix3, Matrix3Trait, Matrix3x2, Matrix3x2Trait, Matrix4, Matrix4Trait,
    Matrix4x2, Matrix4x2Trait, Matrix6, Matrix6Trait, Matrix6x2, Matrix6x2Trait, MatrixTrMul,
    Vector2, Vector2Trait, Vector3, Vector3Trait, Vector4, Vector4Trait, Vector6, Vector6Trait,
};
use nalgebra_testing::black_box;
use nalgebra_tests_utils::{
    Cholesky2PartialEq, Cholesky3PartialEq, Cholesky4PartialEq, Cholesky6PartialEq, Perm2PartialEq,
    Perm3PartialEq, Perm4PartialEq, Perm6PartialEq, fx, int, ulp_diff,
};
use simba::scalar::Transcendental;

fn a2() -> Matrix2<Fixed> {
    black_box(Matrix2Trait::new(fx(11747301768), fx(1740196400), fx(1306031009), fx(11742695554)))
}

fn s2() -> Matrix2<Fixed> {
    black_box(Matrix2Trait::new(fx(11417955886), fx(-1692320418), fx(-1692320418), fx(11528693499)))
}

fn z2() -> Matrix2<Fixed> {
    black_box(Matrix2Trait::new(fx(-887566630), fx(0), fx(-3277922735), fx(0)))
}

fn v2() -> Vector2<Fixed> {
    black_box(Vector2Trait::new(fx(6897083304), fx(-2438979992)))
}

fn bm2() -> Matrix2<Fixed> {
    black_box(Matrix2Trait::new(fx(-4116256593), fx(5227512656), fx(2261968962), fx(-6018047083)))
}

/// `Lu2`: `unpack`, `l_unpack` and `lu_internal` are the stored factors, moved.
#[test]
fn test_lu2_unpack() {
    let f = Lu2Trait::new(a2());
    let (p, l, u) = f.unpack();
    assert!(p == f.p && l == f.l() && u == f.u());
    assert!(f.l_unpack() == f.l());
    assert!(f.lu_internal() == f.lu);
}

/// `Lu2::solve_mut` on a vector is `solve`, bit for bit; on a 2x2 right-hand side, each
/// column is the `solve` of that column (the shared prepared divisor is bit-identical).
#[test]
fn test_lu2_solve_mut() {
    let f = Lu2Trait::new(a2());
    let mut x = v2();
    assert!(f.solve_mut(ref x));
    assert!(Option::Some(x) == f.solve(v2()));
    let b = bm2();
    let mut xm = b;
    assert!(f.solve_mut(ref xm));
    assert!(Option::Some(xm.column(0)) == f.solve(Vector2Trait::new(b.m11, b.m21)));
    assert!(Option::Some(xm.column(1)) == f.solve(Vector2Trait::new(b.m12, b.m22)));
}

/// `Lu2::try_inverse_to` writes `try_inverse`, bit for bit; `try_invert_to` is the same LU
/// inverse of the matrix, and `try_inverse_mut` the matrix's own `try_inverse`.
#[test]
fn test_lu2_try_inverse_to() {
    let f = Lu2Trait::new(a2());
    let mut out = s2();
    assert!(f.try_inverse_to(ref out));
    assert!(Option::Some(out) == f.try_inverse());
    let mut out2 = s2();
    assert!(try_invert_to(a2(), ref out2));
    assert!(out2 == out);
    let mut m = a2();
    assert!(m.try_inverse_mut());
    assert!(Option::Some(m) == a2().try_inverse());
}

/// A singular matrix (a zero column): `solve_mut`, `try_inverse_to`, `try_invert_to` and
/// `try_inverse_mut` return `false` and leave their output unchanged.
#[test]
fn test_lu2_singular_leaves_outputs_unchanged() {
    let f = Lu2Trait::new(z2());
    assert!(!f.is_invertible());
    let mut x = v2();
    assert!(!f.solve_mut(ref x));
    assert!(x == v2());
    let mut out = s2();
    assert!(!f.try_inverse_to(ref out));
    assert!(out == s2());
    assert!(!try_invert_to(z2(), ref out));
    assert!(out == s2());
    let mut m = z2();
    assert!(!m.try_inverse_mut());
    assert!(m == z2());
}

/// `Cholesky2`: `unpack`, `unpack_dirty` and `l_dirty` are `l()`; `pack_dirty` rebuilds the
/// factor from the lower triangle only (the upper one is ignored); `new_unchecked` and
/// `new_with_substitute` of a positive-definite matrix are `new`; `cholesky()` / `udu()` are the
/// constructors.
#[test]
fn test_cholesky2_api() {
    let c = Cholesky2Trait::new(s2()).unwrap();
    assert!(c.unpack() == c.l() && c.unpack_dirty() == c.l() && c.l_dirty() == c.l());
    assert!(Cholesky2Trait::pack_dirty(c.l()) == c);
    assert!(Cholesky2Trait::pack_dirty(c.l() + c.l().transpose()) != c);
    assert!(Cholesky2Trait::pack_dirty(s2().lower_triangle()) == Cholesky2Trait::pack_dirty(s2()));
    assert!(Cholesky2Trait::new_unchecked(s2()) == c);
    assert!(Cholesky2Trait::new_with_substitute(s2(), int(1)) == Option::Some(c));
    assert!(s2().cholesky() == Option::Some(c));
    let u = s2().udu().unwrap();
    let w = Udu2Trait::new(s2()).unwrap();
    assert!(u.u == w.u && u.d == w.d);
}

/// `Cholesky2::solve_mut` on a vector is `solve`, bit for bit; on a 2x2 right-hand side each
/// column is the `solve` of that column.
#[test]
fn test_cholesky2_solve_mut() {
    let c = Cholesky2Trait::new(s2()).unwrap();
    let mut x = v2();
    c.solve_mut(ref x);
    assert!(x == c.solve(v2()));
    let b = bm2();
    let mut xm = b;
    c.solve_mut(ref xm);
    assert!(xm.column(0) == c.solve(Vector2Trait::new(b.m11, b.m21)));
    assert!(xm.column(1) == c.solve(Vector2Trait::new(b.m12, b.m22)));
}

/// `Cholesky2::ln_determinant` is `ln(determinant)` within the rounding of 2 logarithms.
#[test]
fn test_cholesky2_ln_determinant() {
    let c = Cholesky2Trait::new(s2()).unwrap();
    let expected = Transcendental::ln(c.determinant());
    assert!(ulp_diff(c.ln_determinant(), expected) <= 64);
}

/// `Cholesky2::new_unchecked` of a matrix that is not positive definite panics (upstream
/// computes NaN / infinities).
#[test]
#[should_panic(expected: 'nalgebra: not positive definite')]
fn test_cholesky2_new_unchecked_not_positive_definite_panics() {
    let _c = Cholesky2Trait::new_unchecked(-s2());
}

/// `new_with_substitute` replaces every non-positive pivot by the substitute (here all of them,
/// the matrix being negative definite: the first pivot is the substitute), and fails when the
/// substitute is not positive either.
#[test]
fn test_cholesky2_new_with_substitute() {
    assert!(Cholesky2Trait::new(-s2()).is_none());
    let c = Cholesky2Trait::new_with_substitute(-s2(), int(4)).unwrap();
    assert!(c.l11 == int(2));
    assert!(Cholesky2Trait::new_with_substitute(-s2(), int(0)).is_none());
    assert!(Cholesky2Trait::new_with_substitute(-s2(), int(-1)).is_none());
}

/// `Qr2`: `unpack_r` is `r()`, `qr_internal` the stored `(q, r)`, `q_tr_mul` is `qᵀ * rhs`
/// (`tr_mul`, bit for bit) on a vector and on a 2x2 matrix.
#[test]
fn test_qr2_api() {
    let f = Qr2Trait::new(a2());
    assert!(f.unpack_r() == f.r());
    let (q, r) = f.qr_internal();
    assert!(q == f.q() && r == f.r());
    let mut y = v2();
    f.q_tr_mul(ref y);
    assert!(y == f.q().tr_mul(v2()));
    let mut ym = bm2();
    f.q_tr_mul(ref ym);
    assert!(ym == f.q().tr_mul(bm2()));
}

/// `Qr2::solve_mut` on a vector is `solve`, bit for bit; on a 2x2 right-hand side each
/// column is the `solve` of that column; singular: `false`, `b` unchanged.
#[test]
fn test_qr2_solve_mut() {
    let f = Qr2Trait::new(a2());
    let mut x = v2();
    assert!(f.solve_mut(ref x));
    assert!(Option::Some(x) == f.solve(v2()));
    let b = bm2();
    let mut xm = b;
    assert!(f.solve_mut(ref xm));
    assert!(Option::Some(xm.column(0)) == f.solve(Vector2Trait::new(b.m11, b.m21)));
    assert!(Option::Some(xm.column(1)) == f.solve(Vector2Trait::new(b.m12, b.m22)));
    let g = Qr2Trait::new(z2());
    let mut x = v2();
    assert!(!g.solve_mut(ref x));
    assert!(x == v2());
}

fn a3() -> Matrix3<Fixed> {
    black_box(
        Matrix3Trait::new(
            fx(15390093215),
            fx(1408741137),
            fx(-2878072639),
            fx(1303078404),
            fx(16734959603),
            fx(-1399590462),
            fx(-3580215982),
            fx(-2538998281),
            fx(-16632670516),
        ),
    )
}

fn s3() -> Matrix3<Fixed> {
    black_box(
        Matrix3Trait::new(
            fx(15835044060),
            fx(-3336919386),
            fx(1148726551),
            fx(-3336919386),
            fx(15414999124),
            fx(-2738488008),
            fx(1148726551),
            fx(-2738488008),
            fx(13291517175),
        ),
    )
}

fn z3() -> Matrix3<Fixed> {
    black_box(
        Matrix3Trait::new(
            fx(3219455263),
            fx(118381803),
            fx(0),
            fx(-407155096),
            fx(-2420747488),
            fx(0),
            fx(4124240),
            fx(-3415117240),
            fx(0),
        ),
    )
}

fn v3() -> Vector3<Fixed> {
    black_box(Vector3Trait::new(fx(-7051730908), fx(-3768082032), fx(-8108942827)))
}

fn bm3() -> Matrix3x2<Fixed> {
    black_box(
        Matrix3x2Trait::new(
            fx(4129031092),
            fx(-6633583528),
            fx(442631755),
            fx(-6992486857),
            fx(-336082435),
            fx(3184668949),
        ),
    )
}

/// `Lu3`: `unpack`, `l_unpack` and `lu_internal` are the stored factors, moved.
#[test]
fn test_lu3_unpack() {
    let f = Lu3Trait::new(a3());
    let (p, l, u) = f.unpack();
    assert!(p == f.p && l == f.l() && u == f.u());
    assert!(f.l_unpack() == f.l());
    assert!(f.lu_internal() == f.lu);
}

/// `Lu3::solve_mut` on a vector is `solve`, bit for bit; on a 3x2 right-hand side, each
/// column is the `solve` of that column (the shared prepared divisor is bit-identical).
#[test]
fn test_lu3_solve_mut() {
    let f = Lu3Trait::new(a3());
    let mut x = v3();
    assert!(f.solve_mut(ref x));
    assert!(Option::Some(x) == f.solve(v3()));
    let b = bm3();
    let mut xm = b;
    assert!(f.solve_mut(ref xm));
    assert!(Option::Some(xm.column(0)) == f.solve(Vector3Trait::new(b.m11, b.m21, b.m31)));
    assert!(Option::Some(xm.column(1)) == f.solve(Vector3Trait::new(b.m12, b.m22, b.m32)));
}

/// `Lu3::try_inverse_to` writes `try_inverse`, bit for bit; `try_invert_to` is the same LU
/// inverse of the matrix, and `try_inverse_mut` the matrix's own `try_inverse`.
#[test]
fn test_lu3_try_inverse_to() {
    let f = Lu3Trait::new(a3());
    let mut out = s3();
    assert!(f.try_inverse_to(ref out));
    assert!(Option::Some(out) == f.try_inverse());
    let mut out2 = s3();
    assert!(try_invert_to(a3(), ref out2));
    assert!(out2 == out);
    let mut m = a3();
    assert!(m.try_inverse_mut());
    assert!(Option::Some(m) == a3().try_inverse());
}

/// A singular matrix (a zero column): `solve_mut`, `try_inverse_to`, `try_invert_to` and
/// `try_inverse_mut` return `false` and leave their output unchanged.
#[test]
fn test_lu3_singular_leaves_outputs_unchanged() {
    let f = Lu3Trait::new(z3());
    assert!(!f.is_invertible());
    let mut x = v3();
    assert!(!f.solve_mut(ref x));
    assert!(x == v3());
    let mut out = s3();
    assert!(!f.try_inverse_to(ref out));
    assert!(out == s3());
    assert!(!try_invert_to(z3(), ref out));
    assert!(out == s3());
    let mut m = z3();
    assert!(!m.try_inverse_mut());
    assert!(m == z3());
}

/// `Cholesky3`: `unpack`, `unpack_dirty` and `l_dirty` are `l()`; `pack_dirty` rebuilds the
/// factor from the lower triangle only (the upper one is ignored); `new_unchecked` and
/// `new_with_substitute` of a positive-definite matrix are `new`; `cholesky()` / `udu()` are the
/// constructors.
#[test]
fn test_cholesky3_api() {
    let c = Cholesky3Trait::new(s3()).unwrap();
    assert!(c.unpack() == c.l() && c.unpack_dirty() == c.l() && c.l_dirty() == c.l());
    assert!(Cholesky3Trait::pack_dirty(c.l()) == c);
    assert!(Cholesky3Trait::pack_dirty(c.l() + c.l().transpose()) != c);
    assert!(Cholesky3Trait::pack_dirty(s3().lower_triangle()) == Cholesky3Trait::pack_dirty(s3()));
    assert!(Cholesky3Trait::new_unchecked(s3()) == c);
    assert!(Cholesky3Trait::new_with_substitute(s3(), int(1)) == Option::Some(c));
    assert!(s3().cholesky() == Option::Some(c));
    let u = s3().udu().unwrap();
    let w = Udu3Trait::new(s3()).unwrap();
    assert!(u.u == w.u && u.d == w.d);
}

/// `Cholesky3::solve_mut` on a vector is `solve`, bit for bit; on a 3x2 right-hand side each
/// column is the `solve` of that column.
#[test]
fn test_cholesky3_solve_mut() {
    let c = Cholesky3Trait::new(s3()).unwrap();
    let mut x = v3();
    c.solve_mut(ref x);
    assert!(x == c.solve(v3()));
    let b = bm3();
    let mut xm = b;
    c.solve_mut(ref xm);
    assert!(xm.column(0) == c.solve(Vector3Trait::new(b.m11, b.m21, b.m31)));
    assert!(xm.column(1) == c.solve(Vector3Trait::new(b.m12, b.m22, b.m32)));
}

/// `Cholesky3::ln_determinant` is `ln(determinant)` within the rounding of 3 logarithms.
#[test]
fn test_cholesky3_ln_determinant() {
    let c = Cholesky3Trait::new(s3()).unwrap();
    let expected = Transcendental::ln(c.determinant());
    assert!(ulp_diff(c.ln_determinant(), expected) <= 64);
}

/// `Cholesky3::new_unchecked` of a matrix that is not positive definite panics (upstream
/// computes NaN / infinities).
#[test]
#[should_panic(expected: 'nalgebra: not positive definite')]
fn test_cholesky3_new_unchecked_not_positive_definite_panics() {
    let _c = Cholesky3Trait::new_unchecked(-s3());
}

/// `new_with_substitute` replaces every non-positive pivot by the substitute (here all of them,
/// the matrix being negative definite: the first pivot is the substitute), and fails when the
/// substitute is not positive either.
#[test]
fn test_cholesky3_new_with_substitute() {
    assert!(Cholesky3Trait::new(-s3()).is_none());
    let c = Cholesky3Trait::new_with_substitute(-s3(), int(4)).unwrap();
    assert!(c.l11 == int(2));
    assert!(Cholesky3Trait::new_with_substitute(-s3(), int(0)).is_none());
    assert!(Cholesky3Trait::new_with_substitute(-s3(), int(-1)).is_none());
}

/// `Qr3`: `unpack_r` is `r()`, `qr_internal` the stored `(q, r)`, `q_tr_mul` is `qᵀ * rhs`
/// (`tr_mul`, bit for bit) on a vector and on a 3x2 matrix.
#[test]
fn test_qr3_api() {
    let f = Qr3Trait::new(a3());
    assert!(f.unpack_r() == f.r());
    let (q, r) = f.qr_internal();
    assert!(q == f.q() && r == f.r());
    let mut y = v3();
    f.q_tr_mul(ref y);
    assert!(y == f.q().tr_mul(v3()));
    let mut ym = bm3();
    f.q_tr_mul(ref ym);
    assert!(ym == f.q().tr_mul(bm3()));
}

/// `Qr3::solve_mut` on a vector is `solve`, bit for bit; on a 3x2 right-hand side each
/// column is the `solve` of that column; singular: `false`, `b` unchanged.
#[test]
fn test_qr3_solve_mut() {
    let f = Qr3Trait::new(a3());
    let mut x = v3();
    assert!(f.solve_mut(ref x));
    assert!(Option::Some(x) == f.solve(v3()));
    let b = bm3();
    let mut xm = b;
    assert!(f.solve_mut(ref xm));
    assert!(Option::Some(xm.column(0)) == f.solve(Vector3Trait::new(b.m11, b.m21, b.m31)));
    assert!(Option::Some(xm.column(1)) == f.solve(Vector3Trait::new(b.m12, b.m22, b.m32)));
    let g = Qr3Trait::new(z3());
    let mut x = v3();
    assert!(!g.solve_mut(ref x));
    assert!(x == v3());
}

fn a4() -> Matrix4<Fixed> {
    black_box(
        Matrix4Trait::new(
            fx(17269183691),
            fx(-385113388),
            fx(-129957666),
            fx(-540800101),
            fx(819880116),
            fx(21213427375),
            fx(-218887334),
            fx(-2265480739),
            fx(3139218471),
            fx(3771154975),
            fx(18707166569),
            fx(-3064055840),
            fx(1747801489),
            fx(613726620),
            fx(-4130848011),
            fx(-17363674548),
        ),
    )
}

fn s4() -> Matrix4<Fixed> {
    black_box(
        Matrix4Trait::new(
            fx(19584547695),
            fx(2557281232),
            fx(-3454611197),
            fx(-4291836762),
            fx(2557281232),
            fx(17589437612),
            fx(-2642347692),
            fx(57277778),
            fx(-3454611197),
            fx(-2642347692),
            fx(17288036959),
            fx(185306299),
            fx(-4291836762),
            fx(57277778),
            fx(185306299),
            fx(20635661227),
        ),
    )
}

fn z4() -> Matrix4<Fixed> {
    black_box(
        Matrix4Trait::new(
            fx(4207302736),
            fx(-3317458813),
            fx(3920843017),
            fx(0),
            fx(-4279664817),
            fx(-2145031880),
            fx(5730128),
            fx(0),
            fx(-2509531673),
            fx(1547260538),
            fx(-1442909805),
            fx(0),
            fx(3397740119),
            fx(-4024127580),
            fx(3621832870),
            fx(0),
        ),
    )
}

fn v4() -> Vector4<Fixed> {
    black_box(Vector4Trait::new(fx(4932980320), fx(-5871295550), fx(2483953131), fx(-5199850193)))
}

fn bm4() -> Matrix4x2<Fixed> {
    black_box(
        Matrix4x2Trait::new(
            fx(-167648974),
            fx(-4383097605),
            fx(-1923319738),
            fx(921629411),
            fx(7041891880),
            fx(-206803466),
            fx(7771265358),
            fx(-8047850464),
        ),
    )
}

/// `Lu4`: `unpack`, `l_unpack` and `lu_internal` are the stored factors, moved.
#[test]
fn test_lu4_unpack() {
    let f = Lu4Trait::new(a4());
    let (p, l, u) = f.unpack();
    assert!(p == f.p && l == f.l() && u == f.u());
    assert!(f.l_unpack() == f.l());
    assert!(f.lu_internal() == f.lu);
}

/// `Lu4::solve_mut` on a vector is `solve`, bit for bit; on a 4x2 right-hand side, each
/// column is the `solve` of that column (the shared prepared divisor is bit-identical).
#[test]
fn test_lu4_solve_mut() {
    let f = Lu4Trait::new(a4());
    let mut x = v4();
    assert!(f.solve_mut(ref x));
    assert!(Option::Some(x) == f.solve(v4()));
    let b = bm4();
    let mut xm = b;
    assert!(f.solve_mut(ref xm));
    assert!(Option::Some(xm.column(0)) == f.solve(Vector4Trait::new(b.m11, b.m21, b.m31, b.m41)));
    assert!(Option::Some(xm.column(1)) == f.solve(Vector4Trait::new(b.m12, b.m22, b.m32, b.m42)));
}

/// `Lu4::try_inverse_to` writes `try_inverse`, bit for bit; `try_invert_to` is the same LU
/// inverse of the matrix, and `try_inverse_mut` the matrix's own `try_inverse`.
#[test]
fn test_lu4_try_inverse_to() {
    let f = Lu4Trait::new(a4());
    let mut out = s4();
    assert!(f.try_inverse_to(ref out));
    assert!(Option::Some(out) == f.try_inverse());
    let mut out2 = s4();
    assert!(try_invert_to(a4(), ref out2));
    assert!(out2 == out);
    let mut m = a4();
    assert!(m.try_inverse_mut());
    assert!(Option::Some(m) == a4().try_inverse());
}

/// A singular matrix (a zero column): `solve_mut`, `try_inverse_to`, `try_invert_to` and
/// `try_inverse_mut` return `false` and leave their output unchanged.
#[test]
fn test_lu4_singular_leaves_outputs_unchanged() {
    let f = Lu4Trait::new(z4());
    assert!(!f.is_invertible());
    let mut x = v4();
    assert!(!f.solve_mut(ref x));
    assert!(x == v4());
    let mut out = s4();
    assert!(!f.try_inverse_to(ref out));
    assert!(out == s4());
    assert!(!try_invert_to(z4(), ref out));
    assert!(out == s4());
    let mut m = z4();
    assert!(!m.try_inverse_mut());
    assert!(m == z4());
}

/// `Cholesky4`: `unpack`, `unpack_dirty` and `l_dirty` are `l()`; `pack_dirty` rebuilds the
/// factor from the lower triangle only (the upper one is ignored); `new_unchecked` and
/// `new_with_substitute` of a positive-definite matrix are `new`; `cholesky()` / `udu()` are the
/// constructors.
#[test]
fn test_cholesky4_api() {
    let c = Cholesky4Trait::new(s4()).unwrap();
    assert!(c.unpack() == c.l() && c.unpack_dirty() == c.l() && c.l_dirty() == c.l());
    assert!(Cholesky4Trait::pack_dirty(c.l()) == c);
    assert!(Cholesky4Trait::pack_dirty(c.l() + c.l().transpose()) != c);
    assert!(Cholesky4Trait::pack_dirty(s4().lower_triangle()) == Cholesky4Trait::pack_dirty(s4()));
    assert!(Cholesky4Trait::new_unchecked(s4()) == c);
    assert!(Cholesky4Trait::new_with_substitute(s4(), int(1)) == Option::Some(c));
    assert!(s4().cholesky() == Option::Some(c));
    let u = s4().udu().unwrap();
    let w = Udu4Trait::new(s4()).unwrap();
    assert!(u.u == w.u && u.d == w.d);
}

/// `Cholesky4::solve_mut` on a vector is `solve`, bit for bit; on a 4x2 right-hand side each
/// column is the `solve` of that column.
#[test]
fn test_cholesky4_solve_mut() {
    let c = Cholesky4Trait::new(s4()).unwrap();
    let mut x = v4();
    c.solve_mut(ref x);
    assert!(x == c.solve(v4()));
    let b = bm4();
    let mut xm = b;
    c.solve_mut(ref xm);
    assert!(xm.column(0) == c.solve(Vector4Trait::new(b.m11, b.m21, b.m31, b.m41)));
    assert!(xm.column(1) == c.solve(Vector4Trait::new(b.m12, b.m22, b.m32, b.m42)));
}

/// `Cholesky4::ln_determinant` is `ln(determinant)` within the rounding of 4 logarithms.
#[test]
fn test_cholesky4_ln_determinant() {
    let c = Cholesky4Trait::new(s4()).unwrap();
    let expected = Transcendental::ln(c.determinant());
    assert!(ulp_diff(c.ln_determinant(), expected) <= 64);
}

/// `Cholesky4::new_unchecked` of a matrix that is not positive definite panics (upstream
/// computes NaN / infinities).
#[test]
#[should_panic(expected: 'nalgebra: not positive definite')]
fn test_cholesky4_new_unchecked_not_positive_definite_panics() {
    let _c = Cholesky4Trait::new_unchecked(-s4());
}

/// `new_with_substitute` replaces every non-positive pivot by the substitute (here all of them,
/// the matrix being negative definite: the first pivot is the substitute), and fails when the
/// substitute is not positive either.
#[test]
fn test_cholesky4_new_with_substitute() {
    assert!(Cholesky4Trait::new(-s4()).is_none());
    let c = Cholesky4Trait::new_with_substitute(-s4(), int(4)).unwrap();
    assert!(c.l11 == int(2));
    assert!(Cholesky4Trait::new_with_substitute(-s4(), int(0)).is_none());
    assert!(Cholesky4Trait::new_with_substitute(-s4(), int(-1)).is_none());
}

/// `Qr4`: `unpack_r` is `r()`, `qr_internal` the stored `(q, r)`, `q_tr_mul` is `qᵀ * rhs`
/// (`tr_mul`, bit for bit) on a vector and on a 4x2 matrix.
#[test]
fn test_qr4_api() {
    let f = Qr4Trait::new(a4());
    assert!(f.unpack_r() == f.r());
    let (q, r) = f.qr_internal();
    assert!(q == f.q() && r == f.r());
    let mut y = v4();
    f.q_tr_mul(ref y);
    assert!(y == f.q().tr_mul(v4()));
    let mut ym = bm4();
    f.q_tr_mul(ref ym);
    assert!(ym == f.q().tr_mul(bm4()));
}

/// `Qr4::solve_mut` on a vector is `solve`, bit for bit; on a 4x2 right-hand side each
/// column is the `solve` of that column; singular: `false`, `b` unchanged.
#[test]
fn test_qr4_solve_mut() {
    let f = Qr4Trait::new(a4());
    let mut x = v4();
    assert!(f.solve_mut(ref x));
    assert!(Option::Some(x) == f.solve(v4()));
    let b = bm4();
    let mut xm = b;
    assert!(f.solve_mut(ref xm));
    assert!(Option::Some(xm.column(0)) == f.solve(Vector4Trait::new(b.m11, b.m21, b.m31, b.m41)));
    assert!(Option::Some(xm.column(1)) == f.solve(Vector4Trait::new(b.m12, b.m22, b.m32, b.m42)));
    let g = Qr4Trait::new(z4());
    let mut x = v4();
    assert!(!g.solve_mut(ref x));
    assert!(x == v4());
}

fn a6() -> Matrix6<Fixed> {
    black_box(
        Matrix6Trait::new(
            fx(-29723545019),
            fx(-606384481),
            fx(2349805781),
            fx(-3525154351),
            fx(2638402502),
            fx(-3179137267),
            fx(-3826296531),
            fx(-27600920026),
            fx(-2477680815),
            fx(-111427939),
            fx(-3215404509),
            fx(3409705432),
            fx(-3858663949),
            fx(-3381284655),
            fx(-29525610721),
            fx(354551899),
            fx(-698341559),
            fx(-473665585),
            fx(-3520820150),
            fx(2650313966),
            fx(-2818358646),
            fx(-26062398528),
            fx(3506558990),
            fx(4135225961),
            fx(3093314986),
            fx(2748615638),
            fx(4249220539),
            fx(819927082),
            fx(25962504484),
            fx(1745577666),
            fx(669036570),
            fx(-271539886),
            fx(3944630750),
            fx(-1522735108),
            fx(-3636010632),
            fx(29576353138),
        ),
    )
}

fn s6() -> Matrix6<Fixed> {
    black_box(
        Matrix6Trait::new(
            fx(26413474944),
            fx(3295801372),
            fx(3130634015),
            fx(-3596929345),
            fx(69475712),
            fx(-1217355462),
            fx(3295801372),
            fx(27782877177),
            fx(1875084641),
            fx(-1963298288),
            fx(415739647),
            fx(-3336428873),
            fx(3130634015),
            fx(1875084641),
            fx(25855387401),
            fx(-1873604884),
            fx(2546136079),
            fx(-3234756262),
            fx(-3596929345),
            fx(-1963298288),
            fx(-1873604884),
            fx(28318260116),
            fx(1931801535),
            fx(-1009117923),
            fx(69475712),
            fx(415739647),
            fx(2546136079),
            fx(1931801535),
            fx(25803069853),
            fx(-1774916521),
            fx(-1217355462),
            fx(-3336428873),
            fx(-3234756262),
            fx(-1009117923),
            fx(-1774916521),
            fx(28881591293),
        ),
    )
}

fn z6() -> Matrix6<Fixed> {
    black_box(
        Matrix6Trait::new(
            fx(-1430570494),
            fx(-2127411319),
            fx(-1872632194),
            fx(571929587),
            fx(2307011053),
            fx(0),
            fx(-2437340190),
            fx(723357621),
            fx(3426187036),
            fx(-4075609158),
            fx(1254794308),
            fx(0),
            fx(-1375208155),
            fx(-991175565),
            fx(872135579),
            fx(-2146606100),
            fx(2262514557),
            fx(0),
            fx(4050041548),
            fx(2423169025),
            fx(-1059935752),
            fx(1851197918),
            fx(-2194979431),
            fx(0),
            fx(4071943687),
            fx(2235087054),
            fx(2814904463),
            fx(-2618764655),
            fx(3005235639),
            fx(0),
            fx(2502749189),
            fx(-400283080),
            fx(1390897699),
            fx(-3863338302),
            fx(-2665494932),
            fx(0),
        ),
    )
}

fn v6() -> Vector6<Fixed> {
    black_box(
        Vector6Trait::new(
            fx(1783208915),
            fx(-3727106204),
            fx(8498362422),
            fx(-694422499),
            fx(6861364063),
            fx(-7416583215),
        ),
    )
}

fn bm6() -> Matrix6x2<Fixed> {
    black_box(
        Matrix6x2Trait::new(
            fx(-7311949733),
            fx(3746879639),
            fx(3746059695),
            fx(-671910311),
            fx(-5110429262),
            fx(-6746729010),
            fx(-2339488579),
            fx(-4843714591),
            fx(3117769388),
            fx(-2852200256),
            fx(587599210),
            fx(8248840103),
        ),
    )
}

/// `Lu6`: `unpack`, `l_unpack` and `lu_internal` are the stored factors, moved.
#[test]
fn test_lu6_unpack() {
    let f = Lu6Trait::new(a6());
    let (p, l, u) = f.unpack();
    assert!(p == f.p && l == f.l() && u == f.u());
    assert!(f.l_unpack() == f.l());
    assert!(f.lu_internal() == f.lu);
}

/// `Lu6::solve_mut` on a vector is `solve`, bit for bit; on a 6x2 right-hand side, each
/// column is the `solve` of that column (the shared prepared divisor is bit-identical).
#[test]
fn test_lu6_solve_mut() {
    let f = Lu6Trait::new(a6());
    let mut x = v6();
    assert!(f.solve_mut(ref x));
    assert!(Option::Some(x) == f.solve(v6()));
    let b = bm6();
    let mut xm = b;
    assert!(f.solve_mut(ref xm));
    assert!(
        Option::Some(xm.column(0)) == f
            .solve(Vector6Trait::new(b.m11, b.m21, b.m31, b.m41, b.m51, b.m61)),
    );
    assert!(
        Option::Some(xm.column(1)) == f
            .solve(Vector6Trait::new(b.m12, b.m22, b.m32, b.m42, b.m52, b.m62)),
    );
}

/// `Lu6::try_inverse_to` writes `try_inverse`, bit for bit; `try_invert_to` is the same LU
/// inverse of the matrix, and `try_inverse_mut` the matrix's own `try_inverse`.
#[test]
fn test_lu6_try_inverse_to() {
    let f = Lu6Trait::new(a6());
    let mut out = s6();
    assert!(f.try_inverse_to(ref out));
    assert!(Option::Some(out) == f.try_inverse());
    let mut out2 = s6();
    assert!(try_invert_to(a6(), ref out2));
    assert!(out2 == out);
    let mut m = a6();
    assert!(m.try_inverse_mut());
    assert!(Option::Some(m) == a6().try_inverse());
}

/// A singular matrix (a zero column): `solve_mut`, `try_inverse_to`, `try_invert_to` and
/// `try_inverse_mut` return `false` and leave their output unchanged.
#[test]
fn test_lu6_singular_leaves_outputs_unchanged() {
    let f = Lu6Trait::new(z6());
    assert!(!f.is_invertible());
    let mut x = v6();
    assert!(!f.solve_mut(ref x));
    assert!(x == v6());
    let mut out = s6();
    assert!(!f.try_inverse_to(ref out));
    assert!(out == s6());
    assert!(!try_invert_to(z6(), ref out));
    assert!(out == s6());
    let mut m = z6();
    assert!(!m.try_inverse_mut());
    assert!(m == z6());
}

/// `Cholesky6`: `unpack`, `unpack_dirty` and `l_dirty` are `l()`; `pack_dirty` rebuilds the
/// factor from the lower triangle only (the upper one is ignored); `new_unchecked` and
/// `new_with_substitute` of a positive-definite matrix are `new`; `cholesky()` / `udu()` are the
/// constructors.
#[test]
fn test_cholesky6_api() {
    let c = Cholesky6Trait::new(s6()).unwrap();
    assert!(c.unpack() == c.l() && c.unpack_dirty() == c.l() && c.l_dirty() == c.l());
    assert!(Cholesky6Trait::pack_dirty(c.l()) == c);
    assert!(Cholesky6Trait::pack_dirty(c.l() + c.l().transpose()) != c);
    assert!(Cholesky6Trait::pack_dirty(s6().lower_triangle()) == Cholesky6Trait::pack_dirty(s6()));
    assert!(Cholesky6Trait::new_unchecked(s6()) == c);
    assert!(Cholesky6Trait::new_with_substitute(s6(), int(1)) == Option::Some(c));
    assert!(s6().cholesky() == Option::Some(c));
    let u = s6().udu().unwrap();
    let w = Udu6Trait::new(s6()).unwrap();
    assert!(u.u == w.u && u.d == w.d);
}

/// `Cholesky6::solve_mut` on a vector is `solve`, bit for bit; on a 6x2 right-hand side each
/// column is the `solve` of that column.
#[test]
fn test_cholesky6_solve_mut() {
    let c = Cholesky6Trait::new(s6()).unwrap();
    let mut x = v6();
    c.solve_mut(ref x);
    assert!(x == c.solve(v6()));
    let b = bm6();
    let mut xm = b;
    c.solve_mut(ref xm);
    assert!(xm.column(0) == c.solve(Vector6Trait::new(b.m11, b.m21, b.m31, b.m41, b.m51, b.m61)));
    assert!(xm.column(1) == c.solve(Vector6Trait::new(b.m12, b.m22, b.m32, b.m42, b.m52, b.m62)));
}

/// `Cholesky6::ln_determinant` is `ln(determinant)` within the rounding of 6 logarithms.
#[test]
fn test_cholesky6_ln_determinant() {
    let c = Cholesky6Trait::new(s6()).unwrap();
    let expected = Transcendental::ln(c.determinant());
    assert!(ulp_diff(c.ln_determinant(), expected) <= 64);
}

/// `Cholesky6::new_unchecked` of a matrix that is not positive definite panics (upstream
/// computes NaN / infinities).
#[test]
#[should_panic(expected: 'nalgebra: not positive definite')]
fn test_cholesky6_new_unchecked_not_positive_definite_panics() {
    let _c = Cholesky6Trait::new_unchecked(-s6());
}

/// `new_with_substitute` replaces every non-positive pivot by the substitute (here all of them,
/// the matrix being negative definite: the first pivot is the substitute), and fails when the
/// substitute is not positive either.
#[test]
fn test_cholesky6_new_with_substitute() {
    assert!(Cholesky6Trait::new(-s6()).is_none());
    let c = Cholesky6Trait::new_with_substitute(-s6(), int(4)).unwrap();
    assert!(c.l11 == int(2));
    assert!(Cholesky6Trait::new_with_substitute(-s6(), int(0)).is_none());
    assert!(Cholesky6Trait::new_with_substitute(-s6(), int(-1)).is_none());
}
