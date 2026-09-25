//! `Perspective3`: a 3D perspective projection stored as its homogeneous `Matrix4` (upstream
//! `nalgebra::Perspective3`), WP 8.4-P11b.
//!
//! The matrix has upstream's OpenGL layout: `m11 = m22 / aspect`, `m22 = 1 / tan(fovy / 2)`,
//! `m33 = (zfar + znear) / (znear - zfar)`, `m34 = 2 · zfar · znear / (znear - zfar)`,
//! `m43 = -1`, every other entry zero. Like upstream, the accessors, the setters and the
//! projections read and write those five entries only; a matrix given to `from_matrix_unchecked`
//! is trusted to have that structure.
//!
//! API split (the pattern of `UnitQuaternionTrait` / `UnitQuaternionAngleTrait`):
//! - `Perspective3Trait` (`Real` scalar): `from_matrix_unchecked`, the conversions, `inverse`, the
//!   accessors, the projections and the setters that need no trigonometry;
//! - `Perspective3AngleTrait` (`Real` + `Transcendental`): `new`, `fovy`, `set_fovy`;
//! - `Matrix4PerspectiveTrait`: upstream's `Matrix4::new_perspective` (`base/cg.rs`), which is
//!   `Perspective3::new(..).into_inner()`;
//! - `Into<Perspective3, Matrix4>`: upstream's `From<Perspective3> for Matrix4`.
//!
//! Numeric contract (AGENTS.md): every division is ONE correctly rounded `Real::div` (to nearest,
//! ties to even, like `f64 /`), every product one floored fixed-point product, every sum of
//! products one fused kernel; overflow and division by zero panic (upstream's `inf` / `NaN`).
//! Upstream's `relative_eq!(a, b)` assertions use the default tolerances of DESIGN D3
//! (`default_epsilon` = 1 ulp, absolute and relative). Where a formula is reassociated to round
//! fewer times (`znear`, `zfar`, `project_*`), the doc comment says so: the result is closer to
//! upstream's (near-exact `f64`) value than the literal fixed-point transcription, whose variants
//! are kept and measured in the benchmarks of `nalgebra_tests_geometry_projections`.

use simba::scalar::{Real, Transcendental};
use crate::base::matrix4::Matrix4;
use crate::base::point3::Point3;
use crate::base::vector3::Vector3;
use super::projective3::Projective3;
use super::quaternion::ApproxEqTrait;

/// Panic messages of `Perspective3` (upstream's assertion messages do not fit a `felt252`: each
/// constant quotes its upstream message).
pub mod errors {
    /// Upstream: "The near-plane and far-plane must not be superimposed." (`new`, `znear`
    /// `relative_eq` to `zfar`).
    pub const SUPERIMPOSED_PLANES: felt252 = 'nalgebra: superimposed planes';
    /// Upstream: "The aspect ratio must not be zero." (`new`, `set_aspect`: `aspect`
    /// `relative_eq` to zero, i.e. within 1 ulp of it).
    pub const ZERO_ASPECT: felt252 = 'nalgebra: zero aspect ratio';
}

/// A 3D perspective projection: the homogeneous matrix `matrix` (private, like upstream's field;
/// read it with `as_matrix` / `into_inner`). `PartialEq` compares the matrices exactly, `Serde`
/// is the matrix's (column-major, upstream's serde layout). Upstream: `Perspective3<T>`.
#[derive(Copy, Drop, PartialEq, Serde, Debug)]
pub struct Perspective3<T> {
    matrix: Matrix4<T>,
}

/// Crate-internal builders of `Perspective3` (methods of a generic impl, AGENTS.md rule 6).
#[generate_trait]
pub(crate) impl Perspective3InternalImpl<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Add<T>, +Sub<T>, +Mul<T>, +Neg<T>,
> of Perspective3InternalTrait<T> {
    /// The matrix of a perspective projection from its four non-constant entries.
    #[inline(always)]
    fn perspective_matrix(m11: T, m22: T, m33: T, m34: T) -> Matrix4<T> {
        let z = R::zero();
        Matrix4 {
            m11,
            m21: z,
            m31: z,
            m41: z,
            m12: z,
            m22,
            m32: z,
            m42: z,
            m13: z,
            m23: z,
            m33,
            m43: -R::one(),
            m14: z,
            m24: z,
            m34,
            m44: z,
        }
    }

    /// `(m33, m34)` of the planes `znear`, `zfar`: `(zfar + znear) / (znear - zfar)` and
    /// `zfar · (2 · znear) / (znear - zfar)` (upstream's `zfar * znear * 2`: the doubling is
    /// exact in `f64` as in fixed point, so doubling `znear` first rounds the product once, not
    /// twice).
    #[inline(always)]
    fn depth_entries(znear: T, zfar: T) -> (T, T) {
        let d = znear - zfar;
        (R::div(zfar + znear, d), R::div(zfar * (znear + znear), d))
    }
}

/// Operations of `Perspective3<T>` over a `Real` scalar. By value, unrolled, no loop.
#[generate_trait]
pub impl Perspective3Impl<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Add<T>, +Sub<T>, +Mul<T>, +Neg<T>, +PartialEq<T>,
> of Perspective3Trait<T> {
    /// The projection whose homogeneous matrix is `matrix`, trusted to have the structure of a
    /// perspective projection (not checked). Exact. Upstream: `from_matrix_unchecked`.
    #[inline(always)]
    fn from_matrix_unchecked(matrix: Matrix4<T>) -> Perspective3<T> {
        Perspective3 { matrix }
    }

    /// The homogeneous matrix (a copy: Cairo values are passed by value). Exact. Upstream:
    /// `as_matrix`.
    #[inline(always)]
    fn as_matrix(self: Perspective3<T>) -> Matrix4<T> {
        self.matrix
    }

    /// The homogeneous matrix. Exact. Upstream: `into_inner`.
    #[inline(always)]
    fn into_inner(self: Perspective3<T>) -> Matrix4<T> {
        self.matrix
    }

    /// The homogeneous matrix: `into_inner` (deprecated upstream, "use `.into_inner()`
    /// instead"). Exact. Upstream: `unwrap`.
    #[inline(always)]
    fn unwrap(self: Perspective3<T>) -> Matrix4<T> {
        self.matrix
    }

    /// The homogeneous matrix. Exact. Upstream: `to_homogeneous`.
    #[inline(always)]
    fn to_homogeneous(self: Perspective3<T>) -> Matrix4<T> {
        self.matrix
    }

    /// This projection seen as a `Projective3` (the same matrix; upstream reinterprets a
    /// reference, Cairo copies the value). Exact. Upstream: `as_projective`.
    #[inline(always)]
    fn as_projective(self: Perspective3<T>) -> Projective3<T> {
        Projective3 { matrix: self.matrix }
    }

    /// This projection as a `Projective3` (the same matrix). Exact. Upstream: `to_projective`.
    #[inline(always)]
    fn to_projective(self: Perspective3<T>) -> Projective3<T> {
        Projective3 { matrix: self.matrix }
    }

    /// The inverse of the projection matrix, in closed form: the matrix with `m11 = 1 / m11`,
    /// `m22 = 1 / m22`, `m33 = 0`, `m34 = 1 / m43`, `m43 = 1 / m34`, `m44 = -m33 / (m34 · m43)`
    /// and the other entries of `self` (each quotient correctly rounded; the product `m34 · m43`
    /// is exact for the `m43 = -1` of a perspective). Panics on a zero entry among `m11`, `m22`,
    /// `m34`, `m43` and on overflow. Upstream: `inverse`.
    #[inline(always)]
    fn inverse(self: Perspective3<T>) -> Matrix4<T> {
        let m = self.matrix;
        Matrix4 {
            m11: R::recip(m.m11),
            m21: m.m21,
            m31: m.m31,
            m41: m.m41,
            m12: m.m12,
            m22: R::recip(m.m22),
            m32: m.m32,
            m42: m.m42,
            m13: m.m13,
            m23: m.m23,
            m33: R::zero(),
            m43: R::recip(m.m34),
            m14: m.m14,
            m24: m.m24,
            m34: R::recip(m.m43),
            m44: R::div(-m.m33, m.m34 * m.m43),
        }
    }

    /// The aspect ratio (width over height), `m22 / m11`, correctly rounded. Upstream: `aspect`.
    #[inline(always)]
    fn aspect(self: Perspective3<T>) -> T {
        let m = self.matrix;
        R::div(m.m22, m.m11)
    }

    /// The near clipping plane, `m34 / (m33 - 1)`: ONE correctly rounded division. Upstream
    /// evaluates the same value as `m34 / (2 · ratio) - m34 / 2`, `ratio = (1 - m33) / (-m33 -
    /// 1)` (four roundings, harmless in `f64`); the single quotient is the fixed-point value
    /// nearest to it, and cheaper (4,130 gas against 13,850 for the literal form). Panics on
    /// `m33 = 1` (upstream's `inf`). Upstream: `znear`.
    #[inline(always)]
    fn znear(self: Perspective3<T>) -> T {
        let m = self.matrix;
        R::div(m.m34, m.m33 - R::one())
    }

    /// The far clipping plane, `m34 / (m33 + 1)`: ONE correctly rounded division (upstream's
    /// `(m34 - ratio · m34) / 2`, see `znear`). Panics on `m33 = -1`. Upstream: `zfar`.
    #[inline(always)]
    fn zfar(self: Perspective3<T>) -> T {
        let m = self.matrix;
        R::div(m.m34, m.m33 + R::one())
    }

    /// Projects the point `p` into normalized device coordinates: `(m11 · x, m22 · y, m33 · z +
    /// m34) / -z`. The numerators are floored products (the third one fused, `Real::mul_add`) and
    /// the three quotients share ONE prepared divisor (`Real::div3`, each correctly rounded):
    /// upstream multiplies by `inverse_denom = -1 / z` instead, which rounds twice (13,540 gas
    /// against 15,050, kept as the loser for that extra rounding; three separate quotients cost
    /// 15,400). Panics on `z = 0` and on overflow. Upstream: `project_point`.
    #[inline(always)]
    fn project_point(self: Perspective3<T>, p: Point3<T>) -> Point3<T> {
        let m = self.matrix;
        let (x, y, z) = R::div3(m.m11 * p.x, m.m22 * p.y, R::mul_add(m.m33, p.z, m.m34), -p.z);
        Point3 { x, y, z }
    }

    /// Un-projects the point `p` of normalized device coordinates back to view space: with `w =
    /// m34 / (z + m33)`, the point `(x · w / m11, y · w / m22, -w)` (upstream's order of
    /// operations: one quotient, then a floored product and a quotient per coordinate). Panics on
    /// `z = -m33` and on overflow. Upstream: `unproject_point`.
    #[inline(always)]
    fn unproject_point(self: Perspective3<T>, p: Point3<T>) -> Point3<T> {
        let m = self.matrix;
        let w = R::div(m.m34, p.z + m.m33);
        Point3 { x: R::div(p.x * w, m.m11), y: R::div(p.y * w, m.m22), z: -w }
    }

    /// Projects the vector `v`: `(m11 · x / -z, m22 · y / -z, m33)` (upstream's third component
    /// is `m33` itself). One floored product and one correctly rounded quotient per component
    /// (two plain quotients: 10,390 gas against 12,580 for a prepared divisor). Panics on `z = 0`
    /// and on overflow. Upstream: `project_vector`.
    #[inline(always)]
    fn project_vector(self: Perspective3<T>, v: Vector3<T>) -> Vector3<T> {
        let m = self.matrix;
        let d = -v.z;
        Vector3 { x: R::div(m.m11 * v.x, d), y: R::div(m.m22 * v.y, d), z: m.m33 }
    }

    /// Sets the aspect ratio: `m11 = m22 / aspect`, correctly rounded. Panics with
    /// `errors::ZERO_ASPECT` when `aspect` is `relative_eq` to zero (within 1 ulp). Upstream:
    /// `set_aspect`.
    #[inline(always)]
    fn set_aspect(ref self: Perspective3<T>, aspect: T) {
        assert(
            !ApproxEqTrait::relative_eq(aspect, R::zero(), 1, R::default_epsilon()),
            errors::ZERO_ASPECT,
        );
        let mut m = self.matrix;
        m.m11 = R::div(m.m22, aspect);
        self.matrix = m;
    }

    /// Sets the near plane, keeping the far plane (`set_znear_and_zfar(znear, self.zfar())`).
    /// Upstream: `set_znear`.
    #[inline(always)]
    fn set_znear(ref self: Perspective3<T>, znear: T) {
        let zfar = Self::zfar(self);
        Self::set_znear_and_zfar(ref self, znear, zfar);
    }

    /// Sets the far plane, keeping the near plane (`set_znear_and_zfar(self.znear(), zfar)`).
    /// Upstream: `set_zfar`.
    #[inline(always)]
    fn set_zfar(ref self: Perspective3<T>, zfar: T) {
        let znear = Self::znear(self);
        Self::set_znear_and_zfar(ref self, znear, zfar);
    }

    /// Sets both clipping planes: `m33 = (zfar + znear) / (znear - zfar)`, `m34 = 2 · zfar ·
    /// znear / (znear - zfar)` (two correctly rounded quotients, the product floored once). Like
    /// upstream, no check: equal planes panic with the division by zero. Upstream:
    /// `set_znear_and_zfar`.
    #[inline(always)]
    fn set_znear_and_zfar(ref self: Perspective3<T>, znear: T, zfar: T) {
        let (m33, m34) = Perspective3InternalTrait::depth_entries(znear, zfar);
        let mut m = self.matrix;
        m.m33 = m33;
        m.m34 = m34;
        self.matrix = m;
    }
}

/// The operations of `Perspective3<T>` that need trigonometry (`Real` + `Transcendental`).
#[generate_trait]
pub impl Perspective3AngleImpl<
    T,
    impl R: Real<T>,
    impl Tr: Transcendental<T>,
    +Copy<T>,
    +Drop<T>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
    +PartialEq<T>,
> of Perspective3AngleTrait<T> {
    /// The perspective projection of aspect ratio `aspect` (width over height), vertical field of
    /// view `fovy` (radians) and clipping planes `znear`, `zfar`: `m22 = 1 / tan(fovy / 2)`,
    /// `m11 = m22 / aspect`, then `set_znear_and_zfar`, `m43 = -1`, `m44 = 0` (upstream's
    /// sequence; `fovy / 2` and every quotient correctly rounded, `tan` is
    /// `Transcendental::tan`). Panics with `errors::SUPERIMPOSED_PLANES` when `zfar` is
    /// `relative_eq` to `znear`, with `errors::ZERO_ASPECT` when `aspect` is `relative_eq` to zero
    /// (DESIGN D3 default tolerances), and on overflow. Upstream: `Perspective3::new`.
    fn new(aspect: T, fovy: T, znear: T, zfar: T) -> Perspective3<T> {
        assert(
            !ApproxEqTrait::relative_eq(zfar, znear, 1, R::default_epsilon()),
            errors::SUPERIMPOSED_PLANES,
        );
        assert(
            !ApproxEqTrait::relative_eq(aspect, R::zero(), 1, R::default_epsilon()),
            errors::ZERO_ASPECT,
        );
        // Upstream starts from the identity: `set_fovy` scales `m11 = 1` by `m22 / 1` (exact).
        let m22 = R::recip(Tr::tan(R::div(fovy, R::from_int(2))));
        let (m33, m34) = Perspective3InternalTrait::depth_entries(znear, zfar);
        Perspective3 {
            matrix: Perspective3InternalTrait::perspective_matrix(
                R::div(m22, aspect), m22, m33, m34,
            ),
        }
    }

    /// The vertical field of view, `2 · atan(1 / m22)` (the reciprocal correctly rounded, the
    /// doubling exact; `atan` is `Transcendental::atan`). Upstream: `fovy`.
    #[inline(always)]
    fn fovy(self: Perspective3<T>) -> T {
        let a = Tr::atan(R::recip(self.matrix.m22));
        a + a
    }

    /// Sets the vertical field of view: `m22 = 1 / tan(fovy / 2)` and `m11 *= m22_new / m22_old`
    /// (the aspect ratio is kept; upstream's order of operations: the ratio correctly rounded,
    /// then one floored product). Panics on overflow and on a zero `tan`. Upstream: `set_fovy`.
    fn set_fovy(ref self: Perspective3<T>, fovy: T) {
        let mut m = self.matrix;
        let new_m22 = R::recip(Tr::tan(R::div(fovy, R::from_int(2))));
        m.m11 = m.m11 * R::div(new_m22, m.m22);
        m.m22 = new_m22;
        self.matrix = m;
    }
}

/// Upstream's `Matrix4::new_perspective` (`base/cg.rs`).
#[generate_trait]
pub impl Matrix4PerspectiveImpl<
    T,
    impl R: Real<T>,
    impl Tr: Transcendental<T>,
    +Copy<T>,
    +Drop<T>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
    +PartialEq<T>,
> of Matrix4PerspectiveTrait<T> {
    /// The homogeneous matrix of a perspective projection: `Perspective3AngleTrait::new(aspect,
    /// fovy, znear, zfar).into_inner()` (same rounding and panics). Upstream:
    /// `Matrix4::new_perspective`.
    #[inline(always)]
    fn new_perspective(aspect: T, fovy: T, znear: T, zfar: T) -> Matrix4<T> {
        let p: Perspective3<T> = Perspective3AngleTrait::new(aspect, fovy, znear, zfar);
        p.matrix
    }
}

/// `p.into()`: the homogeneous matrix (`into_inner`). Exact. Upstream: `From<Perspective3> for
/// Matrix4`.
pub impl Matrix4FromPerspective3<T> of Into<Perspective3<T>, Matrix4<T>> {
    #[inline(always)]
    fn into(self: Perspective3<T>) -> Matrix4<T> {
        self.matrix
    }
}
