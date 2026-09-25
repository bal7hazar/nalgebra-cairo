//! `Orthographic3`: a 3D orthographic projection stored as its homogeneous `Matrix4` (upstream
//! `nalgebra::Orthographic3`), WP 8.4-P11b.
//!
//! The matrix has upstream's OpenGL layout: `m11 = 2 / (right - left)`, `m14 = -(right + left) /
//! (right - left)`, `m22 = 2 / (top - bottom)`, `m24 = -(top + bottom) / (top - bottom)`, `m33 =
//! -2 / (zfar - znear)`, `m34 = -(zfar + znear) / (zfar - znear)`, `m44 = 1`, every other entry
//! zero. Like upstream, the accessors, the setters and the projections read and write those seven
//! entries only; a matrix given to `from_matrix_unchecked` is trusted to have that structure.
//!
//! API split (the pattern of `UnitQuaternionTrait` / `UnitQuaternionAngleTrait`):
//! - `Orthographic3Trait` (`Real` scalar): everything but `from_fov`;
//! - `Orthographic3AngleTrait` (`Real` + `Transcendental`): `from_fov`;
//! - `Matrix4OrthographicTrait`: upstream's `Matrix4::new_orthographic` (`base/cg.rs`), which is
//!   `Orthographic3::new(..).into_inner()`;
//! - `Into<Orthographic3, Matrix4>`: upstream's `From<Orthographic3> for Matrix4`.
//! Upstream's `as_projective` / `to_projective` need `Projective3` (WP 8.4-P11a).
//!
//! Numeric contract (AGENTS.md): every division is ONE correctly rounded `Real::div` (to nearest,
//! ties to even, like `f64 /`), every product one floored fixed-point product, every product-sum
//! one fused kernel (`Real::mul_add`); overflow and division by zero panic (upstream's `inf` /
//! `NaN`). Upstream's `relative_eq!(a, b)` assertions use the default tolerances of DESIGN D3
//! (`default_epsilon` = 1 ulp, absolute and relative).

use simba::scalar::{Real, Transcendental};
use crate::base::matrix4::Matrix4;
use crate::base::point3::Point3;
use crate::base::vector3::Vector3;
use super::quaternion::ApproxEqTrait;

/// Panic messages of `Orthographic3` (upstream's assertion messages do not fit a `felt252`: each
/// constant quotes its upstream message).
pub mod errors {
    /// Upstream: "The near-plane and far-plane must not be superimposed." (`set_znear_and_zfar`,
    /// hence `new`, `set_znear`, `set_zfar`: `znear == zfar`).
    pub const SUPERIMPOSED_PLANES: felt252 = 'nalgebra: superimposed planes';
    /// Upstream: "The far plane must not be equal to the near plane." (`from_fov`: `znear ==
    /// zfar`).
    pub const FAR_EQUALS_NEAR: felt252 = 'nalgebra: far plane == near';
    /// Upstream: "The aspect ratio must not be zero." (`from_fov`: `aspect` `relative_eq` to
    /// zero, i.e. within 1 ulp of it).
    pub const ZERO_ASPECT: felt252 = 'nalgebra: zero aspect ratio';
    /// Upstream: "The left corner must not be equal to the right corner."
    /// (`set_left_and_right`, hence `new`, `set_left`, `set_right`).
    pub const LEFT_EQUALS_RIGHT: felt252 = 'nalgebra: left == right';
    /// Upstream: "The top corner must not be equal to the bottom corner." (`set_bottom_and_top`,
    /// hence `new`, `set_bottom`, `set_top`).
    pub const BOTTOM_EQUALS_TOP: felt252 = 'nalgebra: bottom == top';
}

/// A 3D orthographic projection: the homogeneous matrix `matrix` (private, like upstream's field;
/// read it with `as_matrix` / `into_inner`). `PartialEq` compares the matrices exactly, `Serde`
/// is the matrix's (column-major, upstream's serde layout). Upstream: `Orthographic3<T>`.
#[derive(Copy, Drop, PartialEq, Serde, Debug)]
pub struct Orthographic3<T> {
    matrix: Matrix4<T>,
}

/// Crate-internal kernel of `Orthographic3` (a method of a generic impl, AGENTS.md rule 6).
#[generate_trait]
pub(crate) impl Orthographic3InternalImpl<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Add<T>, +Sub<T>, +Neg<T>,
> of Orthographic3InternalTrait<T> {
    /// `(2 / (hi - lo), -(hi + lo) / (hi - lo))`: the scale and offset of one axis (two correctly
    /// rounded quotients). The caller checks `lo != hi`.
    #[inline(always)]
    fn axis_entries(lo: T, hi: T) -> (T, T) {
        let d = hi - lo;
        (R::div(R::from_int(2), d), R::div(-(hi + lo), d))
    }
}

/// Operations of `Orthographic3<T>` over a `Real` scalar. By value, unrolled, no loop.
#[generate_trait]
pub impl Orthographic3Impl<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Add<T>, +Sub<T>, +Mul<T>, +Neg<T>, +PartialEq<T>,
> of Orthographic3Trait<T> {
    /// The orthographic projection of the box `[left, right] x [bottom, top] x [-znear, -zfar]`
    /// (view space, camera looking down `-z`) onto the unit cube: upstream's
    /// `set_left_and_right`, `set_bottom_and_top`, `set_znear_and_zfar` on the identity (six
    /// correctly rounded quotients). Panics with `errors::LEFT_EQUALS_RIGHT`,
    /// `errors::BOTTOM_EQUALS_TOP` or `errors::SUPERIMPOSED_PLANES` on an empty extent, and on
    /// overflow. Upstream: `Orthographic3::new`.
    fn new(left: T, right: T, bottom: T, top: T, znear: T, zfar: T) -> Orthographic3<T> {
        assert(left != right, errors::LEFT_EQUALS_RIGHT);
        assert(bottom != top, errors::BOTTOM_EQUALS_TOP);
        assert(zfar != znear, errors::SUPERIMPOSED_PLANES);
        let (m11, m14) = Orthographic3InternalTrait::axis_entries(left, right);
        let (m22, m24) = Orthographic3InternalTrait::axis_entries(bottom, top);
        let (m33, m34) = Orthographic3InternalTrait::axis_entries(znear, zfar);
        let z = R::zero();
        Orthographic3 {
            matrix: Matrix4 {
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
                m33: -m33,
                m43: z,
                m14,
                m24,
                m34,
                m44: R::one(),
            },
        }
    }

    /// The projection whose homogeneous matrix is `matrix`, trusted to have the structure of an
    /// orthographic projection (not checked). Exact. Upstream: `from_matrix_unchecked`.
    #[inline(always)]
    fn from_matrix_unchecked(matrix: Matrix4<T>) -> Orthographic3<T> {
        Orthographic3 { matrix }
    }

    /// The homogeneous matrix (a copy: Cairo values are passed by value). Exact. Upstream:
    /// `as_matrix`.
    #[inline(always)]
    fn as_matrix(self: Orthographic3<T>) -> Matrix4<T> {
        self.matrix
    }

    /// The homogeneous matrix. Exact. Upstream: `into_inner`.
    #[inline(always)]
    fn into_inner(self: Orthographic3<T>) -> Matrix4<T> {
        self.matrix
    }

    /// The homogeneous matrix: `into_inner` (deprecated upstream, "use `.into_inner()`
    /// instead"). Exact. Upstream: `unwrap`.
    #[inline(always)]
    fn unwrap(self: Orthographic3<T>) -> Matrix4<T> {
        self.matrix
    }

    /// The homogeneous matrix. Exact. Upstream: `to_homogeneous`.
    #[inline(always)]
    fn to_homogeneous(self: Orthographic3<T>) -> Matrix4<T> {
        self.matrix
    }

    /// The inverse of the projection matrix, in closed form: `m11 = 1 / m11`, `m22 = 1 / m22`,
    /// `m33 = 1 / m33`, `m14 = -m14 / m11`, `m24 = -m24 / m22`, `m34 = -m34 / m33` and the other
    /// entries of `self`. Every entry is ONE correctly rounded quotient: upstream multiplies
    /// `-m14` by the reciprocal `1 / m11` (two roundings, harmless in `f64`), a variant kept in
    /// the benchmarks. Panics on a zero diagonal entry and on overflow. Upstream: `inverse`.
    #[inline(always)]
    fn inverse(self: Orthographic3<T>) -> Matrix4<T> {
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
            m33: R::recip(m.m33),
            m43: m.m43,
            m14: R::div(-m.m14, m.m11),
            m24: R::div(-m.m24, m.m22),
            m34: R::div(-m.m34, m.m33),
            m44: m.m44,
        }
    }

    /// The left offset of the view cuboid, `(-1 - m14) / m11`, correctly rounded. Upstream:
    /// `left`.
    #[inline(always)]
    fn left(self: Orthographic3<T>) -> T {
        let m = self.matrix;
        R::div(-R::one() - m.m14, m.m11)
    }

    /// The right offset of the view cuboid, `(1 - m14) / m11`, correctly rounded. Upstream:
    /// `right`.
    #[inline(always)]
    fn right(self: Orthographic3<T>) -> T {
        let m = self.matrix;
        R::div(R::one() - m.m14, m.m11)
    }

    /// The bottom offset of the view cuboid, `(-1 - m24) / m22`, correctly rounded. Upstream:
    /// `bottom`.
    #[inline(always)]
    fn bottom(self: Orthographic3<T>) -> T {
        let m = self.matrix;
        R::div(-R::one() - m.m24, m.m22)
    }

    /// The top offset of the view cuboid, `(1 - m24) / m22`, correctly rounded. Upstream: `top`.
    #[inline(always)]
    fn top(self: Orthographic3<T>) -> T {
        let m = self.matrix;
        R::div(R::one() - m.m24, m.m22)
    }

    /// The near plane offset, `(1 + m34) / m33`, correctly rounded. Upstream: `znear`.
    #[inline(always)]
    fn znear(self: Orthographic3<T>) -> T {
        let m = self.matrix;
        R::div(R::one() + m.m34, m.m33)
    }

    /// The far plane offset, `(-1 + m34) / m33`, correctly rounded. Upstream: `zfar`.
    #[inline(always)]
    fn zfar(self: Orthographic3<T>) -> T {
        let m = self.matrix;
        R::div(-R::one() + m.m34, m.m33)
    }

    /// Projects the point `p`: `(m11 · x + m14, m22 · y + m24, m33 · z + m34)`, one fused
    /// `Real::mul_add` (a single floor) per coordinate. Panics on overflow. Upstream:
    /// `project_point`.
    #[inline(always)]
    fn project_point(self: Orthographic3<T>, p: Point3<T>) -> Point3<T> {
        let m = self.matrix;
        Point3 {
            x: R::mul_add(m.m11, p.x, m.m14),
            y: R::mul_add(m.m22, p.y, m.m24),
            z: R::mul_add(m.m33, p.z, m.m34),
        }
    }

    /// Un-projects the point `p`: `((x - m14) / m11, (y - m24) / m22, (z - m34) / m33)`, one
    /// correctly rounded quotient per coordinate. Panics on a zero diagonal entry and on overflow.
    /// Upstream: `unproject_point`.
    #[inline(always)]
    fn unproject_point(self: Orthographic3<T>, p: Point3<T>) -> Point3<T> {
        let m = self.matrix;
        Point3 {
            x: R::div(p.x - m.m14, m.m11),
            y: R::div(p.y - m.m24, m.m22),
            z: R::div(p.z - m.m34, m.m33),
        }
    }

    /// Projects the vector `v`: `(m11 · x, m22 · y, m33 · z)`, one floored product per component
    /// (vectors ignore the translation). Panics on overflow. Upstream: `project_vector`.
    #[inline(always)]
    fn project_vector(self: Orthographic3<T>, v: Vector3<T>) -> Vector3<T> {
        let m = self.matrix;
        Vector3 { x: m.m11 * v.x, y: m.m22 * v.y, z: m.m33 * v.z }
    }

    /// Sets the left offset, keeping the right one (`set_left_and_right(left, self.right())`).
    /// Upstream: `set_left`.
    #[inline(always)]
    fn set_left(ref self: Orthographic3<T>, left: T) {
        let right = Self::right(self);
        Self::set_left_and_right(ref self, left, right);
    }

    /// Sets the right offset, keeping the left one. Upstream: `set_right`.
    #[inline(always)]
    fn set_right(ref self: Orthographic3<T>, right: T) {
        let left = Self::left(self);
        Self::set_left_and_right(ref self, left, right);
    }

    /// Sets the bottom offset, keeping the top one. Upstream: `set_bottom`.
    #[inline(always)]
    fn set_bottom(ref self: Orthographic3<T>, bottom: T) {
        let top = Self::top(self);
        Self::set_bottom_and_top(ref self, bottom, top);
    }

    /// Sets the top offset, keeping the bottom one. Upstream: `set_top`.
    #[inline(always)]
    fn set_top(ref self: Orthographic3<T>, top: T) {
        let bottom = Self::bottom(self);
        Self::set_bottom_and_top(ref self, bottom, top);
    }

    /// Sets the near plane, keeping the far one. Upstream: `set_znear`.
    #[inline(always)]
    fn set_znear(ref self: Orthographic3<T>, znear: T) {
        let zfar = Self::zfar(self);
        Self::set_znear_and_zfar(ref self, znear, zfar);
    }

    /// Sets the far plane, keeping the near one. Upstream: `set_zfar`.
    #[inline(always)]
    fn set_zfar(ref self: Orthographic3<T>, zfar: T) {
        let znear = Self::znear(self);
        Self::set_znear_and_zfar(ref self, znear, zfar);
    }

    /// Sets the left and right offsets: `m11 = 2 / (right - left)`, `m14 = -(right + left) /
    /// (right - left)` (correctly rounded). Panics with `errors::LEFT_EQUALS_RIGHT` when `left ==
    /// right`, and on overflow. Upstream: `set_left_and_right`.
    #[inline(always)]
    fn set_left_and_right(ref self: Orthographic3<T>, left: T, right: T) {
        assert(left != right, errors::LEFT_EQUALS_RIGHT);
        let (m11, m14) = Orthographic3InternalTrait::axis_entries(left, right);
        let mut m = self.matrix;
        m.m11 = m11;
        m.m14 = m14;
        self.matrix = m;
    }

    /// Sets the bottom and top offsets: `m22 = 2 / (top - bottom)`, `m24 = -(top + bottom) / (top
    /// - bottom)` (correctly rounded). Panics with `errors::BOTTOM_EQUALS_TOP` when `bottom ==
    /// top`, and on overflow. Upstream: `set_bottom_and_top`.
    #[inline(always)]
    fn set_bottom_and_top(ref self: Orthographic3<T>, bottom: T, top: T) {
        assert(bottom != top, errors::BOTTOM_EQUALS_TOP);
        let (m22, m24) = Orthographic3InternalTrait::axis_entries(bottom, top);
        let mut m = self.matrix;
        m.m22 = m22;
        m.m24 = m24;
        self.matrix = m;
    }

    /// Sets the near and far planes: `m33 = -2 / (zfar - znear)`, `m34 = -(zfar + znear) / (zfar -
    /// znear)` (correctly rounded; `-2 / d` is `-(2 / d)` exactly, rounding to nearest being
    /// symmetric). Panics with `errors::SUPERIMPOSED_PLANES` when `znear == zfar`, and on
    /// overflow. Upstream: `set_znear_and_zfar`.
    #[inline(always)]
    fn set_znear_and_zfar(ref self: Orthographic3<T>, znear: T, zfar: T) {
        assert(zfar != znear, errors::SUPERIMPOSED_PLANES);
        let (m33, m34) = Orthographic3InternalTrait::axis_entries(znear, zfar);
        let mut m = self.matrix;
        m.m33 = -m33;
        m.m34 = m34;
        self.matrix = m;
    }
}

/// The operations of `Orthographic3<T>` that need trigonometry (`Real` + `Transcendental`).
#[generate_trait]
pub impl Orthographic3AngleImpl<
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
> of Orthographic3AngleTrait<T> {
    /// The orthographic projection of the frustum section at the far plane of a perspective of
    /// aspect `aspect` and vertical field of view `vfov`: `width = zfar · tan(vfov · 0.5)`,
    /// `height = width / aspect`, then `new(-width · 0.5, width · 0.5, -height · 0.5, height ·
    /// 0.5, znear, zfar)` (upstream's operations: the products by `0.5` are floored products of
    /// the negated / plain operands, the quotient correctly rounded; `tan` is
    /// `Transcendental::tan`). Panics with `errors::FAR_EQUALS_NEAR` when `znear == zfar`, with
    /// `errors::ZERO_ASPECT` when `aspect` is `relative_eq` to zero (within 1 ulp), and like
    /// `new`. Upstream: `Orthographic3::from_fov`.
    fn from_fov(aspect: T, vfov: T, znear: T, zfar: T) -> Orthographic3<T> {
        assert(znear != zfar, errors::FAR_EQUALS_NEAR);
        assert(
            !ApproxEqTrait::relative_eq(aspect, R::zero(), 1, R::default_epsilon()),
            errors::ZERO_ASPECT,
        );
        let half = R::from_ratio(1, 2);
        let width = zfar * Tr::tan(vfov * half);
        let height = R::div(width, aspect);
        Orthographic3Trait::new(
            -width * half, width * half, -height * half, height * half, znear, zfar,
        )
    }
}

/// Upstream's `Matrix4::new_orthographic` (`base/cg.rs`).
#[generate_trait]
pub impl Matrix4OrthographicImpl<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Add<T>, +Sub<T>, +Mul<T>, +Neg<T>, +PartialEq<T>,
> of Matrix4OrthographicTrait<T> {
    /// The homogeneous matrix of an orthographic projection: `Orthographic3Trait::new(left,
    /// right, bottom, top, znear, zfar).into_inner()` (same rounding and panics). Upstream:
    /// `Matrix4::new_orthographic`.
    #[inline(always)]
    fn new_orthographic(left: T, right: T, bottom: T, top: T, znear: T, zfar: T) -> Matrix4<T> {
        let o: Orthographic3<T> = Orthographic3Trait::new(left, right, bottom, top, znear, zfar);
        o.matrix
    }
}

/// `o.into()`: the homogeneous matrix (`into_inner`). Exact. Upstream: `From<Orthographic3> for
/// Matrix4`.
pub impl Matrix4FromOrthographic3<T> of Into<Orthographic3<T>, Matrix4<T>> {
    #[inline(always)]
    fn into(self: Orthographic3<T>) -> Matrix4<T> {
        self.matrix
    }
}
