//! The kernel trait `MatrixInfSup` of the crate-root `inf` / `sup` / `inf_sup`
//! (`nalgebra::root::MatrixInfSup`, re-exported by `root`) and its impls, one per static shape
//! (their public paths are the shapes' modules, `nalgebra::base::matrix3::Matrix3InfSup`).
//!
//! WP 9-NS11a: the impls call the shapes' methods (`nalgebra_static3`..`nalgebra_static6_wide`),
//! so they cannot sit in their type's module (`nalgebra_core`, `nalgebra_shapes5`,
//! `nalgebra_shapes6`, below the method crates); Cairo finds them in the trait's module, this one
//! (docs/SPLIT.md §1, §3.2). The module is crate-visible so that it adds no public path: 0.1.0's
//! surface is exactly kept (`tools/split/public_paths.py --check`).

use nalgebra_core::base::matrix1::Matrix1;
use nalgebra_core::base::matrix2::Matrix2;
use nalgebra_core::base::matrix2x3::Matrix2x3;
use nalgebra_core::base::matrix2x4::Matrix2x4;
use nalgebra_core::base::matrix3x2::Matrix3x2;
use nalgebra_core::base::matrix3x4::Matrix3x4;
use nalgebra_core::base::matrix4x2::Matrix4x2;
use nalgebra_core::base::matrix4x3::Matrix4x3;
use nalgebra_core::base::row_vector2::RowVector2;
use nalgebra_core::base::row_vector3::RowVector3;
use nalgebra_core::base::row_vector4::RowVector4;
use nalgebra_core::base::vector2::Vector2;
use nalgebra_core::base::vector3::Vector3;
use nalgebra_core::base::vector4::Vector4;
use nalgebra_shapes5::base::matrix2x5::Matrix2x5;
use nalgebra_shapes5::base::matrix3x5::Matrix3x5;
use nalgebra_shapes5::base::matrix4x5::Matrix4x5;
use nalgebra_shapes5::base::matrix5::Matrix5;
use nalgebra_shapes5::base::matrix5x2::Matrix5x2;
use nalgebra_shapes5::base::matrix5x3::Matrix5x3;
use nalgebra_shapes5::base::matrix5x4::Matrix5x4;
use nalgebra_shapes5::base::row_vector5::RowVector5;
use nalgebra_shapes5::base::vector5::Vector5;
use nalgebra_shapes6::base::matrix2x6::Matrix2x6;
use nalgebra_shapes6::base::matrix3x6::Matrix3x6;
use nalgebra_shapes6::base::matrix4x6::Matrix4x6;
use nalgebra_shapes6::base::matrix5x6::Matrix5x6;
use nalgebra_shapes6::base::matrix6::Matrix6;
use nalgebra_shapes6::base::matrix6x2::Matrix6x2;
use nalgebra_shapes6::base::matrix6x3::Matrix6x3;
use nalgebra_shapes6::base::matrix6x4::Matrix6x4;
use nalgebra_shapes6::base::matrix6x5::Matrix6x5;
use nalgebra_shapes6::base::row_vector6::RowVector6;
use nalgebra_shapes6::base::vector6::Vector6;
use simba::scalar::Real;
use crate::base::matrix1::Matrix1Trait;
use crate::base::matrix2::Matrix2Trait;
use crate::base::matrix2x3::Matrix2x3Trait;
use crate::base::matrix2x4::Matrix2x4Trait;
use crate::base::matrix2x5::Matrix2x5Trait;
use crate::base::matrix2x6::Matrix2x6Trait;
use crate::base::matrix3::Matrix3Trait;
use crate::base::matrix3x2::Matrix3x2Trait;
use crate::base::matrix3x4::Matrix3x4Trait;
use crate::base::matrix3x5::Matrix3x5Trait;
use crate::base::matrix3x6::Matrix3x6Trait;
use crate::base::matrix4::Matrix4Trait;
use crate::base::matrix4x2::Matrix4x2Trait;
use crate::base::matrix4x3::Matrix4x3Trait;
use crate::base::matrix4x5::Matrix4x5Trait;
use crate::base::matrix4x6::Matrix4x6Trait;
use crate::base::matrix5::Matrix5Trait;
use crate::base::matrix5x2::Matrix5x2Trait;
use crate::base::matrix5x3::Matrix5x3Trait;
use crate::base::matrix5x4::Matrix5x4Trait;
use crate::base::matrix5x6::Matrix5x6Trait;
use crate::base::matrix6::Matrix6Trait;
use crate::base::matrix6x2::Matrix6x2Trait;
use crate::base::matrix6x3::Matrix6x3Trait;
use crate::base::matrix6x4::Matrix6x4Trait;
use crate::base::matrix6x5::Matrix6x5Trait;
use crate::base::row_vector2::RowVector2Trait;
use crate::base::row_vector3::RowVector3Trait;
use crate::base::row_vector4::RowVector4Trait;
use crate::base::row_vector5::RowVector5Trait;
use crate::base::row_vector6::RowVector6Trait;
use crate::base::vector2::Vector2Trait;
use crate::base::vector3::Vector3Trait;
use crate::base::vector4::Vector4Trait;

/// The kernel of `inf` / `sup` / `inf_sup` (upstream's `SimdPartialOrd` bound on the
/// components): implemented by each static shape in its module, as its methods of the same
/// names. Static functions (no `self`), so they never compete with the shapes' methods.
pub trait MatrixInfSup<M> {
    /// The component-wise minimum.
    fn inf(a: M, b: M) -> M;
    /// The component-wise maximum.
    fn sup(a: M, b: M) -> M;
    /// `(inf(a, b), sup(a, b))`.
    fn inf_sup(a: M, b: M) -> (M, M);
}
use nalgebra_core::base::matrix3::Matrix3;
use nalgebra_core::base::matrix4::Matrix4;
use nalgebra_static5::base::vector5::Vector5Trait;
use nalgebra_static6_tall::base::vector6::Vector6Trait;

// crate-map: generated items (tools/split/cratemap.py) [shapegen]
// crate-map: from base/matrix1.cairo
// crate-map: from base/matrix2.cairo
// crate-map: from base/matrix2x3.cairo
// crate-map: from base/matrix2x4.cairo
// crate-map: from base/matrix2x5.cairo
// crate-map: from base/matrix2x6.cairo
// crate-map: from base/matrix3.cairo
// crate-map: from base/matrix3x2.cairo
// crate-map: from base/matrix3x4.cairo
// crate-map: from base/matrix3x5.cairo
// crate-map: from base/matrix3x6.cairo
// crate-map: from base/matrix4.cairo
// crate-map: from base/matrix4x2.cairo
// crate-map: from base/matrix4x3.cairo
// crate-map: from base/matrix4x5.cairo
// crate-map: from base/matrix4x6.cairo
// crate-map: from base/matrix5.cairo
// crate-map: from base/matrix5x2.cairo
// crate-map: from base/matrix5x3.cairo
// crate-map: from base/matrix5x4.cairo
// crate-map: from base/matrix5x6.cairo
// crate-map: from base/matrix6.cairo
// crate-map: from base/matrix6x2.cairo
// crate-map: from base/matrix6x3.cairo
// crate-map: from base/matrix6x4.cairo
// crate-map: from base/matrix6x5.cairo
// crate-map: from base/row_vector2.cairo
// crate-map: from base/row_vector3.cairo
// crate-map: from base/row_vector4.cairo
// crate-map: from base/row_vector5.cairo
// crate-map: from base/row_vector6.cairo
// crate-map: from base/vector2.cairo
// crate-map: from base/vector3.cairo
// crate-map: from base/vector4.cairo
// crate-map: from base/vector5.cairo
// crate-map: from base/vector6.cairo
/// The kernel of the crate-root `nalgebra::inf` / `sup` / `inf_sup` on `Matrix1`: the shape's
/// `inf` / `sup` / `inf_sup`.
pub impl Matrix1InfSup<
    T,
    impl R: Real<T>,
    +Copy<T>,
    +Drop<T>,
    +Drop<R::Wide>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
    +PartialEq<T>,
    +PartialOrd<T>,
> of crate::root::MatrixInfSup<Matrix1<T>> {
    #[inline(always)]
    fn inf(a: Matrix1<T>, b: Matrix1<T>) -> Matrix1<T> {
        Matrix1Trait::inf(a, b)
    }
    #[inline(always)]
    fn sup(a: Matrix1<T>, b: Matrix1<T>) -> Matrix1<T> {
        Matrix1Trait::sup(a, b)
    }
    #[inline(always)]
    fn inf_sup(a: Matrix1<T>, b: Matrix1<T>) -> (Matrix1<T>, Matrix1<T>) {
        Matrix1Trait::inf_sup(a, b)
    }
}

/// The kernel of the crate-root `nalgebra::inf` / `sup` / `inf_sup` on `RowVector2`: the shape's
/// `inf` / `sup` / `inf_sup`.
pub impl RowVector2InfSup<
    T,
    impl R: Real<T>,
    +Copy<T>,
    +Drop<T>,
    +Drop<R::Wide>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
    +PartialEq<T>,
    +PartialOrd<T>,
> of crate::root::MatrixInfSup<RowVector2<T>> {
    #[inline(always)]
    fn inf(a: RowVector2<T>, b: RowVector2<T>) -> RowVector2<T> {
        RowVector2Trait::inf(a, b)
    }
    #[inline(always)]
    fn sup(a: RowVector2<T>, b: RowVector2<T>) -> RowVector2<T> {
        RowVector2Trait::sup(a, b)
    }
    #[inline(always)]
    fn inf_sup(a: RowVector2<T>, b: RowVector2<T>) -> (RowVector2<T>, RowVector2<T>) {
        RowVector2Trait::inf_sup(a, b)
    }
}

/// The kernel of the crate-root `nalgebra::inf` / `sup` / `inf_sup` on `RowVector3`: the shape's
/// `inf` / `sup` / `inf_sup`.
pub impl RowVector3InfSup<
    T,
    impl R: Real<T>,
    +Copy<T>,
    +Drop<T>,
    +Drop<R::Wide>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
    +PartialEq<T>,
    +PartialOrd<T>,
> of crate::root::MatrixInfSup<RowVector3<T>> {
    #[inline(always)]
    fn inf(a: RowVector3<T>, b: RowVector3<T>) -> RowVector3<T> {
        RowVector3Trait::inf(a, b)
    }
    #[inline(always)]
    fn sup(a: RowVector3<T>, b: RowVector3<T>) -> RowVector3<T> {
        RowVector3Trait::sup(a, b)
    }
    #[inline(always)]
    fn inf_sup(a: RowVector3<T>, b: RowVector3<T>) -> (RowVector3<T>, RowVector3<T>) {
        RowVector3Trait::inf_sup(a, b)
    }
}

/// The kernel of the crate-root `nalgebra::inf` / `sup` / `inf_sup` on `RowVector4`: the shape's
/// `inf` / `sup` / `inf_sup`.
pub impl RowVector4InfSup<
    T,
    impl R: Real<T>,
    +Copy<T>,
    +Drop<T>,
    +Drop<R::Wide>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
    +PartialEq<T>,
    +PartialOrd<T>,
> of crate::root::MatrixInfSup<RowVector4<T>> {
    #[inline(always)]
    fn inf(a: RowVector4<T>, b: RowVector4<T>) -> RowVector4<T> {
        RowVector4Trait::inf(a, b)
    }
    #[inline(always)]
    fn sup(a: RowVector4<T>, b: RowVector4<T>) -> RowVector4<T> {
        RowVector4Trait::sup(a, b)
    }
    #[inline(always)]
    fn inf_sup(a: RowVector4<T>, b: RowVector4<T>) -> (RowVector4<T>, RowVector4<T>) {
        RowVector4Trait::inf_sup(a, b)
    }
}

/// The kernel of the crate-root `nalgebra::inf` / `sup` / `inf_sup` on `RowVector5`: the shape's
/// `inf` / `sup` / `inf_sup`.
pub impl RowVector5InfSup<
    T,
    impl R: Real<T>,
    +Copy<T>,
    +Drop<T>,
    +Drop<R::Wide>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
    +PartialEq<T>,
    +PartialOrd<T>,
> of crate::root::MatrixInfSup<RowVector5<T>> {
    #[inline(always)]
    fn inf(a: RowVector5<T>, b: RowVector5<T>) -> RowVector5<T> {
        RowVector5Trait::inf(a, b)
    }
    #[inline(always)]
    fn sup(a: RowVector5<T>, b: RowVector5<T>) -> RowVector5<T> {
        RowVector5Trait::sup(a, b)
    }
    #[inline(always)]
    fn inf_sup(a: RowVector5<T>, b: RowVector5<T>) -> (RowVector5<T>, RowVector5<T>) {
        RowVector5Trait::inf_sup(a, b)
    }
}

/// The kernel of the crate-root `nalgebra::inf` / `sup` / `inf_sup` on `RowVector6`: the shape's
/// `inf` / `sup` / `inf_sup`.
pub impl RowVector6InfSup<
    T,
    impl R: Real<T>,
    +Copy<T>,
    +Drop<T>,
    +Drop<R::Wide>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
    +PartialEq<T>,
    +PartialOrd<T>,
> of crate::root::MatrixInfSup<RowVector6<T>> {
    #[inline(always)]
    fn inf(a: RowVector6<T>, b: RowVector6<T>) -> RowVector6<T> {
        RowVector6Trait::inf(a, b)
    }
    #[inline(always)]
    fn sup(a: RowVector6<T>, b: RowVector6<T>) -> RowVector6<T> {
        RowVector6Trait::sup(a, b)
    }
    #[inline(always)]
    fn inf_sup(a: RowVector6<T>, b: RowVector6<T>) -> (RowVector6<T>, RowVector6<T>) {
        RowVector6Trait::inf_sup(a, b)
    }
}

/// The kernel of the crate-root `nalgebra::inf` / `sup` / `inf_sup` on `Vector2`: the shape's
/// `inf` / `sup` / `inf_sup`.
pub impl Vector2InfSup<
    T,
    impl R: Real<T>,
    +Copy<T>,
    +Drop<T>,
    +Drop<R::Wide>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
    +PartialEq<T>,
    +PartialOrd<T>,
> of crate::root::MatrixInfSup<Vector2<T>> {
    #[inline(always)]
    fn inf(a: Vector2<T>, b: Vector2<T>) -> Vector2<T> {
        Vector2Trait::inf(a, b)
    }
    #[inline(always)]
    fn sup(a: Vector2<T>, b: Vector2<T>) -> Vector2<T> {
        Vector2Trait::sup(a, b)
    }
    #[inline(always)]
    fn inf_sup(a: Vector2<T>, b: Vector2<T>) -> (Vector2<T>, Vector2<T>) {
        Vector2Trait::inf_sup(a, b)
    }
}

/// The kernel of the crate-root `nalgebra::inf` / `sup` / `inf_sup` on `Matrix2`: the shape's
/// `inf` / `sup` / `inf_sup`.
pub impl Matrix2InfSup<
    T,
    impl R: Real<T>,
    +Copy<T>,
    +Drop<T>,
    +Drop<R::Wide>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
    +PartialEq<T>,
    +PartialOrd<T>,
> of crate::root::MatrixInfSup<Matrix2<T>> {
    #[inline(always)]
    fn inf(a: Matrix2<T>, b: Matrix2<T>) -> Matrix2<T> {
        Matrix2Trait::inf(a, b)
    }
    #[inline(always)]
    fn sup(a: Matrix2<T>, b: Matrix2<T>) -> Matrix2<T> {
        Matrix2Trait::sup(a, b)
    }
    #[inline(always)]
    fn inf_sup(a: Matrix2<T>, b: Matrix2<T>) -> (Matrix2<T>, Matrix2<T>) {
        Matrix2Trait::inf_sup(a, b)
    }
}

/// The kernel of the crate-root `nalgebra::inf` / `sup` / `inf_sup` on `Matrix2x3`: the shape's
/// `inf` / `sup` / `inf_sup`.
pub impl Matrix2x3InfSup<
    T,
    impl R: Real<T>,
    +Copy<T>,
    +Drop<T>,
    +Drop<R::Wide>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
    +PartialEq<T>,
    +PartialOrd<T>,
> of crate::root::MatrixInfSup<Matrix2x3<T>> {
    #[inline(always)]
    fn inf(a: Matrix2x3<T>, b: Matrix2x3<T>) -> Matrix2x3<T> {
        Matrix2x3Trait::inf(a, b)
    }
    #[inline(always)]
    fn sup(a: Matrix2x3<T>, b: Matrix2x3<T>) -> Matrix2x3<T> {
        Matrix2x3Trait::sup(a, b)
    }
    #[inline(always)]
    fn inf_sup(a: Matrix2x3<T>, b: Matrix2x3<T>) -> (Matrix2x3<T>, Matrix2x3<T>) {
        Matrix2x3Trait::inf_sup(a, b)
    }
}

/// The kernel of the crate-root `nalgebra::inf` / `sup` / `inf_sup` on `Matrix2x4`: the shape's
/// `inf` / `sup` / `inf_sup`.
pub impl Matrix2x4InfSup<
    T,
    impl R: Real<T>,
    +Copy<T>,
    +Drop<T>,
    +Drop<R::Wide>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
    +PartialEq<T>,
    +PartialOrd<T>,
> of crate::root::MatrixInfSup<Matrix2x4<T>> {
    #[inline(always)]
    fn inf(a: Matrix2x4<T>, b: Matrix2x4<T>) -> Matrix2x4<T> {
        Matrix2x4Trait::inf(a, b)
    }
    #[inline(always)]
    fn sup(a: Matrix2x4<T>, b: Matrix2x4<T>) -> Matrix2x4<T> {
        Matrix2x4Trait::sup(a, b)
    }
    #[inline(always)]
    fn inf_sup(a: Matrix2x4<T>, b: Matrix2x4<T>) -> (Matrix2x4<T>, Matrix2x4<T>) {
        Matrix2x4Trait::inf_sup(a, b)
    }
}

/// The kernel of the crate-root `nalgebra::inf` / `sup` / `inf_sup` on `Matrix2x5`: the shape's
/// `inf` / `sup` / `inf_sup`.
pub impl Matrix2x5InfSup<
    T,
    impl R: Real<T>,
    +Copy<T>,
    +Drop<T>,
    +Drop<R::Wide>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
    +PartialEq<T>,
    +PartialOrd<T>,
> of crate::root::MatrixInfSup<Matrix2x5<T>> {
    #[inline(always)]
    fn inf(a: Matrix2x5<T>, b: Matrix2x5<T>) -> Matrix2x5<T> {
        Matrix2x5Trait::inf(a, b)
    }
    #[inline(always)]
    fn sup(a: Matrix2x5<T>, b: Matrix2x5<T>) -> Matrix2x5<T> {
        Matrix2x5Trait::sup(a, b)
    }
    #[inline(always)]
    fn inf_sup(a: Matrix2x5<T>, b: Matrix2x5<T>) -> (Matrix2x5<T>, Matrix2x5<T>) {
        Matrix2x5Trait::inf_sup(a, b)
    }
}

/// The kernel of the crate-root `nalgebra::inf` / `sup` / `inf_sup` on `Matrix2x6`: the shape's
/// `inf` / `sup` / `inf_sup`.
pub impl Matrix2x6InfSup<
    T,
    impl R: Real<T>,
    +Copy<T>,
    +Drop<T>,
    +Drop<R::Wide>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
    +PartialEq<T>,
    +PartialOrd<T>,
> of crate::root::MatrixInfSup<Matrix2x6<T>> {
    #[inline(always)]
    fn inf(a: Matrix2x6<T>, b: Matrix2x6<T>) -> Matrix2x6<T> {
        Matrix2x6Trait::inf(a, b)
    }
    #[inline(always)]
    fn sup(a: Matrix2x6<T>, b: Matrix2x6<T>) -> Matrix2x6<T> {
        Matrix2x6Trait::sup(a, b)
    }
    #[inline(always)]
    fn inf_sup(a: Matrix2x6<T>, b: Matrix2x6<T>) -> (Matrix2x6<T>, Matrix2x6<T>) {
        Matrix2x6Trait::inf_sup(a, b)
    }
}

/// The kernel of the crate-root `nalgebra::inf` / `sup` / `inf_sup` on `Vector3`: the shape's
/// `inf` / `sup` / `inf_sup`.
pub impl Vector3InfSup<
    T,
    impl R: Real<T>,
    +Copy<T>,
    +Drop<T>,
    +Drop<R::Wide>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
    +PartialEq<T>,
    +PartialOrd<T>,
> of crate::root::MatrixInfSup<Vector3<T>> {
    #[inline(always)]
    fn inf(a: Vector3<T>, b: Vector3<T>) -> Vector3<T> {
        Vector3Trait::inf(a, b)
    }
    #[inline(always)]
    fn sup(a: Vector3<T>, b: Vector3<T>) -> Vector3<T> {
        Vector3Trait::sup(a, b)
    }
    #[inline(always)]
    fn inf_sup(a: Vector3<T>, b: Vector3<T>) -> (Vector3<T>, Vector3<T>) {
        Vector3Trait::inf_sup(a, b)
    }
}

/// The kernel of the crate-root `nalgebra::inf` / `sup` / `inf_sup` on `Matrix3x2`: the shape's
/// `inf` / `sup` / `inf_sup`.
pub impl Matrix3x2InfSup<
    T,
    impl R: Real<T>,
    +Copy<T>,
    +Drop<T>,
    +Drop<R::Wide>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
    +PartialEq<T>,
    +PartialOrd<T>,
> of crate::root::MatrixInfSup<Matrix3x2<T>> {
    #[inline(always)]
    fn inf(a: Matrix3x2<T>, b: Matrix3x2<T>) -> Matrix3x2<T> {
        Matrix3x2Trait::inf(a, b)
    }
    #[inline(always)]
    fn sup(a: Matrix3x2<T>, b: Matrix3x2<T>) -> Matrix3x2<T> {
        Matrix3x2Trait::sup(a, b)
    }
    #[inline(always)]
    fn inf_sup(a: Matrix3x2<T>, b: Matrix3x2<T>) -> (Matrix3x2<T>, Matrix3x2<T>) {
        Matrix3x2Trait::inf_sup(a, b)
    }
}

/// The kernel of the crate-root `nalgebra::inf` / `sup` / `inf_sup` on `Matrix3`: the shape's
/// `inf` / `sup` / `inf_sup`.
pub impl Matrix3InfSup<
    T,
    impl R: Real<T>,
    +Copy<T>,
    +Drop<T>,
    +Drop<R::Wide>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
    +PartialEq<T>,
    +PartialOrd<T>,
> of crate::root::MatrixInfSup<Matrix3<T>> {
    #[inline(always)]
    fn inf(a: Matrix3<T>, b: Matrix3<T>) -> Matrix3<T> {
        Matrix3Trait::inf(a, b)
    }
    #[inline(always)]
    fn sup(a: Matrix3<T>, b: Matrix3<T>) -> Matrix3<T> {
        Matrix3Trait::sup(a, b)
    }
    #[inline(always)]
    fn inf_sup(a: Matrix3<T>, b: Matrix3<T>) -> (Matrix3<T>, Matrix3<T>) {
        Matrix3Trait::inf_sup(a, b)
    }
}

/// The kernel of the crate-root `nalgebra::inf` / `sup` / `inf_sup` on `Matrix3x4`: the shape's
/// `inf` / `sup` / `inf_sup`.
pub impl Matrix3x4InfSup<
    T,
    impl R: Real<T>,
    +Copy<T>,
    +Drop<T>,
    +Drop<R::Wide>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
    +PartialEq<T>,
    +PartialOrd<T>,
> of crate::root::MatrixInfSup<Matrix3x4<T>> {
    #[inline(always)]
    fn inf(a: Matrix3x4<T>, b: Matrix3x4<T>) -> Matrix3x4<T> {
        Matrix3x4Trait::inf(a, b)
    }
    #[inline(always)]
    fn sup(a: Matrix3x4<T>, b: Matrix3x4<T>) -> Matrix3x4<T> {
        Matrix3x4Trait::sup(a, b)
    }
    #[inline(always)]
    fn inf_sup(a: Matrix3x4<T>, b: Matrix3x4<T>) -> (Matrix3x4<T>, Matrix3x4<T>) {
        Matrix3x4Trait::inf_sup(a, b)
    }
}

/// The kernel of the crate-root `nalgebra::inf` / `sup` / `inf_sup` on `Matrix3x5`: the shape's
/// `inf` / `sup` / `inf_sup`.
pub impl Matrix3x5InfSup<
    T,
    impl R: Real<T>,
    +Copy<T>,
    +Drop<T>,
    +Drop<R::Wide>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
    +PartialEq<T>,
    +PartialOrd<T>,
> of crate::root::MatrixInfSup<Matrix3x5<T>> {
    #[inline(always)]
    fn inf(a: Matrix3x5<T>, b: Matrix3x5<T>) -> Matrix3x5<T> {
        Matrix3x5Trait::inf(a, b)
    }
    #[inline(always)]
    fn sup(a: Matrix3x5<T>, b: Matrix3x5<T>) -> Matrix3x5<T> {
        Matrix3x5Trait::sup(a, b)
    }
    #[inline(always)]
    fn inf_sup(a: Matrix3x5<T>, b: Matrix3x5<T>) -> (Matrix3x5<T>, Matrix3x5<T>) {
        Matrix3x5Trait::inf_sup(a, b)
    }
}

/// The kernel of the crate-root `nalgebra::inf` / `sup` / `inf_sup` on `Matrix3x6`: the shape's
/// `inf` / `sup` / `inf_sup`.
pub impl Matrix3x6InfSup<
    T,
    impl R: Real<T>,
    +Copy<T>,
    +Drop<T>,
    +Drop<R::Wide>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
    +PartialEq<T>,
    +PartialOrd<T>,
> of crate::root::MatrixInfSup<Matrix3x6<T>> {
    #[inline(always)]
    fn inf(a: Matrix3x6<T>, b: Matrix3x6<T>) -> Matrix3x6<T> {
        Matrix3x6Trait::inf(a, b)
    }
    #[inline(always)]
    fn sup(a: Matrix3x6<T>, b: Matrix3x6<T>) -> Matrix3x6<T> {
        Matrix3x6Trait::sup(a, b)
    }
    #[inline(always)]
    fn inf_sup(a: Matrix3x6<T>, b: Matrix3x6<T>) -> (Matrix3x6<T>, Matrix3x6<T>) {
        Matrix3x6Trait::inf_sup(a, b)
    }
}

/// The kernel of the crate-root `nalgebra::inf` / `sup` / `inf_sup` on `Vector4`: the shape's
/// `inf` / `sup` / `inf_sup`.
pub impl Vector4InfSup<
    T,
    impl R: Real<T>,
    +Copy<T>,
    +Drop<T>,
    +Drop<R::Wide>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
    +PartialEq<T>,
    +PartialOrd<T>,
> of crate::root::MatrixInfSup<Vector4<T>> {
    #[inline(always)]
    fn inf(a: Vector4<T>, b: Vector4<T>) -> Vector4<T> {
        Vector4Trait::inf(a, b)
    }
    #[inline(always)]
    fn sup(a: Vector4<T>, b: Vector4<T>) -> Vector4<T> {
        Vector4Trait::sup(a, b)
    }
    #[inline(always)]
    fn inf_sup(a: Vector4<T>, b: Vector4<T>) -> (Vector4<T>, Vector4<T>) {
        Vector4Trait::inf_sup(a, b)
    }
}

/// The kernel of the crate-root `nalgebra::inf` / `sup` / `inf_sup` on `Matrix4x2`: the shape's
/// `inf` / `sup` / `inf_sup`.
pub impl Matrix4x2InfSup<
    T,
    impl R: Real<T>,
    +Copy<T>,
    +Drop<T>,
    +Drop<R::Wide>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
    +PartialEq<T>,
    +PartialOrd<T>,
> of crate::root::MatrixInfSup<Matrix4x2<T>> {
    #[inline(always)]
    fn inf(a: Matrix4x2<T>, b: Matrix4x2<T>) -> Matrix4x2<T> {
        Matrix4x2Trait::inf(a, b)
    }
    #[inline(always)]
    fn sup(a: Matrix4x2<T>, b: Matrix4x2<T>) -> Matrix4x2<T> {
        Matrix4x2Trait::sup(a, b)
    }
    #[inline(always)]
    fn inf_sup(a: Matrix4x2<T>, b: Matrix4x2<T>) -> (Matrix4x2<T>, Matrix4x2<T>) {
        Matrix4x2Trait::inf_sup(a, b)
    }
}

/// The kernel of the crate-root `nalgebra::inf` / `sup` / `inf_sup` on `Matrix4x3`: the shape's
/// `inf` / `sup` / `inf_sup`.
pub impl Matrix4x3InfSup<
    T,
    impl R: Real<T>,
    +Copy<T>,
    +Drop<T>,
    +Drop<R::Wide>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
    +PartialEq<T>,
    +PartialOrd<T>,
> of crate::root::MatrixInfSup<Matrix4x3<T>> {
    #[inline(always)]
    fn inf(a: Matrix4x3<T>, b: Matrix4x3<T>) -> Matrix4x3<T> {
        Matrix4x3Trait::inf(a, b)
    }
    #[inline(always)]
    fn sup(a: Matrix4x3<T>, b: Matrix4x3<T>) -> Matrix4x3<T> {
        Matrix4x3Trait::sup(a, b)
    }
    #[inline(always)]
    fn inf_sup(a: Matrix4x3<T>, b: Matrix4x3<T>) -> (Matrix4x3<T>, Matrix4x3<T>) {
        Matrix4x3Trait::inf_sup(a, b)
    }
}

/// The kernel of the crate-root `nalgebra::inf` / `sup` / `inf_sup` on `Matrix4`: the shape's
/// `inf` / `sup` / `inf_sup`.
pub impl Matrix4InfSup<
    T,
    impl R: Real<T>,
    +Copy<T>,
    +Drop<T>,
    +Drop<R::Wide>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
    +PartialEq<T>,
    +PartialOrd<T>,
> of crate::root::MatrixInfSup<Matrix4<T>> {
    #[inline(always)]
    fn inf(a: Matrix4<T>, b: Matrix4<T>) -> Matrix4<T> {
        Matrix4Trait::inf(a, b)
    }
    #[inline(always)]
    fn sup(a: Matrix4<T>, b: Matrix4<T>) -> Matrix4<T> {
        Matrix4Trait::sup(a, b)
    }
    #[inline(always)]
    fn inf_sup(a: Matrix4<T>, b: Matrix4<T>) -> (Matrix4<T>, Matrix4<T>) {
        Matrix4Trait::inf_sup(a, b)
    }
}

/// The kernel of the crate-root `nalgebra::inf` / `sup` / `inf_sup` on `Matrix4x5`: the shape's
/// `inf` / `sup` / `inf_sup`.
pub impl Matrix4x5InfSup<
    T,
    impl R: Real<T>,
    +Copy<T>,
    +Drop<T>,
    +Drop<R::Wide>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
    +PartialEq<T>,
    +PartialOrd<T>,
> of crate::root::MatrixInfSup<Matrix4x5<T>> {
    #[inline(always)]
    fn inf(a: Matrix4x5<T>, b: Matrix4x5<T>) -> Matrix4x5<T> {
        Matrix4x5Trait::inf(a, b)
    }
    #[inline(always)]
    fn sup(a: Matrix4x5<T>, b: Matrix4x5<T>) -> Matrix4x5<T> {
        Matrix4x5Trait::sup(a, b)
    }
    #[inline(always)]
    fn inf_sup(a: Matrix4x5<T>, b: Matrix4x5<T>) -> (Matrix4x5<T>, Matrix4x5<T>) {
        Matrix4x5Trait::inf_sup(a, b)
    }
}

/// The kernel of the crate-root `nalgebra::inf` / `sup` / `inf_sup` on `Matrix4x6`: the shape's
/// `inf` / `sup` / `inf_sup`.
pub impl Matrix4x6InfSup<
    T,
    impl R: Real<T>,
    +Copy<T>,
    +Drop<T>,
    +Drop<R::Wide>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
    +PartialEq<T>,
    +PartialOrd<T>,
> of crate::root::MatrixInfSup<Matrix4x6<T>> {
    #[inline(always)]
    fn inf(a: Matrix4x6<T>, b: Matrix4x6<T>) -> Matrix4x6<T> {
        Matrix4x6Trait::inf(a, b)
    }
    #[inline(always)]
    fn sup(a: Matrix4x6<T>, b: Matrix4x6<T>) -> Matrix4x6<T> {
        Matrix4x6Trait::sup(a, b)
    }
    #[inline(always)]
    fn inf_sup(a: Matrix4x6<T>, b: Matrix4x6<T>) -> (Matrix4x6<T>, Matrix4x6<T>) {
        Matrix4x6Trait::inf_sup(a, b)
    }
}

/// The kernel of the crate-root `nalgebra::inf` / `sup` / `inf_sup` on `Vector5`: the shape's
/// `inf` / `sup` / `inf_sup`.
pub impl Vector5InfSup<
    T,
    impl R: Real<T>,
    +Copy<T>,
    +Drop<T>,
    +Drop<R::Wide>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
    +PartialEq<T>,
    +PartialOrd<T>,
> of crate::root::MatrixInfSup<Vector5<T>> {
    #[inline(always)]
    fn inf(a: Vector5<T>, b: Vector5<T>) -> Vector5<T> {
        Vector5Trait::inf(a, b)
    }
    #[inline(always)]
    fn sup(a: Vector5<T>, b: Vector5<T>) -> Vector5<T> {
        Vector5Trait::sup(a, b)
    }
    #[inline(always)]
    fn inf_sup(a: Vector5<T>, b: Vector5<T>) -> (Vector5<T>, Vector5<T>) {
        Vector5Trait::inf_sup(a, b)
    }
}

/// The kernel of the crate-root `nalgebra::inf` / `sup` / `inf_sup` on `Matrix5x2`: the shape's
/// `inf` / `sup` / `inf_sup`.
pub impl Matrix5x2InfSup<
    T,
    impl R: Real<T>,
    +Copy<T>,
    +Drop<T>,
    +Drop<R::Wide>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
    +PartialEq<T>,
    +PartialOrd<T>,
> of crate::root::MatrixInfSup<Matrix5x2<T>> {
    #[inline(always)]
    fn inf(a: Matrix5x2<T>, b: Matrix5x2<T>) -> Matrix5x2<T> {
        Matrix5x2Trait::inf(a, b)
    }
    #[inline(always)]
    fn sup(a: Matrix5x2<T>, b: Matrix5x2<T>) -> Matrix5x2<T> {
        Matrix5x2Trait::sup(a, b)
    }
    #[inline(always)]
    fn inf_sup(a: Matrix5x2<T>, b: Matrix5x2<T>) -> (Matrix5x2<T>, Matrix5x2<T>) {
        Matrix5x2Trait::inf_sup(a, b)
    }
}

/// The kernel of the crate-root `nalgebra::inf` / `sup` / `inf_sup` on `Matrix5x3`: the shape's
/// `inf` / `sup` / `inf_sup`.
pub impl Matrix5x3InfSup<
    T,
    impl R: Real<T>,
    +Copy<T>,
    +Drop<T>,
    +Drop<R::Wide>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
    +PartialEq<T>,
    +PartialOrd<T>,
> of crate::root::MatrixInfSup<Matrix5x3<T>> {
    #[inline(always)]
    fn inf(a: Matrix5x3<T>, b: Matrix5x3<T>) -> Matrix5x3<T> {
        Matrix5x3Trait::inf(a, b)
    }
    #[inline(always)]
    fn sup(a: Matrix5x3<T>, b: Matrix5x3<T>) -> Matrix5x3<T> {
        Matrix5x3Trait::sup(a, b)
    }
    #[inline(always)]
    fn inf_sup(a: Matrix5x3<T>, b: Matrix5x3<T>) -> (Matrix5x3<T>, Matrix5x3<T>) {
        Matrix5x3Trait::inf_sup(a, b)
    }
}

/// The kernel of the crate-root `nalgebra::inf` / `sup` / `inf_sup` on `Matrix5x4`: the shape's
/// `inf` / `sup` / `inf_sup`.
pub impl Matrix5x4InfSup<
    T,
    impl R: Real<T>,
    +Copy<T>,
    +Drop<T>,
    +Drop<R::Wide>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
    +PartialEq<T>,
    +PartialOrd<T>,
> of crate::root::MatrixInfSup<Matrix5x4<T>> {
    #[inline(always)]
    fn inf(a: Matrix5x4<T>, b: Matrix5x4<T>) -> Matrix5x4<T> {
        Matrix5x4Trait::inf(a, b)
    }
    #[inline(always)]
    fn sup(a: Matrix5x4<T>, b: Matrix5x4<T>) -> Matrix5x4<T> {
        Matrix5x4Trait::sup(a, b)
    }
    #[inline(always)]
    fn inf_sup(a: Matrix5x4<T>, b: Matrix5x4<T>) -> (Matrix5x4<T>, Matrix5x4<T>) {
        Matrix5x4Trait::inf_sup(a, b)
    }
}

/// The kernel of the crate-root `nalgebra::inf` / `sup` / `inf_sup` on `Matrix5`: the shape's
/// `inf` / `sup` / `inf_sup`.
pub impl Matrix5InfSup<
    T,
    impl R: Real<T>,
    +Copy<T>,
    +Drop<T>,
    +Drop<R::Wide>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
    +PartialEq<T>,
    +PartialOrd<T>,
> of crate::root::MatrixInfSup<Matrix5<T>> {
    #[inline(always)]
    fn inf(a: Matrix5<T>, b: Matrix5<T>) -> Matrix5<T> {
        Matrix5Trait::inf(a, b)
    }
    #[inline(always)]
    fn sup(a: Matrix5<T>, b: Matrix5<T>) -> Matrix5<T> {
        Matrix5Trait::sup(a, b)
    }
    #[inline(always)]
    fn inf_sup(a: Matrix5<T>, b: Matrix5<T>) -> (Matrix5<T>, Matrix5<T>) {
        Matrix5Trait::inf_sup(a, b)
    }
}

/// The kernel of the crate-root `nalgebra::inf` / `sup` / `inf_sup` on `Matrix5x6`: the shape's
/// `inf` / `sup` / `inf_sup`.
pub impl Matrix5x6InfSup<
    T,
    impl R: Real<T>,
    +Copy<T>,
    +Drop<T>,
    +Drop<R::Wide>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
    +PartialEq<T>,
    +PartialOrd<T>,
> of crate::root::MatrixInfSup<Matrix5x6<T>> {
    #[inline(always)]
    fn inf(a: Matrix5x6<T>, b: Matrix5x6<T>) -> Matrix5x6<T> {
        Matrix5x6Trait::inf(a, b)
    }
    #[inline(always)]
    fn sup(a: Matrix5x6<T>, b: Matrix5x6<T>) -> Matrix5x6<T> {
        Matrix5x6Trait::sup(a, b)
    }
    #[inline(always)]
    fn inf_sup(a: Matrix5x6<T>, b: Matrix5x6<T>) -> (Matrix5x6<T>, Matrix5x6<T>) {
        Matrix5x6Trait::inf_sup(a, b)
    }
}

/// The kernel of the crate-root `nalgebra::inf` / `sup` / `inf_sup` on `Vector6`: the shape's
/// `inf` / `sup` / `inf_sup`.
pub impl Vector6InfSup<
    T,
    impl R: Real<T>,
    +Copy<T>,
    +Drop<T>,
    +Drop<R::Wide>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
    +PartialEq<T>,
    +PartialOrd<T>,
> of crate::root::MatrixInfSup<Vector6<T>> {
    #[inline(always)]
    fn inf(a: Vector6<T>, b: Vector6<T>) -> Vector6<T> {
        Vector6Trait::inf(a, b)
    }
    #[inline(always)]
    fn sup(a: Vector6<T>, b: Vector6<T>) -> Vector6<T> {
        Vector6Trait::sup(a, b)
    }
    #[inline(always)]
    fn inf_sup(a: Vector6<T>, b: Vector6<T>) -> (Vector6<T>, Vector6<T>) {
        Vector6Trait::inf_sup(a, b)
    }
}

/// The kernel of the crate-root `nalgebra::inf` / `sup` / `inf_sup` on `Matrix6x2`: the shape's
/// `inf` / `sup` / `inf_sup`.
pub impl Matrix6x2InfSup<
    T,
    impl R: Real<T>,
    +Copy<T>,
    +Drop<T>,
    +Drop<R::Wide>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
    +PartialEq<T>,
    +PartialOrd<T>,
> of crate::root::MatrixInfSup<Matrix6x2<T>> {
    #[inline(always)]
    fn inf(a: Matrix6x2<T>, b: Matrix6x2<T>) -> Matrix6x2<T> {
        Matrix6x2Trait::inf(a, b)
    }
    #[inline(always)]
    fn sup(a: Matrix6x2<T>, b: Matrix6x2<T>) -> Matrix6x2<T> {
        Matrix6x2Trait::sup(a, b)
    }
    #[inline(always)]
    fn inf_sup(a: Matrix6x2<T>, b: Matrix6x2<T>) -> (Matrix6x2<T>, Matrix6x2<T>) {
        Matrix6x2Trait::inf_sup(a, b)
    }
}

/// The kernel of the crate-root `nalgebra::inf` / `sup` / `inf_sup` on `Matrix6x3`: the shape's
/// `inf` / `sup` / `inf_sup`.
pub impl Matrix6x3InfSup<
    T,
    impl R: Real<T>,
    +Copy<T>,
    +Drop<T>,
    +Drop<R::Wide>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
    +PartialEq<T>,
    +PartialOrd<T>,
> of crate::root::MatrixInfSup<Matrix6x3<T>> {
    #[inline(always)]
    fn inf(a: Matrix6x3<T>, b: Matrix6x3<T>) -> Matrix6x3<T> {
        Matrix6x3Trait::inf(a, b)
    }
    #[inline(always)]
    fn sup(a: Matrix6x3<T>, b: Matrix6x3<T>) -> Matrix6x3<T> {
        Matrix6x3Trait::sup(a, b)
    }
    #[inline(always)]
    fn inf_sup(a: Matrix6x3<T>, b: Matrix6x3<T>) -> (Matrix6x3<T>, Matrix6x3<T>) {
        Matrix6x3Trait::inf_sup(a, b)
    }
}

/// The kernel of the crate-root `nalgebra::inf` / `sup` / `inf_sup` on `Matrix6x4`: the shape's
/// `inf` / `sup` / `inf_sup`.
pub impl Matrix6x4InfSup<
    T,
    impl R: Real<T>,
    +Copy<T>,
    +Drop<T>,
    +Drop<R::Wide>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
    +PartialEq<T>,
    +PartialOrd<T>,
> of crate::root::MatrixInfSup<Matrix6x4<T>> {
    #[inline(always)]
    fn inf(a: Matrix6x4<T>, b: Matrix6x4<T>) -> Matrix6x4<T> {
        Matrix6x4Trait::inf(a, b)
    }
    #[inline(always)]
    fn sup(a: Matrix6x4<T>, b: Matrix6x4<T>) -> Matrix6x4<T> {
        Matrix6x4Trait::sup(a, b)
    }
    #[inline(always)]
    fn inf_sup(a: Matrix6x4<T>, b: Matrix6x4<T>) -> (Matrix6x4<T>, Matrix6x4<T>) {
        Matrix6x4Trait::inf_sup(a, b)
    }
}

/// The kernel of the crate-root `nalgebra::inf` / `sup` / `inf_sup` on `Matrix6x5`: the shape's
/// `inf` / `sup` / `inf_sup`.
pub impl Matrix6x5InfSup<
    T,
    impl R: Real<T>,
    +Copy<T>,
    +Drop<T>,
    +Drop<R::Wide>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
    +PartialEq<T>,
    +PartialOrd<T>,
> of crate::root::MatrixInfSup<Matrix6x5<T>> {
    #[inline(always)]
    fn inf(a: Matrix6x5<T>, b: Matrix6x5<T>) -> Matrix6x5<T> {
        Matrix6x5Trait::inf(a, b)
    }
    #[inline(always)]
    fn sup(a: Matrix6x5<T>, b: Matrix6x5<T>) -> Matrix6x5<T> {
        Matrix6x5Trait::sup(a, b)
    }
    #[inline(always)]
    fn inf_sup(a: Matrix6x5<T>, b: Matrix6x5<T>) -> (Matrix6x5<T>, Matrix6x5<T>) {
        Matrix6x5Trait::inf_sup(a, b)
    }
}

/// The kernel of the crate-root `nalgebra::inf` / `sup` / `inf_sup` on `Matrix6`: the shape's
/// `inf` / `sup` / `inf_sup`.
pub impl Matrix6InfSup<
    T,
    impl R: Real<T>,
    +Copy<T>,
    +Drop<T>,
    +Drop<R::Wide>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
    +PartialEq<T>,
    +PartialOrd<T>,
> of crate::root::MatrixInfSup<Matrix6<T>> {
    #[inline(always)]
    fn inf(a: Matrix6<T>, b: Matrix6<T>) -> Matrix6<T> {
        Matrix6Trait::inf(a, b)
    }
    #[inline(always)]
    fn sup(a: Matrix6<T>, b: Matrix6<T>) -> Matrix6<T> {
        Matrix6Trait::sup(a, b)
    }
    #[inline(always)]
    fn inf_sup(a: Matrix6<T>, b: Matrix6<T>) -> (Matrix6<T>, Matrix6<T>) {
        Matrix6Trait::inf_sup(a, b)
    }
}
// crate-map: end
