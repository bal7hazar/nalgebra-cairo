//! Internal, no stability promise: the crate-private items of `linalg::householder_steps` that the
//! packages above this one use (docs/SPLIT.md §12.3). Never re-exported by the facade `nalgebra`.

use nalgebra_core::base::errors::SLICE_LENGTH;
use nalgebra_core::base::matrix1::Matrix1;
use nalgebra_core::base::matrix2::Matrix2;
use nalgebra_core::base::matrix2x3::Matrix2x3;
use nalgebra_core::base::matrix2x4::Matrix2x4;
use nalgebra_core::base::matrix3::Matrix3;
use nalgebra_core::base::matrix3x2::Matrix3x2;
use nalgebra_core::base::matrix3x4::Matrix3x4;
use nalgebra_core::base::matrix4::Matrix4;
use nalgebra_core::base::matrix4x2::Matrix4x2;
use nalgebra_core::base::matrix4x3::Matrix4x3;
use nalgebra_core::base::row_vector2::RowVector2;
use nalgebra_core::base::row_vector3::RowVector3;
use nalgebra_core::base::row_vector4::RowVector4;
use nalgebra_core::base::vector2::Vector2;
use nalgebra_core::base::vector3::Vector3;
use nalgebra_core::base::vector4::Vector4;

/// Run-time column-major access to a static shape: the interface of the building blocks below
/// (their indices are run-time values, like upstream's). Crate-private: the free functions are
/// upstream's interface.
pub trait ColumnMajor<M, T> {
    /// The number of rows.
    fn nrows() -> usize;
    /// The number of columns.
    fn ncols() -> usize;
    /// The components in column-major order.
    fn to_column_major(self: M) -> Array<T>;
    /// The matrix of the column-major components `data` (panics with `nalgebra: wrong slice
    /// length` unless there are exactly `nrows * ncols`).
    fn from_column_major(data: Span<T>) -> M;
}

pub impl Matrix1ColumnMajor<T, +Copy<T>, +Drop<T>> of ColumnMajor<Matrix1<T>, T> {
    #[inline(always)]
    fn nrows() -> usize {
        1
    }

    #[inline(always)]
    fn ncols() -> usize {
        1
    }

    fn to_column_major(self: Matrix1<T>) -> Array<T> {
        array![self.x]
    }

    fn from_column_major(data: Span<T>) -> Matrix1<T> {
        let boxed: @Box<[T; 1]> = data.try_into().expect(SLICE_LENGTH);
        let [v0] = boxed.unbox();
        Matrix1 { x: v0 }
    }
}

pub impl RowVector2ColumnMajor<T, +Copy<T>, +Drop<T>> of ColumnMajor<RowVector2<T>, T> {
    #[inline(always)]
    fn nrows() -> usize {
        1
    }

    #[inline(always)]
    fn ncols() -> usize {
        2
    }

    fn to_column_major(self: RowVector2<T>) -> Array<T> {
        array![self.x, self.y]
    }

    fn from_column_major(data: Span<T>) -> RowVector2<T> {
        let boxed: @Box<[T; 2]> = data.try_into().expect(SLICE_LENGTH);
        let [v0, v1] = boxed.unbox();
        RowVector2 { x: v0, y: v1 }
    }
}

pub impl RowVector3ColumnMajor<T, +Copy<T>, +Drop<T>> of ColumnMajor<RowVector3<T>, T> {
    #[inline(always)]
    fn nrows() -> usize {
        1
    }

    #[inline(always)]
    fn ncols() -> usize {
        3
    }

    fn to_column_major(self: RowVector3<T>) -> Array<T> {
        array![self.x, self.y, self.z]
    }

    fn from_column_major(data: Span<T>) -> RowVector3<T> {
        let boxed: @Box<[T; 3]> = data.try_into().expect(SLICE_LENGTH);
        let [v0, v1, v2] = boxed.unbox();
        RowVector3 { x: v0, y: v1, z: v2 }
    }
}

pub impl RowVector4ColumnMajor<T, +Copy<T>, +Drop<T>> of ColumnMajor<RowVector4<T>, T> {
    #[inline(always)]
    fn nrows() -> usize {
        1
    }

    #[inline(always)]
    fn ncols() -> usize {
        4
    }

    fn to_column_major(self: RowVector4<T>) -> Array<T> {
        array![self.x, self.y, self.z, self.w]
    }

    fn from_column_major(data: Span<T>) -> RowVector4<T> {
        let boxed: @Box<[T; 4]> = data.try_into().expect(SLICE_LENGTH);
        let [v0, v1, v2, v3] = boxed.unbox();
        RowVector4 { x: v0, y: v1, z: v2, w: v3 }
    }
}

pub impl Vector2ColumnMajor<T, +Copy<T>, +Drop<T>> of ColumnMajor<Vector2<T>, T> {
    #[inline(always)]
    fn nrows() -> usize {
        2
    }

    #[inline(always)]
    fn ncols() -> usize {
        1
    }

    fn to_column_major(self: Vector2<T>) -> Array<T> {
        array![self.x, self.y]
    }

    fn from_column_major(data: Span<T>) -> Vector2<T> {
        let boxed: @Box<[T; 2]> = data.try_into().expect(SLICE_LENGTH);
        let [v0, v1] = boxed.unbox();
        Vector2 { x: v0, y: v1 }
    }
}

pub impl Matrix2ColumnMajor<T, +Copy<T>, +Drop<T>> of ColumnMajor<Matrix2<T>, T> {
    #[inline(always)]
    fn nrows() -> usize {
        2
    }

    #[inline(always)]
    fn ncols() -> usize {
        2
    }

    fn to_column_major(self: Matrix2<T>) -> Array<T> {
        array![self.m11, self.m21, self.m12, self.m22]
    }

    fn from_column_major(data: Span<T>) -> Matrix2<T> {
        let boxed: @Box<[T; 4]> = data.try_into().expect(SLICE_LENGTH);
        let [v0, v1, v2, v3] = boxed.unbox();
        Matrix2 { m11: v0, m21: v1, m12: v2, m22: v3 }
    }
}

pub impl Matrix2x3ColumnMajor<T, +Copy<T>, +Drop<T>> of ColumnMajor<Matrix2x3<T>, T> {
    #[inline(always)]
    fn nrows() -> usize {
        2
    }

    #[inline(always)]
    fn ncols() -> usize {
        3
    }

    fn to_column_major(self: Matrix2x3<T>) -> Array<T> {
        array![self.m11, self.m21, self.m12, self.m22, self.m13, self.m23]
    }

    fn from_column_major(data: Span<T>) -> Matrix2x3<T> {
        let boxed: @Box<[T; 6]> = data.try_into().expect(SLICE_LENGTH);
        let [v0, v1, v2, v3, v4, v5] = boxed.unbox();
        Matrix2x3 { m11: v0, m21: v1, m12: v2, m22: v3, m13: v4, m23: v5 }
    }
}

pub impl Matrix2x4ColumnMajor<T, +Copy<T>, +Drop<T>> of ColumnMajor<Matrix2x4<T>, T> {
    #[inline(always)]
    fn nrows() -> usize {
        2
    }

    #[inline(always)]
    fn ncols() -> usize {
        4
    }

    fn to_column_major(self: Matrix2x4<T>) -> Array<T> {
        array![self.m11, self.m21, self.m12, self.m22, self.m13, self.m23, self.m14, self.m24]
    }

    fn from_column_major(data: Span<T>) -> Matrix2x4<T> {
        let boxed: @Box<[T; 8]> = data.try_into().expect(SLICE_LENGTH);
        let [v0, v1, v2, v3, v4, v5, v6, v7] = boxed.unbox();
        Matrix2x4 { m11: v0, m21: v1, m12: v2, m22: v3, m13: v4, m23: v5, m14: v6, m24: v7 }
    }
}

pub impl Vector3ColumnMajor<T, +Copy<T>, +Drop<T>> of ColumnMajor<Vector3<T>, T> {
    #[inline(always)]
    fn nrows() -> usize {
        3
    }

    #[inline(always)]
    fn ncols() -> usize {
        1
    }

    fn to_column_major(self: Vector3<T>) -> Array<T> {
        array![self.x, self.y, self.z]
    }

    fn from_column_major(data: Span<T>) -> Vector3<T> {
        let boxed: @Box<[T; 3]> = data.try_into().expect(SLICE_LENGTH);
        let [v0, v1, v2] = boxed.unbox();
        Vector3 { x: v0, y: v1, z: v2 }
    }
}

pub impl Matrix3x2ColumnMajor<T, +Copy<T>, +Drop<T>> of ColumnMajor<Matrix3x2<T>, T> {
    #[inline(always)]
    fn nrows() -> usize {
        3
    }

    #[inline(always)]
    fn ncols() -> usize {
        2
    }

    fn to_column_major(self: Matrix3x2<T>) -> Array<T> {
        array![self.m11, self.m21, self.m31, self.m12, self.m22, self.m32]
    }

    fn from_column_major(data: Span<T>) -> Matrix3x2<T> {
        let boxed: @Box<[T; 6]> = data.try_into().expect(SLICE_LENGTH);
        let [v0, v1, v2, v3, v4, v5] = boxed.unbox();
        Matrix3x2 { m11: v0, m21: v1, m31: v2, m12: v3, m22: v4, m32: v5 }
    }
}

pub impl Matrix3ColumnMajor<T, +Copy<T>, +Drop<T>> of ColumnMajor<Matrix3<T>, T> {
    #[inline(always)]
    fn nrows() -> usize {
        3
    }

    #[inline(always)]
    fn ncols() -> usize {
        3
    }

    fn to_column_major(self: Matrix3<T>) -> Array<T> {
        array![
            self.m11, self.m21, self.m31, self.m12, self.m22, self.m32, self.m13, self.m23,
            self.m33,
        ]
    }

    fn from_column_major(data: Span<T>) -> Matrix3<T> {
        let boxed: @Box<[T; 9]> = data.try_into().expect(SLICE_LENGTH);
        let [v0, v1, v2, v3, v4, v5, v6, v7, v8] = boxed.unbox();
        Matrix3 { m11: v0, m21: v1, m31: v2, m12: v3, m22: v4, m32: v5, m13: v6, m23: v7, m33: v8 }
    }
}

pub impl Matrix3x4ColumnMajor<T, +Copy<T>, +Drop<T>> of ColumnMajor<Matrix3x4<T>, T> {
    #[inline(always)]
    fn nrows() -> usize {
        3
    }

    #[inline(always)]
    fn ncols() -> usize {
        4
    }

    fn to_column_major(self: Matrix3x4<T>) -> Array<T> {
        array![
            self.m11, self.m21, self.m31, self.m12, self.m22, self.m32, self.m13, self.m23,
            self.m33, self.m14, self.m24, self.m34,
        ]
    }

    fn from_column_major(data: Span<T>) -> Matrix3x4<T> {
        let boxed: @Box<[T; 12]> = data.try_into().expect(SLICE_LENGTH);
        let [v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11] = boxed.unbox();
        Matrix3x4 {
            m11: v0,
            m21: v1,
            m31: v2,
            m12: v3,
            m22: v4,
            m32: v5,
            m13: v6,
            m23: v7,
            m33: v8,
            m14: v9,
            m24: v10,
            m34: v11,
        }
    }
}

pub impl Vector4ColumnMajor<T, +Copy<T>, +Drop<T>> of ColumnMajor<Vector4<T>, T> {
    #[inline(always)]
    fn nrows() -> usize {
        4
    }

    #[inline(always)]
    fn ncols() -> usize {
        1
    }

    fn to_column_major(self: Vector4<T>) -> Array<T> {
        array![self.x, self.y, self.z, self.w]
    }

    fn from_column_major(data: Span<T>) -> Vector4<T> {
        let boxed: @Box<[T; 4]> = data.try_into().expect(SLICE_LENGTH);
        let [v0, v1, v2, v3] = boxed.unbox();
        Vector4 { x: v0, y: v1, z: v2, w: v3 }
    }
}

pub impl Matrix4x2ColumnMajor<T, +Copy<T>, +Drop<T>> of ColumnMajor<Matrix4x2<T>, T> {
    #[inline(always)]
    fn nrows() -> usize {
        4
    }

    #[inline(always)]
    fn ncols() -> usize {
        2
    }

    fn to_column_major(self: Matrix4x2<T>) -> Array<T> {
        array![self.m11, self.m21, self.m31, self.m41, self.m12, self.m22, self.m32, self.m42]
    }

    fn from_column_major(data: Span<T>) -> Matrix4x2<T> {
        let boxed: @Box<[T; 8]> = data.try_into().expect(SLICE_LENGTH);
        let [v0, v1, v2, v3, v4, v5, v6, v7] = boxed.unbox();
        Matrix4x2 { m11: v0, m21: v1, m31: v2, m41: v3, m12: v4, m22: v5, m32: v6, m42: v7 }
    }
}

pub impl Matrix4x3ColumnMajor<T, +Copy<T>, +Drop<T>> of ColumnMajor<Matrix4x3<T>, T> {
    #[inline(always)]
    fn nrows() -> usize {
        4
    }

    #[inline(always)]
    fn ncols() -> usize {
        3
    }

    fn to_column_major(self: Matrix4x3<T>) -> Array<T> {
        array![
            self.m11, self.m21, self.m31, self.m41, self.m12, self.m22, self.m32, self.m42,
            self.m13, self.m23, self.m33, self.m43,
        ]
    }

    fn from_column_major(data: Span<T>) -> Matrix4x3<T> {
        let boxed: @Box<[T; 12]> = data.try_into().expect(SLICE_LENGTH);
        let [v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11] = boxed.unbox();
        Matrix4x3 {
            m11: v0,
            m21: v1,
            m31: v2,
            m41: v3,
            m12: v4,
            m22: v5,
            m32: v6,
            m42: v7,
            m13: v8,
            m23: v9,
            m33: v10,
            m43: v11,
        }
    }
}

pub impl Matrix4ColumnMajor<T, +Copy<T>, +Drop<T>> of ColumnMajor<Matrix4<T>, T> {
    #[inline(always)]
    fn nrows() -> usize {
        4
    }

    #[inline(always)]
    fn ncols() -> usize {
        4
    }

    fn to_column_major(self: Matrix4<T>) -> Array<T> {
        array![
            self.m11, self.m21, self.m31, self.m41, self.m12, self.m22, self.m32, self.m42,
            self.m13, self.m23, self.m33, self.m43, self.m14, self.m24, self.m34, self.m44,
        ]
    }

    fn from_column_major(data: Span<T>) -> Matrix4<T> {
        let boxed: @Box<[T; 16]> = data.try_into().expect(SLICE_LENGTH);
        let [v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15] = boxed.unbox();
        Matrix4 {
            m11: v0,
            m21: v1,
            m31: v2,
            m41: v3,
            m12: v4,
            m22: v5,
            m32: v6,
            m42: v7,
            m13: v8,
            m23: v9,
            m33: v10,
            m43: v11,
            m14: v12,
            m24: v13,
            m34: v14,
            m44: v15,
        }
    }
}
