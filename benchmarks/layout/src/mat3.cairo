//! Question 2 — 3x3 matrix layout. One trait, six layouts, generic over the scalar.
//!
//! * `Mat3F<T>`   : 9 named fields m11..m33
//! * `Mat3C<T>`   : 3 column `Vec3<T>` (nalgebra / glam are column-major), ops written with
//!                  vector ops (scale/add/dot/cross)
//! * `Mat3A<T>`   : `[T; 9]` newtype (row-major)
//! * `Mat3AA<T>`  : `[[T; 3]; 3]` newtype (array of rows)
//! * `DMatIdx<T>` : `Span<T>` + (rows, cols), index arithmetic + loops (orion/alexandria style)
//! * `DMatSeq<T>` : `Span<T>` + (rows, cols), sequential traversal (slices + pop_front), and
//!                  `multi_pop_front::<9>()` for the 3x3-only ops (det / inverse / transpose)

use crate::vec3::{DVecLoop, Vec3, Vec3Ops};

pub type Row<T> = (T, T, T);

pub trait Mat3Ops<M, V, T> {
    fn from_rows(r1: Row<T>, r2: Row<T>, r3: Row<T>) -> M;
    fn rows(self: M) -> (Row<T>, Row<T>, Row<T>);
    fn vector(x: T, y: T, z: T) -> V;
    fn vector_xyz(v: V) -> Row<T>;
    fn mul_vec(self: M, v: V) -> V;
    fn mul_mat(self: M, rhs: M) -> M;
    fn transpose(self: M) -> M;
    fn determinant(self: M) -> T;
    /// Adjugate / determinant (division panics on a singular matrix).
    fn inverse(self: M) -> M;
    fn add(self: M, rhs: M) -> M;
    fn scale(self: M, k: T) -> M;
}

// ---------------------------------------------------------------------------------------------
// 9 named fields
// ---------------------------------------------------------------------------------------------

#[derive(Copy, Drop, Debug, PartialEq)]
pub struct Mat3F<T> {
    pub m11: T,
    pub m12: T,
    pub m13: T,
    pub m21: T,
    pub m22: T,
    pub m23: T,
    pub m31: T,
    pub m32: T,
    pub m33: T,
}

pub impl FieldsMat3<
    T, +Add<T>, +Sub<T>, +Mul<T>, +Div<T>, +Neg<T>, +PartialEq<T>, +Copy<T>, +Drop<T>,
> of Mat3Ops<Mat3F<T>, Vec3<T>, T> {
    fn from_rows(r1: Row<T>, r2: Row<T>, r3: Row<T>) -> Mat3F<T> {
        let (m11, m12, m13) = r1;
        let (m21, m22, m23) = r2;
        let (m31, m32, m33) = r3;
        Mat3F { m11, m12, m13, m21, m22, m23, m31, m32, m33 }
    }
    fn rows(self: Mat3F<T>) -> (Row<T>, Row<T>, Row<T>) {
        (
            (self.m11, self.m12, self.m13),
            (self.m21, self.m22, self.m23),
            (self.m31, self.m32, self.m33),
        )
    }
    fn vector(x: T, y: T, z: T) -> Vec3<T> {
        Vec3 { x, y, z }
    }
    fn vector_xyz(v: Vec3<T>) -> Row<T> {
        (v.x, v.y, v.z)
    }
    fn mul_vec(self: Mat3F<T>, v: Vec3<T>) -> Vec3<T> {
        Vec3 {
            x: self.m11 * v.x + self.m12 * v.y + self.m13 * v.z,
            y: self.m21 * v.x + self.m22 * v.y + self.m23 * v.z,
            z: self.m31 * v.x + self.m32 * v.y + self.m33 * v.z,
        }
    }
    fn mul_mat(self: Mat3F<T>, rhs: Mat3F<T>) -> Mat3F<T> {
        let a = self;
        let b = rhs;
        Mat3F {
            m11: a.m11 * b.m11 + a.m12 * b.m21 + a.m13 * b.m31,
            m12: a.m11 * b.m12 + a.m12 * b.m22 + a.m13 * b.m32,
            m13: a.m11 * b.m13 + a.m12 * b.m23 + a.m13 * b.m33,
            m21: a.m21 * b.m11 + a.m22 * b.m21 + a.m23 * b.m31,
            m22: a.m21 * b.m12 + a.m22 * b.m22 + a.m23 * b.m32,
            m23: a.m21 * b.m13 + a.m22 * b.m23 + a.m23 * b.m33,
            m31: a.m31 * b.m11 + a.m32 * b.m21 + a.m33 * b.m31,
            m32: a.m31 * b.m12 + a.m32 * b.m22 + a.m33 * b.m32,
            m33: a.m31 * b.m13 + a.m32 * b.m23 + a.m33 * b.m33,
        }
    }
    fn transpose(self: Mat3F<T>) -> Mat3F<T> {
        Mat3F {
            m11: self.m11,
            m12: self.m21,
            m13: self.m31,
            m21: self.m12,
            m22: self.m22,
            m23: self.m32,
            m31: self.m13,
            m32: self.m23,
            m33: self.m33,
        }
    }
    fn determinant(self: Mat3F<T>) -> T {
        let a = self;
        a.m11 * (a.m22 * a.m33 - a.m23 * a.m32)
            - a.m12 * (a.m21 * a.m33 - a.m23 * a.m31)
            + a.m13 * (a.m21 * a.m32 - a.m22 * a.m31)
    }
    fn inverse(self: Mat3F<T>) -> Mat3F<T> {
        let a = self;
        let c11 = a.m22 * a.m33 - a.m23 * a.m32;
        let c12 = a.m23 * a.m31 - a.m21 * a.m33;
        let c13 = a.m21 * a.m32 - a.m22 * a.m31;
        let det = a.m11 * c11 + a.m12 * c12 + a.m13 * c13;
        Mat3F {
            m11: c11 / det,
            m12: (a.m13 * a.m32 - a.m12 * a.m33) / det,
            m13: (a.m12 * a.m23 - a.m13 * a.m22) / det,
            m21: c12 / det,
            m22: (a.m11 * a.m33 - a.m13 * a.m31) / det,
            m23: (a.m13 * a.m21 - a.m11 * a.m23) / det,
            m31: c13 / det,
            m32: (a.m12 * a.m31 - a.m11 * a.m32) / det,
            m33: (a.m11 * a.m22 - a.m12 * a.m21) / det,
        }
    }
    fn add(self: Mat3F<T>, rhs: Mat3F<T>) -> Mat3F<T> {
        Mat3F {
            m11: self.m11 + rhs.m11,
            m12: self.m12 + rhs.m12,
            m13: self.m13 + rhs.m13,
            m21: self.m21 + rhs.m21,
            m22: self.m22 + rhs.m22,
            m23: self.m23 + rhs.m23,
            m31: self.m31 + rhs.m31,
            m32: self.m32 + rhs.m32,
            m33: self.m33 + rhs.m33,
        }
    }
    fn scale(self: Mat3F<T>, k: T) -> Mat3F<T> {
        Mat3F {
            m11: self.m11 * k,
            m12: self.m12 * k,
            m13: self.m13 * k,
            m21: self.m21 * k,
            m22: self.m22 * k,
            m23: self.m23 * k,
            m31: self.m31 * k,
            m32: self.m32 * k,
            m33: self.m33 * k,
        }
    }
}

// ---------------------------------------------------------------------------------------------
// 3 column vectors
// ---------------------------------------------------------------------------------------------

#[derive(Copy, Drop, Debug, PartialEq)]
pub struct Mat3C<T> {
    pub c1: Vec3<T>,
    pub c2: Vec3<T>,
    pub c3: Vec3<T>,
}

pub impl ColsMat3<
    T, +Add<T>, +Sub<T>, +Mul<T>, +Div<T>, +Neg<T>, +PartialEq<T>, +Copy<T>, +Drop<T>,
> of Mat3Ops<Mat3C<T>, Vec3<T>, T> {
    fn from_rows(r1: Row<T>, r2: Row<T>, r3: Row<T>) -> Mat3C<T> {
        let (m11, m12, m13) = r1;
        let (m21, m22, m23) = r2;
        let (m31, m32, m33) = r3;
        Mat3C {
            c1: Vec3 { x: m11, y: m21, z: m31 },
            c2: Vec3 { x: m12, y: m22, z: m32 },
            c3: Vec3 { x: m13, y: m23, z: m33 },
        }
    }
    fn rows(self: Mat3C<T>) -> (Row<T>, Row<T>, Row<T>) {
        (
            (self.c1.x, self.c2.x, self.c3.x),
            (self.c1.y, self.c2.y, self.c3.y),
            (self.c1.z, self.c2.z, self.c3.z),
        )
    }
    fn vector(x: T, y: T, z: T) -> Vec3<T> {
        Vec3 { x, y, z }
    }
    fn vector_xyz(v: Vec3<T>) -> Row<T> {
        (v.x, v.y, v.z)
    }
    fn mul_vec(self: Mat3C<T>, v: Vec3<T>) -> Vec3<T> {
        self.c1.scale(v.x).add(self.c2.scale(v.y)).add(self.c3.scale(v.z))
    }
    fn mul_mat(self: Mat3C<T>, rhs: Mat3C<T>) -> Mat3C<T> {
        Mat3C { c1: self.mul_vec(rhs.c1), c2: self.mul_vec(rhs.c2), c3: self.mul_vec(rhs.c3) }
    }
    fn transpose(self: Mat3C<T>) -> Mat3C<T> {
        Mat3C {
            c1: Vec3 { x: self.c1.x, y: self.c2.x, z: self.c3.x },
            c2: Vec3 { x: self.c1.y, y: self.c2.y, z: self.c3.y },
            c3: Vec3 { x: self.c1.z, y: self.c2.z, z: self.c3.z },
        }
    }
    fn determinant(self: Mat3C<T>) -> T {
        self.c1.dot(self.c2.cross(self.c3))
    }
    fn inverse(self: Mat3C<T>) -> Mat3C<T> {
        // Rows of the inverse are the cross products of the columns, over the determinant.
        let r1 = self.c2.cross(self.c3);
        let r2 = self.c3.cross(self.c1);
        let r3 = self.c1.cross(self.c2);
        let det = self.c1.dot(r1);
        Mat3C {
            c1: Vec3 { x: r1.x / det, y: r2.x / det, z: r3.x / det },
            c2: Vec3 { x: r1.y / det, y: r2.y / det, z: r3.y / det },
            c3: Vec3 { x: r1.z / det, y: r2.z / det, z: r3.z / det },
        }
    }
    fn add(self: Mat3C<T>, rhs: Mat3C<T>) -> Mat3C<T> {
        Mat3C { c1: self.c1.add(rhs.c1), c2: self.c2.add(rhs.c2), c3: self.c3.add(rhs.c3) }
    }
    fn scale(self: Mat3C<T>, k: T) -> Mat3C<T> {
        Mat3C { c1: self.c1.scale(k), c2: self.c2.scale(k), c3: self.c3.scale(k) }
    }
}

// ---------------------------------------------------------------------------------------------
// [T; 9] and [[T; 3]; 3] : destructure, run the named-field kernel, rebuild.
// (Destructuring is the only zero-cost access to a fixed-size array; `.span()[i]` is measured
// separately in the `access` groups.)
// ---------------------------------------------------------------------------------------------

#[derive(Copy, Drop)]
pub struct Mat3A<T> {
    pub data: [T; 9],
}

#[derive(Copy, Drop)]
pub struct Mat3AA<T> {
    pub data: [[T; 3]; 3],
}

#[generate_trait]
impl Mat3AConv<T, +Copy<T>, +Drop<T>> of Mat3AConvTrait<T> {
    #[inline(always)]
    fn fields(self: Mat3A<T>) -> Mat3F<T> {
        let [m11, m12, m13, m21, m22, m23, m31, m32, m33] = self.data;
        Mat3F { m11, m12, m13, m21, m22, m23, m31, m32, m33 }
    }
    #[inline(always)]
    fn array(self: Mat3F<T>) -> Mat3A<T> {
        Mat3A {
            data: [
                self.m11, self.m12, self.m13, self.m21, self.m22, self.m23, self.m31, self.m32,
                self.m33,
            ],
        }
    }
}

#[generate_trait]
impl Mat3AAConv<T, +Copy<T>, +Drop<T>> of Mat3AAConvTrait<T> {
    #[inline(always)]
    fn fields(self: Mat3AA<T>) -> Mat3F<T> {
        let [[m11, m12, m13], [m21, m22, m23], [m31, m32, m33]] = self.data;
        Mat3F { m11, m12, m13, m21, m22, m23, m31, m32, m33 }
    }
    #[inline(always)]
    fn nested(self: Mat3F<T>) -> Mat3AA<T> {
        Mat3AA {
            data: [
                [self.m11, self.m12, self.m13], [self.m21, self.m22, self.m23],
                [self.m31, self.m32, self.m33],
            ],
        }
    }
}

pub impl ArrayMat3<
    T, +Add<T>, +Sub<T>, +Mul<T>, +Div<T>, +Neg<T>, +PartialEq<T>, +Copy<T>, +Drop<T>,
> of Mat3Ops<Mat3A<T>, Vec3<T>, T> {
    fn from_rows(r1: Row<T>, r2: Row<T>, r3: Row<T>) -> Mat3A<T> {
        let (m11, m12, m13) = r1;
        let (m21, m22, m23) = r2;
        let (m31, m32, m33) = r3;
        Mat3A { data: [m11, m12, m13, m21, m22, m23, m31, m32, m33] }
    }
    fn rows(self: Mat3A<T>) -> (Row<T>, Row<T>, Row<T>) {
        let [m11, m12, m13, m21, m22, m23, m31, m32, m33] = self.data;
        ((m11, m12, m13), (m21, m22, m23), (m31, m32, m33))
    }
    fn vector(x: T, y: T, z: T) -> Vec3<T> {
        Vec3 { x, y, z }
    }
    fn vector_xyz(v: Vec3<T>) -> Row<T> {
        (v.x, v.y, v.z)
    }
    fn mul_vec(self: Mat3A<T>, v: Vec3<T>) -> Vec3<T> {
        self.fields().mul_vec(v)
    }
    fn mul_mat(self: Mat3A<T>, rhs: Mat3A<T>) -> Mat3A<T> {
        self.fields().mul_mat(rhs.fields()).array()
    }
    fn transpose(self: Mat3A<T>) -> Mat3A<T> {
        let [m11, m12, m13, m21, m22, m23, m31, m32, m33] = self.data;
        Mat3A { data: [m11, m21, m31, m12, m22, m32, m13, m23, m33] }
    }
    fn determinant(self: Mat3A<T>) -> T {
        self.fields().determinant()
    }
    fn inverse(self: Mat3A<T>) -> Mat3A<T> {
        self.fields().inverse().array()
    }
    fn add(self: Mat3A<T>, rhs: Mat3A<T>) -> Mat3A<T> {
        self.fields().add(rhs.fields()).array()
    }
    fn scale(self: Mat3A<T>, k: T) -> Mat3A<T> {
        self.fields().scale(k).array()
    }
}

pub impl NestedMat3<
    T, +Add<T>, +Sub<T>, +Mul<T>, +Div<T>, +Neg<T>, +PartialEq<T>, +Copy<T>, +Drop<T>,
> of Mat3Ops<Mat3AA<T>, Vec3<T>, T> {
    fn from_rows(r1: Row<T>, r2: Row<T>, r3: Row<T>) -> Mat3AA<T> {
        let (m11, m12, m13) = r1;
        let (m21, m22, m23) = r2;
        let (m31, m32, m33) = r3;
        Mat3AA { data: [[m11, m12, m13], [m21, m22, m23], [m31, m32, m33]] }
    }
    fn rows(self: Mat3AA<T>) -> (Row<T>, Row<T>, Row<T>) {
        let [[m11, m12, m13], [m21, m22, m23], [m31, m32, m33]] = self.data;
        ((m11, m12, m13), (m21, m22, m23), (m31, m32, m33))
    }
    fn vector(x: T, y: T, z: T) -> Vec3<T> {
        Vec3 { x, y, z }
    }
    fn vector_xyz(v: Vec3<T>) -> Row<T> {
        (v.x, v.y, v.z)
    }
    fn mul_vec(self: Mat3AA<T>, v: Vec3<T>) -> Vec3<T> {
        self.fields().mul_vec(v)
    }
    fn mul_mat(self: Mat3AA<T>, rhs: Mat3AA<T>) -> Mat3AA<T> {
        self.fields().mul_mat(rhs.fields()).nested()
    }
    fn transpose(self: Mat3AA<T>) -> Mat3AA<T> {
        let [[m11, m12, m13], [m21, m22, m23], [m31, m32, m33]] = self.data;
        Mat3AA { data: [[m11, m21, m31], [m12, m22, m32], [m13, m23, m33]] }
    }
    fn determinant(self: Mat3AA<T>) -> T {
        self.fields().determinant()
    }
    fn inverse(self: Mat3AA<T>) -> Mat3AA<T> {
        self.fields().inverse().nested()
    }
    fn add(self: Mat3AA<T>, rhs: Mat3AA<T>) -> Mat3AA<T> {
        self.fields().add(rhs.fields()).nested()
    }
    fn scale(self: Mat3AA<T>, k: T) -> Mat3AA<T> {
        self.fields().scale(k).nested()
    }
}

// ---------------------------------------------------------------------------------------------
// Dynamic matrix, index arithmetic + loops (row-major `Span<T>` + dims)
// ---------------------------------------------------------------------------------------------

#[derive(Copy, Drop)]
pub struct DMatIdx<T> {
    pub data: Span<T>,
    pub rows: usize,
    pub cols: usize,
}

#[generate_trait]
pub impl DMatIdxImpl<T, +Add<T>, +Mul<T>, +Copy<T>, +Drop<T>> of DMatIdxTrait<T> {
    #[inline(always)]
    fn at(self: DMatIdx<T>, i: usize, j: usize) -> T {
        *self.data[i * self.cols + j]
    }

    /// Generic (rows x cols) * (cols x k) product, orion/alexandria style.
    fn matmul(self: DMatIdx<T>, rhs: DMatIdx<T>) -> DMatIdx<T> {
        assert(self.cols == rhs.rows && self.cols != 0, 'dim mismatch');
        let mut out = array![];
        for i in 0..self.rows {
            for j in 0..rhs.cols {
                let mut acc = self.at(i, 0) * rhs.at(0, j);
                for k in 1..self.cols {
                    acc = acc + self.at(i, k) * rhs.at(k, j);
                }
                out.append(acc);
            }
        }
        DMatIdx { data: out.span(), rows: self.rows, cols: rhs.cols }
    }

    fn transposed(self: DMatIdx<T>) -> DMatIdx<T> {
        let mut out = array![];
        for j in 0..self.cols {
            for i in 0..self.rows {
                out.append(self.at(i, j));
            }
        }
        DMatIdx { data: out.span(), rows: self.cols, cols: self.rows }
    }
}

pub impl IndexDMat3<
    T, +Add<T>, +Sub<T>, +Mul<T>, +Div<T>, +Neg<T>, +PartialEq<T>, +Copy<T>, +Drop<T>,
> of Mat3Ops<DMatIdx<T>, DVecLoop<T>, T> {
    fn from_rows(r1: Row<T>, r2: Row<T>, r3: Row<T>) -> DMatIdx<T> {
        let (m11, m12, m13) = r1;
        let (m21, m22, m23) = r2;
        let (m31, m32, m33) = r3;
        DMatIdx {
            data: array![m11, m12, m13, m21, m22, m23, m31, m32, m33].span(), rows: 3, cols: 3,
        }
    }
    fn rows(self: DMatIdx<T>) -> (Row<T>, Row<T>, Row<T>) {
        let d = self.data;
        ((*d[0], *d[1], *d[2]), (*d[3], *d[4], *d[5]), (*d[6], *d[7], *d[8]))
    }
    fn vector(x: T, y: T, z: T) -> DVecLoop<T> {
        DVecLoop { data: array![x, y, z].span() }
    }
    fn vector_xyz(v: DVecLoop<T>) -> Row<T> {
        (*v.data[0], *v.data[1], *v.data[2])
    }
    fn mul_vec(self: DMatIdx<T>, v: DVecLoop<T>) -> DVecLoop<T> {
        let product = self.matmul(DMatIdx { data: v.data, rows: v.data.len(), cols: 1 });
        DVecLoop { data: product.data }
    }
    fn mul_mat(self: DMatIdx<T>, rhs: DMatIdx<T>) -> DMatIdx<T> {
        self.matmul(rhs)
    }
    fn transpose(self: DMatIdx<T>) -> DMatIdx<T> {
        self.transposed()
    }
    fn determinant(self: DMatIdx<T>) -> T {
        assert(self.rows == 3 && self.cols == 3, 'dim mismatch');
        let a = self;
        a.at(0, 0) * (a.at(1, 1) * a.at(2, 2) - a.at(1, 2) * a.at(2, 1))
            - a.at(0, 1) * (a.at(1, 0) * a.at(2, 2) - a.at(1, 2) * a.at(2, 0))
            + a.at(0, 2) * (a.at(1, 0) * a.at(2, 1) - a.at(1, 1) * a.at(2, 0))
    }
    fn inverse(self: DMatIdx<T>) -> DMatIdx<T> {
        assert(self.rows == 3 && self.cols == 3, 'dim mismatch');
        // Cyclic cofactor loop: inv[j][i] = cofactor(i, j) / det.
        let a = self;
        let det = Self::determinant(a);
        let mut out = array![];
        for j in 0..3_usize {
            let (j1, j2) = ((j + 1) % 3, (j + 2) % 3);
            for i in 0..3_usize {
                let (i1, i2) = ((i + 1) % 3, (i + 2) % 3);
                // out is row-major of the inverse: entry (j, i) = cofactor(i, j) / det.
                let cof = a.at(i1, j1) * a.at(i2, j2) - a.at(i1, j2) * a.at(i2, j1);
                out.append(cof / det);
            }
        }
        DMatIdx { data: out.span(), rows: 3, cols: 3 }
    }
    fn add(self: DMatIdx<T>, rhs: DMatIdx<T>) -> DMatIdx<T> {
        assert(self.rows == rhs.rows && self.cols == rhs.cols, 'dim mismatch');
        let (mut a, mut b) = (self.data, rhs.data);
        let mut out = array![];
        while let Some(x) = a.pop_front() {
            out.append(*x + *b.pop_front().unwrap());
        }
        DMatIdx { data: out.span(), rows: self.rows, cols: self.cols }
    }
    fn scale(self: DMatIdx<T>, k: T) -> DMatIdx<T> {
        let mut a = self.data;
        let mut out = array![];
        while let Some(x) = a.pop_front() {
            out.append(*x * k);
        }
        DMatIdx { data: out.span(), rows: self.rows, cols: self.cols }
    }
}

// ---------------------------------------------------------------------------------------------
// Dynamic matrix, sequential traversal (no index arithmetic in the inner loop)
// ---------------------------------------------------------------------------------------------

#[derive(Copy, Drop)]
pub struct DMatSeq<T> {
    pub data: Span<T>,
    pub rows: usize,
    pub cols: usize,
}

#[generate_trait]
pub impl DMatSeqImpl<T, +Add<T>, +Mul<T>, +Copy<T>, +Drop<T>> of DMatSeqTrait<T> {
    fn transposed(self: DMatSeq<T>) -> DMatSeq<T> {
        let mut out = array![];
        for j in 0..self.cols {
            let mut index = j;
            for _ in 0..self.rows {
                out.append(*self.data[index]);
                index += self.cols;
            }
        }
        DMatSeq { data: out.span(), rows: self.cols, cols: self.rows }
    }

    /// Transposes `rhs` once so that every inner product walks two contiguous slices.
    fn matmul(self: DMatSeq<T>, rhs: DMatSeq<T>) -> DMatSeq<T> {
        assert(self.cols == rhs.rows && self.cols != 0, 'dim mismatch');
        let rhs_t = if rhs.cols == 1 {
            rhs
        } else {
            rhs.transposed()
        };
        let mut out = array![];
        let mut lhs_rows = self.data;
        for _ in 0..self.rows {
            let row = lhs_rows.slice(0, self.cols);
            lhs_rows = lhs_rows.slice(self.cols, lhs_rows.len() - self.cols);
            let mut rhs_cols = rhs_t.data;
            for _ in 0..rhs.cols {
                let mut a = row;
                let mut acc = *a.pop_front().unwrap() * *rhs_cols.pop_front().unwrap();
                while let Some(x) = a.pop_front() {
                    acc = acc + *x * *rhs_cols.pop_front().unwrap();
                }
                out.append(acc);
            }
        }
        DMatSeq { data: out.span(), rows: self.rows, cols: rhs.cols }
    }
}

#[inline]
fn unpack9<T, +Copy<T>, +Drop<T>>(m: DMatSeq<T>) -> Mat3F<T> {
    assert(m.rows == 3 && m.cols == 3, 'dim mismatch');
    let mut span = m.data;
    let [m11, m12, m13, m21, m22, m23, m31, m32, m33] = (*span
        .multi_pop_front::<9>()
        .expect('dim mismatch'))
        .unbox();
    Mat3F { m11, m12, m13, m21, m22, m23, m31, m32, m33 }
}

#[inline]
fn pack9<T, +Copy<T>, +Drop<T>>(m: Mat3F<T>) -> DMatSeq<T> {
    DMatSeq {
        data: array![m.m11, m.m12, m.m13, m.m21, m.m22, m.m23, m.m31, m.m32, m.m33].span(),
        rows: 3,
        cols: 3,
    }
}

/// "Dynamic storage, static kernel": check the dims at runtime, unbox the 9 + 9 elements with
/// two bounds checks, run the unrolled named-field product, re-pack into an array.
pub fn matmul_dispatch<
    T, +Add<T>, +Sub<T>, +Mul<T>, +Div<T>, +Neg<T>, +PartialEq<T>, +Copy<T>, +Drop<T>,
>(
    lhs: DMatSeq<T>, rhs: DMatSeq<T>,
) -> DMatSeq<T> {
    if lhs.rows == 3 && lhs.cols == 3 && rhs.rows == 3 && rhs.cols == 3 {
        pack9(unpack9(lhs).mul_mat(unpack9(rhs)))
    } else {
        lhs.matmul(rhs)
    }
}

pub impl SeqDMat3<
    T, +Add<T>, +Sub<T>, +Mul<T>, +Div<T>, +Neg<T>, +PartialEq<T>, +Copy<T>, +Drop<T>,
> of Mat3Ops<DMatSeq<T>, DVecLoop<T>, T> {
    fn from_rows(r1: Row<T>, r2: Row<T>, r3: Row<T>) -> DMatSeq<T> {
        let (m11, m12, m13) = r1;
        let (m21, m22, m23) = r2;
        let (m31, m32, m33) = r3;
        DMatSeq {
            data: array![m11, m12, m13, m21, m22, m23, m31, m32, m33].span(), rows: 3, cols: 3,
        }
    }
    fn rows(self: DMatSeq<T>) -> (Row<T>, Row<T>, Row<T>) {
        let m = unpack9(self);
        ((m.m11, m.m12, m.m13), (m.m21, m.m22, m.m23), (m.m31, m.m32, m.m33))
    }
    fn vector(x: T, y: T, z: T) -> DVecLoop<T> {
        DVecLoop { data: array![x, y, z].span() }
    }
    fn vector_xyz(v: DVecLoop<T>) -> Row<T> {
        let mut span = v.data;
        let [x, y, z] = (*span.multi_pop_front::<3>().expect('dim mismatch')).unbox();
        (x, y, z)
    }
    fn mul_vec(self: DMatSeq<T>, v: DVecLoop<T>) -> DVecLoop<T> {
        let product = self.matmul(DMatSeq { data: v.data, rows: v.data.len(), cols: 1 });
        DVecLoop { data: product.data }
    }
    fn mul_mat(self: DMatSeq<T>, rhs: DMatSeq<T>) -> DMatSeq<T> {
        self.matmul(rhs)
    }
    fn transpose(self: DMatSeq<T>) -> DMatSeq<T> {
        self.transposed()
    }
    // 3x3-only operations: unbox the 9 elements with a single bounds check, then run the
    // static kernel (what a "dynamic storage, static kernels" design would do).
    fn determinant(self: DMatSeq<T>) -> T {
        unpack9(self).determinant()
    }
    fn inverse(self: DMatSeq<T>) -> DMatSeq<T> {
        pack9(unpack9(self).inverse())
    }
    fn add(self: DMatSeq<T>, rhs: DMatSeq<T>) -> DMatSeq<T> {
        assert(self.rows == rhs.rows && self.cols == rhs.cols, 'dim mismatch');
        let (mut a, mut b) = (self.data, rhs.data);
        let mut out = array![];
        while let Some(x) = a.pop_front() {
            out.append(*x + *b.pop_front().unwrap());
        }
        DMatSeq { data: out.span(), rows: self.rows, cols: self.cols }
    }
    fn scale(self: DMatSeq<T>, k: T) -> DMatSeq<T> {
        let mut a = self.data;
        let mut out = array![];
        while let Some(x) = a.pop_front() {
            out.append(*x * k);
        }
        DMatSeq { data: out.span(), rows: self.rows, cols: self.cols }
    }
}

// ---------------------------------------------------------------------------------------------
// Generic benchmark bodies.
//   A = [1 2 3; 0 1 4; 5 6 0] (det 1), B = [2 0 -1; 1 3 2; -2 1 1], v = (1, -2, 3), k = 7
// ---------------------------------------------------------------------------------------------

#[cfg(test)]
pub mod runners {
    use crate::fixed::{Scalar, input};
    use super::{Mat3Ops, Row};

    #[inline]
    fn lit<T, +Scalar<T>>(n: i64) -> T {
        Scalar::<T>::from_int(n)
    }

    #[inline]
    fn row<T, +Scalar<T>, +Drop<T>>(a: i64, b: i64, c: i64) -> Row<T> {
        (input(a), input(b), input(c))
    }

    /// All groups share the same 22 opaque inputs.
    #[inline]
    fn inputs<
        T, +Scalar<T>, +Drop<T>,
    >() -> ((Row<T>, Row<T>, Row<T>), (Row<T>, Row<T>, Row<T>), Row<T>, T) {
        (
            (row(1, 2, 3), row(0, 1, 4), row(5, 6, 0)),
            (row(2, 0, -1), row(1, 3, 2), row(-2, 1, 1)),
            row(1, -2, 3),
            input(7),
        )
    }

    #[inline]
    fn row_eq<T, +Scalar<T>, +PartialEq<T>, +Drop<T>>(r: Row<T>, a: i64, b: i64, c: i64) -> bool {
        let (x, y, z) = r;
        x == lit(a) && y == lit(b) && z == lit(c)
    }

    #[inline(never)]
    pub fn baseline9<T, +Scalar<T>, +PartialEq<T>, +Drop<T>>() {
        let ((r1, r2, r3), _, _, _) = inputs::<T>();
        assert!(row_eq(r1, 1, 2, 3) && row_eq(r2, 0, 1, 4) && row_eq(r3, 5, 6, 0));
    }

    #[inline(never)]
    pub fn baseline3<T, +Scalar<T>, +PartialEq<T>, +Drop<T>>() {
        let ((r1, _, _), _, _, _) = inputs::<T>();
        assert!(row_eq(r1, 1, 2, 3));
    }

    #[inline(never)]
    pub fn baseline1<T, +Scalar<T>, +PartialEq<T>, +Drop<T>>() {
        let (_, _, _, k) = inputs::<T>();
        assert!(k == lit(7));
    }

    #[inline(never)]
    pub fn construct<M, V, T, +Mat3Ops<M, V, T>, +Scalar<T>, +PartialEq<T>, +Drop<T>, +Drop<M>>() {
        let ((a1, a2, a3), _, _, _) = inputs::<T>();
        let m: M = Mat3Ops::<M, V, T>::from_rows(a1, a2, a3);
        let (r1, r2, r3) = Mat3Ops::<M, V, T>::rows(m);
        assert!(row_eq(r1, 1, 2, 3) && row_eq(r2, 0, 1, 4) && row_eq(r3, 5, 6, 0));
    }

    #[inline(never)]
    pub fn mul_vec<M, V, T, +Mat3Ops<M, V, T>, +Scalar<T>, +PartialEq<T>, +Drop<T>, +Drop<M>>() {
        let ((a1, a2, a3), _, (x, y, z), _) = inputs::<T>();
        let m: M = Mat3Ops::<M, V, T>::from_rows(a1, a2, a3);
        let v = Mat3Ops::<M, V, T>::mul_vec(m, Mat3Ops::<M, V, T>::vector(x, y, z));
        assert!(row_eq(Mat3Ops::<M, V, T>::vector_xyz(v), 6, 10, -7));
    }

    #[inline(never)]
    pub fn mul_mat<M, V, T, +Mat3Ops<M, V, T>, +Scalar<T>, +PartialEq<T>, +Drop<T>, +Drop<M>>() {
        let ((a1, a2, a3), (b1, b2, b3), _, _) = inputs::<T>();
        let a: M = Mat3Ops::<M, V, T>::from_rows(a1, a2, a3);
        let b: M = Mat3Ops::<M, V, T>::from_rows(b1, b2, b3);
        let (r1, r2, r3) = Mat3Ops::<M, V, T>::rows(Mat3Ops::<M, V, T>::mul_mat(a, b));
        assert!(row_eq(r1, -2, 9, 6) && row_eq(r2, -7, 7, 6) && row_eq(r3, 16, 18, 7));
    }

    #[inline(never)]
    pub fn transpose<M, V, T, +Mat3Ops<M, V, T>, +Scalar<T>, +PartialEq<T>, +Drop<T>, +Drop<M>>() {
        let ((a1, a2, a3), _, _, _) = inputs::<T>();
        let a: M = Mat3Ops::<M, V, T>::from_rows(a1, a2, a3);
        let (r1, r2, r3) = Mat3Ops::<M, V, T>::rows(Mat3Ops::<M, V, T>::transpose(a));
        assert!(row_eq(r1, 1, 0, 5) && row_eq(r2, 2, 1, 6) && row_eq(r3, 3, 4, 0));
    }

    #[inline(never)]
    pub fn determinant<
        M, V, T, +Mat3Ops<M, V, T>, +Scalar<T>, +PartialEq<T>, +Drop<T>, +Drop<M>,
    >() {
        let ((a1, a2, a3), _, _, _) = inputs::<T>();
        let a: M = Mat3Ops::<M, V, T>::from_rows(a1, a2, a3);
        assert!(Mat3Ops::<M, V, T>::determinant(a) == lit(1));
    }

    #[inline(never)]
    pub fn inverse<M, V, T, +Mat3Ops<M, V, T>, +Scalar<T>, +PartialEq<T>, +Drop<T>, +Drop<M>>() {
        let ((a1, a2, a3), _, _, _) = inputs::<T>();
        let a: M = Mat3Ops::<M, V, T>::from_rows(a1, a2, a3);
        let (r1, r2, r3) = Mat3Ops::<M, V, T>::rows(Mat3Ops::<M, V, T>::inverse(a));
        assert!(row_eq(r1, -24, 18, 5) && row_eq(r2, 20, -15, -4) && row_eq(r3, -5, 4, 1));
    }

    #[inline(never)]
    pub fn add<M, V, T, +Mat3Ops<M, V, T>, +Scalar<T>, +PartialEq<T>, +Drop<T>, +Drop<M>>() {
        let ((a1, a2, a3), (b1, b2, b3), _, _) = inputs::<T>();
        let a: M = Mat3Ops::<M, V, T>::from_rows(a1, a2, a3);
        let b: M = Mat3Ops::<M, V, T>::from_rows(b1, b2, b3);
        let (r1, r2, r3) = Mat3Ops::<M, V, T>::rows(Mat3Ops::<M, V, T>::add(a, b));
        assert!(row_eq(r1, 3, 2, 2) && row_eq(r2, 1, 4, 6) && row_eq(r3, 3, 7, 1));
    }

    #[inline(never)]
    pub fn scale<M, V, T, +Mat3Ops<M, V, T>, +Scalar<T>, +PartialEq<T>, +Drop<T>, +Drop<M>>() {
        let ((a1, a2, a3), _, _, k) = inputs::<T>();
        let a: M = Mat3Ops::<M, V, T>::from_rows(a1, a2, a3);
        let (r1, r2, r3) = Mat3Ops::<M, V, T>::rows(Mat3Ops::<M, V, T>::scale(a, k));
        assert!(row_eq(r1, 7, 14, 21) && row_eq(r2, 0, 7, 28) && row_eq(r3, 35, 42, 0));
    }
}
