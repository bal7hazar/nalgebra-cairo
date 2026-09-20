//! Static 6x6 matrix as 2x2 blocks of `Mat3F` (how rapier-style spatial matrices would be built
//! from the static 3x3 type). Used by the matmul scaling experiment.

use crate::mat3::{FieldsMat3, Mat3F, Mat3Ops};

#[derive(Copy, Drop)]
pub struct Mat6B<T> {
    pub a: Mat3F<T>,
    pub b: Mat3F<T>,
    pub c: Mat3F<T>,
    pub d: Mat3F<T>,
}

#[generate_trait]
pub impl Mat6BImpl<
    T, +Add<T>, +Sub<T>, +Mul<T>, +Div<T>, +Neg<T>, +PartialEq<T>, +Copy<T>, +Drop<T>,
> of Mat6BTrait<T> {
    /// [a b; c d] * [e f; g h] = [ae+bg af+bh; ce+dg cf+dh]
    fn mul_mat(self: Mat6B<T>, rhs: Mat6B<T>) -> Mat6B<T> {
        Mat6B {
            a: self.a.mul_mat(rhs.a).add(self.b.mul_mat(rhs.c)),
            b: self.a.mul_mat(rhs.b).add(self.b.mul_mat(rhs.d)),
            c: self.c.mul_mat(rhs.a).add(self.d.mul_mat(rhs.c)),
            d: self.c.mul_mat(rhs.b).add(self.d.mul_mat(rhs.d)),
        }
    }
}
