// EXTRACTED from orion @ bac0b42 (src/operators/matrix.cairo; argmax / softmax / sigmoid dropped): only the items needed by the benchmarks are kept;
// function bodies are verbatim unless marked PATCH.

use orion::numbers::NumberTrait;
use orion::operators::vec::{VecTrait, NullableVec, NullableVecImpl};

struct MutMatrix<T> {
    data: NullableVec<T>,
    rows: usize,
    cols: usize,
}

impl MutMatrixDestruct<T, +Drop<T>> of Destruct<MutMatrix<T>> {
    fn destruct(self: MutMatrix<T>) nopanic {
        self.data.destruct()
    }
}

#[generate_trait]
impl MutMatrixImpl<
    T, MAG, +Drop<T>, +Copy<T>, +NumberTrait<T, MAG>, +PartialOrd<T>
> of MutMatrixTrait<T> {
    /// Constructor for the Matrix
    fn new(rows: usize, cols: usize) -> MutMatrix<T> {
        MutMatrix { data: NullableVecImpl::new(), rows: rows, cols: cols }
    }

    /// Get the value at (row, col)
    fn get(ref self: MutMatrix<T>, row: usize, col: usize) -> Option<T> {
        if row >= self.rows || col >= self.cols {
            Option::None
        } else {
            self.data.get(row * self.cols + col)
        }
    }

    /// Get the value at (row, col)
    fn at(ref self: MutMatrix<T>, row: usize, col: usize) -> T {
        match self.get(row, col) {
            Option::Some(val) => val,
            Option::None => NumberTrait::zero(),
        }
    }

    /// Performs the product between a m x n `MutMatrix<T>` and a n x 1 `NullableVec<T>`. 
    /// Returns the resulta as a `NullableVec<T>`.
    fn matrix_vector_product<+Mul<T>, +Add<T>, +Div<T>, +AddEq<T>>(
        ref self: MutMatrix<T>, ref vec: NullableVec<T>
    ) -> NullableVec<T> {
        assert(self.cols == vec.len, 'wrong matrix shape for dot');

        let m = self.rows;
        let n = self.cols;

        let mut result_vec = VecTrait::new();

        let mut i = 0_usize;
        while i != m {
            let mut sum: T = NumberTrait::zero();
            let mut k = 0_usize;
            while k != n {
                sum += MutMatrixImpl::at(ref self, i, k) * VecTrait::at(ref vec, k);

                k += 1;
            };

            VecTrait::set(ref result_vec, i, sum);
            i += 1;
        };

        result_vec
    }

    /// Set the value at (row, col)
    fn set(ref self: MutMatrix<T>, row: usize, col: usize, value: T) {
        if row < self.rows && col < self.cols {
            let index = row * self.cols + col;

            self.data.set(index, value)
        }
    }

    /// Returns the shape of the matrix as (rows, cols)
    fn shape(self: MutMatrix<T>) -> (usize, usize) {
        (self.rows, self.cols)
    }
}
