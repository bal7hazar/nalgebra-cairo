//! Vector convolutions (upstream `linalg/convolution.rs`, `Vector::convolve_full` /
//! `convolve_same` / `convolve_valid`) on the static column vectors (`Matrix1`, `Vector2..6`)
//! and on `DVector`, behind the `dynamic` feature (their results are dynamic vectors).
//!
//! The kernel is reversed once, so every output component is the dot product of two contiguous
//! runs (`self[u0..=u1]` and the reversed kernel), ONE exact accumulation floored once
//! (`DynKernels::dot`); upstream accumulates rounded products. The kernel is any vector that
//! converts `Into<DVector>` (a `DVector`, a static column vector...), like upstream's generic
//! `Vector<T, D2, S2>`.

use simba::scalar::Real;
use super::dvector::DVector;
use super::kernels::DynKernels;
use super::super::matrix1::Matrix1;
use super::super::vector2::Vector2;
use super::super::vector3::Vector3;
use super::super::vector4::Vector4;
use super::super::vector5::Vector5;
use super::super::vector6::Vector6;

/// `convolve_*` with an empty kernel or a kernel longer than the vector (upstream panics with
/// "convolve_full expects `self.len() >= kernel.len() > 0`, ...").
pub const CONVOLUTION_KERNEL: felt252 = 'nalgebra: convolution kernel';

/// The convolutions of a column vector `V` by a kernel. Every form panics with `nalgebra:
/// convolution kernel` unless `self.len() >= kernel.len() > 0`, like upstream's.
pub trait Convolution<V, T> {
    /// The full convolution: `n + m - 1` components (`n = self.len()`, `m = kernel.len()`),
    /// `out[i] = sum_u self[u] * kernel[i - u]`. Upstream: `Vector::convolve_full(kernel)`
    /// (`OVector<T, n + m - 1>`; a `DVector` here: Cairo has no dimension arithmetic on types).
    fn convolve_full<K, +Into<K, DVector<T>>, +Drop<K>>(self: V, kernel: K) -> DVector<T>;
    /// The valid convolution: the `n - m + 1` components where the kernel fits entirely,
    /// `out[i] = sum_j self[i + j] * kernel[m - 1 - j]`. Upstream: `Vector::convolve_valid(kernel)`
    /// (a `DVector` here).
    fn convolve_valid<K, +Into<K, DVector<T>>, +Drop<K>>(self: V, kernel: K) -> DVector<T>;
    /// The same-size convolution: `n` components, `out[i] = sum_j self[i + j - 1] * kernel[m - 1 -
    /// j]` over the indices inside `self` (upstream's offset of one, whatever `m`). Upstream:
    /// `Vector::convolve_same(kernel)` (a vector of `self`'s type).
    fn convolve_same<K, +Into<K, DVector<T>>, +Drop<K>>(self: V, kernel: K) -> V;
}

/// The three convolutions on spans (crate-private).
trait ConvKernels<T> {
    fn full(x: Span<T>, k: Span<T>) -> Span<T>;
    fn valid(x: Span<T>, k: Span<T>) -> Span<T>;
    fn same(x: Span<T>, k: Span<T>) -> Span<T>;
}

/// `k` reversed, after the length check.
fn reversed<T, +Copy<T>, +Drop<T>>(x: Span<T>, mut k: Span<T>) -> Span<T> {
    let m = k.len();
    if m == 0 || m > x.len() {
        core::panic_with_felt252(CONVOLUTION_KERNEL);
    }
    let mut out: Array<T> = array![];
    while let Some(v) = k.pop_back() {
        out.append(*v);
    }
    out.span()
}

impl ConvKernelsImpl<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>, +Add<T>, +Sub<T>, +Mul<T>, +Neg<T>,
> of ConvKernels<T> {
    fn full(x: Span<T>, k: Span<T>) -> Span<T> {
        let kr = reversed(x, k);
        let n = x.len();
        let m = kr.len();
        let mut out: Array<T> = array![];
        let mut i: usize = 0;
        while i != n + m - 1 {
            // u in [u0, u1]: u0 = max(0, i - m + 1), u1 = min(i, n - 1); kernel[i - u] is
            // kr[m - 1 - i + u].
            let u0 = if i + 1 > m {
                i + 1 - m
            } else {
                0
            };
            let u1 = if i < n {
                i
            } else {
                n - 1
            };
            let len = u1 + 1 - u0;
            out.append(DynKernels::dot(x.slice(u0, len), kr.slice(m - 1 + u0 - i, len)));
            i += 1;
        }
        out.span()
    }

    fn valid(x: Span<T>, k: Span<T>) -> Span<T> {
        let kr = reversed(x, k);
        let n = x.len();
        let m = kr.len();
        let mut out: Array<T> = array![];
        let mut i: usize = 0;
        while i != n - m + 1 {
            out.append(DynKernels::dot(x.slice(i, m), kr));
            i += 1;
        }
        out.span()
    }

    fn same(x: Span<T>, k: Span<T>) -> Span<T> {
        let kr = reversed(x, k);
        let n = x.len();
        let m = kr.len();
        let mut out: Array<T> = array![];
        let mut i: usize = 0;
        while i != n {
            // j in [j0, j1] with 0 <= i + j - 1 < n: j0 = max(0, 1 - i), j1 = min(m - 1, n - i);
            // kernel[m - 1 - j] is kr[j].
            let j0 = if i == 0 {
                1
            } else {
                0
            };
            let j1 = if m - 1 < n - i {
                m - 1
            } else {
                n - i
            };
            if j1 + 1 > j0 {
                let len = j1 + 1 - j0;
                out.append(DynKernels::dot(x.slice(i + j0 - 1, len), kr.slice(j0, len)));
            } else {
                out.append(R::zero());
            }
            i += 1;
        }
        out.span()
    }
}

/// The convolutions of a `DVector`.
pub impl DVectorConvolution<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>, +Add<T>, +Sub<T>, +Mul<T>, +Neg<T>,
> of Convolution<DVector<T>, T> {
    fn convolve_full<K, +Into<K, DVector<T>>, +Drop<K>>(self: DVector<T>, kernel: K) -> DVector<T> {
        let k: DVector<T> = kernel.into();
        DVector { data: ConvKernels::full(self.data, k.data) }
    }

    fn convolve_valid<K, +Into<K, DVector<T>>, +Drop<K>>(
        self: DVector<T>, kernel: K,
    ) -> DVector<T> {
        let k: DVector<T> = kernel.into();
        DVector { data: ConvKernels::valid(self.data, k.data) }
    }

    fn convolve_same<K, +Into<K, DVector<T>>, +Drop<K>>(self: DVector<T>, kernel: K) -> DVector<T> {
        let k: DVector<T> = kernel.into();
        DVector { data: ConvKernels::same(self.data, k.data) }
    }
}

/// The convolutions of a `Matrix1`.
pub impl Matrix1Convolution<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>, +Add<T>, +Sub<T>, +Mul<T>, +Neg<T>,
> of Convolution<Matrix1<T>, T> {
    fn convolve_full<K, +Into<K, DVector<T>>, +Drop<K>>(self: Matrix1<T>, kernel: K) -> DVector<T> {
        let k: DVector<T> = kernel.into();
        DVector { data: ConvKernels::full(array![self.x].span(), k.data) }
    }

    fn convolve_valid<K, +Into<K, DVector<T>>, +Drop<K>>(
        self: Matrix1<T>, kernel: K,
    ) -> DVector<T> {
        let k: DVector<T> = kernel.into();
        DVector { data: ConvKernels::valid(array![self.x].span(), k.data) }
    }

    fn convolve_same<K, +Into<K, DVector<T>>, +Drop<K>>(self: Matrix1<T>, kernel: K) -> Matrix1<T> {
        let k: DVector<T> = kernel.into();
        let boxed: @Box<[T; 1]> = ConvKernels::same(array![self.x].span(), k.data)
            .try_into()
            .unwrap();
        let [a0] = boxed.unbox();
        Matrix1 { x: a0 }
    }
}

/// The convolutions of a `Vector2`.
pub impl Vector2Convolution<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>, +Add<T>, +Sub<T>, +Mul<T>, +Neg<T>,
> of Convolution<Vector2<T>, T> {
    fn convolve_full<K, +Into<K, DVector<T>>, +Drop<K>>(self: Vector2<T>, kernel: K) -> DVector<T> {
        let k: DVector<T> = kernel.into();
        DVector { data: ConvKernels::full(array![self.x, self.y].span(), k.data) }
    }

    fn convolve_valid<K, +Into<K, DVector<T>>, +Drop<K>>(
        self: Vector2<T>, kernel: K,
    ) -> DVector<T> {
        let k: DVector<T> = kernel.into();
        DVector { data: ConvKernels::valid(array![self.x, self.y].span(), k.data) }
    }

    fn convolve_same<K, +Into<K, DVector<T>>, +Drop<K>>(self: Vector2<T>, kernel: K) -> Vector2<T> {
        let k: DVector<T> = kernel.into();
        let boxed: @Box<[T; 2]> = ConvKernels::same(array![self.x, self.y].span(), k.data)
            .try_into()
            .unwrap();
        let [a0, a1] = boxed.unbox();
        Vector2 { x: a0, y: a1 }
    }
}

/// The convolutions of a `Vector3`.
pub impl Vector3Convolution<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>, +Add<T>, +Sub<T>, +Mul<T>, +Neg<T>,
> of Convolution<Vector3<T>, T> {
    fn convolve_full<K, +Into<K, DVector<T>>, +Drop<K>>(self: Vector3<T>, kernel: K) -> DVector<T> {
        let k: DVector<T> = kernel.into();
        DVector { data: ConvKernels::full(array![self.x, self.y, self.z].span(), k.data) }
    }

    fn convolve_valid<K, +Into<K, DVector<T>>, +Drop<K>>(
        self: Vector3<T>, kernel: K,
    ) -> DVector<T> {
        let k: DVector<T> = kernel.into();
        DVector { data: ConvKernels::valid(array![self.x, self.y, self.z].span(), k.data) }
    }

    fn convolve_same<K, +Into<K, DVector<T>>, +Drop<K>>(self: Vector3<T>, kernel: K) -> Vector3<T> {
        let k: DVector<T> = kernel.into();
        let boxed: @Box<[T; 3]> = ConvKernels::same(array![self.x, self.y, self.z].span(), k.data)
            .try_into()
            .unwrap();
        let [a0, a1, a2] = boxed.unbox();
        Vector3 { x: a0, y: a1, z: a2 }
    }
}

/// The convolutions of a `Vector4`.
pub impl Vector4Convolution<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>, +Add<T>, +Sub<T>, +Mul<T>, +Neg<T>,
> of Convolution<Vector4<T>, T> {
    fn convolve_full<K, +Into<K, DVector<T>>, +Drop<K>>(self: Vector4<T>, kernel: K) -> DVector<T> {
        let k: DVector<T> = kernel.into();
        DVector { data: ConvKernels::full(array![self.x, self.y, self.z, self.w].span(), k.data) }
    }

    fn convolve_valid<K, +Into<K, DVector<T>>, +Drop<K>>(
        self: Vector4<T>, kernel: K,
    ) -> DVector<T> {
        let k: DVector<T> = kernel.into();
        DVector { data: ConvKernels::valid(array![self.x, self.y, self.z, self.w].span(), k.data) }
    }

    fn convolve_same<K, +Into<K, DVector<T>>, +Drop<K>>(self: Vector4<T>, kernel: K) -> Vector4<T> {
        let k: DVector<T> = kernel.into();
        let boxed: @Box<[T; 4]> = ConvKernels::same(
            array![self.x, self.y, self.z, self.w].span(), k.data,
        )
            .try_into()
            .unwrap();
        let [a0, a1, a2, a3] = boxed.unbox();
        Vector4 { x: a0, y: a1, z: a2, w: a3 }
    }
}

/// The convolutions of a `Vector5`.
pub impl Vector5Convolution<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>, +Add<T>, +Sub<T>, +Mul<T>, +Neg<T>,
> of Convolution<Vector5<T>, T> {
    fn convolve_full<K, +Into<K, DVector<T>>, +Drop<K>>(self: Vector5<T>, kernel: K) -> DVector<T> {
        let k: DVector<T> = kernel.into();
        DVector {
            data: ConvKernels::full(array![self.x, self.y, self.z, self.w, self.a].span(), k.data),
        }
    }

    fn convolve_valid<K, +Into<K, DVector<T>>, +Drop<K>>(
        self: Vector5<T>, kernel: K,
    ) -> DVector<T> {
        let k: DVector<T> = kernel.into();
        DVector {
            data: ConvKernels::valid(array![self.x, self.y, self.z, self.w, self.a].span(), k.data),
        }
    }

    fn convolve_same<K, +Into<K, DVector<T>>, +Drop<K>>(self: Vector5<T>, kernel: K) -> Vector5<T> {
        let k: DVector<T> = kernel.into();
        let boxed: @Box<[T; 5]> = ConvKernels::same(
            array![self.x, self.y, self.z, self.w, self.a].span(), k.data,
        )
            .try_into()
            .unwrap();
        let [a0, a1, a2, a3, a4] = boxed.unbox();
        Vector5 { x: a0, y: a1, z: a2, w: a3, a: a4 }
    }
}

/// The convolutions of a `Vector6`.
pub impl Vector6Convolution<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>, +Add<T>, +Sub<T>, +Mul<T>, +Neg<T>,
> of Convolution<Vector6<T>, T> {
    fn convolve_full<K, +Into<K, DVector<T>>, +Drop<K>>(self: Vector6<T>, kernel: K) -> DVector<T> {
        let k: DVector<T> = kernel.into();
        DVector {
            data: ConvKernels::full(
                array![self.x, self.y, self.z, self.w, self.a, self.b].span(), k.data,
            ),
        }
    }

    fn convolve_valid<K, +Into<K, DVector<T>>, +Drop<K>>(
        self: Vector6<T>, kernel: K,
    ) -> DVector<T> {
        let k: DVector<T> = kernel.into();
        DVector {
            data: ConvKernels::valid(
                array![self.x, self.y, self.z, self.w, self.a, self.b].span(), k.data,
            ),
        }
    }

    fn convolve_same<K, +Into<K, DVector<T>>, +Drop<K>>(self: Vector6<T>, kernel: K) -> Vector6<T> {
        let k: DVector<T> = kernel.into();
        let boxed: @Box<[T; 6]> = ConvKernels::same(
            array![self.x, self.y, self.z, self.w, self.a, self.b].span(), k.data,
        )
            .try_into()
            .unwrap();
        let [a0, a1, a2, a3, a4, a5] = boxed.unbox();
        Vector6 { x: a0, y: a1, z: a2, w: a3, a: a4, b: a5 }
    }
}
