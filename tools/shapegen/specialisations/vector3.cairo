// Specialisations of `Vector3`, spliced verbatim by shapegen.py (format: `library.py`).
// @method cross
/// Cross product, each component fused (`Real::diff_prod`): the exact difference of products
/// is floored once. Panics on overflow of a result component. Upstream: `cross`.
#[inline(always)]
fn cross(self: Vector3<T>, rhs: Vector3<T>) -> Vector3<T> {
    Vector3 {
        x: R::diff_prod(self.y, rhs.z, self.z, rhs.y),
        y: R::diff_prod(self.z, rhs.x, self.x, rhs.z),
        z: R::diff_prod(self.x, rhs.y, self.y, rhs.x),
    }
}
// @item Vector3InternalTrait impl
/// Crate-internal kernels of `Vector3<T>` with no upstream METHOD of that shape (WP 8.0: the public
/// API is strictly upstream's). `orthonormal_basis` is the fixed-cost form of upstream's
/// callback-based `orthonormal_subspace_basis`, used by `Svd3`.
pub(crate) trait Vector3InternalTrait<T> {
    /// Two unit vectors `(u, w)` orthogonal to `self` and to each other, with `u x w = self`.
    /// `self` MUST be a unit vector (not checked). Branches on the sign of `z` only (Duff et
    /// al., "Building an Orthonormal Basis, Revisited"): two divisions by `1 + |z|` in `[1, 2]`
    /// and fused products, every component is within about 3 ulp. Upstream:
    /// `Vector3::orthonormal_subspace_basis(&[v], ..)` (rapier: `orthonormal_basis`, glam:
    /// `any_orthonormal_pair`). The upstream construction is kept as a benchmark
    /// (`bench_vector3_orthonormal_basis__alt_upstream`).
    fn orthonormal_basis(self: Vector3<T>) -> (Vector3<T>, Vector3<T>);
}
// @item Vector3InternalImpl impl
pub(crate) impl Vector3InternalImpl<
    T,
    impl R: Real<T>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
    +PartialEq<T>,
    +PartialOrd<T>,
    +Copy<T>,
    +Drop<T>,
> of Vector3InternalTrait<T> {
    #[inline(always)]
    fn orthonormal_basis(self: Vector3<T>) -> (Vector3<T>, Vector3<T>) {
        // With d = 1 + |z|, p = x^2 / d, q = x * y / d, r = y^2 / d (Duff et al., signs folded):
        //   z >= 0: u = (1 - p, -q, -x), w = (-q, 1 - r, -y);
        //   z <  0: u = (1 - p, -q,  x), w = ( q, r - 1, -y).
        let d = R::one() + R::abs(self.z);
        let xd = R::div(self.x, d);
        let yd = R::div(self.y, d);
        let q = xd * self.y;
        let ux = R::diff_prod(R::one(), R::one(), xd, self.x);
        let wy = R::diff_prod(R::one(), R::one(), yd, self.y);
        if R::is_sign_negative(self.z) {
            (Vector3 { x: ux, y: -q, z: self.x }, Vector3 { x: q, y: -wy, z: -self.y })
        } else {
            (Vector3 { x: ux, y: -q, z: -self.x }, Vector3 { x: -q, y: wy, z: -self.y })
        }
    }
}
