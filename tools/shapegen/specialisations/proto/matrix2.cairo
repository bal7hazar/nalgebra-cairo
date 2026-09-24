// Specialisations of `Matrix2`: spliced verbatim into the generated impl by shapegen.py
// (`// @method <name>` replaces the generated method of that name, or adds it).
// @method determinant
/// `m11 * m22 - m12 * m21` with a single rounding (`diff_prod`): the exact floor of the true
/// determinant. Panics on overflow. Upstream: `determinant`.
#[inline(always)]
fn determinant(self: Matrix2<T>) -> T {
    R::diff_prod(self.m11, self.m22, self.m12, self.m21)
}
