// Specialisations of `Matrix3`: spliced verbatim into the generated impl by shapegen.py
// (`// @method <name>` replaces the generated method of that name, or adds it).
// @method determinant
/// The determinant, by cofactor expansion along the first row: 3 `diff_prod` (2x2 minors,
/// one rounding each) then one `sum_prod3` (second rounding). The error against the exact
/// floor is at most `|m11| + |m12| + |m13| + 1` ulp (values, not raw: 4 ulp for a rotation).
/// Panics on overflow. Upstream: `determinant`.
fn determinant(self: Matrix3<T>) -> T {
    R::sum_prod3(
        self.m11,
        R::diff_prod(self.m22, self.m33, self.m23, self.m32),
        self.m12,
        R::diff_prod(self.m23, self.m31, self.m21, self.m33),
        self.m13,
        R::diff_prod(self.m21, self.m32, self.m22, self.m31),
    )
}
