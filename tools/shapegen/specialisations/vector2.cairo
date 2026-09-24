// Specialisations of `Vector2`, spliced verbatim by shapegen.py (format: `library.py`).
// @method perp
/// Perpendicular (2D cross) product `self.x * rhs.y - self.y * rhs.x`, fused
/// (`Real::diff_prod`): the exact difference of products is floored once. Panics on overflow
/// of the result. Upstream: `perp`.
#[inline(always)]
fn perp(self: Vector2<T>, rhs: Vector2<T>) -> T {
    R::diff_prod(self.x, rhs.y, self.y, rhs.x)
}
