// Specialisations of `Vector3`: spliced verbatim into the generated impl by shapegen.py
// (`// @method <name>` replaces the generated method of that name, or adds it).
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
