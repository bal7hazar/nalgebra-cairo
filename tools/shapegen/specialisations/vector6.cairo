// Specialisations of `Vector6`, spliced verbatim by shapegen.py (format: `library.py`). The
// kernels are the templates' (6-term `Real::Wide` chains); these sections keep the measured notes
// of the hand-written block type in the doc comments.
// @doc module
//!
//! The 6-term sums (`dot`, `norm_squared`, `norm`) use the explicit `Real::Wide` accumulator, so
//! they cost ONE floor rounding and ONE overflow check, exactly like the 3-term `sum_prod3`
//! kernels — never two chained `sum_prod3`, which would round twice.
// @doc unscale
/// `self / k`, each component being the correctly rounded quotient (`Real::div6`: one prepared
/// divisor). Panics on a zero `k` and on overflow. Upstream: `unscale` (`self / k`).
///
/// One division per component on purpose, like `Vector3::unscale`: `scale(k.recip())` is
/// cheaper but rounds `1 / k` first, which costs up to `|self|` ulp instead of 1.
// @doc dot
/// Dot product of the six components: the products are accumulated EXACTLY in the `Real::Wide`
/// accumulator and rescaled ONCE (one floor, one overflow check), so the result is the exact
/// floor of the mathematical dot product. Only the result must fit. Upstream: `dot`.
///
/// Two chained `sum_prod3` (`head.dot(rhs.head) + tail.dot(rhs.tail)`) would round twice — up
/// to 1 ulp off the exact floor AND 1.87x dearer (5 140 against 2 750 net), because a
/// `sum_prod3` pays a full rescale where an extra `wide_add_prod` costs about 200 gas. Kept as
/// `bench_vector6_dot__alt_two_sum_prod3`; one rounded product per term is 5.35x dearer
/// (`__alt_unfused`).
// @doc normalize
/// `self / self.norm()`: the floored norm, then one correctly rounded division per component
/// (`unscale`). The error is about `1 + 1 / norm` ulp per component whatever the magnitude of
/// `self`. Panics with a division by zero when the norm is zero, and on overflow when the norm
/// does not fit. Upstream: `normalize`.
