//! `nalgebra::base`: Vector2/3/4/6, Matrix2/3/4/6, Point2/3.

use super::{dm, flat, ring, sm, sv, Ring};
use crate::engine::{fbool, fm, fs, fv, im, is, it, iv, with, Op, Suite, Tol};
use crate::gen::{Dist, Gen};
use nalgebra::{center, distance, Point, SVector};

const NORMALIZE: Tol = Tol::Sens { k: 2.0, base: 2.0 };

// Generic bilinear kernels, run by upstream on f64 and on exact i128 raws ------------------------

fn dot<T: Ring, const N: usize>(x: &[T]) -> Vec<T> {
    vec![sv::<T, N>(x).dot(&sv::<T, N>(&x[N..]))]
}
fn norm_squared<T: Ring, const N: usize>(x: &[T]) -> Vec<T> {
    // `norm_squared` needs a ComplexField; upstream defines it as `self.dotc(self)`.
    vec![sv::<T, N>(x).dot(&sv::<T, N>(x))]
}
fn scale<T: Ring, const N: usize>(x: &[T]) -> Vec<T> {
    flat(&(sv::<T, N>(x) * x[N]))
}
fn component_mul<T: Ring, const N: usize>(x: &[T]) -> Vec<T> {
    flat(&sv::<T, N>(x).component_mul(&sv::<T, N>(&x[N..])))
}
fn cross<T: Ring>(x: &[T]) -> Vec<T> {
    flat(&sv::<T, 3>(x).cross(&sv::<T, 3>(&x[3..])))
}
fn perp<T: Ring>(x: &[T]) -> Vec<T> {
    vec![sv::<T, 2>(x).perp(&sv::<T, 2>(&x[2..]))]
}
fn outer<T: Ring, const N: usize>(x: &[T]) -> Vec<T> {
    flat(&(sv::<T, N>(x) * sv::<T, N>(&x[N..]).transpose()))
}
fn distance_squared<T: Ring, const N: usize>(x: &[T]) -> Vec<T> {
    let d = sv::<T, N>(x) - sv::<T, N>(&x[N..]);
    vec![d.dot(&d)]
}
pub fn mat_mul<T: Ring, const N: usize>(x: &[T]) -> Vec<T> {
    flat(&(sm::<T, N, N>(x) * sm::<T, N, N>(&x[N * N..])))
}
pub fn mat_tr_mul<T: Ring, const N: usize>(x: &[T]) -> Vec<T> {
    flat(&sm::<T, N, N>(x).tr_mul(&sm::<T, N, N>(&x[N * N..])))
}
pub fn mat_mul_vec<T: Ring, const N: usize>(x: &[T]) -> Vec<T> {
    flat(&(sm::<T, N, N>(x) * sv::<T, N>(&x[N * N..])))
}
fn mat_scale<T: Ring, const N: usize>(x: &[T]) -> Vec<T> {
    flat(&(sm::<T, N, N>(x) * x[N * N]))
}
fn det2<T: Ring>(x: &[T]) -> Vec<T> {
    // Upstream's closed form for 2x2 (`Matrix::determinant` needs a ComplexField).
    vec![x[0] * x[3] - x[1] * x[2]]
}

// Vectors ---------------------------------------------------------------------------------------

fn vector_ops<const N: usize>(full: bool) -> Vec<Op> {
    let p = format!("vector{N}");
    let a = || iv("a", N);
    let b = || iv("b", N);
    let v = |x: &[f64]| sv::<f64, N>(x);
    let mut ops = vec![
        Op::new(format!("{p}_add"), "a + b")
            .input(a())
            .input(b())
            .out(fv("sum", N))
            .eval(move |x| Some(flat(&(v(x) + v(&x[N..]))))),
        Op::new(format!("{p}_sub"), "a - b")
            .input(a())
            .input(b())
            .out(fv("difference", N))
            .eval(move |x| Some(flat(&(v(x) - v(&x[N..]))))),
        Op::new(format!("{p}_neg"), "-a")
            .input(a())
            .out(fv("neg", N))
            .eval(move |x| Some(flat(&(-v(x))))),
        Op::new(format!("{p}_scale"), "a * k")
            .input(a())
            .input(is("k"))
            .out(fv("scaled", N))
            .ring(ring!(scale, N)),
        Op::new(format!("{p}_dot"), "a.dot(&b)")
            .input(a())
            .input(b())
            .out(fs("dot"))
            .ring(ring!(dot, N)),
        Op::new(format!("{p}_norm_squared"), "a.norm_squared()")
            .input(a())
            .out(fs("norm_squared"))
            .ring(ring!(norm_squared, N)),
        Op::new(format!("{p}_norm"), "a.norm()")
            .input(a())
            .out(fs("norm"))
            .tol(Tol::Ulp(2))
            .eval(move |x| Some(vec![v(x).norm()])),
    ];
    if !full {
        return ops;
    }
    ops.extend([
        Op::new(format!("{p}_normalize"), "a.normalize()")
            .input(a())
            .out(fv("unit", N))
            .tol(NORMALIZE)
            .eval(move |x| Some(flat(&v(x).normalize()))),
        Op::new(format!("{p}_lerp"), "a.lerp(&b, t)")
            .input(a())
            .input(b())
            .input(it("t"))
            .out(fv("lerp", N))
            .tol(Tol::Ulp(2))
            .eval(move |x| Some(flat(&v(x).lerp(&v(&x[N..]), x[2 * N])))),
        Op::new(format!("{p}_angle"), "a.angle(&b)")
            .input(a())
            .input(b())
            .out(fs("angle"))
            .dists(&Dist::NO_LARGE)
            .tol(Tol::Sens { k: 8.0, base: 16.0 })
            .eval(move |x| {
                let (a, b) = (v(x), v(&x[N..]));
                let angle = a.angle(&b);
                // Nearly parallel vectors make acos-based implementations lose half the bits.
                (libm::sin(angle) > 0.05).then_some(vec![angle])
            }),
        Op::new(format!("{p}_component_mul"), "a.component_mul(&b)")
            .input(a())
            .input(b())
            .out(fv("product", N))
            .ring(ring!(component_mul, N)),
        Op::new(format!("{p}_component_div"), "a.component_div(&b)")
            .input(a())
            .input(b())
            .out(fv("quotient", N))
            .tol(Tol::Ulp(1))
            .eval(move |x| Some(flat(&v(x).component_div(&v(&x[N..]))))),
        Op::new(format!("{p}_abs"), "a.abs()")
            .input(a())
            .out(fv("abs", N))
            .eval(move |x| Some(flat(&v(x).abs()))),
        Op::new(format!("{p}_min"), "a.min()")
            .input(a())
            .out(fs("min"))
            .eval(move |x| Some(vec![v(x).min()])),
        Op::new(format!("{p}_max"), "a.max()")
            .input(a())
            .out(fs("max"))
            .eval(move |x| Some(vec![v(x).max()])),
        Op::new(format!("{p}_inf"), "a.inf(&b) (component-wise min)")
            .input(a())
            .input(b())
            .out(fv("inf", N))
            .eval(move |x| Some(flat(&v(x).inf(&v(&x[N..]))))),
        Op::new(format!("{p}_sup"), "a.sup(&b) (component-wise max)")
            .input(a())
            .input(b())
            .out(fv("sup", N))
            .eval(move |x| Some(flat(&v(x).sup(&v(&x[N..]))))),
    ]);
    if N == 2 {
        ops.push(
            Op::new("vector2_perp", "a.perp(&b) = a.x * b.y - a.y * b.x")
                .input(a())
                .input(b())
                .out(fs("perp"))
                .ring(ring!(perp)),
        );
    }
    if N == 3 {
        ops.push(
            Op::new("vector3_cross", "a.cross(&b)")
                .input(a())
                .input(b())
                .out(fv("cross", 3))
                .ring(ring!(cross)),
        );
    }
    ops
}

// Matrices --------------------------------------------------------------------------------------

/// Rounding model of a fixed-point determinant (3x3: six triple products or three minors;
/// 4x4: products of 2x2 minors), in ulp.
fn det_budget(n: usize, x: &[f64]) -> f64 {
    let amax = x.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    match n {
        3 => 6.0 * amax + 6.0,
        // The last term is the f64 oracle's own error (24 products of four entries, 2^-52
        // relative, in ulp); the 3x3 expectation is exact.
        4 => 24.0 * amax * amax + 24.0 * amax + 8.0 + 24.0 * amax.powi(4) / 1_048_576.0,
        _ => unreachable!(),
    }
}

/// Exact 3x3 determinant on raws (scale 2^96), `None` on i128 overflow.
fn det3_exact(m: &[i128]) -> Option<Vec<i128>> {
    let minor = |a: usize, b: usize, c: usize, d: usize| -> Option<i128> {
        m[a].checked_mul(m[b])?.checked_sub(m[c].checked_mul(m[d])?)
    };
    let t0 = m[0].checked_mul(minor(4, 8, 5, 7)?)?;
    let t1 = m[1].checked_mul(minor(3, 8, 5, 6)?)?;
    let t2 = m[2].checked_mul(minor(3, 7, 4, 6)?)?;
    Some(vec![t0.checked_sub(t1)?.checked_add(t2)?])
}

fn inverse_tol(n: usize) -> Tol {
    Tol::Sens {
        k: 2.0 * n as f64,
        base: 4.0,
    }
}

fn matrix_ops<const N: usize>(full: bool) -> Vec<Op> {
    let p = format!("matrix{N}");
    let a = || im("a", N, N);
    let b = || im("b", N, N);
    let m = |x: &[f64]| sm::<f64, N, N>(x);
    let mut ops = vec![
        Op::new(format!("{p}_add"), "a + b")
            .input(a())
            .input(b())
            .out(fm("sum", N, N))
            .eval(move |x| Some(flat(&(m(x) + m(&x[N * N..]))))),
        Op::new(format!("{p}_sub"), "a - b")
            .input(a())
            .input(b())
            .out(fm("difference", N, N))
            .eval(move |x| Some(flat(&(m(x) - m(&x[N * N..]))))),
        Op::new(format!("{p}_scale"), "a * k")
            .input(a())
            .input(is("k"))
            .out(fm("scaled", N, N))
            .ring(ring!(mat_scale, N)),
        Op::new(format!("{p}_mul"), "a * b")
            .input(a())
            .input(b())
            .out(fm("product", N, N))
            .ring(ring!(mat_mul, N)),
        Op::new(format!("{p}_mul_vec"), "a * v")
            .input(a())
            .input(iv("v", N))
            .out(fv("product", N))
            .ring(ring!(mat_mul_vec, N)),
        Op::new(format!("{p}_transpose"), "a.transpose()")
            .input(a())
            .out(fm("transpose", N, N))
            .eval(move |x| Some(flat(&m(x).transpose()))),
        Op::new(format!("{p}_trace"), "a.trace()")
            .input(a())
            .out(fs("trace"))
            .eval(move |x| Some(vec![m(x).trace()])),
    ];
    if !full {
        return ops;
    }
    ops.extend([
        Op::new(format!("{p}_tr_mul"), "a.tr_mul(&b) = a.transpose() * b")
            .input(a())
            .input(b())
            .out(fm("product", N, N))
            .ring(ring!(mat_tr_mul, N)),
        Op::new(format!("{p}_outer"), "u * v.transpose()")
            .input(iv("u", N))
            .input(iv("v", N))
            .out(fm("outer", N, N))
            .ring(ring!(outer, N)),
        Op::new(format!("{p}_try_inverse"), "a.try_inverse().unwrap()")
            .input(with(fm("a", N, N), Gen::WellCond(N)))
            .out(fm("inverse", N, N))
            .dists(&Dist::NO_LARGE)
            .tol(inverse_tol(N))
            .eval(move |x| dm(x, N).try_inverse().map(|inv| flat(&inv))),
        Op::new(
            format!("{p}_try_inverse_singular"),
            "a.try_inverse().is_some() on exactly singular integer-valued matrices",
        )
        .input(with(fm("a", N, N), Gen::Singular(N)))
        .out(fbool("is_some"))
        .dists(&Dist::UNIT)
        .eval(move |x| {
            let det = dm(x, N).determinant();
            assert!(
                det.abs() < 1.0e-6,
                "singular generator produced det = {det}"
            );
            Some(vec![0.0])
        }),
        Op::new(
            format!("{p}_try_inverse_near_singular"),
            "a.try_inverse().unwrap(), condition number 1e2..1e4 (FLAGGED: loose tolerance)",
        )
        .input(with(fm("a", N, N), Gen::NearSingular(N)))
        .out(fm("inverse", N, N))
        .dists(&Dist::UNIT)
        .cap(1 << 44)
        .tol(inverse_tol(N))
        .eval(move |x| dm(x, N).try_inverse().map(|inv| flat(&inv))),
    ]);
    let determinant = Op::new(format!("{p}_determinant"), "a.determinant()")
        .input(a())
        .out(fs("determinant"))
        .dists(&Dist::NO_LARGE);
    let eval = move |x: &[f64]| Some(vec![dm(x, N).determinant()]);
    ops.push(match N {
        2 => determinant.dists(&Dist::ALL).ring(ring!(det2)),
        3 => determinant
            .ring((Box::new(eval), Box::new(det3_exact)))
            .degree(3)
            .tol(Tol::Model(
                Box::new(move |x, _| det_budget(3, x)),
                "ceil(6 * amax + 6), amax = max |a_ij|: one-ulp errors on the 2x2 minors (or on \
                 the six pair products), multiplied by the remaining factor",
            )),
        _ => determinant
            .tol(Tol::Model(
                Box::new(move |x, _| det_budget(4, x)),
                "ceil(24 * amax^2 + 24 * amax + 8 + 24 * amax^4 * 2^-20), amax = max |a_ij|: \
                 one-ulp errors on the 2x2 minors multiplied by the complementary minors, plus \
                 the f64 oracle's own error",
            ))
            .eval(eval),
    });
    if N == 3 {
        ops.push(
            Op::new("matrix3_cross_matrix", "v.cross_matrix()")
                .input(iv("v", 3))
                .out(fm("cross_matrix", 3, 3))
                .eval(|x| Some(flat(&sv::<f64, 3>(x).cross_matrix()))),
        );
    }
    ops
}

// Points ----------------------------------------------------------------------------------------

fn point_ops<const N: usize>() -> Vec<Op> {
    let p = format!("point{N}");
    let pt = |x: &[f64]| Point::<f64, N>::from(sv::<f64, N>(x));
    let coords = |q: Point<f64, N>| -> SVector<f64, N> { q.coords };
    vec![
        Op::new(format!("{p}_sub"), "p - q (a vector)")
            .input(iv("p", N))
            .input(iv("q", N))
            .out(fv("difference", N))
            .eval(move |x| Some(flat(&(pt(x) - pt(&x[N..]))))),
        Op::new(format!("{p}_add_vector"), "p + v")
            .input(iv("p", N))
            .input(iv("v", N))
            .out(fv("point", N))
            .eval(move |x| Some(flat(&coords(pt(x) + sv::<f64, N>(&x[N..]))))),
        Op::new(format!("{p}_sub_vector"), "p - v")
            .input(iv("p", N))
            .input(iv("v", N))
            .out(fv("point", N))
            .eval(move |x| Some(flat(&coords(pt(x) - sv::<f64, N>(&x[N..]))))),
        Op::new(format!("{p}_scale"), "p * k")
            .input(iv("p", N))
            .input(is("k"))
            .out(fv("point", N))
            .ring(ring!(scale, N)),
        Op::new(format!("{p}_distance_squared"), "distance_squared(&p, &q)")
            .input(iv("p", N))
            .input(iv("q", N))
            .out(fs("distance_squared"))
            .ring(ring!(distance_squared, N)),
        Op::new(format!("{p}_distance"), "distance(&p, &q)")
            .input(iv("p", N))
            .input(iv("q", N))
            .out(fs("distance"))
            .tol(Tol::Ulp(2))
            .eval(move |x| Some(vec![distance(&pt(x), &pt(&x[N..]))])),
        Op::new(format!("{p}_center"), "center(&p, &q)")
            .input(iv("p", N))
            .input(iv("q", N))
            .out(fv("center", N))
            .tol(Tol::Ulp(1))
            .eval(move |x| Some(flat(&coords(center(&pt(x), &pt(&x[N..])))))),
    ]
}

pub fn suites() -> Vec<Suite> {
    let mut dim6 = vector_ops::<6>(false);
    dim6.extend(matrix_ops::<6>(false));
    vec![
        Suite {
            name: "vector2",
            description:
                "Vector2: add, sub, neg, scale, dot, perp, norms, normalize, lerp, angle, \
                          component-wise ops, min / max",
            ops: vector_ops::<2>(true),
        },
        Suite {
            name: "vector3",
            description:
                "Vector3: add, sub, neg, scale, dot, cross, norms, normalize, lerp, angle, \
                          component-wise ops, min / max",
            ops: vector_ops::<3>(true),
        },
        Suite {
            name: "vector4",
            description: "Vector4: add, sub, neg, scale, dot, norms, normalize, lerp, angle, \
                          component-wise ops, min / max",
            ops: vector_ops::<4>(true),
        },
        Suite {
            name: "matrix2",
            description: "Matrix2: add, sub, scale, mul, mul_vec, tr_mul, transpose, trace, \
                          determinant, try_inverse (regular, singular, near-singular), outer",
            ops: matrix_ops::<2>(true),
        },
        Suite {
            name: "matrix3",
            description: "Matrix3: add, sub, scale, mul, mul_vec, tr_mul, transpose, trace, \
                          determinant, try_inverse (regular, singular, near-singular), outer, \
                          cross_matrix",
            ops: matrix_ops::<3>(true),
        },
        Suite {
            name: "matrix4",
            description: "Matrix4: add, sub, scale, mul, mul_vec, tr_mul, transpose, trace, \
                          determinant, try_inverse (regular, singular, near-singular), outer",
            ops: matrix_ops::<4>(true),
        },
        Suite {
            name: "dim6",
            description: "Vector6 and Matrix6 basics (plain 6-vectors and row-major 6x6 matrices; \
                          the Cairo block layout is an implementation detail)",
            ops: dim6,
        },
        Suite {
            name: "point",
            description: "Point2 / Point3: sub, add / sub vector, scale, distance, center",
            ops: {
                let mut ops = point_ops::<2>();
                ops.extend(point_ops::<3>());
                ops
            },
        },
    ]
}
