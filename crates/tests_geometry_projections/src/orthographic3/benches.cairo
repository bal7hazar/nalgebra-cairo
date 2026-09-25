//! Gas benchmarks of `Orthographic3` (WP 8.4-P11b) (`bench_<group>__<variant>`, net = raw -
//! `baseline` of the group), and the alternative implementations (`alt_*`, AGENTS.md rule 8).
//!
//! Expected values are the results of the kernels themselves, all of which are checked against
//! upstream nalgebra in `tests.cairo`.

use fixed::Fixed;
use nalgebra::base::matrix4::Matrix4;
use nalgebra::base::point3::Point3;
use nalgebra::geometry::orthographic3::{Orthographic3, Orthographic3AngleTrait, Orthographic3Trait};
use nalgebra_testing::black_box;
use nalgebra_tests_utils::{fx, m4, ortho6t, p3t};
use simba::scalar::Real;

/// `(m11, m14, m22, m24, m33, m34)` ≈ `(0.4, -0.2, 0.6, 0.1, -0.05, -1.1)`.
fn o() -> Orthographic3<Fixed> {
    ortho6t((1717986918, -858993459, 2576980378, 429496730, -214748365, -4724464026))
}

/// `(1.5, -0.75, -3.2)`.
fn pt() -> Point3<Fixed> {
    p3t((6442450944, -3221225472, -13743895347))
}

// --- alternative implementations

/// `project_point` as upstream writes it: a floored product, then a sum (the same bits as the
/// fused `mul_add`: the added term is exact).
#[inline(always)]
fn alt_project_point_mul_add_split(o: Orthographic3<Fixed>, p: Point3<Fixed>) -> Point3<Fixed> {
    let m = o.into_inner();
    Point3 { x: m.m11 * p.x + m.m14, y: m.m22 * p.y + m.m24, z: m.m33 * p.z + m.m34 }
}

/// Upstream's literal `inverse`: the offsets are `-m_i4` times the rounded reciprocal (two
/// roundings) instead of one quotient.
#[inline(always)]
fn alt_inverse_recip_mul(o: Orthographic3<Fixed>) -> Matrix4<Fixed> {
    let m = o.into_inner();
    let (i11, i22, i33) = (Real::recip(m.m11), Real::recip(m.m22), Real::recip(m.m33));
    Matrix4 {
        m11: i11, m22: i22, m33: i33, m14: -m.m14 * i11, m24: -m.m24 * i22, m34: -m.m34 * i33, ..m,
    }
}


#[test]
fn test_project_point_alt_agrees() {
    assert!(o().project_point(pt()) == alt_project_point_mul_add_split(o(), pt()));
}

#[test]
fn test_inverse_alt_recip_mul_within_one_ulp() {
    let (f, l) = (o().inverse(), alt_inverse_recip_mul(o()));
    assert!(f.m11 == l.m11 && f.m22 == l.m22 && f.m33 == l.m33);
    assert!(f.m14 == l.m14 && f.m34 == l.m34);
    // Here the product by the rounded reciprocal lands 1 ulp below the single quotient.
    assert!((f.m24 - l.m24).raw == 1);
}

// --- benchmarks

#[test]
#[inline(never)]
fn bench_orthographic3_new__baseline() {
    let _l: Fixed = black_box(fx(-6442450944));
    let _r: Fixed = black_box(fx(8589934592));
    let _b: Fixed = black_box(fx(-4294967296));
    let _t: Fixed = black_box(fx(5368709120));
    let _zn: Fixed = black_box(fx(429496730));
    let _zf: Fixed = black_box(fx(429496729600));
    let e: Orthographic3<Fixed> = black_box(
        ortho6t((2454267026, -613566757, 3817748708, -477218588, -85985331, -4303565829)),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_orthographic3_new__new() {
    let l: Fixed = black_box(fx(-6442450944));
    let r: Fixed = black_box(fx(8589934592));
    let b: Fixed = black_box(fx(-4294967296));
    let t: Fixed = black_box(fx(5368709120));
    let zn: Fixed = black_box(fx(429496730));
    let zf: Fixed = black_box(fx(429496729600));
    let e: Orthographic3<Fixed> = black_box(
        ortho6t((2454267026, -613566757, 3817748708, -477218588, -85985331, -4303565829)),
    );
    assert!(Orthographic3Trait::new(l, r, b, t, zn, zf) == e);
}

#[test]
#[inline(never)]
fn bench_orthographic3_from_fov__baseline() {
    let _aspect: Fixed = black_box(fx(7635497415));
    let _vfov: Fixed = black_box(fx(4497880316));
    let _zn: Fixed = black_box(fx(429496730));
    let _zf: Fixed = black_box(fx(429496729600));
    let e: Orthographic3<Fixed> = black_box(
        ortho6t((148773989, 0, 264487091, 0, -85985331, -4303565829)),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_orthographic3_from_fov__from_fov() {
    let aspect: Fixed = black_box(fx(7635497415));
    let vfov: Fixed = black_box(fx(4497880316));
    let zn: Fixed = black_box(fx(429496730));
    let zf: Fixed = black_box(fx(429496729600));
    let e: Orthographic3<Fixed> = black_box(
        ortho6t((148773989, 0, 264487091, 0, -85985331, -4303565829)),
    );
    assert!(Orthographic3AngleTrait::from_fov(aspect, vfov, zn, zf) == e);
}

#[test]
#[inline(never)]
fn bench_orthographic3_project_point__baseline() {
    let _o: Orthographic3<Fixed> = black_box(o());
    let _p: Point3<Fixed> = black_box(pt());
    let e: Point3<Fixed> = black_box(p3t((1717986918, -1503238554, -4037269259)));
    assert!(e == e);
}

/// The fused kernel (library).
#[test]
#[inline(never)]
fn bench_orthographic3_project_point__mul_add() {
    let o: Orthographic3<Fixed> = black_box(o());
    let p: Point3<Fixed> = black_box(pt());
    let e: Point3<Fixed> = black_box(p3t((1717986918, -1503238554, -4037269259)));
    assert!(o.project_point(p) == e);
}

/// A floored product, then a sum (same bits).
#[test]
#[inline(never)]
fn bench_orthographic3_project_point__alt_split() {
    let o: Orthographic3<Fixed> = black_box(o());
    let p: Point3<Fixed> = black_box(pt());
    let e: Point3<Fixed> = black_box(p3t((1717986918, -1503238554, -4037269259)));
    assert!(alt_project_point_mul_add_split(o, p) == e);
}

#[test]
#[inline(never)]
fn bench_orthographic3_unproject_point__baseline() {
    let _o: Orthographic3<Fixed> = black_box(o());
    let _p: Point3<Fixed> = black_box(pt());
    let e: Point3<Fixed> = black_box(p3t((18253611012, -6084537002, 180388626252)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_orthographic3_unproject_point__unproject() {
    let o: Orthographic3<Fixed> = black_box(o());
    let p: Point3<Fixed> = black_box(pt());
    let e: Point3<Fixed> = black_box(p3t((18253611012, -6084537002, 180388626252)));
    assert!(o.unproject_point(p) == e);
}

#[test]
#[inline(never)]
fn bench_orthographic3_inverse__baseline() {
    let _o: Orthographic3<Fixed> = black_box(o());
    let e: Matrix4<Fixed> = black_box(
        m4(
            [
                [10737418243, 0, 0, 2147483648], [0, 7158278826, 0, -715827883],
                [0, 0, -85899345840, -94489280432], [0, 0, 0, 0x100000000],
            ],
        ),
    );
    assert!(e == e);
}

/// One quotient per offset (winner).
#[test]
#[inline(never)]
fn bench_orthographic3_inverse__div() {
    let o: Orthographic3<Fixed> = black_box(o());
    let e: Matrix4<Fixed> = black_box(
        m4(
            [
                [10737418243, 0, 0, 2147483648], [0, 7158278826, 0, -715827883],
                [0, 0, -85899345840, -94489280432], [0, 0, 0, 0x100000000],
            ],
        ),
    );
    assert!(o.inverse() == e);
}

/// Upstream's literal form (two roundings).
#[test]
#[inline(never)]
fn bench_orthographic3_inverse__alt_recip_mul() {
    let o: Orthographic3<Fixed> = black_box(o());
    let e: Matrix4<Fixed> = black_box(
        m4(
            [
                [10737418243, 0, 0, 2147483648], [0, 7158278826, 0, -715827884],
                [0, 0, -85899345840, -94489280432], [0, 0, 0, 0x100000000],
            ],
        ),
    );
    assert!(alt_inverse_recip_mul(o) == e);
}

#[test]
#[inline(never)]
fn bench_orthographic3_left__baseline() {
    let _o: Orthographic3<Fixed> = black_box(o());
    let e: Fixed = black_box(fx(-8589934595));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_orthographic3_left__left() {
    let o: Orthographic3<Fixed> = black_box(o());
    let e: Fixed = black_box(fx(-8589934595));
    assert!(o.left() == e);
}

#[test]
#[inline(never)]
fn bench_orthographic3_set_left__baseline() {
    let _o: Orthographic3<Fixed> = black_box(o());
    let _l: Fixed = black_box(fx(-6442450944));
    let e: (Fixed, Fixed) = black_box((fx(1908874353), fx(-1431655766)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_orthographic3_set_left__set_left() {
    let mut o: Orthographic3<Fixed> = black_box(o());
    let l: Fixed = black_box(fx(-6442450944));
    let e: (Fixed, Fixed) = black_box((fx(1908874353), fx(-1431655766)));
    o.set_left(l);
    let (a, b) = e;
    assert!(o.into_inner().m11 == a && o.into_inner().m14 == b);
}
