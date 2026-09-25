//! Gas benchmarks of `SimilarityMatrix2` / `SimilarityMatrix3` (`bench_<type>_<op>__<variant>`,
//! net = raw - `baseline` of the group). The poses are the `IsometryMatrix2/3` bench poses with the
//! scales 1.5 and 0.75 (2D), 2 and 0.5 (3D). Expected values are the results of the kernels
//! themselves, all of which are checked against upstream nalgebra in `tests.cairo`.

use fixed::Fixed;
use nalgebra::base::point2::Point2;
use nalgebra::base::point3::Point3;
use nalgebra::geometry::isometry3::Isometry3;
use nalgebra::geometry::isometry_matrix3::IsometryMatrix3;
use nalgebra::geometry::similarity_matrix2::{SimilarityMatrix2, SimilarityMatrix2Trait};
use nalgebra::geometry::similarity_matrix3::{SimilarityMatrix3, SimilarityMatrix3Trait};
use nalgebra_testing::black_box;
use nalgebra_tests_utils::{fx, iso3, p2, p3, simm2t, simm3t};

fn a2() -> SimilarityMatrix2<Fixed> {
    simm2t(
        (6442450944, -9663676416),
        [[3955926847, -1672539044], [1672539044, 3955926847]],
        0x180000000,
    )
}

fn b2() -> SimilarityMatrix2<Fixed> {
    simm2t(
        (-3221225472, 2147483648), [[4235452929, 712518464], [-712518464, 4235452929]], 0xc0000000,
    )
}

fn a3() -> SimilarityMatrix3<Fixed> {
    let q: Isometry3<Fixed> = iso3(
        (6442450944, -9663676416, 16106127360), (4234293283, 534340439, -400755330, 267170219),
    );
    let m: IsometryMatrix3<Fixed> = q.into();
    SimilarityMatrix3Trait::from_isometry(m, fx(0x200000000))
}

fn b3() -> SimilarityMatrix3<Fixed> {
    let q: Isometry3<Fixed> = iso3(
        (-3221225472, 2147483648, 5368709120), (3689020097, -1022754606, 767065954, 1789820560),
    );
    let m: IsometryMatrix3<Fixed> = q.into();
    SimilarityMatrix3Trait::from_isometry(m, fx(0x80000000))
}


#[test]
#[inline(never)]
fn bench_similarity_matrix2_mul__baseline() {
    let _x: SimilarityMatrix2<Fixed> = black_box(a2());
    let _y: SimilarityMatrix2<Fixed> = black_box(b2());
    let e: SimilarityMatrix2<Fixed> = black_box(
        simm2t(
            (737628957, -8578337706),
            [[4178578243, -993090094], [993090093, 4178578243]],
            4831838208,
        ),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_similarity_matrix2_mul__composition() {
    let x: SimilarityMatrix2<Fixed> = black_box(a2());
    let y: SimilarityMatrix2<Fixed> = black_box(b2());
    let e: SimilarityMatrix2<Fixed> = black_box(
        simm2t(
            (737628957, -8578337706),
            [[4178578243, -993090094], [993090093, 4178578243]],
            4831838208,
        ),
    );
    assert!(x * y == e);
}

#[test]
#[inline(never)]
fn bench_similarity_matrix2_inverse__baseline() {
    let _x: SimilarityMatrix2<Fixed> = black_box(a2());
    let e: SimilarityMatrix2<Fixed> = black_box(
        simm2t(
            (-1447118281, 7606429314),
            [[3955926847, 1672539044], [-1672539044, 3955926847]],
            2863311531,
        ),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_similarity_matrix2_inverse__transpose_div() {
    let x: SimilarityMatrix2<Fixed> = black_box(a2());
    let e: SimilarityMatrix2<Fixed> = black_box(
        simm2t(
            (-1447118281, 7606429314),
            [[3955926847, 1672539044], [-1672539044, 3955926847]],
            2863311531,
        ),
    );
    assert!(x.inverse() == e);
}

#[test]
#[inline(never)]
fn bench_similarity_matrix2_transform_point__baseline() {
    let _x: SimilarityMatrix2<Fixed> = black_box(a2());
    let _q: Point2<Fixed> = black_box(p2(-0x280000000, 0x3c0000000));
    let e: Point2<Fixed> = black_box(p2(-17800306856, 6316390683));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_similarity_matrix2_transform_point__fused() {
    let x: SimilarityMatrix2<Fixed> = black_box(a2());
    let q: Point2<Fixed> = black_box(p2(-0x280000000, 0x3c0000000));
    let e: Point2<Fixed> = black_box(p2(-17800306856, 6316390683));
    assert!(x.transform_point(q) == e);
}

#[test]
#[inline(never)]
fn bench_similarity_matrix3_mul__baseline() {
    let _x: SimilarityMatrix3<Fixed> = black_box(a3());
    let _y: SimilarityMatrix3<Fixed> = black_box(b3());
    let e: SimilarityMatrix3<Fixed> = black_box(
        simm3t(
            (-2273744476, -8934121888, 26042915386),
            [
                [2436096684, -3502951961, -491431154], [3413489599, 2171600315, 1441868542],
                [-927505130, -1208397167, 4015750812],
            ],
            4294967296,
        ),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_similarity_matrix3_mul__composition() {
    let x: SimilarityMatrix3<Fixed> = black_box(a3());
    let y: SimilarityMatrix3<Fixed> = black_box(b3());
    let e: SimilarityMatrix3<Fixed> = black_box(
        simm3t(
            (-2273744476, -8934121888, 26042915386),
            [
                [2436096684, -3502951961, -491431154], [3413489599, 2171600315, 1441868542],
                [-927505130, -1208397167, 4015750812],
            ],
            4294967296,
        ),
    );
    assert!(x * y == e);
}

#[test]
#[inline(never)]
fn bench_similarity_matrix3_inverse__baseline() {
    let _x: SimilarityMatrix3<Fixed> = black_box(a3());
    let e: SimilarityMatrix3<Fixed> = black_box(
        simm3t(
            (-4265994061, 3232765539, -8362135509),
            [
                [4186940973, 427075328, 856665640], [-626508539, 4128772954, 1003725567],
                [-723710168, -1103442173, 4087224368],
            ],
            2147483648,
        ),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_similarity_matrix3_inverse__transpose_div() {
    let x: SimilarityMatrix3<Fixed> = black_box(a3());
    let e: SimilarityMatrix3<Fixed> = black_box(
        simm3t(
            (-4265994061, 3232765539, -8362135509),
            [
                [4186940973, 427075328, 856665640], [-626508539, 4128772954, 1003725567],
                [-723710168, -1103442173, 4087224368],
            ],
            2147483648,
        ),
    );
    assert!(x.inverse() == e);
}

#[test]
#[inline(never)]
fn bench_similarity_matrix3_transform_point__baseline() {
    let _x: SimilarityMatrix3<Fixed> = black_box(a3());
    let _q: Point3<Fixed> = black_box(p3(-0x280000000, 0x3c0000000, 0xc0000000));
    let e: Point3<Fixed> = black_box(p3(-20276633216, 17511580838, 25481577464));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_similarity_matrix3_transform_point__fused() {
    let x: SimilarityMatrix3<Fixed> = black_box(a3());
    let q: Point3<Fixed> = black_box(p3(-0x280000000, 0x3c0000000, 0xc0000000));
    let e: Point3<Fixed> = black_box(p3(-20276633216, 17511580838, 25481577464));
    assert!(x.transform_point(q) == e);
}
