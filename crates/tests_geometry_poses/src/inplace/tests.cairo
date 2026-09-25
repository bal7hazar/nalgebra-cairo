//! WP 8.4-R: `inverse_mut` of `IsometryMatrix2/3` and `SimilarityMatrix2/3`, and their scaling
//! mutators (`prepend_scaling_mut`, `append_scaling_mut`), each checked bit for bit against its
//! by-value form, with the zero-scale panics.

use nalgebra::geometry::isometry_matrix2::IsometryMatrix2Trait;
use nalgebra::geometry::isometry_matrix3::IsometryMatrix3Trait;
use nalgebra::geometry::similarity_matrix2::SimilarityMatrix2Trait;
use nalgebra::geometry::similarity_matrix3::SimilarityMatrix3Trait;
use nalgebra_tests_utils::{ONE_RAW, fx, int, isom2t, isom3t, simm2t, simm3t};
use simba::scalar::Real;

const R2: [[i64; 2]; 2] = [[0xCCCCCCCD, -0x99999999], [0x99999999, 0xCCCCCCCD]];
const R3: [[i64; 3]; 3] = [
    [0xCCCCCCCD, -0x99999999, 0], [0x99999999, 0xCCCCCCCD, 0], [0, 0, ONE_RAW],
];

#[test]
fn test_isometry_matrix_inverse_mut_matches_inverse() {
    let a = isom2t((0x123456789, -0x2BCDEF01), R2);
    let mut x = a;
    x.inverse_mut();
    assert!(x == a.inverse());
    let b = isom3t((0x123456789, -0x2BCDEF01, 0x1FFFFFFF), R3);
    let mut y = b;
    y.inverse_mut();
    assert!(y == b.inverse());
    // Exact: the inverse of the identity rotation is a negated translation.
    let mut z = isom2t((int(2).raw, int(-3).raw), [[ONE_RAW, 0], [0, ONE_RAW]]);
    z.inverse_mut();
    assert!(z == isom2t((int(-2).raw, int(3).raw), [[ONE_RAW, 0], [0, ONE_RAW]]));
}

#[test]
fn test_similarity_matrix_inverse_mut_and_scaling_mutators() {
    let a = simm2t((0x123456789, -0x2BCDEF01), R2, 0x180000000);
    let mut x = a;
    x.inverse_mut();
    assert!(x == a.inverse());
    let mut x = a;
    x.prepend_scaling_mut(fx(0x2AAAAAAB));
    assert!(x == a.prepend_scaling(fx(0x2AAAAAAB)));
    let mut x = a;
    x.append_scaling_mut(fx(0x2AAAAAAB));
    assert!(x == a.append_scaling(fx(0x2AAAAAAB)));

    let b = simm3t((0x123456789, -0x2BCDEF01, 0x1FFFFFFF), R3, 0x180000000);
    let mut y = b;
    y.inverse_mut();
    assert!(y == b.inverse());
    let mut y = b;
    y.prepend_scaling_mut(fx(-0x2AAAAAAB));
    assert!(y == b.prepend_scaling(fx(-0x2AAAAAAB)));
    let mut y = b;
    y.append_scaling_mut(fx(-0x2AAAAAAB));
    assert!(y == b.append_scaling(fx(-0x2AAAAAAB)));
    // Exact: scale 2 then 3 gives 6, the translation is scaled by 3.
    let mut z = simm3t((1, 2, 3), [[ONE_RAW, 0, 0], [0, ONE_RAW, 0], [0, 0, ONE_RAW]], 2 * ONE_RAW);
    z.append_scaling_mut(int(3));
    assert!(z.scaling == int(6));
    assert!(z.isometry.translation.vector.z == fx(9));
}

#[test]
#[should_panic(expected: ('nalgebra: zero scale',))]
fn test_similarity_matrix3_append_scaling_mut_zero_panics() {
    let mut z = simm3t((1, 2, 3), R3, ONE_RAW);
    z.append_scaling_mut(Real::zero());
}

#[test]
#[should_panic(expected: ('nalgebra: zero scale',))]
fn test_similarity_matrix2_prepend_scaling_mut_zero_panics() {
    let mut z = simm2t((1, 2), R2, ONE_RAW);
    z.prepend_scaling_mut(Real::zero());
}
