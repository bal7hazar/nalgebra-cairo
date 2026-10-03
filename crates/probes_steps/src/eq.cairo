//! Field-wise equality of the decompositions the probes assert (the library structs have no
//! `PartialEq`); the same comparison runs in the baseline and in the operation of a probe.

use fixed::Fixed;
use nalgebra_linalg3::linalg::cholesky::Cholesky3;
use nalgebra_linalg3::linalg::lu::lu3::Lu3;
use nalgebra_linalg6::linalg::cholesky::Cholesky6;
use nalgebra_linalg_svd_eigen3::linalg::svd3::Svd3;
use nalgebra_linalg_svd_eigen3::linalg::symmetric_eigen3::SymmetricEigen3;
use nalgebra_static6_wide::linalg::lu::Perm6;
use nalgebra_static6_wide::linalg::lu::lu6::Lu6;
use nalgebra_types3::linalg::lu::Perm3;

pub impl Cholesky3Eq of PartialEq<Cholesky3<Fixed>> {
    fn eq(lhs: @Cholesky3<Fixed>, rhs: @Cholesky3<Fixed>) -> bool {
        lhs.l11 == rhs.l11
            && lhs.l21 == rhs.l21
            && lhs.l31 == rhs.l31
            && lhs.l22 == rhs.l22
            && lhs.l32 == rhs.l32
            && lhs.l33 == rhs.l33
    }
}

pub impl Cholesky6Eq of PartialEq<Cholesky6<Fixed>> {
    fn eq(lhs: @Cholesky6<Fixed>, rhs: @Cholesky6<Fixed>) -> bool {
        lhs.l11 == rhs.l11
            && lhs.l21 == rhs.l21
            && lhs.l31 == rhs.l31
            && lhs.l41 == rhs.l41
            && lhs.l51 == rhs.l51
            && lhs.l61 == rhs.l61
            && lhs.l22 == rhs.l22
            && lhs.l32 == rhs.l32
            && lhs.l42 == rhs.l42
            && lhs.l52 == rhs.l52
            && lhs.l62 == rhs.l62
            && lhs.l33 == rhs.l33
            && lhs.l43 == rhs.l43
            && lhs.l53 == rhs.l53
            && lhs.l63 == rhs.l63
            && lhs.l44 == rhs.l44
            && lhs.l54 == rhs.l54
            && lhs.l64 == rhs.l64
            && lhs.l55 == rhs.l55
            && lhs.l65 == rhs.l65
            && lhs.l66 == rhs.l66
    }
}

pub impl Perm3Eq of PartialEq<Perm3> {
    fn eq(lhs: @Perm3, rhs: @Perm3) -> bool {
        lhs.p1 == rhs.p1 && lhs.p2 == rhs.p2
    }
}

pub impl Perm6Eq of PartialEq<Perm6> {
    fn eq(lhs: @Perm6, rhs: @Perm6) -> bool {
        lhs.p1 == rhs.p1
            && lhs.p2 == rhs.p2
            && lhs.p3 == rhs.p3
            && lhs.p4 == rhs.p4
            && lhs.p5 == rhs.p5
    }
}

pub impl Lu3Eq of PartialEq<Lu3<Fixed>> {
    fn eq(lhs: @Lu3<Fixed>, rhs: @Lu3<Fixed>) -> bool {
        lhs.lu == rhs.lu && lhs.p == rhs.p
    }
}

pub impl Lu6Eq of PartialEq<Lu6<Fixed>> {
    fn eq(lhs: @Lu6<Fixed>, rhs: @Lu6<Fixed>) -> bool {
        lhs.lu == rhs.lu && lhs.p == rhs.p
    }
}

pub impl SymmetricEigen3Eq of PartialEq<SymmetricEigen3<Fixed>> {
    fn eq(lhs: @SymmetricEigen3<Fixed>, rhs: @SymmetricEigen3<Fixed>) -> bool {
        lhs.eigenvalues == rhs.eigenvalues && lhs.eigenvectors == rhs.eigenvectors
    }
}

pub impl Svd3Eq of PartialEq<Svd3<Fixed>> {
    fn eq(lhs: @Svd3<Fixed>, rhs: @Svd3<Fixed>) -> bool {
        lhs.u == rhs.u && lhs.singular_values == rhs.singular_values && lhs.v_t == rhs.v_t
    }
}
