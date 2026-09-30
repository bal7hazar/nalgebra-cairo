//! The decompositions of `nalgebra_linalg_pivot6` (upstream `nalgebra::linalg`) of dimension 6:
//! `col_piv_qr`, `full_piv_lu`, `lblt`; the rest of the module is in the other `nalgebra_linalg*`
//! crates, `nalgebra_types*`, `nalgebra_static6_wide` (`Lu6`) and the facade `nalgebra`, which
//! re-exports every item at its 0.1.0 path. Features (docs/SPLIT.md §4): `full_piv_lu`
//! (`linalg::full_piv_lu`); `col_piv_qr` (`linalg::col_piv_qr`); `lblt` (`linalg::lblt`), as in
//! `nalgebra`.

#[cfg(feature: 'col_piv_qr')]
pub mod col_piv_qr;
#[cfg(feature: 'full_piv_lu')]
pub mod full_piv_lu;
#[cfg(feature: 'lblt')]
pub mod lblt;
