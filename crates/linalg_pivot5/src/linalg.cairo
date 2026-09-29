//! The pivoting decompositions of `nalgebra_linalg_pivot5` (upstream `nalgebra::linalg`) of
//! dimension 5: `col_piv_qr`, `full_piv_lu`, `lblt`; the rest of the module is in the other
//! `nalgebra_linalg*` crates (the facade `nalgebra` re-exports every item at its 0.1.0 path).
//! Features (docs/SPLIT.md §4): `col_piv_qr`, `full_piv_lu`, `lblt`, as in `nalgebra`.

#[cfg(feature: 'col_piv_qr')]
pub mod col_piv_qr;
#[cfg(feature: 'full_piv_lu')]
pub mod full_piv_lu;
#[cfg(feature: 'lblt')]
pub mod lblt;
