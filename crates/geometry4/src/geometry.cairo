//! Projective / affine / general transforms, perspective and orthographic projections, scales and
//! reflections of dimension 1 to 4 (upstream `nalgebra::geometry`; the facade `nalgebra` re-exports
//! them at the same paths).

pub mod affine2;
pub mod affine3;
pub mod orthographic3;
pub mod perspective3;
pub mod projective2;
pub mod projective3;
pub mod reflection1;
pub mod reflection2;
pub mod reflection3;
pub mod reflection4;
pub mod scale1;
pub mod scale2;
pub mod scale3;
pub mod scale4;
pub mod transform;
pub mod transform2;
pub mod transform3;
