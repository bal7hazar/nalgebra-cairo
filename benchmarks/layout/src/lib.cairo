pub mod abstraction;

#[cfg(test)]
mod abstraction_gen;
pub mod composite;
pub mod fixed;
#[cfg(test)]
mod inline_gen;
pub mod lazy;
pub mod mat3;
#[cfg(test)]
mod mat3_gen;
pub mod mat6;
#[cfg(test)]
mod matn_gen;
pub mod passing;
pub mod scalar_variants_gen;
pub mod static_mats_gen;
pub mod vec3;
#[cfg(test)]
mod vec3_gen;
pub mod vec3_variants_gen;
pub mod vecn;
