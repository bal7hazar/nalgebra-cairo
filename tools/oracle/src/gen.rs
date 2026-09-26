//! Deterministic input generators. Every input is produced as raw Q32.32 (`i64`), so the Rust
//! oracle and the Cairo code under test see bit-identical inputs.
//!
//! Only `next_u64` of ChaCha8 is consumed and every transformation is written here on top of the
//! pure-Rust `libm`, so the stream does not depend on `rand`'s distribution code nor on the
//! platform libm.

use crate::fixed::{quantize, to_f64};
use nalgebra::DMatrix;
use rand_chacha::rand_core::{RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::f64::consts::PI;

/// Magnitude class of the generated inputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dist {
    /// Uniform in (-1, 1).
    Small,
    /// Magnitude uniform in [0.5, 2), random sign. Unit vectors / rotations are always exact-ish unit.
    Unit,
    /// Magnitude log-uniform in [2, 1e3), random sign.
    Medium,
    /// Magnitude log-uniform in [1e3, 4e4), random sign (squares still fit Q32.32).
    Large,
}

impl Dist {
    pub const ALL: [Dist; 4] = [Dist::Small, Dist::Unit, Dist::Medium, Dist::Large];
    pub const NO_LARGE: [Dist; 3] = [Dist::Small, Dist::Unit, Dist::Medium];
    pub const UNIT: [Dist; 1] = [Dist::Unit];

    pub fn label(self) -> &'static str {
        match self {
            Dist::Small => "small",
            Dist::Unit => "unit",
            Dist::Medium => "medium",
            Dist::Large => "large",
        }
    }

    pub fn parse(text: &str) -> Option<Dist> {
        Dist::ALL.into_iter().find(|d| d.label() == text)
    }
}

/// ChaCha8 stream dedicated to one op: adding or reordering ops never changes existing vectors.
pub struct Rng(ChaCha8Rng);

fn fnv1a(text: &str) -> u64 {
    text.bytes().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

impl Rng {
    pub fn for_op(seed: u64, op: &str) -> Rng {
        let mut key = [0u8; 32];
        key[..8].copy_from_slice(&seed.to_le_bytes());
        key[8..16].copy_from_slice(&fnv1a(op).to_le_bytes());
        Rng(ChaCha8Rng::from_seed(key))
    }

    /// Uniform in [0, 1) with 53 bits.
    pub fn unit(&mut self) -> f64 {
        (self.0.next_u64() >> 11) as f64 / 9_007_199_254_740_992.0
    }

    pub fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }

    pub fn sign(&mut self) -> f64 {
        if self.0.next_u64() & 1 == 0 {
            1.0
        } else {
            -1.0
        }
    }

    pub fn log_uniform(&mut self, lo: f64, hi: f64) -> f64 {
        libm::exp(self.range(libm::log(lo), libm::log(hi)))
    }

    /// Inclusive integer range.
    pub fn int(&mut self, lo: i64, hi: i64) -> i64 {
        lo + (self.0.next_u64() % (hi - lo + 1) as u64) as i64
    }

    /// Standard normal (Box-Muller).
    pub fn normal(&mut self) -> f64 {
        let u = 1.0 - self.unit();
        let v = self.unit();
        libm::sqrt(-2.0 * libm::log(u)) * libm::cos(2.0 * PI * v)
    }

    /// One scalar of the given magnitude class.
    pub fn scalar(&mut self, dist: Dist) -> f64 {
        match dist {
            Dist::Small => self.range(-1.0, 1.0),
            Dist::Unit => self.sign() * self.range(0.5, 2.0),
            Dist::Medium => self.sign() * self.log_uniform(2.0, 1.0e3),
            Dist::Large => self.sign() * self.log_uniform(1.0e3, 4.0e4),
        }
    }

    /// Global scale of a structured matrix (well-conditioned, SPD, ...).
    fn matrix_scale(&mut self, dist: Dist) -> f64 {
        match dist {
            Dist::Small => self.range(0.05, 0.5),
            Dist::Unit => 1.0,
            Dist::Medium => self.log_uniform(2.0, 1.0e2),
            Dist::Large => self.log_uniform(1.0e3, 1.0e4),
        }
    }

    fn unit_vector(&mut self, n: usize) -> Vec<f64> {
        loop {
            let v: Vec<f64> = (0..n).map(|_| self.normal()).collect();
            let norm = libm::sqrt(v.iter().map(|x| x * x).sum::<f64>());
            if norm > 1.0e-3 {
                return v.iter().map(|x| x / norm).collect();
            }
        }
    }

    /// Haar-ish random orthogonal matrix (Q of a Gaussian matrix).
    fn orthogonal(&mut self, n: usize) -> DMatrix<f64> {
        let data: Vec<f64> = (0..n * n).map(|_| self.normal()).collect();
        DMatrix::from_vec(n, n, data).qr().q()
    }

    /// `Q1 * Sigma * Q2^T` for a `rows x cols` `Sigma` with `sigma` on its diagonal
    /// (WP 8.5-P14b).
    fn with_rect_singular_values(&mut self, r: usize, c: usize, sigma: &[f64]) -> DMatrix<f64> {
        let q1 = self.orthogonal(r);
        let q2 = self.orthogonal(c);
        let mut s = DMatrix::zeros(r, c);
        for (i, v) in sigma.iter().enumerate() {
            s[(i, i)] = *v;
        }
        q1 * s * q2.transpose()
    }

    /// `Q1 * diag(sigma) * Q2^T`.
    fn with_singular_values(&mut self, sigma: &[f64]) -> DMatrix<f64> {
        let n = sigma.len();
        let q1 = self.orthogonal(n);
        let q2 = self.orthogonal(n);
        q1 * DMatrix::from_diagonal(&nalgebra::DVector::from_column_slice(sigma)) * q2.transpose()
    }
}

/// How an input is drawn.
#[derive(Clone, Debug)]
pub enum Gen {
    /// Scalar of the case's magnitude class.
    S,
    /// Strictly positive scalar of the case's magnitude class.
    Pos,
    /// Uniform scalar in a fixed range, whatever the magnitude class (angles, `t`, ...).
    Range(f64, f64),
    /// Vector with independent components of the case's magnitude class.
    V(usize),
    /// Unit vector (unit complex = `U(2)` as (re, im), unit quaternion = `U(4)` as (w, i, j, k)):
    /// normalised in f64, then quantised component-wise, so `|norm - 1| <= ~2 ulp`.
    U(usize),
    /// Rotation vector `axis * angle`, angle uniform in [0.05, 3.0].
    ScaledAxis,
    /// `rows x cols` matrix with independent entries of the case's magnitude class.
    M(usize, usize),
    /// Symmetric matrix with entries of the case's magnitude class (exactly symmetric in raw).
    Sym(usize),
    /// Well-conditioned square matrix: singular values log-uniform in [0.25, 2] x scale.
    WellCond(usize),
    /// Symmetric positive-definite: eigenvalues log-uniform in [0.25, 2] x scale, exactly symmetric.
    Spd(usize),
    /// Near-singular: one singular value log-uniform in [1e-4, 1e-2], the others in [0.5, 2].
    NearSingular(usize),
    /// Exactly singular with small integer entries (every fixed-point product is exact).
    Singular(usize),
    /// Rotation matrix (2 or 3), quantised entry-wise.
    Rot(usize),
    /// Concatenation (isometry = translation + rotation, ...).
    Group(Vec<Gen>),
    /// Unit dual quaternion `(real (w, i, j, k), dual (w, i, j, k))` (WP 8.4-P12): a quantised
    /// unit quaternion `r` and a translation `t` of the case's magnitude class, then upstream's
    /// `UnitDualQuaternion::from_parts(t, r)` evaluated in f64 on those raws and its dual part
    /// quantised, so `real · dual* + dual · real*` vanishes within a few ulp, not exactly.
    UnitDual,
    /// WP 8.5-P14b: `rows x cols` well-conditioned matrix, `Q1 * Sigma * Q2^T` with the
    /// `min(rows, cols)` singular values log-uniform in [0.25, 2] x scale.
    WellCondRect(usize, usize),
    /// WP 8.5-P14b: `rows x cols` near-rank-deficient matrix: one singular value log-uniform in
    /// [1e-4, 1e-2], the others in [0.5, 2] (FLAGGED ops, loose tolerance by construction).
    NearSingularRect(usize, usize),
    /// WP 8.5-P14b: `rows x cols` EXACTLY rank-deficient matrix with small integer entries: the
    /// product of integer `rows x k` and `k x cols` factors (entries in [-2, 2]), `k` uniform in
    /// `0 .. min(rows, cols) - 1`. Every fixed-point product of it is exact.
    RankDeficient(usize, usize),
    /// WP 8.5-P14b: symmetric positive-SEMI-definite `B^T B` of an integer `k x n` factor
    /// (entries in [-2, 2], `k < n`): exact zero eigenvalues.
    SymDeficient(usize),
    /// WP 8.5-P14b: symmetric positive-definite with CLUSTERED eigenvalues: `scale * base * (1 +
    /// e_i)`, `e_i` in {0, 1e-6, 1e-3, 0.5}, the hard case of Jacobi and QR alike.
    Clustered(usize),
}

fn quantize_all(values: &[f64]) -> Option<Vec<i64>> {
    values.iter().map(|x| quantize(*x)).collect()
}

/// Row-major flattening.
fn rows(m: &DMatrix<f64>) -> Vec<f64> {
    let mut out = Vec::with_capacity(m.len());
    for i in 0..m.nrows() {
        for j in 0..m.ncols() {
            out.push(m[(i, j)]);
        }
    }
    out
}

fn mirror_lower(n: usize, raw: &mut [i64]) {
    for i in 0..n {
        for j in (i + 1)..n {
            raw[i * n + j] = raw[j * n + i];
        }
    }
}

impl Gen {
    /// Draws one input as flat raw values (matrices row-major). `None` asks for a resample.
    pub fn sample(&self, rng: &mut Rng, dist: Dist) -> Option<Vec<i64>> {
        match self {
            Gen::S => quantize_all(&[rng.scalar(dist)]),
            Gen::Pos => {
                let raw = quantize(rng.scalar(dist).abs())?;
                (raw > 0).then_some(vec![raw])
            }
            Gen::Range(lo, hi) => quantize_all(&[rng.range(*lo, *hi)]),
            Gen::V(n) => {
                let v: Vec<f64> = (0..*n).map(|_| rng.scalar(dist)).collect();
                quantize_all(&v)
            }
            Gen::U(2) => {
                let angle = rng.range(-PI, PI);
                quantize_all(&[libm::cos(angle), libm::sin(angle)])
            }
            Gen::U(n) => quantize_all(&rng.unit_vector(*n)),
            Gen::ScaledAxis => {
                let angle = rng.range(0.05, 3.0);
                let axis = rng.unit_vector(3);
                quantize_all(&[axis[0] * angle, axis[1] * angle, axis[2] * angle])
            }
            Gen::M(r, c) => {
                let m: Vec<f64> = (0..r * c).map(|_| rng.scalar(dist)).collect();
                quantize_all(&m)
            }
            Gen::Sym(n) => {
                let m: Vec<f64> = (0..n * n).map(|_| rng.scalar(dist)).collect();
                let mut raw = quantize_all(&m)?;
                mirror_lower(*n, &mut raw);
                Some(raw)
            }
            Gen::WellCond(n) => {
                let scale = rng.matrix_scale(dist);
                let sigma: Vec<f64> = (0..*n)
                    .map(|_| scale * rng.log_uniform(0.25, 2.0))
                    .collect();
                quantize_all(&rows(&rng.with_singular_values(&sigma)))
            }
            Gen::Spd(n) => {
                let scale = rng.matrix_scale(dist);
                let lambda: Vec<f64> = (0..*n)
                    .map(|_| scale * rng.log_uniform(0.25, 2.0))
                    .collect();
                let q = rng.orthogonal(*n);
                let d = DMatrix::from_diagonal(&nalgebra::DVector::from_column_slice(&lambda));
                let mut raw = quantize_all(&rows(&(&q * d * q.transpose())))?;
                mirror_lower(*n, &mut raw);
                Some(raw)
            }
            Gen::NearSingular(n) => {
                let mut sigma: Vec<f64> = (0..*n).map(|_| rng.range(0.5, 2.0)).collect();
                sigma[n - 1] = rng.log_uniform(1.0e-4, 1.0e-2);
                quantize_all(&rows(&rng.with_singular_values(&sigma)))
            }
            Gen::Singular(n) => Some(singular_integer_matrix(rng, *n)),
            Gen::Rot(2) => {
                let angle = rng.range(-PI, PI);
                let (s, c) = (libm::sin(angle), libm::cos(angle));
                quantize_all(&[c, -s, s, c])
            }
            Gen::Rot(3) => {
                let q = rng.unit_vector(4);
                let q = nalgebra::Unit::new_normalize(nalgebra::Quaternion::new(
                    q[0], q[1], q[2], q[3],
                ));
                let m = q.to_rotation_matrix();
                let m = m.matrix();
                let mut out = Vec::with_capacity(9);
                for i in 0..3 {
                    for j in 0..3 {
                        out.push(m[(i, j)]);
                    }
                }
                quantize_all(&out)
            }
            Gen::Rot(n) => panic!("no rotation matrix of dimension {n}"),
            Gen::UnitDual => {
                let r = quantize_all(&rng.unit_vector(4))?;
                let t: Vec<f64> = (0..3).map(|_| rng.scalar(dist)).collect();
                let t = to_f64_all(&quantize_all(&t)?);
                let q = to_f64_all(&r);
                let dq = nalgebra::UnitDualQuaternion::from_parts(
                    nalgebra::Translation3::new(t[0], t[1], t[2]),
                    nalgebra::Unit::new_unchecked(nalgebra::Quaternion::new(
                        q[0], q[1], q[2], q[3],
                    )),
                );
                let d = dq.as_ref().dual;
                let mut out = r;
                out.extend(quantize_all(&[d.w, d.i, d.j, d.k])?);
                Some(out)
            }
            Gen::WellCondRect(r, c) => {
                let scale = rng.matrix_scale(dist);
                let sigma: Vec<f64> = (0..*r.min(c))
                    .map(|_| scale * rng.log_uniform(0.25, 2.0))
                    .collect();
                quantize_all(&rows(&rng.with_rect_singular_values(*r, *c, &sigma)))
            }
            Gen::NearSingularRect(r, c) => {
                let k = *r.min(c);
                let mut sigma: Vec<f64> = (0..k).map(|_| rng.range(0.5, 2.0)).collect();
                sigma[k - 1] = rng.log_uniform(1.0e-4, 1.0e-2);
                quantize_all(&rows(&rng.with_rect_singular_values(*r, *c, &sigma)))
            }
            Gen::RankDeficient(r, c) => {
                let k = rng.int(0, (*r.min(c) as i64) - 1) as usize;
                let a: Vec<i64> = (0..r * k).map(|_| rng.int(-2, 2)).collect();
                let b: Vec<i64> = (0..k * c).map(|_| rng.int(-2, 2)).collect();
                let mut m = vec![0i64; r * c];
                for i in 0..*r {
                    for j in 0..*c {
                        m[i * c + j] = (0..k).map(|l| a[i * k + l] * b[l * c + j]).sum::<i64>() << 32;
                    }
                }
                Some(m)
            }
            Gen::SymDeficient(n) => {
                let k = rng.int(1, (*n as i64) - 1) as usize;
                let b: Vec<i64> = (0..k * n).map(|_| rng.int(-2, 2)).collect();
                let mut m = vec![0i64; n * n];
                for i in 0..*n {
                    for j in 0..*n {
                        m[i * n + j] = (0..k).map(|l| b[l * n + i] * b[l * n + j]).sum::<i64>() << 32;
                    }
                }
                Some(m)
            }
            Gen::Clustered(n) => {
                let scale = rng.matrix_scale(dist);
                let base = scale * rng.log_uniform(0.5, 2.0);
                let lambda: Vec<f64> = (0..*n)
                    .map(|_| {
                        let e = [0.0, 1.0e-6, 1.0e-3, 0.5][rng.int(0, 3) as usize];
                        base * (1.0 + e)
                    })
                    .collect();
                let q = rng.orthogonal(*n);
                let d = DMatrix::from_diagonal(&nalgebra::DVector::from_column_slice(&lambda));
                let mut raw = quantize_all(&rows(&(&q * d * q.transpose())))?;
                mirror_lower(*n, &mut raw);
                Some(raw)
            }
            Gen::Group(parts) => {
                let mut out = Vec::new();
                for part in parts {
                    out.extend(part.sample(rng, dist)?);
                }
                Some(out)
            }
        }
    }
}

/// Rank `n - 1` matrix with integer entries in about [-40, 40]: the last row is an integer
/// combination of the others, then rows are shuffled. Every product of the cofactor expansion is
/// an exact integer in Q32.32, so a fixed-point determinant is exactly zero.
fn singular_integer_matrix(rng: &mut Rng, n: usize) -> Vec<i64> {
    let mut m = vec![0i64; n * n];
    for value in m.iter_mut().take((n - 1) * n) {
        *value = rng.int(-5, 5);
    }
    let coefficients: Vec<i64> = (0..n - 1).map(|_| rng.int(-2, 2)).collect();
    for j in 0..n {
        m[(n - 1) * n + j] = (0..n - 1).map(|i| coefficients[i] * m[i * n + j]).sum();
    }
    // Fisher-Yates on rows.
    for i in (1..n).rev() {
        let k = rng.int(0, i as i64) as usize;
        for j in 0..n {
            m.swap(i * n + j, k * n + j);
        }
    }
    m.iter().map(|x| x << 32).collect()
}

/// Inputs as seen by the oracle.
pub fn to_f64_all(raw: &[i64]) -> Vec<f64> {
    raw.iter().map(|x| to_f64(*x)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streams_are_deterministic_and_independent() {
        let a: Vec<f64> = (0..4).map(|_| Rng::for_op(7, "x").unit()).collect();
        assert!(a.iter().all(|x| *x == a[0]));
        assert_ne!(Rng::for_op(7, "x").unit(), Rng::for_op(7, "y").unit());
        assert_ne!(Rng::for_op(7, "x").unit(), Rng::for_op(8, "x").unit());
    }

    #[test]
    fn test_unit_vectors_are_unit_after_quantisation() {
        let mut rng = Rng::for_op(1, "unit");
        for n in 2..=4 {
            for _ in 0..50 {
                let raw = Gen::U(n).sample(&mut rng, Dist::Unit).unwrap();
                let norm2: f64 = to_f64_all(&raw).iter().map(|x| x * x).sum();
                assert!((norm2 - 1.0).abs() < 4.0 / crate::fixed::ONE);
            }
        }
    }

    #[test]
    fn test_structured_matrices() {
        let mut rng = Rng::for_op(1, "structured");
        for n in 2..=6 {
            let spd = Gen::Spd(n).sample(&mut rng, Dist::Medium).unwrap();
            for i in 0..n {
                for j in 0..n {
                    assert_eq!(spd[i * n + j], spd[j * n + i]);
                }
            }
            let m = DMatrix::from_row_slice(n, n, &to_f64_all(&spd));
            assert!(m.cholesky().is_some());

            let wc = Gen::WellCond(n).sample(&mut rng, Dist::Unit).unwrap();
            let sv = DMatrix::from_row_slice(n, n, &to_f64_all(&wc)).singular_values();
            assert!(sv.max() / sv.min() < 8.5);

            let singular = Gen::Singular(n).sample(&mut rng, Dist::Unit).unwrap();
            let m = DMatrix::from_row_slice(n, n, &to_f64_all(&singular));
            assert!(m.determinant().abs() < 1.0e-6);
        }
    }
}
