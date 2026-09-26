//! WP 8.5-P17, suite `matrix_functions`: upstream's matrix exponential (`linalg/exp.rs`, the
//! Al-Mohy-Higham Padé approximant with scaling and squaring) and integer power (`linalg/pow.rs`,
//! exponentiation by squaring) on the squares 1..6.
//!
//! Upstream runs the same generic code on static and dynamic matrices; `DMatrix` is used so that
//! one definition covers every size. Upstream compiles `exp.rs` only with its `std` feature, which
//! this crate leaves off (see `Cargo.toml`: `std` pulls `matrixmultiply` into the other suites), so
//! `upstream_exp` below is a line-by-line transcription of `nalgebra-0.35.0/src/linalg/exp.rs` on
//! `DMatrix<f64>`, its `f64` functions through `libm` like the rest of the oracle.
//!
//! The inputs stay in the range where `exp(a)` fits Q32.32: `small` / `unit` general matrices, skew-symmetric matrices (rotation generators: `exp` is
//! orthogonal), and matrices whose 1-norm is drawn in [3, 12] (the inputs that need scaling and
//! squaring). Cases whose outputs do not fit are resampled by the engine.

use super::flat;
use crate::engine::{fm, fs, with, Op, Suite, Tol};
use crate::gen::{Dist, Gen};
use nalgebra::DMatrix;

/// `exp`: a Padé approximant evaluated with a handful of rounded matrix products and one LU
/// solve, then up to a few squarings, each of which doubles the relative error already committed
/// (WP 8.5-P17 model: a few ulp per unit of `max |exp(a)|`). The sensitivity `A` of `exp` is
/// itself of the order of `n * max |exp(a)|`. Measured on the Cairo side (`tests_linalg_exp`):
/// `k = 1, base = 4` is exceeded by up to 5 ulp (4x4 / 5x5 general and scaled inputs).
const EXP: Tol = Tol::Sens { k: 2.0, base: 8.0 };

/// `pow`: `k` rounded matrix products (exponentiation by squaring), each one floor per entry
/// amplified by the later products.
const POW: Tol = Tol::Sens { k: 1.0, base: 4.0 };

fn mat(x: &[f64], n: usize) -> DMatrix<f64> {
    DMatrix::from_row_slice(n, n, &x[..n * n])
}

fn exp_op(n: usize, suffix: &str, doc: &str, gen: Gen, dists: &[Dist]) -> Op {
    Op::new(
        format!("exp{n}{suffix}"),
        format!("a.exp() (upstream's Padé approximant with scaling and squaring), {doc}"),
    )
    .input(with(fm("a", n, n), gen))
    .out(fm("e", n, n))
    .dists(dists)
    .tol(EXP)
    .eval(move |x| Some(flat(&upstream_exp::exp(&mat(x, n)))))
}

fn exp_ops(n: usize) -> Vec<Op> {
    let mut ops = vec![exp_op(
        n,
        "",
        "general a",
        Gen::M(n, n),
        &[Dist::Small, Dist::Unit],
    )];
    if n > 1 {
        // A 1x1 skew-symmetric matrix is 0.
        ops.push(exp_op(
            n,
            "_skew",
            "skew-symmetric a (exp(a) orthogonal)",
            Gen::Skew(n),
            &[Dist::Small, Dist::Unit],
        ));
    }
    ops.push(exp_op(
        n,
        "_scaled",
        "a with entries uniform in (-1, 1) rescaled to a 1-norm uniform in [3, 12] (scaling \
             and squaring)",
        Gen::NormScaled(n, 3.0, 12.0),
        &Dist::UNIT,
    ));
    ops
}

fn pow_op(n: usize) -> Op {
    Op::new(
        format!("pow{n}"),
        format!(
            "a.pow(k) (upstream's exponentiation by squaring), general a, k an integer in 0..=8 \
             (emitted as a scalar, k * 2^32 raw); {n}x{n}"
        ),
    )
    .input(with(fm("a", n, n), Gen::M(n, n)))
    .input(with(fs("k"), Gen::Int(0, 8)))
    .out(fm("p", n, n))
    .dists(&[Dist::Small, Dist::Unit])
    .tol(POW)
    .eval(move |x| Some(flat(&mat(x, n).pow(x[n * n].round() as u32))))
}

pub fn suites() -> Vec<Suite> {
    let mut ops: Vec<Op> = (1..=6).flat_map(exp_ops).collect();
    ops.extend((1..=6).map(pow_op));
    vec![Suite {
        name: "matrix_functions",
        description: "Matrix exponential (general, skew-symmetric, scaled 1-norm) and integer \
                      power of the squares 1..6 (WP 8.5-P17)",
        ops,
    }]
}

/// Transcription of upstream's `nalgebra-0.35.0/src/linalg/exp.rs` (`ExpmPadeHelper`, `ell`,
/// `solve_p_q`, `OMatrix::exp`) on `DMatrix<f64>`, `use_exact_norm = true` as upstream calls it.
mod upstream_exp {
    use nalgebra::{DMatrix, DVector};

    fn factorial(n: usize) -> u128 {
        (1..=n as u128).product()
    }

    struct Helper {
        ident: DMatrix<f64>,
        a: DMatrix<f64>,
        a2: Option<DMatrix<f64>>,
        a4: Option<DMatrix<f64>>,
        a6: Option<DMatrix<f64>>,
        a8: Option<DMatrix<f64>>,
        a10: Option<DMatrix<f64>>,
        d4_exact: Option<f64>,
        d6_exact: Option<f64>,
        d8_exact: Option<f64>,
        d10_exact: Option<f64>,
    }

    impl Helper {
        fn new(a: DMatrix<f64>) -> Helper {
            let n = a.nrows();
            Helper {
                ident: DMatrix::identity(n, n),
                a,
                a2: None,
                a4: None,
                a6: None,
                a8: None,
                a10: None,
                d4_exact: None,
                d6_exact: None,
                d8_exact: None,
                d10_exact: None,
            }
        }

        fn calc_a2(&mut self) {
            if self.a2.is_none() {
                self.a2 = Some(&self.a * &self.a);
            }
        }

        fn calc_a4(&mut self) {
            if self.a4.is_none() {
                self.calc_a2();
                let a2 = self.a2.as_ref().unwrap();
                self.a4 = Some(a2 * a2);
            }
        }

        fn calc_a6(&mut self) {
            if self.a6.is_none() {
                self.calc_a2();
                self.calc_a4();
                self.a6 = Some(self.a4.as_ref().unwrap() * self.a2.as_ref().unwrap());
            }
        }

        fn calc_a8(&mut self) {
            if self.a8.is_none() {
                self.calc_a2();
                self.calc_a6();
                self.a8 = Some(self.a6.as_ref().unwrap() * self.a2.as_ref().unwrap());
            }
        }

        fn calc_a10(&mut self) {
            if self.a10.is_none() {
                self.calc_a4();
                self.calc_a6();
                self.a10 = Some(self.a6.as_ref().unwrap() * self.a4.as_ref().unwrap());
            }
        }

        fn d4_tight(&mut self) -> f64 {
            if self.d4_exact.is_none() {
                self.calc_a4();
                self.d4_exact = Some(libm::pow(one_norm(self.a4.as_ref().unwrap()), 0.25));
            }
            self.d4_exact.unwrap()
        }

        fn d6_tight(&mut self) -> f64 {
            if self.d6_exact.is_none() {
                self.calc_a6();
                self.d6_exact = Some(libm::pow(one_norm(self.a6.as_ref().unwrap()), 1.0 / 6.0));
            }
            self.d6_exact.unwrap()
        }

        fn d8_tight(&mut self) -> f64 {
            if self.d8_exact.is_none() {
                self.calc_a8();
                self.d8_exact = Some(libm::pow(one_norm(self.a8.as_ref().unwrap()), 1.0 / 8.0));
            }
            self.d8_exact.unwrap()
        }

        fn d10_tight(&mut self) -> f64 {
            if self.d10_exact.is_none() {
                self.calc_a10();
                self.d10_exact = Some(libm::pow(one_norm(self.a10.as_ref().unwrap()), 1.0 / 10.0));
            }
            self.d10_exact.unwrap()
        }

        // `use_exact_norm = true`: every `d*_loose` is the tight one.
        fn d4_loose(&mut self) -> f64 {
            self.d4_tight()
        }

        fn d6_loose(&mut self) -> f64 {
            self.d6_tight()
        }

        fn d8_loose(&mut self) -> f64 {
            self.d8_tight()
        }

        fn d10_loose(&mut self) -> f64 {
            self.d10_tight()
        }

        fn pade3(&mut self) -> (DMatrix<f64>, DMatrix<f64>) {
            let b = [120.0, 60.0, 12.0, 1.0];
            self.calc_a2();
            let a2 = self.a2.as_ref().unwrap();
            let u = &self.a * (a2 * b[3] + &self.ident * b[1]);
            let v = a2 * b[2] + &self.ident * b[0];
            (u, v)
        }

        fn pade5(&mut self) -> (DMatrix<f64>, DMatrix<f64>) {
            let b = [30240.0, 15120.0, 3360.0, 420.0, 30.0, 1.0];
            self.calc_a2();
            self.calc_a6();
            let a2 = self.a2.as_ref().unwrap();
            let a4 = self.a4.as_ref().unwrap();
            let u = &self.a * (a4 * b[5] + a2 * b[3] + &self.ident * b[1]);
            let v = a4 * b[4] + a2 * b[2] + &self.ident * b[0];
            (u, v)
        }

        fn pade7(&mut self) -> (DMatrix<f64>, DMatrix<f64>) {
            let b = [
                17_297_280.0,
                8_648_640.0,
                1_995_840.0,
                277_200.0,
                25_200.0,
                1_512.0,
                56.0,
                1.0,
            ];
            self.calc_a2();
            self.calc_a4();
            self.calc_a6();
            let a2 = self.a2.as_ref().unwrap();
            let a4 = self.a4.as_ref().unwrap();
            let a6 = self.a6.as_ref().unwrap();
            let u = &self.a * (a6 * b[7] + a4 * b[5] + a2 * b[3] + &self.ident * b[1]);
            let v = a6 * b[6] + a4 * b[4] + a2 * b[2] + &self.ident * b[0];
            (u, v)
        }

        fn pade9(&mut self) -> (DMatrix<f64>, DMatrix<f64>) {
            let b = [
                17_643_225_600.0,
                8_821_612_800.0,
                2_075_673_600.0,
                302_702_400.0,
                30_270_240.0,
                2_162_160.0,
                110_880.0,
                3_960.0,
                90.0,
                1.0,
            ];
            self.calc_a2();
            self.calc_a4();
            self.calc_a6();
            self.calc_a8();
            let a2 = self.a2.as_ref().unwrap();
            let a4 = self.a4.as_ref().unwrap();
            let a6 = self.a6.as_ref().unwrap();
            let a8 = self.a8.as_ref().unwrap();
            let u = &self.a * (a8 * b[9] + a6 * b[7] + a4 * b[5] + a2 * b[3] + &self.ident * b[1]);
            let v = a8 * b[8] + a6 * b[6] + a4 * b[4] + a2 * b[2] + &self.ident * b[0];
            (u, v)
        }

        fn pade13_scaled(&mut self, s: u64) -> (DMatrix<f64>, DMatrix<f64>) {
            let b = [
                64_764_752_532_480_000.0,
                32_382_376_266_240_000.0,
                7_771_770_303_897_600.0,
                1_187_353_796_428_800.0,
                129_060_195_264_000.0,
                10_559_470_521_600.0,
                670_442_572_800.0,
                33_522_128_640.0,
                1_323_241_920.0,
                40_840_800.0,
                960_960.0,
                16_380.0,
                182.0,
                1.0,
            ];
            let s = s as f64;
            let mb = &self.a * libm::pow(2.0, -s);
            self.calc_a2();
            self.calc_a4();
            self.calc_a6();
            let mb2 = self.a2.as_ref().unwrap() * libm::pow(2.0, -2.0 * s);
            let mb4 = self.a4.as_ref().unwrap() * libm::pow(2.0, -4.0 * s);
            let mb6 = self.a6.as_ref().unwrap() * libm::pow(2.0, -6.0 * s);
            let u2 = &mb6 * (&mb6 * b[13] + &mb4 * b[11] + &mb2 * b[9]);
            let u = &mb * (&u2 + &mb6 * b[7] + &mb4 * b[5] + &mb2 * b[3] + &self.ident * b[1]);
            let v2 = &mb6 * (&mb6 * b[12] + &mb4 * b[10] + &mb2 * b[8]);
            let v = v2 + &mb6 * b[6] + &mb4 * b[4] + &mb2 * b[2] + &self.ident * b[0];
            (u, v)
        }
    }

    fn one_norm(m: &DMatrix<f64>) -> f64 {
        m.column_iter()
            .map(|c| c.iter().map(|x| x.abs()).sum::<f64>())
            .fold(0.0, f64::max)
    }

    fn onenorm_matrix_power_nonm(a: &DMatrix<f64>, p: usize) -> f64 {
        let n = a.nrows();
        let mut v = DVector::<f64>::repeat(n, 1.0);
        let m = a.transpose();
        for _ in 0..p {
            v = &m * v;
        }
        v.max()
    }

    fn ell(a: &DMatrix<f64>, m: usize) -> u64 {
        let a_abs = a.map(|x| x.abs());
        let a_abs_onenorm = onenorm_matrix_power_nonm(&a_abs, 2 * m + 1);
        if a_abs_onenorm == 0.0 {
            return 0;
        }
        let m_factorial = factorial(m);
        let choose_2m_m = factorial(2 * m) / (m_factorial * m_factorial);
        let abs_c_recip = choose_2m_m * factorial(2 * m + 1);
        let alpha = a_abs_onenorm / one_norm(a);
        let alpha = alpha / abs_c_recip as f64;
        let u = libm::pow(2.0, -53.0);
        let log2_alpha_div_u = libm::log2(alpha / u);
        let value = libm::ceil(log2_alpha_div_u / (2.0 * m as f64));
        if value > 0.0 {
            value as u64
        } else {
            0
        }
    }

    fn solve_p_q(u: DMatrix<f64>, v: DMatrix<f64>) -> DMatrix<f64> {
        let p = &u + &v;
        let q = &v - &u;
        q.lu().solve(&p).unwrap()
    }

    pub fn exp(a: &DMatrix<f64>) -> DMatrix<f64> {
        if a.nrows() == 1 {
            return a.map(libm::exp);
        }
        let mut helper = Helper::new(a.clone());
        let eta_1 = helper.d4_loose().max(helper.d6_loose());
        if eta_1 < 1.495_585_217_958_292e-2 && ell(&helper.a, 3) == 0 {
            let (u, v) = helper.pade3();
            return solve_p_q(u, v);
        }
        let eta_2 = helper.d4_tight().max(helper.d6_loose());
        if eta_2 < 2.539_398_330_063_23e-1 && ell(&helper.a, 5) == 0 {
            let (u, v) = helper.pade5();
            return solve_p_q(u, v);
        }
        let eta_3 = helper.d6_tight().max(helper.d8_loose());
        if eta_3 < 9.504_178_996_162_932e-1 && ell(&helper.a, 7) == 0 {
            let (u, v) = helper.pade7();
            return solve_p_q(u, v);
        }
        if eta_3 < 2.097_847_961_257_068e0 && ell(&helper.a, 9) == 0 {
            let (u, v) = helper.pade9();
            return solve_p_q(u, v);
        }
        let eta_4 = helper.d8_loose().max(helper.d10_loose());
        let eta_5 = eta_3.min(eta_4);
        let theta_13 = 4.25;
        let mut s = if eta_5 == 0.0 {
            0
        } else {
            let l2 = libm::ceil(libm::log2(eta_5 / theta_13));
            if l2 < 0.0 {
                0
            } else {
                l2 as u64
            }
        };
        s += ell(&(&helper.a * libm::pow(2.0, -(s as f64))), 13);
        let (u, v) = helper.pade13_scaled(s);
        let mut x = solve_p_q(u, v);
        for _ in 0..s {
            x = &x * &x;
        }
        x
    }
}
