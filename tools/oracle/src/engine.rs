//! Turns an op definition into golden cases: draw raw inputs, evaluate upstream nalgebra in f64
//! (or on exact `i128` raws for bilinear ops), floor the outputs to Q32.32, derive a tolerance.

use crate::fixed::{is_ambiguous, quantize, rescale_exact};
use crate::gen::{to_f64_all, Dist, Gen, Rng};
use crate::model::{Case, Field, Kind, OpData, SuiteData};

/// Upstream evaluation in f64 on flat inputs (matrices row-major). `None` rejects the case.
pub type EvalFn = Box<dyn Fn(&[f64]) -> Option<Vec<f64>>>;
/// Evaluation on exact raws for homogeneous polynomial ops. Outputs are wide integers at scale
/// `2^(32 + shift)` (`shift = 32` for bilinear ops). `None` on `i128` overflow.
pub type ExactFn = Box<dyn Fn(&[i128]) -> Option<Vec<i128>>>;
/// Rounding-error model in ulp, from the f64 inputs and outputs.
pub type ModelFn = Box<dyn Fn(&[f64], &[f64]) -> f64>;

/// Default rejection threshold for a case tolerance: 2^22 ulp (1e-3).
pub const DEFAULT_CAP: u64 = 1 << 22;
/// Step of the central differences (value units): 2^-20, i.e. 4096 ulp.
const STEP: f64 = 1.0 / 1_048_576.0;
/// Relative error budget of the f64 oracle itself, in units of the output: 2^-49.
const ORACLE_REL_ERR: f64 = 1.0 / 562_949_953_421_312.0;
const MAX_ATTEMPTS: usize = 200_000;

/// Tolerance policy of an op (see README, "Tolerance policy").
pub enum Tol {
    /// The result is exactly representable and involves no rounding: 0.
    Exact,
    /// Constant budget in ulp: the rounding error of a sane implementation does not depend on
    /// the inputs (single rescale, exact-input kernels such as sqrt).
    Ulp(u64),
    /// `base + k * A`, `A = max_out sum_in |d out / d in|`: `k` one-ulp rounding errors committed
    /// on intermediates, amplified to first order like input perturbations.
    Sens { k: f64, base: f64 },
    /// `Sens` plus `mag * max |input|`: algorithms that apply rounded unit-scale factors
    /// (Givens / Jacobi rotations, Householder reflectors) to the input matrix.
    SensMag { k: f64, base: f64, mag: f64 },
    /// Op-specific model, with its description.
    Model(ModelFn, &'static str),
}

pub struct Input {
    pub field: Field,
    pub gen: Gen,
}

pub struct Op {
    pub name: String,
    pub doc: String,
    pub inputs: Vec<Input>,
    pub outputs: Vec<Field>,
    pub dists: Vec<Dist>,
    pub tol: Tol,
    pub cap: u64,
    pub eval: Option<EvalFn>,
    pub exact: Option<ExactFn>,
    /// Scale of the exact outputs above Q32.32, in bits.
    pub exact_shift: u32,
    /// Hand-picked inputs (f64, floored to raw) with an explicit tolerance, emitted first.
    pub specials: Vec<(Vec<f64>, u64)>,
}

impl Op {
    pub fn new(name: impl Into<String>, doc: impl Into<String>) -> Op {
        Op {
            name: name.into(),
            doc: doc.into(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            dists: Dist::ALL.to_vec(),
            tol: Tol::Exact,
            cap: DEFAULT_CAP,
            eval: None,
            exact: None,
            exact_shift: 32,
            specials: Vec::new(),
        }
    }

    pub fn input(mut self, input: Input) -> Op {
        self.inputs.push(input);
        self
    }

    pub fn out(mut self, field: Field) -> Op {
        self.outputs.push(field);
        self
    }

    pub fn dists(mut self, dists: &[Dist]) -> Op {
        self.dists = dists.to_vec();
        self
    }

    pub fn tol(mut self, tol: Tol) -> Op {
        self.tol = tol;
        self
    }

    pub fn cap(mut self, cap: u64) -> Op {
        self.cap = cap;
        self
    }

    pub fn eval(mut self, f: impl Fn(&[f64]) -> Option<Vec<f64>> + 'static) -> Op {
        self.eval = Some(Box::new(f));
        self
    }

    /// Bilinear op evaluated on both `f64` and exact `i128` raws: the expected outputs are the
    /// floors of the exact results. Tolerance 0 unless `tol` is called afterwards.
    pub fn ring(mut self, pair: (EvalFn, ExactFn)) -> Op {
        self.eval = Some(pair.0);
        self.exact = Some(pair.1);
        self.tol = Tol::Exact;
        self
    }

    /// Exact outputs are products of `degree` raws.
    pub fn degree(mut self, degree: u32) -> Op {
        self.exact_shift = 32 * (degree - 1);
        self
    }

    pub fn special(mut self, inputs: &[f64], tol: u64) -> Op {
        self.specials.push((inputs.to_vec(), tol));
        self
    }

    fn input_len(&self) -> usize {
        self.inputs.iter().map(|i| i.field.kind.len()).sum()
    }

    fn tolerance_doc(&self) -> String {
        let prefix = if self.exact.is_some() {
            "expected = floor(exact result), computed on i128 raws; "
        } else {
            ""
        };
        let budget = match &self.tol {
            Tol::Exact if self.exact.is_some() => {
                "0: one rescale per output scalar reproduces it bit for bit".into()
            }
            Tol::Exact => "0: exactly representable, no rounding involved".into(),
            Tol::Ulp(n) => format!("{n} ulp, independent of the inputs"),
            Tol::Sens { k, base } => format!(
                "ceil({base} + {k} * A), A = max over outputs of sum over inputs of \
                 |d out / d in| (central differences)"
            ),
            Tol::SensMag { k, base, mag } => format!(
                "ceil({base} + {k} * A + {mag} * max |input|), A = max over outputs of sum over \
                 inputs of |d out / d in| (central differences)"
            ),
            Tol::Model(_, doc) => (*doc).into(),
        };
        format!("{prefix}{budget}")
    }

    /// Number of random cases for a requested `--cases` budget: big inputs get fewer cases so
    /// that files stay small.
    fn budget(&self, cases: usize) -> usize {
        let len = self.input_len();
        let divisor = if len > 64 {
            4
        } else if len > 24 {
            2
        } else {
            1
        };
        (cases / divisor).max(self.dists.len())
    }
}

struct Evaluated {
    outputs: Vec<i64>,
    tol: u64,
}

fn output_mask(op: &Op) -> Vec<bool> {
    op.outputs.iter().flat_map(|f| f.kind.bool_mask()).collect()
}

/// `A = max_j sum_i |d f_j / d x_i|` by central differences. `None` when the function is not
/// smooth around `x` (branch cut, domain edge, flag flip).
fn sensitivity(eval: &EvalFn, mask: &[bool], x: &[f64], y: &[f64]) -> Option<f64> {
    let mut sums = vec![0.0f64; y.len()];
    let mut probe = x.to_vec();
    for i in 0..x.len() {
        probe[i] = x[i] + STEP;
        let plus = eval(&probe)?;
        probe[i] = x[i] - STEP;
        let minus = eval(&probe)?;
        probe[i] = x[i];
        for j in 0..y.len() {
            if mask[j] {
                if plus[j] != y[j] || minus[j] != y[j] {
                    return None;
                }
                continue;
            }
            let d = (plus[j] - minus[j]).abs() / (2.0 * STEP);
            if !d.is_finite() {
                return None;
            }
            sums[j] += d;
        }
    }
    Some(sums.into_iter().fold(0.0, f64::max))
}

fn evaluate(op: &Op, raw: &[i64], forced_tol: Option<u64>) -> Option<Evaluated> {
    let eval = op
        .eval
        .as_ref()
        .unwrap_or_else(|| panic!("{}: no eval function", op.name));
    let mask = output_mask(op);
    let x = to_f64_all(raw);
    let y = eval(&x)?;
    assert_eq!(y.len(), mask.len(), "{}: output arity", op.name);
    if y.iter().any(|v| !v.is_finite()) {
        return None;
    }

    let exact_outputs = match &op.exact {
        Some(exact) => {
            let wide: Vec<i128> = raw.iter().map(|v| i128::from(*v)).collect();
            let outputs: Option<Vec<i64>> = exact(&wide)?
                .into_iter()
                .map(|w| rescale_exact(w, op.exact_shift))
                .collect();
            let outputs = outputs?;
            // Cross-check of the two evaluations (f64 vs exact integers): the f64 error is
            // relative to the largest product, not to the (possibly cancelled) result.
            let xmax = x.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            let degree = (op.exact_shift / 32 + 1) as i32;
            let slack =
                2.0 + x.len() as f64 * xmax.powi(degree) * crate::fixed::ONE * ORACLE_REL_ERR;
            for (o, v) in outputs.iter().zip(&y) {
                let scaled = v * crate::fixed::ONE;
                assert!(
                    (*o as f64 - scaled).abs() <= slack,
                    "{}: exact {o} vs f64 {scaled}",
                    op.name
                );
            }
            Some(outputs)
        }
        None => None,
    };
    let is_exact = exact_outputs.is_some();

    let outputs = match exact_outputs {
        Some(outputs) => outputs,
        None => {
            let mut outputs = Vec::with_capacity(y.len());
            for (v, is_bool) in y.iter().zip(&mask) {
                if *is_bool {
                    outputs.push(i64::from(*v != 0.0));
                } else {
                    outputs.push(quantize(*v)?);
                }
            }
            outputs
        }
    };
    if let Some(tol) = forced_tol {
        return Some(Evaluated { outputs, tol });
    }

    if matches!(op.tol, Tol::Exact) {
        for (v, is_bool) in y.iter().zip(&mask) {
            let scaled = v * crate::fixed::ONE;
            assert!(
                is_exact || *is_bool || scaled == scaled.floor(),
                "{}: declared exact but {v} is not representable",
                op.name
            );
        }
        return Some(Evaluated { outputs, tol: 0 });
    }
    if !is_exact && y.iter().zip(&mask).any(|(v, b)| !*b && is_ambiguous(*v)) {
        return None;
    }

    let largest = outputs
        .iter()
        .zip(&mask)
        .filter(|(_, b)| !**b)
        .map(|(o, _)| o.unsigned_abs())
        .max()
        .unwrap_or(0) as f64;
    // Below 2^45 raw the ambiguity check guarantees the floor; above, the f64 oracle's own error
    // (2^-49 relative) is added to the budget.
    let oracle_err = if !is_exact && largest >= 35_184_372_088_832.0 {
        (largest * ORACLE_REL_ERR).ceil() + 1.0
    } else {
        0.0
    };
    let budget = match &op.tol {
        Tol::Exact => unreachable!(),
        Tol::Ulp(n) => *n as f64,
        Tol::Sens { k, base } => base + k * sensitivity(eval, &mask, &x, &y)?,
        Tol::SensMag { k, base, mag } => {
            let xmax = x.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            base + k * sensitivity(eval, &mask, &x, &y)? + mag * xmax
        }
        Tol::Model(model, _) => model(&x, &y),
    };
    let tol = budget.ceil() + oracle_err;
    if !tol.is_finite() || tol > op.cap as f64 {
        return None;
    }
    Some(Evaluated {
        outputs,
        tol: tol as u64,
    })
}

fn pack(fields: &[&Field], flat: &[i64]) -> Vec<crate::model::Value> {
    let mut offset = 0;
    fields
        .iter()
        .map(|field| {
            let len = field.kind.len();
            let value = field.kind.pack(&flat[offset..offset + len]);
            offset += len;
            value
        })
        .collect()
}

/// Generates the cases of one op.
pub fn run_op(op: &Op, seed: u64, cases: usize) -> OpData {
    let mut rng = Rng::for_op(seed, &op.name);
    let input_fields: Vec<&Field> = op.inputs.iter().map(|i| &i.field).collect();
    let output_fields: Vec<&Field> = op.outputs.iter().collect();
    let mut out = Vec::new();

    for (values, tol) in &op.specials {
        let raw: Vec<i64> = values
            .iter()
            .map(|v| quantize(*v).expect("special input out of range"))
            .collect();
        assert_eq!(raw.len(), op.input_len(), "{}: special arity", op.name);
        let evaluated = evaluate(op, &raw, Some(*tol))
            .unwrap_or_else(|| panic!("{}: special case {values:?} rejected", op.name));
        out.push(Case {
            dist: "special".into(),
            inputs: pack(&input_fields, &raw),
            outputs: pack(&output_fields, &evaluated.outputs),
            tol: evaluated.tol,
        });
    }

    let per_dist = (op.budget(cases) / op.dists.len()).max(1);
    for dist in &op.dists {
        for _ in 0..per_dist {
            let mut attempts = 0;
            let (raw, evaluated) = loop {
                attempts += 1;
                assert!(
                    attempts <= MAX_ATTEMPTS,
                    "{}: no acceptable '{}' case after {MAX_ATTEMPTS} attempts",
                    op.name,
                    dist.label()
                );
                let mut raw = Vec::with_capacity(op.input_len());
                let mut complete = true;
                for input in &op.inputs {
                    match input.gen.sample(&mut rng, *dist) {
                        Some(part) => raw.extend(part),
                        None => {
                            complete = false;
                            break;
                        }
                    }
                }
                if !complete {
                    continue;
                }
                if let Some(evaluated) = evaluate(op, &raw, None) {
                    break (raw, evaluated);
                }
            };
            out.push(Case {
                dist: dist.label().into(),
                inputs: pack(&input_fields, &raw),
                outputs: pack(&output_fields, &evaluated.outputs),
                tol: evaluated.tol,
            });
        }
    }

    OpData {
        name: op.name.clone(),
        doc: op.doc.clone(),
        tolerance: op.tolerance_doc(),
        inputs: op.inputs.iter().map(|i| i.field.clone()).collect(),
        outputs: op.outputs.clone(),
        cases: out,
    }
}

/// A named list of ops: one JSON file, one Cairo module.
pub struct Suite {
    pub name: &'static str,
    pub description: &'static str,
    pub ops: Vec<Op>,
}

pub fn run_suite(suite: &Suite, seed: u64, cases: usize) -> SuiteData {
    SuiteData {
        suite: suite.name.into(),
        description: suite.description.into(),
        format: Default::default(),
        generator: crate::model::Generator {
            name: env!("CARGO_PKG_NAME").into(),
            version: env!("CARGO_PKG_VERSION").into(),
            nalgebra: crate::NALGEBRA_VERSION.into(),
            seed,
            cases,
        },
        ops: suite.ops.iter().map(|op| run_op(op, seed, cases)).collect(),
    }
}

// Field / input constructors -------------------------------------------------------------------

fn field(name: &str, kind: Kind, layout: &str) -> Field {
    Field {
        name: name.into(),
        kind,
        layout: layout.into(),
    }
}

const XYZ: [&str; 7] = [
    "",
    "x",
    "(x, y)",
    "(x, y, z)",
    "(x, y, z, w)",
    "",
    "(x, y, z, a, b, c)",
];

/// Scalar field.
pub fn fs(name: &str) -> Field {
    field(name, Kind::Scalar, "raw")
}
/// Angle in radians.
pub fn fangle(name: &str) -> Field {
    field(name, Kind::Scalar, "radians")
}
pub fn fbool(name: &str) -> Field {
    field(name, Kind::Bool, "bool")
}
/// Vector or point with `n` components.
pub fn fv(name: &str, n: usize) -> Field {
    let layout = if n < XYZ.len() && !XYZ[n].is_empty() {
        XYZ[n].to_string()
    } else {
        format!(
            "({})",
            (1..=n)
                .map(|i| format!("v{i}"))
                .collect::<Vec<_>>()
                .join(", ")
        )
    };
    field(name, Kind::Vector(n), &layout)
}
/// Quaternion `(w, i, j, k)`.
pub fn fq(name: &str) -> Field {
    field(name, Kind::Vector(4), "(w, i, j, k)")
}
/// `n` quaternions, each `(w, i, j, k)`.
pub fn fquats(name: &str, n: usize) -> Field {
    field(
        name,
        Kind::Group(vec![Kind::Vector(4); n]),
        &format!("({n} quaternions, each (w, i, j, k))"),
    )
}
/// Unit complex `(re, im)`.
pub fn fc(name: &str) -> Field {
    field(name, Kind::Vector(2), "(re, im)")
}
/// Euler angles as upstream: `(roll, pitch, yaw)`.
pub fn feuler(name: &str) -> Field {
    field(name, Kind::Vector(3), "(roll, pitch, yaw) radians")
}
/// Row-major matrix.
pub fn fm(name: &str, r: usize, c: usize) -> Field {
    let layout = format!("row-major [[m11, .., m1{c}], .., [m{r}1, .., m{r}{c}]]");
    field(name, Kind::Matrix(r, c), &layout)
}
pub fn fiso2(name: &str) -> Field {
    field(
        name,
        Kind::Group(vec![Kind::Vector(2), Kind::Vector(2)]),
        "(translation (x, y), rotation (re, im))",
    )
}
pub fn fiso3(name: &str) -> Field {
    field(
        name,
        Kind::Group(vec![Kind::Vector(3), Kind::Vector(4)]),
        "(translation (x, y, z), rotation (w, i, j, k))",
    )
}
pub fn fsim2(name: &str) -> Field {
    field(
        name,
        Kind::Group(vec![Kind::Vector(2), Kind::Vector(2), Kind::Scalar]),
        "(translation (x, y), rotation (re, im), scaling)",
    )
}
pub fn fsim3(name: &str) -> Field {
    field(
        name,
        Kind::Group(vec![Kind::Vector(3), Kind::Vector(4), Kind::Scalar]),
        "(translation (x, y, z), rotation (w, i, j, k), scaling)",
    )
}

/// Pairs a field with its generator.
pub fn with(field: Field, gen: Gen) -> Input {
    assert_eq!(field.kind.len(), gen_len(&gen), "{}", field.name);
    Input { field, gen }
}

fn gen_len(gen: &Gen) -> usize {
    match gen {
        Gen::S | Gen::Pos | Gen::Range(..) => 1,
        Gen::V(n) | Gen::U(n) => *n,
        Gen::ScaledAxis => 3,
        Gen::M(r, c) => r * c,
        Gen::Sym(n)
        | Gen::WellCond(n)
        | Gen::Spd(n)
        | Gen::NearSingular(n)
        | Gen::Singular(n)
        | Gen::Rot(n) => n * n,
        Gen::Group(parts) => parts.iter().map(gen_len).sum(),
        Gen::UnitDual => 8,
    }
}

/// Scalar of the case's magnitude class.
pub fn is(name: &str) -> Input {
    with(fs(name), Gen::S)
}
/// Interpolation parameter in [0, 1].
pub fn it(name: &str) -> Input {
    with(fs(name), Gen::Range(0.0, 1.0))
}
/// Angle in (-pi, pi).
pub fn iangle(name: &str) -> Input {
    with(
        fangle(name),
        Gen::Range(-std::f64::consts::PI, std::f64::consts::PI),
    )
}
pub fn iv(name: &str, n: usize) -> Input {
    with(fv(name, n), Gen::V(n))
}
/// Unit vector.
pub fn iu(name: &str, n: usize) -> Input {
    with(fv(name, n), Gen::U(n))
}
pub fn iq(name: &str) -> Input {
    with(fq(name), Gen::V(4))
}
pub fn iuq(name: &str) -> Input {
    with(fq(name), Gen::U(4))
}
pub fn iuc(name: &str) -> Input {
    with(fc(name), Gen::U(2))
}
pub fn im(name: &str, r: usize, c: usize) -> Input {
    with(fm(name, r, c), Gen::M(r, c))
}
pub fn iiso2(name: &str) -> Input {
    with(fiso2(name), Gen::Group(vec![Gen::V(2), Gen::U(2)]))
}
pub fn iiso3(name: &str) -> Input {
    with(fiso3(name), Gen::Group(vec![Gen::V(3), Gen::U(4)]))
}
/// Similarity with a scaling factor in [0.25, 4].
pub fn isim2(name: &str) -> Input {
    with(
        fsim2(name),
        Gen::Group(vec![Gen::V(2), Gen::U(2), Gen::Range(0.25, 4.0)]),
    )
}
pub fn isim3(name: &str) -> Input {
    with(
        fsim3(name),
        Gen::Group(vec![Gen::V(3), Gen::U(4), Gen::Range(0.25, 4.0)]),
    )
}

/// Dual quaternion `(real (w, i, j, k), dual (w, i, j, k))` (WP 8.4-P12).
pub fn fdq(name: &str) -> Field {
    field(
        name,
        Kind::Group(vec![Kind::Vector(4), Kind::Vector(4)]),
        "(real (w, i, j, k), dual (w, i, j, k))",
    )
}
/// General dual quaternion: eight components of the case's magnitude class.
pub fn idq(name: &str) -> Input {
    with(fdq(name), Gen::Group(vec![Gen::V(4), Gen::V(4)]))
}
/// Unit dual quaternion (`Gen::UnitDual`).
pub fn iudq(name: &str) -> Input {
    with(fdq(name), Gen::UnitDual)
}
