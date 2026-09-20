//! `oracle`: golden test vectors for nalgebra.cairo from upstream nalgebra. See README.md.

mod emit;
mod engine;
mod fixed;
mod gen;
mod model;
mod suites;

use std::path::{Path, PathBuf};
use std::process::ExitCode;

/// Version of upstream nalgebra the vectors come from (checked against Cargo.lock in tests).
pub const NALGEBRA_VERSION: &str = "0.35.0";
pub const DEFAULT_SEED: u64 = 1;
pub const DEFAULT_CASES: usize = 32;
const DEFAULT_DIR: &str = "vectors";

const USAGE: &str = "\
oracle: golden test vectors for nalgebra.cairo (upstream nalgebra in f64 on Q32.32 inputs)

USAGE (from tools/oracle):
  cargo run --release -- list
  cargo run --release -- all   [--out DIR] [--cases N] [--seed S]
  cargo run --release -- gen <suite>... [--out DIR] [--cases N] [--seed S]
  cargo run --release -- check [--out DIR] [--cases N] [--seed S]
  cargo run --release -- emit-cairo <suite> --out <file.cairo>
                         [--cases N] [--seed S] [--from DIR]
                         [--dist small,unit,medium,large,special] [--ops op1,op2] [--max-per-dist K]

  list        suites, ops and case counts
  all / gen   write DIR/<suite>.json (default DIR = vectors, N = 32 cases per op, S = 1)
  check       regenerate in memory and fail if DIR/<suite>.json differs
  emit-cairo  write a Cairo module of `pub fn <op>_cases() -> Span<(inputs.., outputs.., u64)>`;
              --from DIR reads DIR/<suite>.json instead of regenerating
";

struct Args {
    positional: Vec<String>,
    out: Option<PathBuf>,
    from: Option<PathBuf>,
    cases: usize,
    seed: u64,
    dists: Option<Vec<String>>,
    ops: Option<Vec<String>>,
    max_per_dist: Option<usize>,
}

fn parse(args: &[String]) -> Result<Args, String> {
    let mut parsed = Args {
        positional: Vec::new(),
        out: None,
        from: None,
        cases: DEFAULT_CASES,
        seed: DEFAULT_SEED,
        dists: None,
        ops: None,
        max_per_dist: None,
    };
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        let mut value = |name: &str| {
            iter.next()
                .cloned()
                .ok_or_else(|| format!("{name} expects a value"))
        };
        let list = |text: String| text.split(',').map(str::to_string).collect::<Vec<_>>();
        match arg.as_str() {
            "--out" => parsed.out = Some(PathBuf::from(value("--out")?)),
            "--from" => parsed.from = Some(PathBuf::from(value("--from")?)),
            "--cases" => {
                parsed.cases = value("--cases")?
                    .parse()
                    .map_err(|e| format!("--cases: {e}"))?;
            }
            "--seed" => {
                parsed.seed = value("--seed")?
                    .parse()
                    .map_err(|e| format!("--seed: {e}"))?;
            }
            "--dist" => {
                let dists = list(value("--dist")?);
                for dist in &dists {
                    if dist != "special" && gen::Dist::parse(dist).is_none() {
                        return Err(format!("unknown distribution '{dist}'"));
                    }
                }
                parsed.dists = Some(dists);
            }
            "--ops" => parsed.ops = Some(list(value("--ops")?)),
            "--max-per-dist" => {
                parsed.max_per_dist = Some(
                    value("--max-per-dist")?
                        .parse()
                        .map_err(|e| format!("--max-per-dist: {e}"))?,
                );
            }
            flag if flag.starts_with("--") => return Err(format!("unknown flag '{flag}'")),
            _ => parsed.positional.push(arg.clone()),
        }
    }
    if parsed.cases == 0 {
        return Err("--cases must be positive".into());
    }
    Ok(parsed)
}

fn select(names: &[String]) -> Result<Vec<engine::Suite>, String> {
    let mut all = suites::all();
    if names.is_empty() {
        return Ok(all);
    }
    let mut selected = Vec::new();
    for name in names {
        let index = all
            .iter()
            .position(|s| s.name == name)
            .ok_or_else(|| format!("unknown suite '{name}' (try `list`)"))?;
        selected.push(all.remove(index));
    }
    Ok(selected)
}

fn json_path(dir: &Path, suite: &str) -> PathBuf {
    dir.join(format!("{suite}.json"))
}

fn run(args: &[String]) -> Result<(), String> {
    let Some((command, rest)) = args.split_first() else {
        return Err(USAGE.into());
    };
    let args = parse(rest)?;
    let dir = args
        .out
        .clone()
        .unwrap_or_else(|| PathBuf::from(DEFAULT_DIR));
    match command.as_str() {
        "list" => {
            for suite in suites::all() {
                let data = engine::run_suite(&suite, args.seed, args.cases);
                let total: usize = data.ops.iter().map(|op| op.cases.len()).sum();
                println!(
                    "{} ({} ops, {total} cases): {}",
                    data.suite,
                    data.ops.len(),
                    data.description
                );
                for op in &data.ops {
                    let max = op.cases.iter().map(|c| c.tol).max().unwrap_or(0);
                    println!(
                        "  {:<46} {:>3} cases, max tol {max}",
                        op.name,
                        op.cases.len()
                    );
                }
            }
            Ok(())
        }
        "all" | "gen" => {
            if command == "all" && !args.positional.is_empty() {
                return Err("`all` takes no suite name (use `gen`)".into());
            }
            if command == "gen" && args.positional.is_empty() {
                return Err("`gen` expects at least one suite name".into());
            }
            std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
            for suite in select(&args.positional)? {
                let data = engine::run_suite(&suite, args.seed, args.cases);
                let path = json_path(&dir, suite.name);
                std::fs::write(&path, emit::to_json(&data))
                    .map_err(|e| format!("{}: {e}", path.display()))?;
                let total: usize = data.ops.iter().map(|op| op.cases.len()).sum();
                println!("{}: {} ops, {total} cases", path.display(), data.ops.len());
            }
            Ok(())
        }
        "check" => {
            let mut stale = Vec::new();
            for suite in select(&args.positional)? {
                let data = engine::run_suite(&suite, args.seed, args.cases);
                let path = json_path(&dir, suite.name);
                let committed = std::fs::read_to_string(&path).unwrap_or_default();
                if committed != emit::to_json(&data) {
                    stale.push(path.display().to_string());
                }
            }
            if stale.is_empty() {
                println!("vectors are up to date");
                Ok(())
            } else {
                Err(format!("stale vectors (run `all`): {}", stale.join(", ")))
            }
        }
        "emit-cairo" => {
            let [name] = args.positional.as_slice() else {
                return Err("`emit-cairo` expects exactly one suite name".into());
            };
            let out = args
                .out
                .clone()
                .ok_or("`emit-cairo` expects --out <file>")?;
            let data = match &args.from {
                Some(from) => {
                    let path = json_path(from, name);
                    let text = std::fs::read_to_string(&path)
                        .map_err(|e| format!("{}: {e}", path.display()))?;
                    emit::from_json(&text)?
                }
                None => {
                    let suite = select(std::slice::from_ref(name))?.remove(0);
                    engine::run_suite(&suite, args.seed, args.cases)
                }
            };
            let filter = emit::CairoFilter {
                dists: args.dists,
                ops: args.ops,
                max_per_dist: args.max_per_dist,
            };
            if let Some(parent) = out.parent().filter(|p| !p.as_os_str().is_empty()) {
                std::fs::create_dir_all(parent)
                    .map_err(|e| format!("{}: {e}", parent.display()))?;
            }
            std::fs::write(&out, emit::to_cairo(&data, &filter))
                .map_err(|e| format!("{}: {e}", out.display()))?;
            println!("{}", out.display());
            Ok(())
        }
        "help" | "--help" | "-h" => {
            print!("{USAGE}");
            Ok(())
        }
        other => Err(format!("unknown command '{other}'\n\n{USAGE}")),
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("{message}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generation_is_deterministic() {
        for suite in suites::all() {
            let first = emit::to_json(&engine::run_suite(&suite, 7, 8));
            let second = emit::to_json(&engine::run_suite(&suite, 7, 8));
            assert_eq!(first, second, "{}", suite.name);
        }
    }

    #[test]
    fn test_seed_changes_the_vectors() {
        let suite = &select(&["vector3".to_string()]).unwrap()[0];
        let a = engine::run_suite(suite, 1, 8);
        let b = engine::run_suite(suite, 2, 8);
        assert_ne!(a.ops[0].cases, b.ops[0].cases);
    }

    #[test]
    fn test_json_round_trip() {
        let suite = &select(&["isometry3".to_string()]).unwrap()[0];
        let data = engine::run_suite(suite, DEFAULT_SEED, 8);
        assert_eq!(emit::from_json(&emit::to_json(&data)).unwrap(), data);
    }

    #[test]
    fn test_op_names_are_unique_and_shapes_consistent() {
        let mut names = std::collections::BTreeSet::new();
        for suite in suites::all() {
            let data = engine::run_suite(&suite, DEFAULT_SEED, 4);
            for op in &data.ops {
                assert!(names.insert(op.name.clone()), "duplicate op {}", op.name);
                assert!(!op.cases.is_empty(), "{}", op.name);
                for case in &op.cases {
                    assert_eq!(case.inputs.len(), op.inputs.len());
                    assert_eq!(case.outputs.len(), op.outputs.len());
                    // Panics if a value does not have the declared shape.
                    for (field, value) in op.inputs.iter().zip(&case.inputs) {
                        field.kind.cairo_literal(value);
                    }
                    for (field, value) in op.outputs.iter().zip(&case.outputs) {
                        field.kind.cairo_literal(value);
                    }
                }
            }
        }
    }

    #[test]
    fn test_known_values() {
        // dot((1, 2, 3), (4, 5, 6)) = 32 and a 90 degree rotation about z, by hand.
        let suite = &select(&["vector3".to_string()]).unwrap()[0];
        let dot = suite
            .ops
            .iter()
            .find(|op| op.name == "vector3_dot")
            .unwrap();
        let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        assert_eq!(dot.eval.as_ref().unwrap()(&x), Some(vec![32.0]));
        let raw: Vec<i128> = x.iter().map(|v| (*v as i128) << 32).collect();
        assert_eq!(dot.exact.as_ref().unwrap()(&raw), Some(vec![32i128 << 64]));

        let suite = &select(&["matrix2".to_string()]).unwrap()[0];
        let mul_vec = suite
            .ops
            .iter()
            .find(|op| op.name == "matrix2_mul_vec")
            .unwrap();
        // ROW-major [[0, -1], [1, 0]] * (1, 0) = (0, 1).
        let y = mul_vec.eval.as_ref().unwrap()(&[0.0, -1.0, 1.0, 0.0, 1.0, 0.0]);
        assert_eq!(y, Some(vec![0.0, 1.0]));
    }

    #[test]
    fn test_nalgebra_version_matches_lockfile() {
        let lock = include_str!("../Cargo.lock");
        let needle = format!("name = \"nalgebra\"\nversion = \"{NALGEBRA_VERSION}\"");
        assert!(lock.contains(&needle));
    }

    #[test]
    fn test_committed_vectors_are_up_to_date() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join(DEFAULT_DIR);
        for suite in suites::all() {
            let data = engine::run_suite(&suite, DEFAULT_SEED, DEFAULT_CASES);
            let committed = std::fs::read_to_string(json_path(&dir, suite.name))
                .unwrap_or_else(|_| panic!("missing vectors for {} (run `all`)", suite.name));
            assert!(
                committed == emit::to_json(&data),
                "{}: stale vectors, run `cargo run --release -- all`",
                suite.name
            );
        }
    }

    #[test]
    fn test_cairo_emission_filters() {
        let suite = &select(&["vector2".to_string()]).unwrap()[0];
        let data = engine::run_suite(suite, DEFAULT_SEED, 8);
        let filter = emit::CairoFilter {
            dists: Some(vec!["small".into(), "unit".into()]),
            ops: Some(vec!["vector2_dot".into()]),
            max_per_dist: Some(1),
        };
        let cairo = emit::to_cairo(&data, &filter);
        assert!(cairo
            .contains("pub fn vector2_dot_cases() -> Span<((i64, i64), (i64, i64), i64, u64)>"));
        assert!(cairo.contains("; 2] = ["));
        assert!(cairo.contains("[0..1) small, [1..2) unit"));
        assert!(!cairo.contains("vector2_add"));
    }
}
