//! Data model shared by the generator, the JSON files and the Cairo emitter.

use serde::{Deserialize, Serialize};

/// Shape of one input or output. Every leaf is a raw Q32.32 `i64`, except `Bool`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    /// One raw `i64`.
    Scalar,
    /// A flag (`is_some`, ...).
    Bool,
    /// `n` components: Cairo tuple `(i64, .., i64)`, JSON array.
    Vector(usize),
    /// `rows x cols`, ROW-major: Cairo `[[i64; cols]; rows]`, JSON array of rows.
    Matrix(usize, usize),
    /// Heterogeneous group (isometry = translation + rotation): Cairo tuple, JSON array.
    Group(Vec<Kind>),
}

impl Kind {
    /// Number of leaves.
    pub fn len(&self) -> usize {
        match self {
            Kind::Scalar | Kind::Bool => 1,
            Kind::Vector(n) => *n,
            Kind::Matrix(r, c) => r * c,
            Kind::Group(kinds) => kinds.iter().map(Kind::len).sum(),
        }
    }

    /// For each leaf, whether it is a boolean.
    pub fn bool_mask(&self) -> Vec<bool> {
        match self {
            Kind::Bool => vec![true],
            Kind::Group(kinds) => kinds.iter().flat_map(Kind::bool_mask).collect(),
            other => vec![false; other.len()],
        }
    }

    /// Packs `self.len()` flat leaves into a structured value.
    pub fn pack(&self, flat: &[i64]) -> Value {
        assert_eq!(flat.len(), self.len());
        match self {
            Kind::Scalar => Value::Int(flat[0]),
            Kind::Bool => Value::Bool(flat[0] != 0),
            Kind::Vector(_) => Value::List(flat.iter().map(|x| Value::Int(*x)).collect()),
            Kind::Matrix(_, c) => Value::List(
                flat.chunks(*c)
                    .map(|row| Value::List(row.iter().map(|x| Value::Int(*x)).collect()))
                    .collect(),
            ),
            Kind::Group(kinds) => {
                let mut offset = 0;
                Value::List(
                    kinds
                        .iter()
                        .map(|kind| {
                            let value = kind.pack(&flat[offset..offset + kind.len()]);
                            offset += kind.len();
                            value
                        })
                        .collect(),
                )
            }
        }
    }

    /// Cairo type of the value.
    pub fn cairo_type(&self) -> String {
        match self {
            Kind::Scalar => "i64".into(),
            Kind::Bool => "bool".into(),
            Kind::Vector(n) => format!("({})", vec!["i64"; *n].join(", ")),
            Kind::Matrix(r, c) => format!("[[i64; {c}]; {r}]"),
            Kind::Group(kinds) => {
                let parts: Vec<String> = kinds.iter().map(Kind::cairo_type).collect();
                format!("({})", parts.join(", "))
            }
        }
    }

    /// Cairo literal of `value`, which must have this shape.
    pub fn cairo_literal(&self, value: &Value) -> String {
        match (self, value) {
            (Kind::Scalar, Value::Int(x)) => x.to_string(),
            (Kind::Bool, Value::Bool(b)) => b.to_string(),
            (Kind::Vector(n), Value::List(items)) if items.len() == *n => {
                let parts: Vec<String> = items
                    .iter()
                    .map(|v| Kind::Scalar.cairo_literal(v))
                    .collect();
                format!("({})", parts.join(", "))
            }
            (Kind::Matrix(r, c), Value::List(rows)) if rows.len() == *r => {
                let rows: Vec<String> = rows
                    .iter()
                    .map(|row| match row {
                        Value::List(items) if items.len() == *c => {
                            let parts: Vec<String> = items
                                .iter()
                                .map(|v| Kind::Scalar.cairo_literal(v))
                                .collect();
                            format!("[{}]", parts.join(", "))
                        }
                        other => panic!("matrix row expected, got {other:?}"),
                    })
                    .collect();
                format!("[{}]", rows.join(", "))
            }
            (Kind::Group(kinds), Value::List(items)) if items.len() == kinds.len() => {
                let parts: Vec<String> = kinds
                    .iter()
                    .zip(items)
                    .map(|(k, v)| k.cairo_literal(v))
                    .collect();
                format!("({})", parts.join(", "))
            }
            (kind, value) => panic!("value {value:?} does not have shape {kind:?}"),
        }
    }
}

/// A structured value (JSON: number, boolean or nested arrays).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Value {
    Bool(bool),
    Int(i64),
    List(Vec<Value>),
}

/// A named input or output.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Field {
    pub name: String,
    pub kind: Kind,
    /// Component layout, e.g. `(x, y, z)` or `(w, i, j, k)` or `[[m11, m12], [m21, m22]]`.
    pub layout: String,
}

/// One golden case.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Case {
    /// Input distribution the case was drawn from (`small`, `unit`, `medium`, `large`, `special`).
    pub dist: String,
    #[serde(rename = "in")]
    pub inputs: Vec<Value>,
    #[serde(rename = "out")]
    pub outputs: Vec<Value>,
    /// Suggested tolerance on every output scalar, in raw units (ulp = 2^-32).
    pub tol: u64,
}

/// All cases of one upstream operation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpData {
    pub name: String,
    /// Upstream expression the expected outputs come from.
    pub doc: String,
    /// How `tol` was derived for this op.
    pub tolerance: String,
    pub inputs: Vec<Field>,
    pub outputs: Vec<Field>,
    pub cases: Vec<Case>,
}

/// Number format and conventions, repeated in every file so that a file is self-describing.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Format {
    pub scalar: String,
    pub rounding: String,
    pub matrix_order: String,
    pub quaternion_order: String,
    pub complex_order: String,
    pub angles: String,
    pub tolerance_unit: String,
}

impl Default for Format {
    fn default() -> Self {
        Format {
            scalar: "Q32.32 raw i64, value = raw / 2^32".into(),
            rounding: "expected = floor(f64 result * 2^32)".into(),
            matrix_order: "row-major: [[m11, m12, ..], [m21, m22, ..], ..]".into(),
            quaternion_order: "(w, i, j, k), as Quaternion::new".into(),
            complex_order: "(re, im) = (cos, sin)".into(),
            angles: "radians".into(),
            tolerance_unit: "raw units (1 ulp = 2^-32), applies to every output scalar".into(),
        }
    }
}

/// Provenance. No timestamp, no host information: the file is a pure function of these fields.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Generator {
    pub name: String,
    pub version: String,
    pub nalgebra: String,
    pub seed: u64,
    pub cases: usize,
}

/// One JSON file.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SuiteData {
    pub suite: String,
    pub description: String,
    pub format: Format,
    pub generator: Generator,
    pub ops: Vec<OpData>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_and_cairo_literal_shapes() {
        let iso = Kind::Group(vec![Kind::Vector(2), Kind::Vector(2), Kind::Scalar]);
        let value = iso.pack(&[1, -2, 3, 4, 5]);
        assert_eq!(iso.cairo_type(), "((i64, i64), (i64, i64), i64)");
        assert_eq!(iso.cairo_literal(&value), "((1, -2), (3, 4), 5)");

        let m = Kind::Matrix(2, 3);
        let value = m.pack(&[1, 2, 3, 4, 5, 6]);
        assert_eq!(m.cairo_type(), "[[i64; 3]; 2]");
        assert_eq!(m.cairo_literal(&value), "[[1, 2, 3], [4, 5, 6]]");
        assert_eq!(serde_json::to_string(&value).unwrap(), "[[1,2,3],[4,5,6]]");
    }

    #[test]
    fn test_value_json_round_trip() {
        let value = Value::List(vec![
            Value::Int(i64::MIN),
            Value::Bool(true),
            Value::List(vec![Value::Int(i64::MAX)]),
        ]);
        let text = serde_json::to_string(&value).unwrap();
        assert_eq!(serde_json::from_str::<Value>(&text).unwrap(), value);
    }
}
