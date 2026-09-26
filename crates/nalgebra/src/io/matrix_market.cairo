//! The Matrix Market coordinate format (upstream `io/matrix_market.rs` and its `pest` grammar
//! `io/matrix_market.pest`), parsed by hand with the same accepted language:
//!
//! ```text
//! Document  = SOI NEWLINE* Header (NEWLINE Comments)* (NEWLINE Shape) (NEWLINE Entry?)*
//! Header    = "%%" (!NEWLINE ANY)*          Comments = "%" (!NEWLINE ANY)*
//! Shape     = Dimension Dimension Dimension Entry    = Dimension Dimension Value
//! Dimension = ASCII_DIGIT+
//! Value     = ("+" | "-")? (DIGIT+ ("." DIGIT*)? | "." DIGIT+) (("e" | "E") ("+" | "-")? DIGIT+)?
//! ```
//!
//! with pest's implicit `WHITESPACE = " "` between the tokens (spaces, not tabs) and `NEWLINE` =
//! `\n`, `\r\n` or `\r`. As upstream: the header line is not interpreted (`coordinate real
//! general` is assumed), the third shape number (the entry count) is ignored, indices are
//! 1-based, and the document ends silently at the first line that is not a complete entry (the
//! grammar has no `EOI`): the rest of the text is ignored.

use simba::scalar::Real;
use crate::base::errors as base_errors;
use crate::sparse::cs_matrix::{CsMatrix, CsMatrixTrait};
use super::errors;

const SPACE: u8 = 0x20;
const LF: u8 = 0x0a;
const CR: u8 = 0x0d;
const PERCENT: u8 = 0x25;
const PLUS: u8 = 0x2b;
const MINUS: u8 = 0x2d;
const DOT: u8 = 0x2e;
const ZERO: u8 = 0x30;
const NINE: u8 = 0x39;
const UPPER_E: u8 = 0x45;
const LOWER_E: u8 = 0x65;
/// The number of fractional digits a value keeps (`10^18 < 2^63`, the `from_ratio` range).
const FRAC_DIGITS: usize = 18;
/// The exponent magnitude beyond which a value is certainly out of range or zero.
const EXP_CAP: u32 = 100000;

/// The byte at `pos`, `None` past the end.
#[inline(always)]
fn peek(data: Span<u8>, pos: usize) -> Option<u8> {
    match data.get(pos) {
        Some(b) => Some(*b.unbox()),
        None => None,
    }
}

/// The bytes of `text`, extracted once (its iterator): the parser peeks at most bytes several
/// times, and one `ByteArray::at` costs about as much as the extraction of a byte. Measured on an
/// 89-byte file (`bench_matrix_market__*`): 1 845 850 gas parsed from the extracted bytes (the
/// extraction alone: 1 068 340 with the iterator, 1 239 230 with `at`), 2 761 160 with `at` at
/// every peek.
fn bytes_of(text: @ByteArray) -> Span<u8> {
    let mut out: Array<u8> = array![];
    for b in text.clone() {
        out.append(b);
    }
    out.span()
}

/// Whether `b` is an ASCII digit.
#[inline(always)]
fn is_digit(b: Option<u8>) -> bool {
    match b {
        Some(b) => b >= ZERO && b <= NINE,
        None => false,
    }
}

/// The position after the spaces at `pos`.
fn skip_spaces(data: Span<u8>, mut pos: usize) -> usize {
    while peek(data, pos) == Some(SPACE) {
        pos += 1;
    }
    pos
}

/// The position after a `NEWLINE` at `pos`, `None` if there is none.
fn newline(data: Span<u8>, pos: usize) -> Option<usize> {
    match peek(data, pos) {
        Some(b) => {
            if b == LF {
                Some(pos + 1)
            } else if b == CR {
                if peek(data, pos + 1) == Some(LF) {
                    Some(pos + 2)
                } else {
                    Some(pos + 1)
                }
            } else {
                None
            }
        },
        None => None,
    }
}

/// The position of the end of the line at `pos` (the next `NEWLINE` or the end of the text).
fn line_end(data: Span<u8>, mut pos: usize) -> usize {
    loop {
        match peek(data, pos) {
            Some(b) => { if b == LF || b == CR {
                break;
            } },
            None => { break; },
        }
        pos += 1;
    }
    pos
}

/// A `Dimension` at `pos`: `None` without a digit, else its value (`None` when it does not fit
/// `usize`, upstream's `parse::<usize>()` failure) and the position after it.
fn dimension(data: Span<u8>, mut pos: usize) -> Option<(Option<usize>, usize)> {
    if !is_digit(peek(data, pos)) {
        return None;
    }
    let mut value: u64 = 0;
    let mut fits = true;
    while let Some(b) = peek(data, pos) {
        if b < ZERO || b > NINE {
            break;
        }
        if fits {
            value = value * 10 + (b - ZERO).into();
            if value > 0xffffffff {
                fits = false;
            }
        }
        pos += 1;
    }
    let value: Option<usize> = if fits {
        Some(value.try_into().unwrap())
    } else {
        None
    };
    Some((value, pos))
}

/// The digits of a run at `pos` appended to `digits`; returns the position after them.
fn digits_into(data: Span<u8>, mut pos: usize, ref digits: Array<u8>) -> usize {
    while let Some(b) = peek(data, pos) {
        if b < ZERO || b > NINE {
            break;
        }
        digits.append(b - ZERO);
        pos += 1;
    }
    pos
}

/// The decimal number `(-1)^neg * 0.d0 d1 d2... * 10^point` (`point` the position of the
/// decimal point in `digits`) as a scalar: `from_int(integer part) + from_ratio(fraction, 10^d)`,
/// the fraction truncated to its first 18 digits (1e-18 at most, far below the 2^-32 step of
/// `Fixed`), ONE rounding (to nearest, ties to even for `Fixed`). Panics with `nalgebra: value
/// out of range` when the integer part reaches `2^31`.
fn decimal<T, impl R: Real<T>, +Drop<T>, +Add<T>>(neg: bool, digits: Span<u8>, point: i64) -> T {
    let len: i64 = digits.len().into();
    // Integer part: digits[0..point], then (point - len) zeros.
    let mut int_part: u64 = 0;
    let mut k: i64 = 0;
    while k < point {
        let d: u64 = if k < len {
            (*digits[k.try_into().unwrap()]).into()
        } else {
            0
        };
        if int_part == 0 && d == 0 && k >= len {
            // Zeros shifted in after a zero integer part: still zero.
            break;
        }
        int_part = int_part * 10 + d;
        if int_part >= 0x80000000 {
            core::panic_with_felt252(errors::VALUE_OUT_OF_RANGE);
        }
        k += 1;
    }
    // Fractional part: (-point) zeros when `point < 0`, then digits[max(point, 0)..], at most 18.
    let mut frac: i64 = 0;
    let mut den: i64 = 1;
    let mut taken: usize = 0;
    let mut k: i64 = point;
    while taken != FRAC_DIGITS && k < len {
        let d: i64 = if k < 0 {
            0
        } else {
            (*digits[k.try_into().unwrap()]).into()
        };
        frac = frac * 10 + d;
        den = den * 10;
        taken += 1;
        k += 1;
    }
    let int_part: i32 = int_part.try_into().unwrap();
    if neg {
        let v = R::from_int(-int_part);
        if frac == 0 {
            v
        } else {
            v + R::from_ratio(-frac, den)
        }
    } else {
        let v = R::from_int(int_part);
        if frac == 0 {
            v
        } else {
            v + R::from_ratio(frac, den)
        }
    }
}

/// A `Value` at `pos` (the atomic rule: no inner spaces): the scalar and the position after
/// it, `None` when the text there is not a value.
fn value<T, impl R: Real<T>, +Drop<T>, +Add<T>>(data: Span<u8>, pos: usize) -> Option<(T, usize)> {
    let mut pos = pos;
    let mut neg = false;
    match peek(data, pos) {
        Some(b) => { if b == PLUS {
            pos += 1;
        } else if b == MINUS {
            neg = true;
            pos += 1;
        } },
        None => { return None; },
    }
    let mut digits: Array<u8> = array![];
    let mut n_int: usize = 0;
    if is_digit(peek(data, pos)) {
        pos = digits_into(data, pos, ref digits);
        n_int = digits.len();
        if peek(data, pos) == Some(DOT) {
            pos = digits_into(data, pos + 1, ref digits);
        }
    } else if peek(data, pos) == Some(DOT) && is_digit(peek(data, pos + 1)) {
        pos = digits_into(data, pos + 1, ref digits);
    } else {
        return None;
    }
    // Optional exponent: taken only when complete.
    let mut exp: i64 = 0;
    let e = peek(data, pos);
    if e == Some(LOWER_E) || e == Some(UPPER_E) {
        let mut q = pos + 1;
        let mut exp_neg = false;
        let s = peek(data, q);
        if s == Some(PLUS) {
            q += 1;
        } else if s == Some(MINUS) {
            exp_neg = true;
            q += 1;
        }
        if is_digit(peek(data, q)) {
            let mut e: u32 = 0;
            while let Some(b) = peek(data, q) {
                if b < ZERO || b > NINE {
                    break;
                }
                if e < EXP_CAP {
                    e = e * 10 + (b - ZERO).into();
                }
                q += 1;
            }
            exp = if exp_neg {
                -e.into()
            } else {
                e.into()
            };
            pos = q;
        }
    }
    let point: i64 = n_int.into() + exp;
    Some((decimal(neg, digits.span(), point), pos))
}

/// An `Entry` at `pos` (`Dimension Dimension Value`, spaces between): the two indices (`None`
/// when one does not fit `usize`), the value and the position after it; `None` when the text
/// there is not an entry.
fn entry<T, impl R: Real<T>, +Drop<T>, +Add<T>>(
    data: Span<u8>, pos: usize,
) -> Option<(Option<usize>, Option<usize>, T, usize)> {
    let (r, q) = dimension(data, pos)?;
    let (c, q) = dimension(data, skip_spaces(data, q))?;
    let (v, q) = value(data, skip_spaces(data, q))?;
    Some((r, c, v, q))
}

/// The sparse matrix described by the Matrix Market text `data` (coordinate format, real values,
/// 1-based indices; see the module doc for the accepted syntax). Returns `None` when a shape
/// number or an index does not fit `usize` (upstream: a failed `parse::<usize>()`); panics with
/// `nalgebra: matrix market syntax` when the header or the shape line does not match the
/// grammar (upstream `unwrap`s the parse), with `nalgebra: index out of bounds` for an index 0
/// (upstream subtracts 1 from it) or beyond the shape (`from_triplet`), with `nalgebra: value
/// out of range` for a value the scalar cannot hold. Duplicated entries are summed. Values are
/// read in decimal and rounded ONCE to the scalar (upstream parses an `f64`, then converts).
/// Upstream: `nalgebra::io::cs_matrix_from_matrix_market_str(&str)`.
pub fn cs_matrix_from_matrix_market_str<
    T,
    impl R: Real<T>,
    +Copy<T>,
    +Drop<T>,
    +Drop<R::Wide>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
    +PartialEq<T>,
>(
    data: @ByteArray,
) -> Option<CsMatrix<T>> {
    let data = bytes_of(data);
    // SOI NEWLINE* (spaces allowed around).
    let mut pos: usize = 0;
    loop {
        let p = skip_spaces(data, pos);
        match newline(data, p) {
            Some(q) => { pos = q; },
            None => {
                pos = p;
                break;
            },
        }
    }
    // Header.
    if peek(data, pos) != Some(PERCENT) || peek(data, pos + 1) != Some(PERCENT) {
        core::panic_with_felt252(errors::MATRIX_MARKET_SYNTAX);
    }
    pos = line_end(data, pos + 2);
    // (NEWLINE Comments)*
    loop {
        let p = skip_spaces(data, pos);
        match newline(data, p) {
            Some(q) => {
                let q = skip_spaces(data, q);
                if peek(data, q) == Some(PERCENT) {
                    pos = line_end(data, q + 1);
                } else {
                    break;
                }
            },
            None => { break; },
        }
    }
    // NEWLINE Shape
    let p = skip_spaces(data, pos);
    let p = match newline(data, p) {
        Some(q) => skip_spaces(data, q),
        None => core::panic_with_felt252(errors::MATRIX_MARKET_SYNTAX),
    };
    let (nrows, p) = dimension(data, p).expect(errors::MATRIX_MARKET_SYNTAX);
    let (ncols, p) = dimension(data, skip_spaces(data, p)).expect(errors::MATRIX_MARKET_SYNTAX);
    let (_, p) = dimension(data, skip_spaces(data, p)).expect(errors::MATRIX_MARKET_SYNTAX);
    pos = p;
    let nrows = nrows?;
    let ncols = ncols?;
    // (NEWLINE Entry?)*
    let mut rows: Array<usize> = array![];
    let mut cols: Array<usize> = array![];
    let mut vals: Array<T> = array![];
    let mut overflow = false;
    while let Some(q) = newline(data, skip_spaces(data, pos)) {
        pos = q;
        // Entry = Dimension Dimension Value, or nothing (then the next NEWLINE is expected).
        if let Some((r, c, v, end)) = entry(data, skip_spaces(data, q)) {
            pos = end;
            match (r, c) {
                (
                    Some(r), Some(c),
                ) => {
                    if r == 0 || c == 0 {
                        core::panic_with_felt252(base_errors::INDEX_OUT_OF_BOUNDS);
                    }
                    rows.append(r - 1);
                    cols.append(c - 1);
                    vals.append(v);
                },
                _ => {
                    overflow = true;
                    break;
                },
            }
        }
    }
    if overflow {
        return None;
    }
    Some(CsMatrixTrait::from_triplet(nrows, ncols, rows.span(), cols.span(), vals.span()))
}
