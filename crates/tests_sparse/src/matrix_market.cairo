//! `cs_matrix_from_matrix_market_str`: upstream's grammar (header, comments, shape, entries,
//! spaces and newlines), values rounded once, `None` / panics where upstream returns `None` /
//! panics.

use fixed::Fixed;
use nalgebra::io::cs_matrix_from_matrix_market_str;
use nalgebra::sparse::{CsMatrix, CsMatrixTrait};
use nalgebra_testing::black_box;
use crate::helpers::{dense, ints, raws};

fn parse(text: ByteArray) -> Option<CsMatrix<Fixed>> {
    cs_matrix_from_matrix_market_str(@black_box(text))
}

#[test]
fn test_matrix_market_basic() {
    let m = parse(
        "%%MatrixMarket matrix coordinate real general\n% a comment\n3 3 4\n1 1 1.5\n2 1 -2\n3 3 .25\n1 3 2e1\n",
    )
        .unwrap();
    assert!(m.shape() == (3, 3));
    assert!(m.len() == 4);
    // Column-major: [1.5, -2, 0], [0, 0, 0], [20, 0, 0.25].
    assert!(
        dense(
            m,
        ) == raws(
            array![0x180000000, -0x200000000, 0, 0, 0, 0, 0x1400000000, 0, 0x40000000].span(),
        ),
    );
}

#[test]
fn test_matrix_market_spaces_and_newlines() {
    // Leading blank lines, CRLF, indented comment, trailing spaces, blank line between entries.
    let m = parse(
        "\n  \r\n%%MatrixMarket\r\n%c1\r\n  % c2\r\n2 2 2\r\n  1 1 3  \r\n\r\n2 2 +4.\r\n",
    )
        .unwrap();
    assert!(dense(m) == ints(array![3, 0, 0, 4].span()));
}

#[test]
fn test_matrix_market_duplicates_summed() {
    let m = parse("%%MatrixMarket\n2 2 2\n1 1 1\n1 1 2\n").unwrap();
    assert!(m.len() == 1);
    assert!(dense(m) == ints(array![3, 0, 0, 0].span()));
}

#[test]
fn test_matrix_market_stops_at_first_non_entry() {
    // The grammar has no EOI: the document ends at the incomplete line "1 2".
    let m = parse("%%MatrixMarket\n2 2 3\n1 1 1\n1 2\n2 2 5\n").unwrap();
    assert!(m.len() == 1);
    // An entry followed by garbage counts, the rest is ignored.
    let m = parse("%%MatrixMarket\n2 2 3\n1 1 1 x\n2 2 5\n").unwrap();
    assert!(m.len() == 1);
    // No entry at all.
    let m = parse("%%MatrixMarket\n2 3 0").unwrap();
    assert!(m.shape() == (2, 3));
    assert!(m.len() == 0);
}

#[test]
fn test_matrix_market_values_rounded_once() {
    let m = parse(
        "%%MatrixMarket\n7 1 7\n1 1 0.1\n2 1 -0.1\n3 1 1.5E-3\n4 1 123.456e2\n5 1 -2.5e-1\n6 1 1e-30\n7 1 5e+0\n",
    )
        .unwrap();
    assert!(
        dense(
            m,
        ) == raws(
            array![429496730, -429496730, 6442451, 53023948249498, -1073741824, 0, 0x500000000]
                .span(),
        ),
    );
}

#[test]
fn test_matrix_market_incomplete_exponent() {
    // "2e" is the value 2 followed by "e": the entry counts, the rest of the text is ignored.
    let m = parse("%%MatrixMarket\n2 1 2\n1 1 2e\n2 1 1\n").unwrap();
    assert!(dense(m) == ints(array![2, 0].span()));
}

#[test]
fn test_matrix_market_dimension_overflow() {
    assert!(parse("%%MatrixMarket\n4294967296 2 0\n").is_none());
    assert!(parse("%%MatrixMarket\n2 2 1\n4294967296 1 1\n").is_none());
}

#[test]
#[should_panic(expected: 'nalgebra: matrix market syntax')]
fn test_matrix_market_missing_header() {
    let _ = parse("3 3 0\n");
}

#[test]
#[should_panic(expected: 'nalgebra: matrix market syntax')]
fn test_matrix_market_blank_line_before_shape() {
    let _ = parse("%%MatrixMarket\n\n3 3 0\n");
}

#[test]
#[should_panic(expected: 'nalgebra: matrix market syntax')]
fn test_matrix_market_short_shape() {
    let _ = parse("%%MatrixMarket\n3 3\n");
}

#[test]
#[should_panic(expected: 'nalgebra: index out of bounds')]
fn test_matrix_market_zero_index() {
    let _ = parse("%%MatrixMarket\n2 2 1\n0 1 1\n");
}

#[test]
#[should_panic(expected: 'nalgebra: index out of bounds')]
fn test_matrix_market_index_beyond_shape() {
    let _ = parse("%%MatrixMarket\n2 2 1\n3 1 1\n");
}

#[test]
#[should_panic(expected: 'nalgebra: value out of range')]
fn test_matrix_market_value_out_of_range() {
    let _ = parse("%%MatrixMarket\n1 1 1\n1 1 3e9\n");
}
