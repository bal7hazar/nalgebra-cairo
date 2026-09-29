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

pub use nalgebra_sparse::io::matrix_market::*;
