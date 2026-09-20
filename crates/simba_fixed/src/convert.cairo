//! Conversions between glam.cairo's `fixed::Fixed` and simba's `simba::fixed::Fixed`.
//!
//! Both are `struct { raw: i64 }` holding the same Q32.32 value (`value = raw / 2^32`), so a
//! conversion is a pure relabelling: no arithmetic, no range check, no rounding. The generated
//! Sierra is a single `struct_deconstruct` / `struct_construct` pair, which the compiler folds
//! away (see `bench_simba_fixed_*` in `crate::benches`: every `via_fixed` variant costs exactly
//! as much as its `via_simba` counterpart).

use fixed::Fixed as Glam;
use simba::fixed::Fixed as Simba;

/// glam.cairo's scalar seen as simba's. Exact, infallible, free.
#[inline(always)]
pub fn to_simba(x: Glam) -> Simba {
    Simba { raw: x.raw }
}

/// simba's scalar seen as glam.cairo's. Exact, infallible, free.
#[inline(always)]
pub fn from_simba(x: Simba) -> Glam {
    Glam { raw: x.raw }
}

/// `let s: simba::fixed::Fixed = g.into();`
pub impl GlamIntoSimba of Into<Glam, Simba> {
    #[inline(always)]
    fn into(self: Glam) -> Simba {
        to_simba(self)
    }
}

/// `let g: fixed::Fixed = s.into();`
pub impl SimbaIntoGlam of Into<Simba, Glam> {
    #[inline(always)]
    fn into(self: Simba) -> Glam {
        from_simba(self)
    }
}

#[cfg(test)]
mod tests {
    use fixed::Fixed as Glam;
    use simba::fixed::Fixed as Simba;
    use super::{GlamIntoSimba, SimbaIntoGlam, from_simba, to_simba};

    #[test]
    fn test_convert_roundtrip_preserves_raw() {
        let raws: [i64; 5] = [0, 1, -1, -0x8000000000000000, 0x7fffffffffffffff];
        for raw in raws.span() {
            let g = Glam { raw: *raw };
            assert!(to_simba(g).raw == *raw);
            assert!(from_simba(Simba { raw: *raw }).raw == *raw);
            assert!(from_simba(to_simba(g)) == g);
        }
    }

    #[test]
    fn test_convert_into_impls() {
        let g = Glam { raw: -0x280000000 };
        let s: Simba = g.into();
        assert!(s == Simba { raw: -0x280000000 });
        let back: Glam = s.into();
        assert!(back == g);
    }
}
