# Origami (dojoengine/origami) — benchmark report for nalgebra.cairo

Source studied: `scratchpad/refs/origami` @ `1ddafcb` (2025-09-23, "chore: bump dojo version"),
workspace version `1.1.2`, Scarb/Cairo `2.12.2`, edition `2024_07`. ~6,000 lines of Cairo.
All paths below are relative to that checkout. Gas figures are `scarb test` ("gas usage est.",
i.e. Sierra gas from `cairo-test`) obtained by running the suite in a scratch copy
(`scratchpad/tmp/origami-run`); the micro-benchmarks in section 3.3 are mine
(`scratchpad/tmp/bench`), not origami's.

---

## 1. Workspace layout

```
Scarb.toml              # [workspace] root, no root package
Scarb.lock
.tool-versions          # scarb 2.12.2 (asdf)
crates/
  Scarb.toml            # STALE: legacy umbrella "origami" package (deps on dojo), not a member
  algebra/  defi/  map/  random/  rating/  security/   # each: Scarb.toml, README.md, src/
scripts/build-all.sh    # STALE: still calls `sozo build --package origami_token/governance`
docs/images, docs/videos  # logo + gif only, no written docs
.github/workflows/ci.yml, release.yaml, ISSUE_TEMPLATE/, pull_request_template.md
```

Root `Scarb.toml`:

```toml
[workspace]
members = ["crates/contracts", "crates/algebra", "crates/defi", "crates/map",
           "crates/random", "crates/rating", "crates/security"]
[workspace.package]
version = "1.1.2"
edition = "2024_07"
[workspace.dependencies]
cubit = { git = "https://github.com/bengineer42/cubit", branch = "bump-cairo-gt-2.8" }
starknet = "^2.12.2"
cairo_test = "^2.12.2"
```

Observations:

- One Scarb package per domain, named `origami_<domain>`; every crate inherits
  `version.workspace = true` / `edition.workspace = true` and declares
  `[dev-dependencies] cairo_test.workspace = true`. Consumers pull a single crate by git:
  `origami_map = { git = "https://github.com/dojoengine/origami" }`.
- **Crates do not depend on each other.** `Scarb.lock` shows only `origami_algebra -> cubit` and
  `origami_defi -> cubit`; `map`, `random`, `rating`, `security` have zero dependencies (not even
  `starknet`). This is why `map` re-implements its own `Seeder` (Poseidon) rather than using
  `origami_random`, and `rating` carries its own private `pow`.
- Hygiene debt: `crates/contracts` is listed as a member but does not exist (Scarb 2.12.2 silently ignores it);
  `.gitmodules` points to a removed `examples/bridge`; `crates/Scarb.toml` and
  `scripts/build-all.sh` are leftovers from the pre-split Dojo era. The cubit dependency is a
  personal fork pinned to a branch, not a tag/rev.
- **Tests** are all inline `#[cfg(test)] mod tests` at the bottom of each source file (127 `#[test]`
  across the repo; no `tests/` dir, no snforge). Test-only helpers are gated at module level:
  `#[cfg(target: "test")] pub mod printer;` (`crates/map/src/lib.cairo`).
- **CI** (`.github/workflows/ci.yml`): `scarb fmt --check` -> `scarb build` -> one job per crate
  running `scarb test --package origami_<x>` in parallel (`needs: [check, build]`), Scarb pinned via
  `env.SCARB_VERSION`. No gas snapshot, no regression gate, no lint beyond fmt.
  `release.yaml` only creates a GitHub release on `v*` tags.
- Docs: a README per crate (install snippet + usage). `///` doc-comments with
  `# Arguments / # Returns / # Effects / # Panics` sections are systematic in `map`, absent in `algebra`.

---

## 2. The `algebra` crate

Files: `crates/algebra/src/{lib,vec2,vector,matrix}.cairo` (3 + 260 + 112 + 384 lines, more than half tests).

**It is effectively unusable as a dependency**: `lib.cairo` is `mod matrix; mod vec2; mod vector;`
(no `pub`) and the structs/traits are not `pub` either; under edition `2024_07` nothing is
exported. It reads as an early experiment that was never revisited, in sharp contrast with `map`.

### 2.1 `Vec2<T>` (vec2.cairo) — glam-style, generic, stack-allocated

```cairo
struct Vec2<T> { x: T, y: T }
impl Vec2Copy<T, impl TCopy: Copy<T>> of Copy<Vec2<T>>;
impl Vec2Drop<T, impl TDrop: Drop<T>> of Drop<Vec2<T>>;

impl Vec2Impl<T, impl TCopy: Copy<T>, impl TDrop: Drop<T>> of Vec2Trait<T> {
    #[inline(always)]
    fn new(x: T, y: T) -> Vec2<T> { Vec2 { x: x, y: y } }
    // #[inline(always)] is not allowed for functions with impl generic parameters.
    fn dot<impl TMul: Mul<T>, impl TAdd: Add<T>>(self: Vec2<T>, rhs: Vec2<T>) -> T {
        (self.x * rhs.x) + (self.y * rhs.y)
    }
```

- Ops: `new`, `splat`, `select(mask: Vec2<bool>, ..)`, `dot`, `dot_into_vec`, swizzles `xx/xy/yx/yy`.
  That is all. No `Add/Sub/Mul/Neg`, no scalar mul, `length`, `normalize`, `cross/perp`, `lerp`,
  `Zero/One`, `PartialEq`, `Serde`, `Store`.
- Good: trait bounds are declared **per method** (`dot` needs `Mul + Add`, constructors need only
  `Copy + Drop`), so `Vec2<bool>` works. Tests use `cubit::f128::Fixed` as `T`.
- Measured: `test_dot` (two Fixed mul + one add, plus asserts, twice) = 75,070 gas. The cost is
  entirely cubit's sign-magnitude 64.64 `mul` (`wide_mul` then `u256_safe_div_rem` by `ONE`).

### 2.2 `Vector<T>` (vector.cairo) and `Matrix<T>` (matrix.cairo) — dynamic, `Span`-backed

```cairo
#[derive(Copy, Drop)] struct Vector<T> { data: Span<T> }
#[derive(Copy, Drop)] struct Matrix<T> { data: Span<T>, rows: u8, cols: u8 }
```

- `Vector`: `new/get/size/dot`, `Add`, `Sub`. `dot` consumes both spans with `pop_front` (the one
  efficient loop in the crate); `Add/Sub` loop with a `u8` index, call bounds-checked
  `get(index)` twice per element, and `append` into a fresh `Array`.
- `Matrix`: `new/get/transpose/minor/det/inv`, `Add/Sub/Mul`. Row-major flat `Span<T>`, runtime
  `rows/cols: u8`, runtime dimension asserts with `felt252` error consts in a `mod errors`.
- `det` is **recursive Laplace expansion** allocating a new `Array` per minor (O(n!));
  `inv` is adjugate / det, computing `rows*cols` minors each with its own recursive `det`:

```cairo
let col = index / self.rows;  let row = index % self.rows;   // two checked u8 divisions per cell
let mut minor = self.minor(row, col);
let cofactor = if (row + col) % 2 == 0 { minor.det() } else { -minor.det() };
values.append(cofactor / determinant);
```

- Every impl repeats an 11-bound list (`+Mul +Div +Add +AddAssign +Sub +SubAssign +Neg +Zero +Copy
  +Drop`) even for `Add`, because `get` lives in the fully-bounded `MatrixImpl`.
- Oddities: `get(ref self ..)` takes `ref` on a `Copy` struct for no reason (forces `let mut`
  everywhere); index arithmetic in `u8` overflows beyond 255 cells; `minor` builds
  `new(.., self.cols - 1, self.rows - 1)` (swapped, harmless only for square input).
- Measured with `i128` elements: `add` 2x3 = 62,930; `mul` 2x2 = 159,600; `mul` 2x3*3x2 = 217,000;
  `det` 3x3 = 312,070; `inv` 2x2 = 218,120; **`inv` 3x3 = 1,230,370 gas**. A hand-unrolled 3x3
  inverse is 9 cofactors of 2 mul each + 3 mul for det + 9 div; my estimate (not measured) is that the generic
  path costs roughly an order of magnitude more than that.

### 2.3 Gaps vs nalgebra

| nalgebra | origami_algebra |
|---|---|
| Statically-sized `SMatrix<T,R,C>`, `Vector2/3/4`, `Matrix2/3/4` | only `Vec2<T>`; matrices are runtime-dimensioned |
| `Point`, `Translation`, `Rotation2/3`, `UnitComplex`, `UnitQuaternion`, `Isometry`, `Similarity`, `Transform` | none |
| norm / normalize / `Unit<T>` / cross / outer / component-wise ops / lerp / slerp | none (`dot` only) |
| scalar ops, `Neg`, `*Assign`, `Zero/One/identity`, `PartialEq`, approx-eq | matrix `Add/Sub/Mul` only; no `PartialEq`, no identity |
| LU / QR / Cholesky / SVD / eigen, `solve` | cofactor `inv`/`det` only |
| scalar abstraction (`RealField`, `ComplexField`, `SimdValue`) | ad-hoc `+Mul<T> +Add<T> ...` bound lists, no scalar trait |
| serde / storage | none (`Span` cannot be `starknet::Store`d anyway) |

Verdict: nothing to reuse here except two patterns (generic struct + manual `Copy/Drop` impls;
per-method bounds). The design to study in origami is `map`, not `algebra`.

---

## 3. Code-efficiency idioms (the `map` crate)

Core design: a whole 2D grid (up to 252 cells, `MAX_SIZE: u8 = 252` in
`helpers/asserter.cairo`) is **one `felt252` bitmap**; positions are a single `u8` index
(`x + y*width`); generators are pure functions `felt252 -> felt252`.
`#[derive(Copy, Drop)] pub struct Map { width: u8, height: u8, grid: felt252, seed: felt252 }`
(`map.cairo`) is 4 felts, copyable, storable in one slot, no `Array`, no `Felt252Dict`.

### 3.1 Catalogue

**I1 — Precomputed power-of-two table as a `const` fixed-size array** — `helpers/power.cairo`

```cairo
const TWO_POWER: [u256; 256] = [0x1, 0x2, 0x4, ... ];
#[inline]
fn pow(exp: u8) -> u256 { *TWO_POWER.span().at(exp.into()) }
```
Replaces a `while exp != 0 { r *= 2 }` loop. Const arrays live in the program's data segment:
lookup is O(1) plus one bounds check. Measured (origami test) 2,100–2,770 gas per call including
harness; my bench: table 2,470 vs 8-arm `match` 670 vs loop (exp=7) **109,320**.

**I2 — Shifts expressed as mul/div by `2^n`** — everywhere (`bitmap.cairo`, `caver.cairo`, `deck.cairo`)

```cairo
let default: u256 = seed.into() / TwoPower::pow(252 - size);   // caver.cairo: seed >> (252-size)
x /= 0x100000000;                                              // bitmap.cairo: x >>= 32
bitmap /= TWO_POW_1;                                           // deck.cairo:   bitmap >>= 1
```
Cairo core has no `<<`/`>>` for integers; a shift *is* a division (range-check builtin only).
Dividing by a literal lets the compiler bake the `NonZero` divisor in.

**I3 — Bit test / set / clear with arithmetic only, branch-free** — `helpers/bitmap.cairo`

```cairo
fn get(x: felt252, index: u8) -> u8 { let x: u256 = x.into(); (x / TwoPower::pow(index) % 2).try_into().unwrap() }
fn set(x: felt252, index: u8) -> felt252 {
    let x: u256 = x.into();  let offset: u256 = TwoPower::pow(index);
    let bit = x / offset % 2;
    let offset: u256 = offset * (1 - bit);          // adds 0 if already set: idempotent, no `if`
    (x + offset).try_into().unwrap()
}
fn unset(..) { .. let offset = offset * bit; (x - offset).try_into().unwrap() }
```
Intent: avoid the bitwise builtin and avoid branches (both arms of a Cairo `if` cost steps plus
branch-alignment). `get` returns `u8` (0/1), not `bool`, so callers **sum** results instead of
branching — see I6. Measured: `get` 15,540, `set` 34,350, `unset` 32,080 gas (see 3.3 for why this
is *not* actually cheap: it is u256 div/mod/mul).

**I4 — SWAR popcount, shifts replaced by divisions, in 32-bit limbs** — `helpers/bitmap.cairo`

```cairo
fn _popcount(mut x: u32) -> u8 {
    x -= ((x / 2) & 0x55555555);
    x = (x & 0x33333333) + ((x / 4) & 0x33333333);
    x = (x + (x / 16)) & 0x0f0f0f0f;
    x += (x / 256);  x += (x / 65536);
    return (x % 64).try_into().unwrap();
}
fn popcount(x: felt252) -> u8 {            // outer loop: 8 limbs max, early exit on x == 0
    let mut x: u256 = x.into();  let mut count: u8 = 0;
    while (x > 0) { count += Self::_popcount((x % 0x100000000).try_into().unwrap()); x /= 0x100000000; }
    count
}
```
At most 8 iterations instead of 252; the outer `while x > 0` exits early for sparse maps. Masks that
have no arithmetic equivalent stay as `&` — the pragmatic mix. 210,244 gas for a 222-bit input.

**I5 — LSB index by unrolled binary search (8 probes, no loop)** — `helpers/bitmap.cairo`

```cairo
let mut r: u8 = 255;
if (x & 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF) > 0 { r -= 128; } else { x /= 0x100000000000000000000000000000000; }
if (x & 0xFFFFFFFFFFFFFFFF) > 0 { r -= 64; } else { x /= 0x10000000000000000; }
... // 32, 16, 8, 4, 2, 1
```
log2(256) = 8 fixed probes, constant masks/divisors, early `return 0` for `x == 0`. 39,728 gas.
(No MSB counterpart exists in the repo.) A cheaper first probe was available: `u256.low != 0`.

**I6 — Branch-free neighbour counting (sum of bits)** — `generators/caver.cairo`

```cairo
if y < height - 1 { let index = (y + 1) * width + x; floor_count += Bitmap::get(grid, index); }
```
Cellular-automaton smoothing adds `get()` results (0/1) into a counter; the only branches are the
bounds checks. Thresholds are top-level `const WALL_TO_FLOOR_THRESHOLD: u8 = 4;`.

**I7 — Whole-grid operations as one bitwise op** — `mazer.cairo`, `digger.cairo`, `spreader.cairo`

```cairo
let merge: u256 = grid.into() | maze.into();          // union of two maps
let objects: u256 = grid.into() ^ merge.into();       // cells that changed = placed objects
```
One `|`/`^` on u256 (2 bitwise-builtin uses) replaces a 252-iteration loop. Bitwise is used where it
replaces a *loop*, arithmetic where it replaces a *single-bit* op.

**I8 — Row/grid construction by felt252 Horner arithmetic** — `map.cairo::Private::empty`

```cairo
let offset: u256 = TwoPower::pow(width);
let row: felt252 = ((offset - 1) / 2).try_into().unwrap() - 1;   // 0b0111..110: strip head+tail bits
let offset: felt252 = offset.try_into().unwrap();
loop { if index == 0 { break; } default += row; default *= offset; index -= 1; }
```
A row mask is computed in closed form (`(2^w - 1)/2 - 1`), then rows are stacked with
`felt252` `+=`/`*=` — native field ops, **no range check, no overflow check** (about 1 step each).
Loop runs `height` times, not `width*height`. 103,206 gas for a full 18x14 map incl. corridor.

**I9 — Packed permutation + `match` lookup table + div/mod "pop"** — `types/direction.cairo`

```cairo
fn compute_shuffled_directions(seed: felt252) -> u32 {
    let mut random: u32 = (seed.into() % 24_u256).try_into().unwrap();
    match random { 0 => 0x2468, 1 => 0x2486, ... 22 => 0x8624, _ => 0x8642 }
}
fn pop_front(ref directions: u32) -> Direction {
    let direciton: u8 = (directions % DIRECTION_SIZE).try_into().unwrap();   // DIRECTION_SIZE = 0x10
    directions /= DIRECTION_SIZE;
    direciton.into()
}
```
All 4! = 24 orderings of N/E/S/W are precomputed as nibble-packed `u32`s; shuffling is one `% 24`
plus a jump table (consecutive-integer `match` compiles to a jump table, O(1)); no Fisher-Yates, no
`Array<Direction>`. The "queue" is an integer consumed by `% 16`, `/ 16`; BFS even uses
`while directions != 0` as the termination test (`finders/bfs.cairo`). `u8 -> Direction` and
`Direction -> felt252` are also `match` tables.

**I10 — Manually unrolled fixed-count loops** — `mazer.cairo`, `walker.cairo`, `digger.cairo`, `astar.cairo`

The 4-direction body is copy-pasted four times ("[Assess] Direction 1..4") instead of looping:
saves loop-counter arithmetic, the recursive-call frame Cairo generates per loop, and lets
`#[inline]` helpers specialise. Same motivation as the fully unrolled LSB.

**I11 — Two logical maps in one `Felt252Dict` via key offset** — `helpers/heap.cairo`

```cairo
const KEY_OFFSET: felt252 = 252;
/// Both information is stored in the same map to save gas.
pub keys: Felt252Dict<u8>,   // [0,252): heap index -> key ; [252,504): key -> heap index
fn contains(..) { let index = self.keys.get(key.into() + KEY_OFFSET); let item_key = self.keys.get(index.into()); index < self.len && item_key == key }
```
One dict to squash at destruction instead of two (squash cost is per-dict and per-access).
`swap` moves only two `u8` keys, never the `Nullable<T>` payloads. The `contains` double-lookup
avoids a separate "present" flag. Explicit `Destruct` impl calls `squash()` on both dicts.

**I12 — Sparse virtual array (lazy Fisher-Yates)** — `random/src/deck.cairo`

`cards: Felt252Dict<u8>` where a missing key (0) means "card == key"; `draw`/`withdraw` swap the
drawn slot with the last slot. A 52-card (or 2^32-card) deck is never materialised: O(1) per draw.
`from_bitmap` consumes a `u128` with `& MASK_1` / `/= TWO_POW_1` and exits early on `bitmap == 0`.

**I13 — Early exits** — `astar.cairo` (`return array![].span()` if start/target not walkable, before
any dict allocation), `spreader.cairo`/`walker.cairo` (tail-recursive `iter` with
`if count == 0 { return grid; }`), `popcount` (`while x > 0`), `least_significant_bit` (`x == 0`),
short-circuit `&&` chains in `Mazer::check` ordered cheapest-first (coordinate compares before
`Bitmap::get`).

**I14 — Type-width choices**

- `felt252` for storage/transport of packed data and for overflow-free accumulation (I8);
  converted to `u256` only at the point where div/mod/bitwise is needed, then `try_into().unwrap()` back.
- `u8` for positions/dimensions, `u16` for costs (`Node { position: u8, source: u8, gcost: u16, hcost: u16 }`),
  `u32` for packed directions. Small ints keep structs `Copy` and dict values single-felt.
- `u256` rather than `u128` for the bitmap because the grid exceeds 128 bits — this is the expensive
  choice (see 3.3). There is no two-limb (`low`/`high`) specialisation anywhere.
- Integer `Sqrt` from core (`core::num::traits::Sqrt`) — a hint-verified libfunc — instead of a
  Newton loop (`finder.cairo::euclidean`, `elo.cairo`).

**I15 — `#[inline]` policy** — 83 `#[inline]` in `map` (every helper, including large recursive ones
like `Mazer::iter` where it is a no-op hint), 14 `#[inline(always)]` in `algebra`/`random` on
trivial constructors. Note the comment in `vec2.cairo`: `#[inline(always)]` is rejected on functions
with their own `impl` generic parameters. No evidence of measured inlining decisions.

**I16 — Consts and error modules** — magic numbers are top-level `const`
(`MULTIPLIER: u256 = 10000`, `KEY_OFFSET`, `DIRECTION_SIZE`); errors are `felt252` short-string
consts in `pub mod errors`, used with `assert(cond, errors::X)` (cheaper than `assert!` with
`ByteArray` formatting — although the tests do use `assert_eq!`).

**I17 — Stateless namespaces via `#[generate_trait] pub impl Bitmap of BitmapTrait`** with
`Self`-less functions (`Bitmap::get(grid, i)`), and a second non-`pub` `impl Private of PrivateTrait`
in the same file for internals. No struct wrapping the felt: zero construction/destructuring cost.

**I18 — Seeds**: `Seeder::shuffle` = Poseidon of two felts (`helpers/seeder.cairo`); random in-range
values by `seed.into() % n` on u256; re-seeding by `shuffle(seed.low, seed.high)` to reuse limbs.

### 3.2 Where origami is *not* efficient (useful negative examples)

- No `DivRem` anywhere (grep: 0 hits). `let (x, y) = (position % width, position / width);` appears in
  `finder.cairo` (x3), `mazer.cairo`, `walker.cairo`, `caver.cairo`, `asserter.cairo` — two checked
  divisions where one `DivRem::div_rem` suffices (measured: 2,960 vs 1,940 gas net).
  `x / offset % 2` on u256 likewise.
- `Caver::assess` performs 1 + up to 8 `Bitmap::get` (each a table lookup + u256 div + u256 mod)
  per cell per pass: `test_caver_generate` = **86.7M gas** for 18x14, order 2. A row-window
  approach (extract 3 rows once, slide a 3-bit window) would cut this by an order of magnitude.
- `Felt252Dict<bool>` for `visited` in all finders, although a second `felt252` bitmap would do.
- `hex.cairo::tiles_within_range` is O(n^2) linear `visited` scans over `Array<Hex>` (9.5M gas for a
  small range); `neighbors()` allocates an `Array` of 6 to test membership.
- `algebra` (section 2): allocation + bounds-checked `get` + div/mod index decomposition per element.

### 3.3 Micro-benchmarks: does "math > bitwise > loops" hold? (Cairo 2.12.2, `cairo-test` gas, net of call baseline)

| operation | net gas | | operation | net gas |
|---|---|---|---|---|
| felt252 `x*y + y` | **100** | | u64 `wide_mul` -> u128 | **~0** |
| u64/u128 checked `+` | 2,370 | | u64 checked `*` | 2,370 |
| u128 checked `*` | 4,930 | | felt252 -> u128 `try_into().unwrap()` | 3,210 |
| u128 `/ 2^64` (const) | 2,480 | | u128 `& (2^64-1)` (const) | 1,283 |
| u128 `x / p % 2` | 3,860 | | u128 `x & p != 0` | 1,583 |
| u128 two `DivRem` | 3,160 | | u128 `sqrt` | 1,580 |
| **u256 `x / p % 2`** (origami `get`) | **13,100** | | u256 `x & p != 0` | 2,466 |
| **u256 math `set`** (origami I3) | **32,280** | | u256 `x \| p` | 1,866 |
| u8 `(p % w, p / w)` | 2,960 | | u8 `DivRem::div_rem` | 1,940 |
| 2^n: const `[u256; 8]` table | 2,470 | | 2^n: `match` 670 / loop(7) 109,320 | |

Reading:

1. **Loops are catastrophically worse** than either alternative (40x–160x). The user's rule 3 is solid,
   and origami's tables/unrolling (I1, I5, I9, I10) are the right reflex.
2. **"Math over bitwise" is true only for `felt252` math and small-int math.** felt252 add/mul is
   ~1 step; but u256 division is a heavyweight libfunc (`u256_safe_divmod`, many range checks) and
   u256 mul is a checked 4-limb product. In Sierra gas the bitwise builtin is priced at 583, so on
   u256 **origami's arithmetic bit-set is ~17x more expensive than `x | p`**, and bit-get ~5x more
   than `x & p`. On u128 the two are within ~2x of each other, bitwise still ahead.
   (Caveat: under the legacy Cairo-steps fee table the bitwise builtin weighed 64 steps vs 16 for a
   range check, which is when origami's idiom was designed; with Sierra-gas pricing the balance flipped.
   This is exactly why nalgebra.cairo must measure rather than assume.)
3. The real hierarchy measured: **felt252 arithmetic < `wide_mul` < bitwise on u128 <= single DivRem on
   <=u128 < checked u128 arithmetic < anything on u256 <<< loops.** Overflow checks and
   `try_into().unwrap()` (panic paths) cost more than the arithmetic itself.

---

## 4. Fixed-point / math / trig / sqrt utilities

Origami ships **none of its own**. Everything is delegated:

- **cubit `f128`** (git fork `bengineer42/cubit`, 64.64 **sign-magnitude** `Fixed { mag: u128, sign: bool }`)
  used by `algebra` tests and `defi` (`exp`, `pow`, `ln` in `auction/gda.cairo`, `auction/vrgda.cairo`).
  Cubit's strategy (from the resolved checkout): `mul` = `a.mag.wide_mul(b.mag)` then
  `u256_safe_div_rem(.., ONE_u256)`, assert `high == 0`, sign = `a.sign ^ b.sign`; `div` = widen,
  u256 divide; `sqrt` = core `u128.sqrt()` rescaled; `exp/ln` via `exp2/log2` with a LUT for the
  integer part and polynomial for the fraction; trig via LUT (`math/lut.cairo`, `trig.cairo`).
  Every mul therefore pays a u256 division (the ~13k-gas class op of 3.3) plus sign branching in
  `add` (three-way `if`). That is the 75k-gas `Vec2::dot`.
- **core `Sqrt`**: `finder.cairo::euclidean` scales before rooting to keep precision in integers —
  `(multiplier * (dx*dx + dy*dy)).sqrt()`; `elo.cairo` computes a 16th root as four nested sqrt
  `Sqrt::<u32>::sqrt(Sqrt::<u64>::sqrt(Sqrt::<u128>::sqrt(Sqrt::<u256>::sqrt(powered))))`, exploiting
  that each sqrt halves the width.
- **Algebraic rewriting to stay in integers** (`rating/src/elo.cairo`): `10^(d/400)` becomes
  `(10^((800+d)/25))^(1/16)` with an offset to stay unsigned; result kept as `(magnitude, negative: bool)`;
  `round_div` does round-half-up with `%` and `/` (again no DivRem). Private generic `pow` is
  recursive square-and-multiply (O(log n)) over a 9-bound generic `T`.
- `manhattan` with `if x1 > x2 { x1 - x2 } else { x2 - x1 }` abs-diff on unsigned.

No trig, no vector norm, no angle type, no approximate-equality helper (tests in `defi` hand-roll
`assert_rel_approx_eq` with a `TOLERANCE` const).

---

## 5. Testing conventions and gas practice

- Inline `#[cfg(test)] mod tests { use super::{...}; }` per file (a few files use `mod test`).
  Section comments `// Local imports`, `// Constants` (`const SEED: felt252 = 'SEED';`).
- Naming `test_<module>_<function>_<case>`: `test_bitmap_set_unchanged`, `test_astar_search_impossible`,
  `test_mazer_generate_order_1`, `test_two_power_exp_255`. Panics:
  `#[should_panic(expected: ('Mazer: order > 1 not supported',))]` matching the `errors::` const.
- **Golden-value tests with ASCII art**: grid tests draw the expected 18x14 bitmap in comments and
  assert a single hex literal (`assert_eq!(cave, 0xC039F0...)`). Excellent for readability and for
  catching any behavioural drift when refactoring for gas.
- `algebra` tests use old-style `assert(cond, 'msg')`; `map` uses `assert_eq!`. Determinism via fixed seeds.
- **Gas: nothing is tracked.** Zero `#[available_gas]` in the repo, no `get_available_gas` deltas, no
  snapshot file, no CI gate, no benchmark crate, no alternative implementations kept side by side.
  The only gas signal is the `gas usage est.` that `cairo-test` prints, which nobody records.
  The "to save gas" comments (`heap.cairo`) are unverified claims; section 3.3 shows one central
  assumption (arithmetic bitmap cheaper than bitwise) is false on today's cost model.
- Tests also conflate the measured op with harness and `assert_eq!` formatting cost, and literal inputs
  let the compiler constant-fold (`test_finder_manhattan` = 300 gas, i.e. nothing was executed).

**Against the user's guidance** ("math > bitwise > loops", "every feature ships with gas-tracking tests
comparing alternatives"): origami embodies the first as a *style* (I1–I10) but provides **no
infrastructure for the second**, and as a result carries at least one mis-optimisation. It is a
source of idioms, not of methodology.

---

## 6. Synthesis for nalgebra.cairo

### 6.1 Copy

1. **Workspace shape**: `[workspace]` root with `[workspace.package]` version/edition and
   `[workspace.dependencies]`; one package per concern (`nalgebra_core` scalar/fixed-point,
   `nalgebra` types, later `nalgebra_geometry`, `bench`); `.tool-versions` pin; CI =
   fmt -> build -> per-package test matrix. Unlike origami, **let crates depend on each other**
   (one scalar crate) and pin git deps by `rev`/tag — or better, have no external fixed-point dep.
2. **Pure, stateless, `Copy` value types made of a handful of felts** (`Map` is 4 felts). `Vector2/3`,
   `Matrix2/3`, `Rotation2`, `Isometry2` should be plain structs of scalars: no `Span`, no `Array`,
   no runtime dimensions, no bounds checks. Follow `Vec2<T>`'s manual `Copy`/`Drop` impls and
   **per-method trait bounds**; replace origami's 11-bound lists by a single `Scalar`/`RealField`-like
   trait alias.
3. **Unroll everything of fixed size** (I10/I5): dot, cross, mat mul, det and inverse for 2x2/3x3/4x4 as
   closed-form expressions. Origami's own numbers make the case: generic 3x3 `inv` = 1.23M gas.
4. **Lookup tables over loops** (I1, I9): `const [T; N]` arrays or consecutive-integer `match` for
   powers, trig (sin/cos/atan LUT + interpolation), reciprocal seeds. Prefer `match` for small dense
   tables (670 vs 2,470 gas), const arrays for large ones.
5. **Do accumulation in `felt252`, range-check once** (I8): sum-of-products (dot, matmul rows) as felt
   ops at ~1 step each, then a single conversion + single `DivRem` rescale per output component,
   rather than per-multiply checked u128/u256 arithmetic. Use `wide_mul` (measured ~free) instead of
   checked `*`. This is the single biggest lever vs cubit's per-mul `u256_safe_div_rem`.
6. **Shifts as mul/div by constant powers of two with `DivRem` and `NonZero` consts** (I2) — choose a
   fixed-point scale of 2^k so rescale is one `div_rem` by a literal; consider a biased/two's-complement
   felt representation over sign-magnitude so `add`/`sub` are branch-free (cubit's `add` is a 3-way `if`).
7. **Branch-free selection by arithmetic** (I3/I6: `offset * (1 - bit)`, summing 0/1) for `abs`, `sign`,
   `min/max`, `clamp`, `select(mask, a, b)` — *but only in felt/<=u128 space, and only if the bench says so*.
8. **Packing** (I9, I11): pack small vectors/AABBs/cell coords into one felt for storage and dict
   keys with mul/add, unpack with `DivRem` by const; single dict with key offsets for broad-phase grids.
9. **Early exits and cheap-first `&&` ordering** (I13); **core `Sqrt` libfunc** for norms, pre-scaled to
   keep precision (`euclidean`), never a Newton loop.
10. **Conventions**: `#[generate_trait] pub impl X of XTrait` + private `impl Private`; `pub mod errors`
    with `'Type: message'` felt consts; `/// # Arguments/# Returns/# Panics` docs; per-crate README;
    inline `mod tests`, `test_<type>_<op>_<case>` names; golden-value assertions with explanatory
    ASCII/maths comments; `#[cfg(target: "test")]` for debug printers.

### 6.2 Avoid

1. **The entire `algebra` crate design**: `Span`-backed runtime-dimension matrices, per-element
   `get()` with `expect`, `index / cols` + `index % cols` decomposition, `Array` allocation per op,
   recursive Laplace `det`, `ref self` on `Copy` types, `u8` index arithmetic, non-`pub` modules.
2. **u256 as a working type.** Origami's bitmap `get` = 13k gas, `set` = 32k because of u256 div/mul.
   Keep scalars <= 128 bits (or felt252); if 252 bits are needed, operate on `low`/`high` limbs explicitly.
3. **Dogma without measurement.** Origami's arithmetic bit ops lose to `&`/`|` by 5–17x on the current
   Sierra-gas model. The rule "math > bitwise > loops" must be a *default hypothesis*, verified per
   primitive and per integer width.
4. **Separate `%` and `/`** on the same operands — always `DivRem::div_rem`.
5. **Blanket `#[inline]`** on large/recursive functions (no effect, or code bloat); decide from benches.
   Remember `#[inline(always)]` is refused on functions with own impl generics — put bounds on the impl.
6. **Sign-magnitude `{mag, sign: bool}` fixed point à la cubit** as the hot-path scalar (branches in add,
   u256 division in mul) and **branch-pinned git forks** as dependencies.
7. **Tests that measure nothing**: literal inputs get constant-folded (300-gas tests); asserts and
   harness dominate small ops.
8. Stale workspace members/scripts, crates that duplicate helpers because they cannot depend on each other,
   `Felt252Dict<bool>` where a bitmap works, O(n^2) `Array` scans (`hex.cairo`).

### 6.3 What nalgebra.cairo must add that origami lacks (gas methodology)

- A `bench`/test convention where every primitive has **N named implementations** (`*_math`, `*_bitwise`,
  `*_loop`, `*_lut`) behind one trait, one shared golden test-vector set, and one gas test each.
- Inputs routed through an `#[inline(never)] fn id<T>(x: T) -> T` (or read from an array) to defeat
  constant folding; a **baseline test** per signature so net cost = measured - baseline (as in 3.3).
- Gas captured either with `core::testing::get_available_gas()` deltas inside the test (asserting
  `used <= BUDGET` so regressions fail CI) or by parsing `gas usage est.` into a committed snapshot
  (`gas-snapshot.txt`) diffed in CI; snforge's `--detailed-resources` (steps / range_check / bitwise
  counts) is worth adding because step count and builtin mix — not just Sierra gas — drive proving cost.
- Record the Cairo/Scarb version next to every number: the ordering bitwise-vs-math depends on the
  cost table and has already flipped once.
