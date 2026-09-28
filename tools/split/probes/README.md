# Coherence probes (WP 9-NS1)

A throwaway workspace of six tiny packages, kept to reproduce the Cairo 2.19.4 / Scarb 2.19.4
facts of `docs/SPLIT.md` §1. Not a workspace member of nalgebra-cairo. Copy it to a temporary
directory and build there (`scarb build -p pd`, `scarb build -p pe`); the build ERRORS are the
result.

| package | role |
|---|---|
| `pa` | struct `A` (module `shapes`, generated trait `ATrait`), traits `Tr<T>` and `MulM<L, R>` (module `ops`), feature `extra` |
| `pb` | struct `B` (module `bmod`); `MulM<A, B>` and `Sub<A>` implemented in `B`'s module |
| `pc` | a third crate: `Tr<B>`, `Add<A>`, `MulM<B, A>`, a `#[generate_trait]` on `B`, a macro `mk!` through `$defsite::` |
| `pd` | consumer of `pa`, `pb`, `pc` (the lookup rules) |
| `pf` | facade: glob re-exports of `pa`, `pb`, `pc`, an explicit re-export shadowing a glob, its own macro, feature forwarding `extra = ["pa/extra"]` |
| `pe` | consumer of the facade only |

Expected `pd` errors (each probe module is named after what it tests; `lookup_rules.cairo` is an
earlier variant kept for reference): `m2` (a core-trait impl in a third crate is not found), `m3`
(same, `Add`), `m4` / `m6` (a trait impl for foreign types in a third crate, not imported),
`m7` (`Into<A, B>` in a third crate), `q1` (a core-trait impl in the type's crate but NOT in the
type's module). Everything else compiles: `m1` (`MulM<A, B>` found in `B`'s module, another
crate than the trait's and `A`'s), the imported variants `m2b`, `m3b`, `m3c`, `m4b`, `m6b`,
`m7b`, `m5` (a generated trait on a foreign type), `m8` (a macro of another crate, by path or
imported), `q2` (`Into<A, B>` in `B`'s module), `q3`.

Expected `pe` errors: `e2` (the impl of a third crate is not reachable by importing the type
only), `e7` (a transitive dependency cannot be named), `e9` (a method needs its trait in scope,
as today). Everything else compiles: the facade's globs, module paths through a glob (`e4`),
re-exported macros (`e5`, `e6`), method resolution through a re-exported trait (`e8`), an
explicit re-export winning over two ambiguous globs (`e10`), a forwarded feature (`e11`, and with
`default-features = false` on the facade it disappears unless another crate of the unit enables
it: features unify per compilation unit).
