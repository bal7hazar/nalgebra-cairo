# nalgebra-cairo

Linear algebra for **provable game physics**: a Cairo port of the Rust
[nalgebra](https://nalgebra.rs) crate on fixed-cairo's Q32.32 fixed point, designed gas-first.

Part of a stack porting reputable Rust crates to Cairo, repository by repository like the Rust
ecosystem: [fixed-cairo](https://github.com/bal7hazar/fixed-cairo) (the Q32.32 scalar, the `f64` of
the stack), [simba-cairo](https://github.com/bal7hazar/simba-cairo) (the scalar traits),
nalgebra-cairo (this repository), [glam-cairo](https://github.com/bal7hazar/glam-cairo),
[glamx-cairo](https://github.com/bal7hazar/glamx-cairo) and
[rapier-cairo](https://github.com/bal7hazar/rapier-cairo), towards games whose whole physics is
provable.

> Status: coverage of nalgebra-rs 0.35.0 at 64 % ([API_PARITY.md](docs/API_PARITY.md)): the 36
> static shapes and 54 aliases, geometry (rotations, quaternions, dual quaternions, isometries,
> similarities, scale, reflection) and small decompositions; ~5,000 tests. Release 0.1.0 when the
> coverage is complete (see the [plan](docs/PLAN.md)).

## Packages

| Package | Content |
|---|---|
| [`nalgebra`](crates/nalgebra) | `base` (vectors, matrices), `geometry` (rotations, isometries), `linalg` (decompositions), generic over `simba::scalar::Real` |

Its scalar layer is the registry package `simba = "0.2.0"`
([simba-cairo](https://github.com/bal7hazar/simba-cairo): `Real` / `Transcendental` implemented for
fixed-cairo's Q32.32 [`fixed::Fixed`](https://github.com/bal7hazar/fixed-cairo) 0.4.0, the scalar
shared by the whole stack), like nalgebra-rs depends on simba-rs.

## Why it is fast

Measured on Cairo 2.19.4 ([full synthesis](docs/BENCHMARK.md)):

| | existing Cairo libraries | nalgebra-cairo kernels |
|---|---:|---:|
| fixed-point add | 4,050 gas | 640 |
| dot3 | 15,450 - 37,820 | 1,980 |
| 3x3 inverse | 1,076,420 | 88,360 |
| sin_cos (pair) | 30,320 - 130,360 (`sin` alone) | 31,500 (`fixed::trig`) |

No loops, no dynamic containers in static types, products accumulated unscaled and rescaled once,
range checks deferred to a single downcast per kernel. Every function ships with gas benchmarks,
and CI fails on any unreviewed gas change.

## Documentation

- [Benchmark synthesis](docs/BENCHMARK.md) — what was measured and learned
- [Design decisions](docs/DESIGN.md)
- [Orchestration strategy](docs/ORCHESTRATOR.md) — how work is split between sub-agents
- [Execution plan](docs/PLAN.md)
- [Research reports](docs/research) — nalgebra, alexandria, starknet-agentic, origami, cubit, orion
- [Design-time benchmarks](benchmarks) — ~2,100 reproducible gas measurements

## Development

```bash
asdf install            # scarb 2.19.4, starknet-foundry 0.61.0
./scripts/check.sh      # fmt, lint, build, tests, gas snapshot
```

See [AGENTS.md](AGENTS.md) for the contribution rules (they apply to humans too).

## License

MIT
