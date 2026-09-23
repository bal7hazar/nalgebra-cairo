# nalgebra.cairo

Linear algebra for **provable game physics**: a Cairo port of the Rust
[nalgebra](https://nalgebra.rs) crate on Q32.32 fixed point, designed gas-first.

Part of a stack porting reputable Rust crates to Cairo — with
[glam.cairo](https://github.com/bal7hazar/glam.cairo) and
[rapier.cairo](https://github.com/bal7hazar/rapier.cairo) — towards games whose whole physics
is provable.

> Status: M1-M4 complete (scalar, static 2/3/4/6 types, geometry, small decompositions) with
> ~3,700 tests; see the [plan](docs/PLAN.md) for what remains.

## Packages

| Package | Content |
|---|---|
| [`simba`](crates/simba) | Scalar traits (`Real`, `Transcendental`: fused kernels, wide accumulator, transcendentals) implemented for glam.cairo's Q32.32 [`fixed::Fixed`](https://github.com/bal7hazar/glam.cairo) 0.2.0, the scalar shared by the whole stack |
| [`nalgebra`](crates/nalgebra) | `base` (vectors, matrices), `geometry` (rotations, isometries), `linalg` (decompositions), generic over `Real` |

## Why it is fast

Measured on Cairo 2.19.4 ([full synthesis](docs/BENCHMARK.md)):

| | existing Cairo libraries | nalgebra.cairo kernels |
|---|---:|---:|
| fixed-point add | 4,050 gas | 640 |
| dot3 | 15,450 - 37,820 | 2,250 |
| 3x3 inverse | 1,076,420 | < 98,490 |
| sin | 30,320 (2e-6) - 130,360 | 12,500 (5e-10) |

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
