# mle-bench

Optimised Rust implementation of **multilinear extension (MLE) evaluation** and **linear combination** over a 192-bit prime field, with [Criterion](https://github.com/bheisler/criterion.rs) micro-benchmarks.

## Overview

This library provides core operations on vectors of field elements (using the [arkworks](https://github.com/arkworks-rs) finite-field stack), in both **single-instance** and **batch-parallel** variants.

### Single-instance operations

| Function | Description |
|---|---|
| `linear_combination` | Computes $\text{result}[j] = \sum_i a_i \cdot v_i[j]$ for $m$ vectors of length $n$. Cache-friendly chunked parallelism via [Rayon](https://github.com/rayon-rs/rayon) (suited for large $n$). |
| `linear_combination_seq` | Same computation, fully sequential — optimal when $n$ is small and rayon overhead would dominate. |
| `mle_eval` | Evaluates the multilinear extension of a vector $v$ of length $2^k$ at a point $(x_0, \dots, x_{k-1})$ using iterative folding. Large folds are parallelised. |
| `mle_eval_seq` | Fully sequential MLE evaluation. |
| `linear_combination_then_mle_eval` | Composes linear combination + MLE evaluation (parallel inner ops). |
| `lincomb_then_mle_eval_seq` | Same composition, fully sequential. |

### Batch-parallel operations

For workloads with many independent computations of moderate size (e.g. 32 instances at $m = 40$, $n = 2^{10}$), parallelising **across** instances is far more efficient than parallelising within each one. The batch API runs each instance sequentially but distributes them across threads via Rayon:

| Function | Description |
|---|---|
| `batch_linear_combination` | Parallel batch of independent linear combinations. |
| `batch_mle_eval` | Parallel batch of independent MLE evaluations. |
| `batch_lincomb_then_mle_eval` | Parallel batch of independent (lincomb + MLE eval). |
| `batch_lincomb_then_mle_eval_seq` | Sequential batch baseline for comparison. |

### Field

Arithmetic is performed over a **192-bit prime field** defined by the NIST P-192 prime:

$$p = 2^{192} - 2^{64} - 1$$

The field is instantiated via `ark-ff`'s Montgomery backend (`Fp192<MontBackend<…, 3>>`).

## Getting Started

### Prerequisites

- **Rust** (edition 2021+) — install via [rustup](https://rustup.rs)

### Build

```bash
cargo build --release
```

### Run Tests

```bash
cargo test
```

Tests cover:

- **MLE evaluation correctness** — checked against a brute-force $O(2^k \cdot k)$ reference implementation.
- **Linear combination correctness** — element-wise manual summation check.
- **Composition consistency** — verifies that the combined function matches running the two steps separately.

### Run Benchmarks

```bash
cargo bench
```

Benchmarks use [Criterion.rs](https://github.com/bheisler/criterion.rs) with HTML reports (generated in `target/criterion/`). The benchmark suite targets **32 independent computations** with $m = 40$ vectors of length $n = 2^{10}$ (1 024 elements each).

Seven benchmark groups are defined:

| Group | What it measures |
|---|---|
| `single_linear_combination` | Single sequential linear combination ($m=40$, $n=2^{10}$) |
| `single_mle_eval` | Single sequential MLE evaluation ($n=2^{10}$) |
| `single_lincomb_then_mle` | Single sequential lincomb + MLE ($m=40$, $n=2^{10}$) |
| `batch_lincomb_mle_seq` | 32× sequential batch (no parallelism baseline) |
| `batch_lincomb_mle_par` | 32× parallel batch (parallelism across instances) |
| `batch_linear_combination_par` | 32× parallel linear combinations only |
| `batch_mle_eval_par` | 32× parallel MLE evaluations only |

#### Sample results

| Benchmark | Time | Throughput |
|---|---|---|
| Single lincomb + MLE | ~233 µs | ~176 Melem/s |
| Batch 32× sequential | ~7.5 ms | ~175 Melem/s |
| **Batch 32× parallel** | **~1.5 ms** | **~855 Melem/s** |

The parallel batch achieves a **~4.9× speedup** over sequential on 32 instances.

## Project Structure

```
├── Cargo.toml           # Crate manifest and build profiles
├── src/
│   └── lib.rs           # Core library (field def, algorithms, batch ops, tests)
├── benches/
│   └── mle_bench.rs     # Criterion benchmarks (32× m=40, n=2^10)
└── check_prime.py       # Helper script to verify the prime and find a generator
```

## Performance Notes

- **Release / bench profiles** are configured with `opt-level = 3`, `lto = "fat"`, and `codegen-units = 1` for maximum throughput.
- The `linear_combination` (parallel variant) uses a chunk size of 4 096 elements to keep the working set in L1/L2 cache across the $m$ passes. For small $n$ (e.g. $2^{10}$), the sequential variant avoids rayon overhead entirely.
- MLE folding switches from parallel to sequential when the remaining buffer drops below 2 048 elements.
- **Batch-level parallelism** is the recommended strategy for many small-to-moderate computations — Rayon distributes independent instances across cores with near-zero coordination overhead.

## Dependencies

| Crate | Purpose |
|---|---|
| `ark-ff` | Finite field arithmetic (Montgomery form) |
| `ark-std` | Deterministic test RNG |
| `ark-serialize` | Serialisation traits (required by ark-ff) |
| `rayon` | Data-parallel iterators |
| `rand` | Random number generation |
| `criterion` | Benchmarking framework (dev-only) |

## License

See the repository for license details.
