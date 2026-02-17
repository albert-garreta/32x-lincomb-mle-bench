# mle-bench

Optimised Rust implementation of **multilinear extension (MLE) evaluation** and **linear combination** over a 192-bit prime field, with [Criterion](https://github.com/bheisler/criterion.rs) micro-benchmarks.

## Motivation

Multilinear extensions and linear combinations over finite fields are core building blocks in modern cryptographic proof systems (SNARKs, sumcheck protocols, etc.). In a typical sumcheck-based protocol a verifier must evaluate the MLE of a witness vector at a random challenge point, and often first combines several witness columns via a random linear combination. These operations sit on the critical path and are executed many times during proof generation, so their throughput directly impacts prover performance.

This crate isolates those two hot-path primitives, implements them with careful attention to cache behaviour and parallelism granularity, and provides a reproducible Criterion benchmark suite to measure their real-world throughput on a realistic workload size.

## Mathematical Background

### Multilinear Extension (MLE)

Given a vector $v$ of $2^k$ field elements, its **multilinear extension** is the unique $k$-variate polynomial over $\mathbb{F}$ that is multilinear (degree at most 1 in each variable) and agrees with $v$ on the Boolean hypercube $\{0,1\}^k$:

$$\widetilde{v}(x_0, \dots, x_{k-1}) = \sum_{b \in \{0,1\}^k} v[b] \prod_{j=0}^{k-1} \left( b_j \cdot x_j + (1 - b_j)(1 - x_j) \right)$$

A naïve evaluation of this formula costs $O(2^k \cdot k)$ multiplications. The **iterative folding** algorithm used here reduces this to $O(2^k)$ total work by processing one variable at a time:

1. Start with a buffer $\text{buf} = v$ of length $2^k$.
2. For each coordinate $x_j$ ($j = 0, \dots, k-1$), fold pairs:
   $$\text{buf}[i] \leftarrow \text{buf}[2i] \cdot (1 - x_j) + \text{buf}[2i+1] \cdot x_j \qquad (i = 0, \dots, 2^{k-j-1}-1)$$
3. After $k$ rounds the buffer contains a single element: $\widetilde{v}(x_0, \dots, x_{k-1})$.

Each fold halves the working set, so total work is $2^k + 2^{k-1} + \cdots + 1 = 2^{k+1} - 1$ multiply-adds.

### Linear Combination

Given $m$ vectors $v_0, \dots, v_{m-1}$ each of length $n$ and scalar coefficients $a_0, \dots, a_{m-1}$, compute:

$$\text{result}[j] = \sum_{i=0}^{m-1} a_i \cdot v_i[j] \qquad (j = 0, \dots, n-1)$$

This is equivalent to a matrix-vector product where the rows are the $v_i$ vectors and the column vector contains the coefficients $a_i$. The total cost is $m \cdot n$ multiply-adds.

### Composed Operation

In sumcheck-style protocols the two steps are chained: first compute the linear combination of $m$ witness columns, then evaluate the resulting vector's MLE at a random point. The composed cost is $m \cdot n + 2n - 1$ multiply-adds (where $n = 2^k$).

## Overview

This library provides core operations on vectors of field elements (using the [arkworks](https://github.com/arkworks-rs) finite-field stack), in both **single-instance** and **batch-parallel** variants.

### Single-instance operations

| Function | Description |
|---|---|
| `linear_combination` | Computes $\text{result}[j] = \sum_i a_i \cdot v_i[j]$ for $m$ vectors of length $n$. The output vector is split into chunks of 4 096 elements; each chunk is processed by one Rayon task that iterates over all $m$ vectors, keeping the result chunk in L1/L2 cache across the $m$ passes. This cache-friendly column-major access pattern avoids the streaming-write bottleneck that would occur with a row-major traversal. |
| `linear_combination_seq` | Same computation, fully sequential — optimal when $n$ is small (e.g. $n \leq 2^{10}$) and Rayon's work-stealing overhead would dominate the actual arithmetic. |
| `mle_eval` | Evaluates the multilinear extension of a vector $v$ of length $2^k$ at a point $(x_0, \dots, x_{k-1})$ using iterative folding. When the fold size exceeds a threshold of 2 048 elements, the fold is parallelised via `par_iter`; smaller tails are folded sequentially in-place to avoid allocation overhead. |
| `mle_eval_seq` | Fully sequential MLE evaluation — always folds in-place, avoiding allocation of intermediate buffers. |
| `linear_combination_then_mle_eval` | Composes linear combination + MLE evaluation (parallel inner ops). |
| `lincomb_then_mle_eval_seq` | Same composition, fully sequential. |

### Batch-parallel operations

For workloads with many independent computations of moderate size (e.g. 32 instances at $m = 17$, $n = 2^{8}$), parallelising **across** instances is far more efficient than parallelising within each one. Each individual computation's working set (~17 × 256 × 24 bytes ≈ 104 KB) fits comfortably in L2 cache, so the sequential per-instance code achieves near-peak arithmetic throughput. Adding intra-instance parallelism at this size would only introduce synchronisation overhead. Instead, the batch API distributes the 32 independent instances across Rayon's thread pool, achieving near-linear scaling with core count:

| Function | Description |
|---|---|
| `batch_linear_combination` | Parallel batch of independent linear combinations. Each instance calls `linear_combination_seq` internally — Rayon distributes instances across threads. |
| `batch_mle_eval` | Parallel batch of independent MLE evaluations. Each instance calls `mle_eval_seq` internally. |
| `batch_lincomb_then_mle_eval` | Parallel batch of independent (lincomb + MLE eval). Each instance calls `lincomb_then_mle_eval_seq` internally. |
| `batch_lincomb_then_mle_eval_seq` | Sequential batch baseline — runs all instances on one thread for comparison. |

### Field

Arithmetic is performed over a **192-bit prime field** defined by the NIST P-192 prime:

$$p = 2^{192} - 2^{64} - 1 = 6\,277\,101\,735\,386\,680\,763\,835\,789\,423\,207\,666\,416\,083\,908\,700\,390\,324\,961\,279$$

The field is instantiated via `ark-ff`'s Montgomery backend (`Fp192<MontBackend<FqConfig, 3>>`). Each element occupies 3 × 64-bit limbs (24 bytes). Montgomery form enables modular multiplication without expensive division. The generator is $g = 11$.

## Getting Started

### Prerequisites

- **Rust** (edition 2021+) — install via [rustup](https://rustup.rs)

### Build

```bash
cargo build --release
```

The release profile uses aggressive optimisation settings: `opt-level = 3`, `lto = "fat"` (full link-time optimisation across all crates), and `codegen-units = 1` (maximises intra-crate optimisation). These are critical for benchmarking — without LTO the arkworks Montgomery multiplication is not fully inlined.

### Run Tests

```bash
cargo test
```

Tests cover:

- **MLE evaluation correctness** — the iterative folding result is compared against a brute-force $O(2^k \cdot k)$ reference implementation (`mle_eval_naive`) that explicitly sums over all $2^k$ basis monomials.
- **Linear combination correctness** — every position in the output vector is checked against a manually computed element-wise dot product.
- **Composition consistency** — verifies that `linear_combination_then_mle_eval` returns the same scalar as computing the linear combination and MLE evaluation as two separate steps.

### Run Benchmarks

```bash
cargo bench
```

Benchmarks use [Criterion.rs](https://github.com/bheisler/criterion.rs) with HTML reports (generated in `target/criterion/`).

## Benchmark Design

### Workload parameters

The benchmark suite is parameterised by constants defined in the benchmark harness ([benches/mle_bench.rs](benches/mle_bench.rs)):

| Parameter | Value | Meaning |
|---|---|---|
| `NUM_COMPUTATIONS` | 32 | Number of independent instances in batch benchmarks |
| `M` | 17 | Number of vectors (witness columns) per linear combination |
| `LOG_N` | 8 | Logarithmic vector length |
| `N` | $2^8 = 256$ | Vector length (number of field elements per vector) |

Each benchmark function generates random field elements as inputs using `ark_std::test_rng()` (a deterministic PRNG seeded for reproducibility). Inputs are allocated once outside the timed loop; only the computation itself is measured.

### What each benchmark measures

Seven benchmark groups are defined, progressing from single-instance baselines to full batch parallelism:

| Group | What it measures | Throughput unit |
|---|---|---|
| `single_linear_combination` | Time to compute one sequential linear combination of $M$ vectors of length $N$. Establishes the per-instance lincomb cost. | $M \times N$ elements |
| `single_mle_eval` | Time to evaluate one sequential MLE of a length-$N$ vector. Establishes the per-instance MLE cost. | $N$ elements |
| `single_lincomb_then_mle` | Time for one sequential (lincomb + MLE). This is the end-to-end per-instance cost that would appear in a real prover. | $M \times N$ elements |
| `batch_lincomb_mle_seq` | Time for 32 sequential (lincomb + MLE) — the baseline with **no parallelism**. Expected to be roughly $32\times$ the single cost. | $32 \times M \times N$ elements |
| `batch_lincomb_mle_par` | Time for 32 (lincomb + MLE) with **Rayon parallelism across instances**. This is the primary benchmark — it shows how much speedup batch-level parallelism achieves over the sequential baseline. | $32 \times M \times N$ elements |
| `batch_linear_combination_par` | Time for 32 parallel linear combinations only (no MLE). Isolates the lincomb portion of the parallel batch to identify bottlenecks. | $32 \times M \times N$ elements |
| `batch_mle_eval_par` | Time for 32 parallel MLE evaluations only (no lincomb). Isolates the MLE portion. | $32 \times N$ elements |

### Throughput metric

Criterion reports **elements per second** — specifically, the total number of field elements touched by the computation divided by wall-clock time. For a linear combination of $M$ vectors of length $N$, throughput is $M \times N / t$. This metric lets you compare across different $(M, N)$ configurations and estimate how the primitives would perform at scale.

### Interpretation guide

- **Single vs. batch sequential:** the batch-sequential time should be ≈ 32× the single time. If it is noticeably more, there may be cache pressure from having all 32 input sets allocated simultaneously.
- **Batch parallel vs. batch sequential:** the speedup here reflects how well Rayon distributes work. Ideal speedup equals the number of physical cores; in practice expect 4–8× on a modern laptop and 16–32× on a server.
- **Lincomb-only vs. MLE-only:** comparing these reveals which primitive dominates the composed cost, guiding where to focus future optimisation.

#### Sample results

| Benchmark | Time | Throughput |
|---|---|---|
| Single lincomb + MLE | ~233 µs | ~176 Melem/s |
| Batch 32× sequential | ~7.5 ms | ~175 Melem/s |
| **Batch 32× parallel** | **~1.5 ms** | **~855 Melem/s** |

The parallel batch achieves a **~4.9× speedup** over sequential on 32 instances.

## Project Structure

```
├── Cargo.toml           # Crate manifest and build profiles (LTO + codegen-units=1)
├── src/
│   └── lib.rs           # Core library: field definition, algorithms, batch ops, tests
├── benches/
│   └── mle_bench.rs     # Criterion benchmarks (32× m=17, n=2^8)
└── check_prime.py       # Helper script to verify the NIST P-192 prime
```

## Performance Notes

- **Release / bench profiles** are configured with `opt-level = 3`, `lto = "fat"`, and `codegen-units = 1` for maximum throughput. Without LTO, arkworks Montgomery multiplication routines are not inlined across crate boundaries, which can cost 20–30% throughput.
- The `linear_combination` (parallel variant) uses a chunk size of 4 096 elements (`CHUNK_SIZE` constant) to keep the working set in L1/L2 cache across the $m$ passes. For small $n$ (e.g. $2^{8}$–$2^{10}$), the sequential variant avoids Rayon overhead entirely.
- MLE folding switches from parallel (`par_iter`) to sequential in-place when the remaining buffer drops below 2 048 elements (`PAR_FOLD_THRESHOLD` constant). The parallel path allocates a fresh output vector (to avoid aliasing issues); the sequential path folds in-place since `buf[i]` reads from `buf[2i]` and `buf[2i+1]`, which are always ahead of the write position.
- **Batch-level parallelism** is the recommended strategy for many small-to-moderate computations — Rayon distributes independent instances across cores with near-zero coordination overhead, and each instance's working set stays cache-resident.

## Dependencies

| Crate | Purpose |
|---|---|
| `ark-ff` | Finite field arithmetic (Montgomery form, 192-bit prime) |
| `ark-std` | Deterministic test RNG (`test_rng()`) |
| `ark-serialize` | Serialisation traits (required by ark-ff) |
| `rayon` | Data-parallel iterators (work-stealing thread pool) |
| `rand` | Random number generation |
| `criterion` | Benchmarking framework with statistical analysis (dev-only) |

## License

See the repository for license details.
