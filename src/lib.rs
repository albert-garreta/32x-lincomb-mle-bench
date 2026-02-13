use ark_ff::{AdditiveGroup, Field, Fp192, MontBackend, MontConfig, UniformRand};
use rayon::prelude::*;

// ---------------------------------------------------------------------------
// Field definition: 192-bit NIST P-192 prime
// p = 2^192 - 2^64 - 1
//   = 6277101735386680763835789423207666416083908700390324961279
// ---------------------------------------------------------------------------
#[derive(MontConfig)]
#[modulus = "6277101735386680763835789423207666416083908700390324961279"]
#[generator = "11"]
pub struct FqConfig;

pub type Fq = Fp192<MontBackend<FqConfig, 3>>;

// ---------------------------------------------------------------------------
// Linear combination: result[j] = Σ_i  a_i · v_i[j]
//
// Strategy: split output positions into cache-friendly chunks.  Each chunk is
// handled by one rayon task which iterates over all m vectors, keeping the
// result chunk in L1/L2 cache across the m passes.
// ---------------------------------------------------------------------------
const CHUNK_SIZE: usize = 4096;

pub fn linear_combination(vectors: &[Vec<Fq>], coeffs: &[Fq]) -> Vec<Fq> {
    assert!(!vectors.is_empty());
    let n = vectors[0].len();
    assert!(
        vectors.iter().all(|v| v.len() == n),
        "all vectors must have the same length"
    );
    assert_eq!(
        vectors.len(),
        coeffs.len(),
        "number of vectors must match number of coefficients"
    );

    let m = vectors.len();

    // Pre-borrow slices to avoid per-chunk index checks
    let vecs: Vec<&[Fq]> = vectors.iter().map(|v| v.as_slice()).collect();

    let mut result = vec![Fq::ZERO; n];

    result
        .par_chunks_mut(CHUNK_SIZE)
        .enumerate()
        .for_each(|(chunk_idx, chunk)| {
            let base = chunk_idx * CHUNK_SIZE;
            let len = chunk.len();
            for i in 0..m {
                let a = coeffs[i];
                let v = &vecs[i][base..base + len];
                for j in 0..len {
                    chunk[j] += a * v[j];
                }
            }
        });

    result
}

// ---------------------------------------------------------------------------
// Multilinear extension evaluation via folding.
//
//   MLE(v)(x_0, ..., x_{k-1}) where |v| = 2^k
//
// For each coordinate x_j we fold:
//     v'[i] = v[2i]·(1 - x_j) + v[2i+1]·x_j
//
// Large folds are parallelised; small tails run sequentially.
// ---------------------------------------------------------------------------
const PAR_FOLD_THRESHOLD: usize = 2048;

pub fn mle_eval(evals: &[Fq], point: &[Fq]) -> Fq {
    let k = point.len();
    assert_eq!(evals.len(), 1 << k, "evals length must be 2^k");

    let mut buf = evals.to_vec();
    let mut len = buf.len();

    for j in 0..k {
        let half = len >> 1;
        let x = point[j];
        let one_minus_x = Fq::ONE - x;

        if half >= PAR_FOLD_THRESHOLD {
            // Parallel fold – produce a new vec to avoid aliasing issues
            let src = &buf[..len];
            let new: Vec<Fq> = (0..half)
                .into_par_iter()
                .map(|i| src[2 * i] * one_minus_x + src[2 * i + 1] * x)
                .collect();
            buf = new;
        } else {
            // Sequential in-place fold (safe because we read 2i before writing i)
            for i in 0..half {
                buf[i] = buf[2 * i] * one_minus_x + buf[2 * i + 1] * x;
            }
            buf.truncate(half);
        }
        len = half;
    }

    buf[0]
}

// ---------------------------------------------------------------------------
// Combined: linear combination followed by MLE evaluation
// ---------------------------------------------------------------------------
pub fn linear_combination_then_mle_eval(
    vectors: &[Vec<Fq>],
    coeffs: &[Fq],
    point: &[Fq],
) -> Fq {
    let lc = linear_combination(vectors, coeffs);
    mle_eval(&lc, point)
}

// ---------------------------------------------------------------------------
// Sequential variants – no rayon overhead, ideal for small n
// ---------------------------------------------------------------------------

/// Sequential linear combination (no parallelism).
pub fn linear_combination_seq(vectors: &[Vec<Fq>], coeffs: &[Fq]) -> Vec<Fq> {
    assert!(!vectors.is_empty());
    let n = vectors[0].len();
    assert!(
        vectors.iter().all(|v| v.len() == n),
        "all vectors must have the same length"
    );
    assert_eq!(
        vectors.len(),
        coeffs.len(),
        "number of vectors must match number of coefficients"
    );

    let m = vectors.len();
    let vecs: Vec<&[Fq]> = vectors.iter().map(|v| v.as_slice()).collect();

    let mut result = vec![Fq::ZERO; n];
    for i in 0..m {
        let a = coeffs[i];
        let v = vecs[i];
        for j in 0..n {
            result[j] += a * v[j];
        }
    }
    result
}

/// Sequential MLE evaluation (no parallelism).
pub fn mle_eval_seq(evals: &[Fq], point: &[Fq]) -> Fq {
    let k = point.len();
    assert_eq!(evals.len(), 1 << k, "evals length must be 2^k");

    let mut buf = evals.to_vec();
    let mut len = buf.len();

    for j in 0..k {
        let half = len >> 1;
        let x = point[j];
        let one_minus_x = Fq::ONE - x;
        for i in 0..half {
            buf[i] = buf[2 * i] * one_minus_x + buf[2 * i + 1] * x;
        }
        buf.truncate(half);
        len = half;
    }
    buf[0]
}

/// Sequential combined: linear combination followed by MLE evaluation.
pub fn lincomb_then_mle_eval_seq(
    vectors: &[Vec<Fq>],
    coeffs: &[Fq],
    point: &[Fq],
) -> Fq {
    let lc = linear_combination_seq(vectors, coeffs);
    mle_eval_seq(&lc, point)
}

// ---------------------------------------------------------------------------
// Batch operations – parallelise across independent computations
//
// For small per-computation sizes (e.g. n = 2^10, m = 40) the inner work
// fits comfortably in L1/L2 cache.  Parallelising at the batch level is
// far more efficient than trying to parallelise within each computation.
// ---------------------------------------------------------------------------

/// Run many independent linear combinations in parallel.
pub fn batch_linear_combination(
    all_vectors: &[Vec<Vec<Fq>>],
    all_coeffs: &[Vec<Fq>],
) -> Vec<Vec<Fq>> {
    assert_eq!(all_vectors.len(), all_coeffs.len());
    all_vectors
        .par_iter()
        .zip(all_coeffs.par_iter())
        .map(|(vectors, coeffs)| linear_combination_seq(vectors, coeffs))
        .collect()
}

/// Run many independent MLE evaluations in parallel.
pub fn batch_mle_eval(
    all_evals: &[Vec<Fq>],
    all_points: &[Vec<Fq>],
) -> Vec<Fq> {
    assert_eq!(all_evals.len(), all_points.len());
    all_evals
        .par_iter()
        .zip(all_points.par_iter())
        .map(|(evals, point)| mle_eval_seq(evals, point))
        .collect()
}

/// Run many independent (linear-combination + MLE eval) in parallel.
pub fn batch_lincomb_then_mle_eval(
    all_vectors: &[Vec<Vec<Fq>>],
    all_coeffs: &[Vec<Fq>],
    all_points: &[Vec<Fq>],
) -> Vec<Fq> {
    assert_eq!(all_vectors.len(), all_coeffs.len());
    assert_eq!(all_vectors.len(), all_points.len());
    all_vectors
        .par_iter()
        .zip(all_coeffs.par_iter())
        .zip(all_points.par_iter())
        .map(|((vectors, coeffs), point)| {
            lincomb_then_mle_eval_seq(vectors, coeffs, point)
        })
        .collect()
}

/// Run many independent (linear-combination + MLE eval) sequentially (baseline).
pub fn batch_lincomb_then_mle_eval_seq(
    all_vectors: &[Vec<Vec<Fq>>],
    all_coeffs: &[Vec<Fq>],
    all_points: &[Vec<Fq>],
) -> Vec<Fq> {
    assert_eq!(all_vectors.len(), all_coeffs.len());
    assert_eq!(all_vectors.len(), all_points.len());
    all_vectors
        .iter()
        .zip(all_coeffs.iter())
        .zip(all_points.iter())
        .map(|((vectors, coeffs), point)| {
            lincomb_then_mle_eval_seq(vectors, coeffs, point)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Test helpers / random data generation
// ---------------------------------------------------------------------------
pub fn random_field_vec(n: usize) -> Vec<Fq> {
    let mut rng = ark_std::test_rng();
    (0..n).map(|_| Fq::rand(&mut rng)).collect()
}

pub fn random_vectors(m: usize, n: usize) -> Vec<Vec<Fq>> {
    (0..m).map(|_| random_field_vec(n)).collect()
}

/// Generate random inputs for a batch of `count` computations.
pub fn random_batch_inputs(
    count: usize,
    m: usize,
    n: usize,
    log_n: usize,
) -> (Vec<Vec<Vec<Fq>>>, Vec<Vec<Fq>>, Vec<Vec<Fq>>) {
    let all_vectors: Vec<_> = (0..count).map(|_| random_vectors(m, n)).collect();
    let all_coeffs: Vec<_> = (0..count).map(|_| random_field_vec(m)).collect();
    let all_points: Vec<_> = (0..count).map(|_| random_field_vec(log_n)).collect();
    (all_vectors, all_coeffs, all_points)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Brute-force reference MLE evaluation for correctness checking.
    fn mle_eval_naive(evals: &[Fq], point: &[Fq]) -> Fq {
        use ark_ff::AdditiveGroup;
        let k = point.len();
        assert_eq!(evals.len(), 1 << k);
        let mut sum = Fq::ZERO;
        for i in 0..(1usize << k) {
            let mut basis = Fq::ONE;
            for j in 0..k {
                if (i >> j) & 1 == 1 {
                    basis *= point[j];
                } else {
                    basis *= Fq::ONE - point[j];
                }
            }
            sum += evals[i] * basis;
        }
        sum
    }

    #[test]
    fn test_mle_eval_small() {
        let k = 4;
        let n = 1 << k;
        let evals = random_field_vec(n);
        let point = random_field_vec(k);

        let fast = mle_eval(&evals, &point);
        let naive = mle_eval_naive(&evals, &point);
        assert_eq!(fast, naive);
    }

    #[test]
    fn test_linear_combination() {
        let m = 5;
        let n = 64;
        let vecs = random_vectors(m, n);
        let coeffs = random_field_vec(m);

        let result = linear_combination(&vecs, &coeffs);

        // Check a few positions manually
        for j in 0..n {
            let mut expected = <Fq as AdditiveGroup>::ZERO;
            for i in 0..m {
                expected += coeffs[i] * vecs[i][j];
            }
            assert_eq!(result[j], expected, "mismatch at position {j}");
        }
    }

    #[test]
    fn test_combined() {
        let k = 6;
        let n = 1usize << k;
        let m = 4;
        let vecs = random_vectors(m, n);
        let coeffs = random_field_vec(m);
        let point = random_field_vec(k);

        let combined = linear_combination_then_mle_eval(&vecs, &coeffs, &point);

        let lc = linear_combination(&vecs, &coeffs);
        let separate = mle_eval(&lc, &point);

        assert_eq!(combined, separate);
    }
}
