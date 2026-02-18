use criterion::{
    black_box, criterion_group, criterion_main, Criterion, Throughput,
};
use mle_bench::*;

// ---------------------------------------------------------------------------
// Target parameters: 32 independent computations, each m=40, n=2^10
// ---------------------------------------------------------------------------
const M: usize = 272;
const LOG_N: usize = 10;
const N: usize = 1 << LOG_N;
const NUM_COMPUTATIONS: usize = 32;

// ---------------------------------------------------------------------------
// Benchmark: single sequential computation (baseline per-instance cost)
// ---------------------------------------------------------------------------
fn bench_single_lincomb_mle(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_lincomb_then_mle");
    group.sample_size(100);

    let vectors = random_vectors(M, N);
    let coeffs = random_field_vec(M);
    let point = random_field_vec(LOG_N);

    group.throughput(Throughput::Elements((M * N) as u64));
    group.bench_function(&format!("m={M},n=2^{LOG_N}"), |b| {
        b.iter(|| {
            black_box(lincomb_then_mle_eval_seq(&vectors, &coeffs, &point));
        })
    });
    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: single sequential linear-combination
// ---------------------------------------------------------------------------
fn bench_single_lincomb(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_linear_combination");
    group.sample_size(100);

    let vectors = random_vectors(M, N);
    let coeffs = random_field_vec(M);

    group.throughput(Throughput::Elements((M * N) as u64));
    group.bench_function(&format!("m={M},n=2^{LOG_N}"), |b| {
        b.iter(|| {
            black_box(linear_combination_seq(&vectors, &coeffs));
        })
    });
    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: single sequential MLE eval
// ---------------------------------------------------------------------------
fn bench_single_mle(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_mle_eval");
    group.sample_size(100);

    let evals = random_field_vec(N);
    let point = random_field_vec(LOG_N);

    group.throughput(Throughput::Elements(N as u64));
    group.bench_function(&format!("n=2^{LOG_N}"), |b| {
        b.iter(|| {
            black_box(mle_eval_seq(&evals, &point));
        })
    });
    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: batch of 32 – sequential (no parallelism baseline)
// ---------------------------------------------------------------------------
fn bench_batch_seq(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_lincomb_mle_seq");
    group.sample_size(50);

    let (all_vectors, all_coeffs, all_points) =
        random_batch_inputs(NUM_COMPUTATIONS, M, N, LOG_N);

    group.throughput(Throughput::Elements(
        (NUM_COMPUTATIONS * M * N) as u64,
    ));
    group.bench_function(&format!("{NUM_COMPUTATIONS}x_m={M},n=2^{LOG_N}"), |b| {
        b.iter(|| {
            black_box(batch_lincomb_then_mle_eval_seq(
                &all_vectors,
                &all_coeffs,
                &all_points,
            ));
        })
    });
    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: batch of 32 – parallel across computations
// ---------------------------------------------------------------------------
fn bench_batch_par(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_lincomb_mle_par");
    group.sample_size(50);

    let (all_vectors, all_coeffs, all_points) =
        random_batch_inputs(NUM_COMPUTATIONS, M, N, LOG_N);

    group.throughput(Throughput::Elements(
        (NUM_COMPUTATIONS * M * N) as u64,
    ));
    group.bench_function(&format!("{NUM_COMPUTATIONS}x_m={M},n=2^{LOG_N}"), |b| {
        b.iter(|| {
            black_box(batch_lincomb_then_mle_eval(
                &all_vectors,
                &all_coeffs,
                &all_points,
            ));
        })
    });
    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: batch of 32 linear combinations – parallel
// ---------------------------------------------------------------------------
fn bench_batch_lincomb_par(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_linear_combination_par");
    group.sample_size(50);

    let (all_vectors, all_coeffs, _) =
        random_batch_inputs(NUM_COMPUTATIONS, M, N, LOG_N);

    group.throughput(Throughput::Elements(
        (NUM_COMPUTATIONS * M * N) as u64,
    ));
    group.bench_function(&format!("{NUM_COMPUTATIONS}x_m={M},n=2^{LOG_N}"), |b| {
        b.iter(|| {
            black_box(batch_linear_combination(&all_vectors, &all_coeffs));
        })
    });
    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: batch of 32 MLE evals – parallel
// ---------------------------------------------------------------------------
fn bench_batch_mle_par(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_mle_eval_par");
    group.sample_size(50);

    let all_evals: Vec<_> = (0..NUM_COMPUTATIONS)
        .map(|_| random_field_vec(N))
        .collect();
    let all_points: Vec<_> = (0..NUM_COMPUTATIONS)
        .map(|_| random_field_vec(LOG_N))
        .collect();

    group.throughput(Throughput::Elements(
        (NUM_COMPUTATIONS * N) as u64,
    ));
    group.bench_function(&format!("{NUM_COMPUTATIONS}x_n=2^{LOG_N}"), |b| {
        b.iter(|| {
            black_box(batch_mle_eval(&all_evals, &all_points));
        })
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_single_lincomb,
    bench_single_mle,
    bench_single_lincomb_mle,
    bench_batch_seq,
    bench_batch_par,
    bench_batch_lincomb_par,
    bench_batch_mle_par,
);
criterion_main!(benches);
