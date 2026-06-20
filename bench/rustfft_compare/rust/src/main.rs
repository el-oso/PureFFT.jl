// Benchmark the rustfft crate: scalar planner (same-algorithm checkpoint vs the Julia Radix4
// port) and AVX planner (the north-star target). Same copy-subtract harness as
// ../../lang_compare, same deterministic input → checksums comparable with the Julia side.
//
// per-transform = (time(copy+fft) - time(copy)) / K, minimum over 25 trials. Scratch is
// preallocated (process_with_scratch) so we time the transform, not scratch allocation.

use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlannerAvx, FftPlannerScalar};
use std::hint::black_box;
use std::sync::Arc;
use std::time::Instant;

fn bench(label: &str, mut plan: impl FnMut(usize) -> Arc<dyn Fft<f64>>) {
    let sizes = [64usize, 256, 1024, 4096, 16384, 65536, 262144];
    println!("# rustfft {label}");
    println!("# n\tns_per_transform\tGFLOPS\tchecksum");
    for &n in &sizes {
        let fft = plan(n);
        let mut scratch = vec![Complex::new(0.0f64, 0.0); fft.get_inplace_scratch_len()];
        // deterministic input, identical values to the Julia side
        let src: Vec<Complex<f64>> = (0..n)
            .map(|k| {
                Complex::new(
                    ((k * 2 + 1) % 17) as f64 / 17.0 - 0.5,
                    ((k * 3 + 2) % 19) as f64 / 19.0 - 0.5,
                )
            })
            .collect();
        let mut work = src.clone();
        let k_iters =
            std::cmp::max(1, (2.0e8 / ((n as f64) * (n as f64).log2())).round() as usize);
        // warm up
        work.copy_from_slice(&src);
        fft.process_with_scratch(&mut work, &mut scratch);
        let mut best = f64::INFINITY;
        let mut chk = 0.0f64;
        for _ in 0..25 {
            let t1 = Instant::now();
            for _ in 0..k_iters {
                work.copy_from_slice(&src);
                fft.process_with_scratch(black_box(&mut work), &mut scratch);
            }
            let d1 = t1.elapsed().as_nanos() as f64;
            chk += work[0].re + work[1].im; // FFT result here (also defeats DCE)
            let t2 = Instant::now();
            for _ in 0..k_iters {
                work.copy_from_slice(&src);
                black_box(&mut work);
            }
            let d2 = t2.elapsed().as_nanos() as f64;
            let per = (d1 - d2) / (k_iters as f64);
            if per < best {
                best = per;
            }
        }
        let gf = 5.0 * (n as f64) * (n as f64).log2() / best;
        println!("{}\t{:.1}\t{:.1}\t{:.3}", n, best, gf, chk);
    }
}

fn main() {
    let mut scalar = FftPlannerScalar::<f64>::new();
    bench("scalar", |n| scalar.plan_fft_forward(n));

    match FftPlannerAvx::<f64>::new() {
        Ok(mut avx) => bench("avx", |n| avx.plan_fft_forward(n)),
        Err(_) => println!("# rustfft avx: NOT AVAILABLE on this CPU"),
    }
}
