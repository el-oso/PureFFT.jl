// Controlled language experiment — RUST side.
//
// The SAME radix-2 DIT algorithm as ../julia_kernel.jl: split-layout (separate re/im f64),
// precomputed twiddles indexed with a stride, in-place bit-reversal, iterative butterfly stages,
// FMA-fused complex multiply via `mul_add`, unchecked indexing (matching Julia's `@inbounds`).
// Same memory layout, same operations, same numerics. Both compile through LLVM.
//
// Timing matches the Julia harness exactly: per-transform = (time(copy+fft) - time(copy)) / K,
// minimum over trials. Deterministic input identical to Julia → checksums must match.

use std::hint::black_box;
use std::time::Instant;

#[inline(never)]
fn radix2_dit(xr: &mut [f64], xi: &mut [f64], twr: &[f64], twi: &[f64], n: usize) {
    unsafe {
        // bit-reversal permutation (applied to both re and im)
        let mut j = 0usize;
        let mut i = 0usize;
        while i < n - 1 {
            if i < j {
                let a = *xr.get_unchecked(i);
                *xr.get_unchecked_mut(i) = *xr.get_unchecked(j);
                *xr.get_unchecked_mut(j) = a;
                let b = *xi.get_unchecked(i);
                *xi.get_unchecked_mut(i) = *xi.get_unchecked(j);
                *xi.get_unchecked_mut(j) = b;
            }
            let mut m = n >> 1;
            while m >= 1 && j >= m {
                j -= m;
                m >>= 1;
            }
            j += m;
            i += 1;
        }
        // iterative radix-2 decimation-in-time
        let mut len = 2usize;
        while len <= n {
            let half = len >> 1;
            let stride = n / len;
            let mut base = 0usize;
            while base < n {
                let mut ti = 0usize;
                let mut jj = 0usize;
                while jj < half {
                    let wr = *twr.get_unchecked(ti);
                    let wi = *twi.get_unchecked(ti);
                    let pr = *xr.get_unchecked(base + jj + half);
                    let pii = *xi.get_unchecked(base + jj + half);
                    let tr = pr.mul_add(wr, -(pii * wi));
                    let tii = pr.mul_add(wi, pii * wr);
                    let ar = *xr.get_unchecked(base + jj);
                    let ai = *xi.get_unchecked(base + jj);
                    *xr.get_unchecked_mut(base + jj) = ar + tr;
                    *xi.get_unchecked_mut(base + jj) = ai + tii;
                    *xr.get_unchecked_mut(base + jj + half) = ar - tr;
                    *xi.get_unchecked_mut(base + jj + half) = ai - tii;
                    ti += stride;
                    jj += 1;
                }
                base += len;
            }
            len <<= 1;
        }
    }
}

fn main() {
    let sizes = [64usize, 256, 1024, 4096, 16384, 65536, 262144];
    println!("# rust  release+lto  target-cpu=native");
    println!("# n\tns_per_transform\tGFLOPS\tchecksum");
    for &n in &sizes {
        let half = n / 2;
        let mut twr = vec![0.0f64; half];
        let mut twi = vec![0.0f64; half];
        for k in 0..half {
            let a = -2.0 * std::f64::consts::PI * (k as f64) / (n as f64);
            twr[k] = a.cos();
            twi[k] = a.sin();
        }
        // deterministic input, identical to the Julia side
        let mut srcr = vec![0.0f64; n];
        let mut srci = vec![0.0f64; n];
        for k in 0..n {
            srcr[k] = ((k * 2 + 1) % 17) as f64 / 17.0 - 0.5;
            srci[k] = ((k * 3 + 2) % 19) as f64 / 19.0 - 0.5;
        }
        let mut workr = srcr.clone();
        let mut worki = srci.clone();
        let k_iters =
            std::cmp::max(1, (2.0e8 / ((n as f64) * (n as f64).log2())).round() as usize);
        // warm up
        workr.copy_from_slice(&srcr);
        worki.copy_from_slice(&srci);
        radix2_dit(&mut workr, &mut worki, &twr, &twi, n);
        let mut best = f64::INFINITY;
        let mut chk = 0.0f64;
        for _ in 0..25 {
            let t1 = Instant::now();
            for _ in 0..k_iters {
                workr.copy_from_slice(&srcr);
                worki.copy_from_slice(&srci);
                radix2_dit(black_box(&mut workr), black_box(&mut worki), &twr, &twi, n);
            }
            let d1 = t1.elapsed().as_nanos() as f64;
            chk += workr[0] + worki[1]; // work holds the FFT result here
            let t2 = Instant::now();
            for _ in 0..k_iters {
                workr.copy_from_slice(&srcr);
                worki.copy_from_slice(&srci);
                black_box(&mut workr);
                black_box(&mut worki);
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
