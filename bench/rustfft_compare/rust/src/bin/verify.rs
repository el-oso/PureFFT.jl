// Golden-value + timing harness for the faithful Julia port of RustFFT's AVX algorithm.
//
//  * PRIMITIVES: the exact __m256d AvxVector expressions (copied verbatim from rustfft 6.4.1
//    src/avx/avx_vector.rs) run on fixed inputs; printed as raw bits (to_bits) for bit-exact diff.
//  * FFT-LEVEL: rustfft's public plan_fft(n) on a fixed seeded input; outputs (bits) + min-time.
//
// Output format (parsed by the Julia side): one record per line:
//   P <name> <b0> <b1> <b2> <b3>                      (a __m256d, 4 lanes, hex bits)
//   F <n> out <2n hex bits...>                        (FFT of size n, interleaved re,im)
//   T <n> <ns_per_transform>
//
// Run: cargo run --release --bin verify
#![allow(unused)]
use std::arch::x86_64::*;
use rustfft::num_complex::Complex;
use rustfft::FftPlannerAvx;
use std::time::Instant;

#[inline(always)]
unsafe fn v(a: f64, b: f64, c: f64, d: f64) -> __m256d { _mm256_set_pd(d, c, b, a) } // lane order a,b,c,d

unsafe fn pr(name: &str, x: __m256d) {
    let mut o = [0.0f64; 4];
    _mm256_storeu_pd(o.as_mut_ptr(), x);
    println!("P {} {:016x} {:016x} {:016x} {:016x}", name, o[0].to_bits(), o[1].to_bits(), o[2].to_bits(), o[3].to_bits());
}

// ---- exact __m256d primitive expressions from avx_vector.rs ----
#[inline(always)] unsafe fn swap(s: __m256d) -> __m256d { _mm256_permute_pd(s, 0x05) }
#[inline(always)] unsafe fn dup_re(s: __m256d) -> __m256d { _mm256_movedup_pd(s) }
#[inline(always)] unsafe fn dup_im(s: __m256d) -> __m256d { _mm256_permute_pd(s, 0x0F) }
#[inline(always)] unsafe fn reverse(s: __m256d) -> __m256d { _mm256_permute2f128_pd(s, s, 0x01) }
#[inline(always)] unsafe fn unpacklo(a: __m256d, b: __m256d) -> __m256d { _mm256_permute2f128_pd(a, b, 0x20) }
#[inline(always)] unsafe fn unpackhi(a: __m256d, b: __m256d) -> __m256d { _mm256_permute2f128_pd(a, b, 0x31) }
#[inline(always)] unsafe fn fmadd(a: __m256d, b: __m256d, c: __m256d) -> __m256d { _mm256_fmadd_pd(a, b, c) }
#[inline(always)] unsafe fn fnmadd(a: __m256d, b: __m256d, c: __m256d) -> __m256d { _mm256_fnmadd_pd(a, b, c) }
#[inline(always)] unsafe fn fmaddsub(a: __m256d, b: __m256d, c: __m256d) -> __m256d { _mm256_fmaddsub_pd(a, b, c) }
#[inline(always)] unsafe fn fmsubadd(a: __m256d, b: __m256d, c: __m256d) -> __m256d { _mm256_fmsubadd_pd(a, b, c) }

#[inline(always)] unsafe fn mul_complex(left: __m256d, right: __m256d) -> __m256d {
    let left_real = dup_re(left);
    let left_imag = dup_im(left);
    let right_shuffled = swap(right);
    let output_right = _mm256_mul_pd(left_imag, right_shuffled);
    fmaddsub(left_real, right, output_right)
}
// rotate90: negate via xor with the rotation mask, then swap re/im.
#[inline(always)] unsafe fn rotate90(s: __m256d, mask: __m256d) -> __m256d { swap(_mm256_xor_pd(s, mask)) }

#[target_feature(enable = "avx,fma")]
unsafe fn primitives() {
    let a = v(1.5, -2.5, 3.25, -0.75);
    let b = v(0.5, 4.0, -1.25, 2.5);
    let c = v(-3.0, 1.0, 0.25, -0.5);
    pr("A", a); pr("B", b); pr("C", c);
    pr("swap_A", swap(a));
    pr("dupre_A", dup_re(a));
    pr("dupim_A", dup_im(a));
    pr("reverse_A", reverse(a));
    pr("unpacklo_AB", unpacklo(a, b));
    pr("unpackhi_AB", unpackhi(a, b));
    pr("fmadd_ABC", fmadd(a, b, c));
    pr("fnmadd_ABC", fnmadd(a, b, c));
    pr("fmaddsub_ABC", fmaddsub(a, b, c));
    pr("fmsubadd_ABC", fmsubadd(a, b, c));
    pr("mulcomplex_AB", mul_complex(a, b));
    // make_rotation90: Forward → broadcast(-0.0, 0.0) => lanes [-0,0,-0,0]; Inverse → (0.0,-0.0) => [0,-0,0,-0]
    let mask_fwd = v(-0.0, 0.0, -0.0, 0.0);
    let mask_inv = v(0.0, -0.0, 0.0, -0.0);
    pr("rotate90fwd_A", rotate90(a, mask_fwd));
    pr("rotate90inv_A", rotate90(a, mask_inv));
}

fn seeded(n: usize) -> Vec<Complex<f64>> {
    (0..n).map(|k| Complex::new(
        ((k * 2 + 1) % 17) as f64 / 17.0 - 0.5,
        ((k * 3 + 2) % 19) as f64 / 19.0 - 0.5,
    )).collect()
}

fn fft_level() {
    let mut planner = FftPlannerAvx::<f64>::new().expect("AVX required");
    for &n in &[5usize, 7, 8, 9, 11, 12, 35, 315, 720, 5760, 11520, 23040, 92160] {
        let fft = planner.plan_fft_forward(n);
        let src = seeded(n);
        let mut work = src.clone();
        let mut scratch = vec![Complex::new(0.0, 0.0); fft.get_inplace_scratch_len()];
        fft.process_with_scratch(&mut work, &mut scratch);
        // outputs (bits) only for small sizes (keep output readable); timing for all
        if n <= 35 {
            print!("F {} out", n);
            for z in &work { print!(" {:016x} {:016x}", z.re.to_bits(), z.im.to_bits()); }
            println!();
        }
        let kit = std::cmp::max(50, (2.0e8 / ((n as f64) * (n as f64).log2())).round() as usize);
        let mut best = f64::INFINITY;
        for _ in 0..40 {
            let t = Instant::now();
            for _ in 0..kit { work.copy_from_slice(&src); fft.process_with_scratch(&mut work, &mut scratch); }
            let d = t.elapsed().as_nanos() as f64;
            let t2 = Instant::now();
            for _ in 0..kit { work.copy_from_slice(&src); std::hint::black_box(&mut work); }
            let d2 = t2.elapsed().as_nanos() as f64;
            best = best.min((d - d2) / kit as f64);
        }
        println!("T {} {:.2}", n, best);
    }
}

fn main() {
    unsafe { primitives(); }
    fft_level();
}
