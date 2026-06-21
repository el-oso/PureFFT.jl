// C-ABI shim so Julia can ccall rustfft's AVX planner and benchmark rust-vs-julia in the SAME process
// (same thermal/frequency state, interleaved per size) — the only way to assess parity reliably.
use rustfft::{Fft, FftPlannerAvx};
use rustfft::num_complex::Complex;
use std::sync::Arc;

pub struct RPlan {
    fft: Arc<dyn Fft<f64>>,
    scratch: Vec<Complex<f64>>,
}

/// Create a forward AVX plan for size `n`. Returns an opaque handle (Box ptr).
#[unsafe(no_mangle)]
pub extern "C" fn rfft_plan(n: usize) -> *mut RPlan {
    let mut planner = FftPlannerAvx::<f64>::new().expect("AVX required");
    let fft = planner.plan_fft_forward(n);
    let scratch = vec![Complex::new(0.0, 0.0); fft.get_inplace_scratch_len()];
    Box::into_raw(Box::new(RPlan { fft, scratch }))
}

/// Run the plan in place on `data` (length n complex, interleaved re/im as 2n f64 — same layout as Julia ComplexF64).
#[unsafe(no_mangle)]
pub extern "C" fn rfft_process(p: *mut RPlan, data: *mut Complex<f64>, n: usize) {
    let plan = unsafe { &mut *p };
    let buf = unsafe { std::slice::from_raw_parts_mut(data, n) };
    plan.fft.process_with_scratch(buf, &mut plan.scratch);
}

#[unsafe(no_mangle)]
pub extern "C" fn rfft_free(p: *mut RPlan) {
    if !p.is_null() {
        unsafe { drop(Box::from_raw(p)); }
    }
}
