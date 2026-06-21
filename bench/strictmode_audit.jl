# StrictMode hot-path audit (manual tool; mirrors bench/alloccheck.jl). Asserts PureFFT's execution
# hot path — `apply_unnormalized!` for every routing path — is type-stable + allocation-free, via
# StrictMode's `check` function API, and prints a report.
#
# NOTE: this audits the HOT PATH specifically (a curated set of concrete plan types), not the whole
# module. A whole-module `check_compiled(PureFFT)` also flags plan-time helpers (`_recursive_factors`,
# `_besttime`, …) that allocate and are type-flexible *by design* — StrictMode has no built-in way to
# scope an audit to "declared" guarantees, so we scope it here by listing the functions we guarantee.
#
# Run (StrictMode checks must be enabled — a compile-time Preference):
#   julia --project=bench -e 'using StrictMode; StrictMode.enable_checks!()'   # once; persists
#   julia --project=bench bench/strictmode_audit.jl

using PureFFT, StrictMode

StrictMode.checks_enabled() || error(
    "StrictMode checks are disabled. Run:\n" *
    "  julia --project=bench -e 'using StrictMode; StrictMode.enable_checks!()'\n" *
    "then re-run this script (the Preference takes effect in a fresh process).")

# A concrete plan per routing path (autoplan picks the winner; a broad size spread exercises pow2
# radix-4 AVX, generated codelet, four-step, the W=4/W=8 faithful trees, Rader, Bluestein).
sizes = (64, 256, 1024, 4096, 16384, 12, 27, 768, 1080, 2520, 6144, 97, 769)
plans = [PureFFT.autoplan(ComplexF64, n) for n in sizes]

findings = StrictFinding[]
for p in plans
    append!(findings, check(PureFFT.apply_unnormalized!, (typeof(p), Vector{ComplexF64})))
end

nf = StrictMode.nfailures(findings)
format_findings(stdout, findings; only_failures = true)
println("\nStrictMode hot-path audit: $(length(findings)) (plan, guarantee) checks across $(length(sizes)) sizes, $nf failure(s).")
nf == 0 || error("StrictMode audit found $nf failure(s) — PureFFT's hot path is not all type-stable + alloc-free.")
println("Every routing path's apply_unnormalized! is type-stable and allocation-free. ✓")
