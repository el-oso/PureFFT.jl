# StrictMode WHOLE-PACKAGE check (manual tool; mirrors bench/alloccheck.jl). Exercises StrictMode's
# package-level functionality — `audit(PureFFT; sweep=true)` walks every method PureFFT actually compiled
# and checks each is type-stable + allocation-free, scoped with `exempt` (the F5 fix) to drop the
# plan-time/autotuner helpers that allocate by design. This auto-covers every hot kernel, incl. new W=8
# ones, with no per-kernel edits.
#
# Uses analysis="fast" (set in bench/Project.toml [preferences.StrictMode]) so the broad sweep is quick —
# return-type concreteness + alloc-freedom; the rigorous JET-based pass is applied per-kernel in the test
# suite (test/strictmode_tests.jl, analysis="full").
#
# Run (StrictMode checks must be enabled — a compile-time Preference, already set in bench/Project.toml):
#   julia --project=bench bench/strictmode_audit.jl

using PureFFT, StrictMode
using AllocCheck, JET   # StrictMode's analysis backend is a weak-dep extension — load it for the sweep

StrictMode.checks_enabled() || error("StrictMode checks disabled — set [preferences.StrictMode] checks_enabled=true")
StrictMode.backend_available() || error("StrictMode analysis backend not loaded — need `using AllocCheck, JET`")

# Warm a representative spread so the usage-driven sweep sees the real hot kernels (autoplan routes each
# to its winning path: pow2 radix-4 AVX, codelet, four-step, the W=4/W=8 faithful trees, Rader, Bluestein).
for n in (64, 256, 1024, 4096, 16384, 12, 27, 576, 768, 1080, 2520, 2880, 6144, 97, 769, 289, 578)
    p = PureFFT.autoplan(ComplexF64, n)
    PureFFT.apply_unnormalized!(p, randn(ComplexF64, n))
end

# Plan-time / autotuner / codegen helpers that allocate or are type-flexible BY DESIGN (not hot path).
# Exempt by base name — kwarg methods' `#name#NN` kwsorters are matched via StrictMode's demangling (F6).
const COLD_HELPERS = (
    :_besttime, :_recursive_factors, :_recursive_candidates, :_foursplit_candidates,
    :_emit_sum!, :_halfcos, :_halfsin, :_primitive_root, :factorize,
    # plan-time routing/candidate-list builders (same category as the above): _gen_pp_prime returns a
    # Union{Int,Nothing} routing gate (autoplan); _bluestein_Ms returns an Int[] of candidate M sizes,
    # called only in the BluesteinPlan constructor. Both construction-only — never on a hot transform path.
    # _gen_pp_composite returns a Union{Tuple{Int,Int},Nothing} routing gate (autoplan), construction-only.
    :_gen_pp_prime, :_bluestein_Ms, :_gen_pp_composite,
)

# Whole-package sweep: type-stability + allocation-freedom + trim-safety over every compiled method. Cheap
# via :fast (analysis="fast" in bench/Project.toml) — the noalloc heuristic is throw-path clean and no
# longer false-positives on pointer/`vload`/`vstore` over preallocated scratch (StrictMode F8/F9, both fixed
# upstream); :trimsafe (TypeContracts, no backend) gives a cheap juliac --trim=safe-style scan across ALL
# kernels, complementing the authoritative TrimCheck @validate on the hot path in test/purefft_tests.jl.
fs = audit(PureFFT; sweep = true, guarantees = (:typestable, :noalloc, :trimsafe),
           exempt = COLD_HELPERS, format = :text)
nf = nfailures(fs)
println("\nStrictMode whole-package sweep: $(length(fs)) (method, guarantee) checks, $nf failure(s) ",
        "(exempt: $(length(COLD_HELPERS)) plan-time helpers).")
nf == 0 || error("StrictMode found $nf failure(s) on PureFFT's compiled hot-path surface.")
println("Every compiled hot-path method is type-stable, allocation-free, and trim-safe. ✓")
