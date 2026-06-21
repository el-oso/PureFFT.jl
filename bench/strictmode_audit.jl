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

StrictMode.checks_enabled() || error("StrictMode checks disabled — set [preferences.StrictMode] checks_enabled=true")

# Warm a representative spread so the usage-driven sweep sees the real hot kernels (autoplan routes each
# to its winning path: pow2 radix-4 AVX, codelet, four-step, the W=4/W=8 faithful trees, Rader, Bluestein).
for n in (64, 256, 1024, 4096, 16384, 12, 27, 576, 768, 1080, 2520, 2880, 6144, 97, 769)
    p = PureFFT.autoplan(ComplexF64, n)
    PureFFT.apply_unnormalized!(p, randn(ComplexF64, n))
end

# Plan-time / autotuner / codegen helpers that allocate or are type-flexible BY DESIGN (not hot path).
# Exempt by base name — kwarg methods' `#name#NN` kwsorters are matched via StrictMode's demangling.
const COLD_HELPERS = (
    :_besttime, :_recursive_factors, :_recursive_candidates, :_foursplit_candidates,
    :_emit_sum!, :_halfcos, :_halfsin, :_primitive_root, :factorize,
)

fs = audit(PureFFT; sweep = true, exempt = COLD_HELPERS, format = :text)

# Robust to StrictMode builds without the kwsorter-demangle fix (FEEDBACK F6): a kwarg helper compiles a
# `#name#NN` kwsorter that base-name `exempt` may not catch — drop failures whose demangled base name is a
# known cold helper. (With F6 present, exempt already handles these and this filter is a no-op.)
_demangle(s) = (m = match(r"^#(.+)#\d+$", s); isnothing(m) ? s : String(m.captures[1]))
realfails = filter(f -> f.status === :fail && Symbol(_demangle(f.func)) ∉ COLD_HELPERS, fs)
nf = length(realfails)
println("\nStrictMode whole-package sweep: $(length(fs)) (method, guarantee) checks, $nf hot-path failure(s) ",
        "(exempt: $(length(COLD_HELPERS)) plan-time helpers).")
nf == 0 || (foreach(f -> println("  ", f), realfails);
            error("StrictMode found $nf failure(s) on PureFFT's compiled hot-path surface."))
println("Every compiled hot-path method is type-stable and allocation-free. ✓")
