# Preflight: is the bench core's clock DETERMINISTIC?  (include + call before any perf-gate measurement)
#
# Why the old min==max test was WRONG on this box: `bench/cpufreq_lock.sh pin <MHz>` clamps
# scaling_min==scaling_max, but on **amd-pstate-epp with boost ON** the boost/CPB range OVERRIDES
# scaling_max — so the core still drifts (measured 1.4–4.5 GHz) while min==max reads "pinned". A whole
# session's ratios were poisoned by this false-positive (110592 read 0.85 → 1.026 → the locked truth 0.83).
# The RELIABLE determinism signal is the **boost state**: boost==0 (`cpufreq_lock.sh lock`) locks the core
# to its base clock (stable ~2 GHz here); boost==1 drifts no matter what min/max say. So we gate on boost,
# and fall back to the min==max heuristic only when the boost knob is unreadable (non-AMD / older kernels).
#
# Relative-to-FFTW/Rust ratios are clock-independent for compute-bound sizes, so this WARNS (doesn't abort);
# pass `abort=true` where you want deterministic-clock numbers gated. See cpufreq_lock.sh + the memory note.

const _BENCH_CORE = 2   # matches `taskset -c 2` throughout the bench pipeline
const _BOOST_KNOB = "/sys/devices/system/cpu/cpufreq/boost"

_cpufreq(core, f) = "/sys/devices/system/cpu/cpu$core/cpufreq/$f"
_readint(path) = try parse(Int, strip(read(path, String))) catch; missing end

"""
    boost_state() -> Union{Int,Missing}

CPB/turbo boost knob: `0` = off (deterministic base clock), `1` = on (drifts under load),
`missing` = unreadable (non-AMD cpufreq / older kernel).
"""
boost_state() = _readint(_BOOST_KNOB)

"""
    clock_deterministic(core=$_BENCH_CORE) -> (ok::Union{Bool,Missing}, mhz::Union{Int,Missing}, why::String)

`ok == true`  → the core's clock is deterministic (boost off → base clock; or boost-unknown + min==max).
`ok == false` → NOT deterministic (boost on → drifts even if min==max; or min≠max).
`ok === missing` → cpufreq sysfs unreadable (non-Linux / different topology).
`mhz` is the clamp/base clock; `why` explains the verdict.
"""
function clock_deterministic(core::Int = _BENCH_CORE)
    lo = _readint(_cpufreq(core, "scaling_min_freq"))
    hi = _readint(_cpufreq(core, "scaling_max_freq"))
    (ismissing(lo) || ismissing(hi)) && return (missing, missing, "cpufreq sysfs unreadable")
    b = boost_state()
    if b === 0
        (true, hi ÷ 1000, "boost OFF → locked to base clock (deterministic)")
    elseif b === 1
        # THE trap: min==max does NOT hold determinism when boost is on (amd-pstate-epp overrides scaling_max)
        (false, hi ÷ 1000, "boost ON → clock drifts under load (min==max does NOT hold on amd-pstate-epp)")
    else
        # boost knob unreadable → fall back to the historical min==max heuristic (correct on non-boost cpufreq)
        lo == hi ? (true, lo ÷ 1000, "min==max (boost state unknown — heuristic)") :
                   (false, hi ÷ 1000, "min≠max (hardware range, drifting)")
    end
end

# back-compat shim for callers of the old name (returns just the ok/mhz pair)
clock_pinned(core::Int = _BENCH_CORE) = (t = clock_deterministic(core); (t[1], t[2]))

"""
    assert_pinned(; abort=false, core=$_BENCH_CORE) -> Bool

Print a one-line banner on whether the bench core's clock is DETERMINISTIC (boost-based, not the old
min==max false-positive) and return it. With `abort=true`, throw on a non-deterministic/undeterminable clock.
"""
function assert_pinned(; abort::Bool = false, core::Int = _BENCH_CORE)
    ok, mhz, why = clock_deterministic(core)
    if ok === true
        printstyled("✓ clock DETERMINISTIC — cpu$core ~$mhz MHz ($why)\n"; color = :green)
        return true
    end
    msg = ok === false ?
        "⚠ clock NOT deterministic — cpu$core: $why. Ratios/floors will be NOISY (drift misleads both ways).\n  Fix:  sudo bench/cpufreq_lock.sh lock   (boost off → stable base clock; ratios are clock-independent)" :
        "⚠ could not read cpu$core cpufreq sysfs — cannot verify the clock."
    printstyled(msg, "\n"; color = :yellow, bold = true)
    abort && error("clock not deterministic (cpu$core): $why — run `sudo bench/cpufreq_lock.sh lock` or pass abort=false")
    return false
end

# Run directly (`julia bench/pin_check.jl`) → just report status.
if abspath(PROGRAM_FILE) == (@__FILE__)
    assert_pinned()
end
