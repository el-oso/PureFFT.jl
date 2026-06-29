# Preflight: is the bench core's clock PINNED?  (include + call before any perf-gate measurement)
#
# Why: `bench/cpufreq_lock.sh pin <MHz>` clamps scaling_min_freq == scaling_max_freq on every core. A
# normal/`restore`d clock leaves min ≠ max (the hardware range, boost drifting under load). So
# `min == max` on the bench core is a reliable "pinned" signal. The pin can SILENTLY reset (e.g. a
# suspend/replug cycle), and an unpinned gate run gives clock-noise floors — the exact failure that
# mis-marked 250/1000 as floors. Print the state loudly so a reset is never silent.
#
# Relative-to-FFTW plots are clock-independent (a pin is OPTIONAL for them — see cpufreq_lock.sh), so
# this WARNS, it does not abort: pass `abort=true` only where you want absolute numbers gated.

const _BENCH_CORE = 2   # matches `taskset -c 2` throughout the bench pipeline

_cpufreq(core, f) = "/sys/devices/system/cpu/cpu$core/cpufreq/$f"

"""
    clock_pinned(core=$_BENCH_CORE) -> (pinned::Union{Bool,Missing}, mhz::Union{Int,Missing})

`pinned == true`  → scaling_min == scaling_max (clamped/pinned), `mhz` = that clock.
`pinned == false` → min ≠ max (boost drifting), `mhz` = current scaling_max.
`pinned === missing` → sysfs unreadable (non-Linux / different topology) — can't tell.
"""
function clock_pinned(core::Int = _BENCH_CORE)
    try
        lo = parse(Int, strip(read(_cpufreq(core, "scaling_min_freq"), String)))
        hi = parse(Int, strip(read(_cpufreq(core, "scaling_max_freq"), String)))
        return (lo == hi, (lo == hi ? lo : hi) ÷ 1000)
    catch
        return (missing, missing)
    end
end

"""
    assert_pinned(; abort=false, core=$_BENCH_CORE) -> Bool

Print a one-line PINNED / NOT-PINNED banner for the bench core and return whether it's pinned.
With `abort=true`, throw on an unpinned (or undetterminable) clock instead of just warning.
"""
function assert_pinned(; abort::Bool = false, core::Int = _BENCH_CORE)
    pinned, mhz = clock_pinned(core)
    if pinned === true
        printstyled("✓ clock PINNED — cpu$core at $mhz MHz (min==max)\n"; color = :green)
        return true
    end
    msg = pinned === false ?
        "⚠ clock NOT PINNED — cpu$core min≠max (current ~$mhz MHz, boost drifting). Gate numbers will be NOISY.\n  Pin first:  sudo bench/cpufreq_lock.sh pin 4500   (relative-to-FFTW plots are clock-independent; absolute numbers are not)" :
        "⚠ could not read cpu$core cpufreq sysfs — cannot verify the clock is pinned."
    printstyled(msg, "\n"; color = :yellow, bold = true)
    abort && error("clock not pinned (cpu$core); re-run after `sudo bench/cpufreq_lock.sh pin 4500` or pass abort=false")
    return false
end

# Run directly (`julia bench/pin_check.jl`) → just report status.
if abspath(PROGRAM_FILE) == (@__FILE__)
    assert_pinned()
end
