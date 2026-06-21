#!/usr/bin/env bash
# Lock the CPU frequency for stable, low-noise benchmarks, then restore.
#
# Why: the AMD Zen5 box uses the amd-pstate-epp driver with the `performance` governor already set, so the
# governor is NOT the noise source. The noise is OPPORTUNISTIC BOOST: cores clock up to ~4.9 GHz but the
# achieved frequency drifts with thermals/power over a multi-minute `bench/plot_compare.jl` run, giving
# cross-size variance (wide σ ribbons). Disabling boost pins the cores to a deterministic base clock —
# lower absolute GFLOP/s, but far tighter and reproducible numbers (which is what the plots need).
#
# This host's BASE clock is only 2.0 GHz (everything above is boost/CPB), so `lock` (boost off) is stable
# but slow. `pin <MHz>` instead keeps boost ON and clamps scaling_min=scaling_max to a fixed high clock —
# stable AND representative. Single-core (`taskset -c 2`) sustains ~4.5 GHz, so `pin 4500` is a good
# default (drift-free, close to the boosted peak). Verify with `status` that cpuN matches under load.
#
# Needs root (writes to /sys). Usage:
#   sudo bench/cpufreq_lock.sh pin 4500   # fixed 4.5 GHz (boost on, clamped) — recommended
#   sudo bench/cpufreq_lock.sh lock       # base clock, boost off (deterministic but ~2 GHz)
#   taskset -c 2 julia -O3 --project=bench bench/plot_compare.jl
#   sudo bench/cpufreq_lock.sh restore    # back to normal turbo
#   bench/cpufreq_lock.sh status          # (no sudo) show governor/boost/freq
#
# Also for clean numbers (no sudo, your responsibility): close other CPU users, and don't run anything on
# core 2's SMT sibling (core 8 here — `cat /sys/devices/system/cpu/cpu2/topology/thread_siblings_list`),
# since `taskset -c 2` shares that physical core. `pkill -f plot_compare; pgrep -af julia` to find strays.

set -euo pipefail
BOOST=/sys/devices/system/cpu/cpufreq/boost
GOV_GLOB='/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor'
MAX_GLOB='/sys/devices/system/cpu/cpu*/cpufreq/scaling_max_freq'
MIN_GLOB='/sys/devices/system/cpu/cpu*/cpufreq/scaling_min_freq'
HW_MIN=$(cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_min_freq)

status() {
    echo "governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"
    echo "boost:    $(cat "$BOOST" 2>/dev/null || echo n/a)  (1 = turbo on/noisy, 0 = locked to base)"
    echo "cpu2 cur: $(( $(cat /sys/devices/system/cpu/cpu2/cpufreq/scaling_cur_freq) / 1000 )) MHz"
}

case "${1:-status}" in
    pin)
        mhz="${2:?usage: $0 pin <MHz>}"; khz=$((mhz * 1000))
        [ -w "$BOOST" ] && echo 1 > "$BOOST"            # boost on so freqs above the 2 GHz base are reachable
        for g in $GOV_GLOB; do echo performance > "$g"; done
        for f in $MAX_GLOB; do echo "$khz" > "$f"; done  # raise the cap first…
        for f in $MIN_GLOB; do echo "$khz" > "$f"; done  # …then clamp the floor to the same value
        echo "Pinned to ${mhz} MHz (boost on, min=max):"; status ;;
    lock)
        for g in $GOV_GLOB; do echo performance > "$g"; done
        [ -w "$BOOST" ] && echo 0 > "$BOOST"
        echo "Locked (performance governor, boost off → base clock):"; status ;;
    restore)
        [ -w "$BOOST" ] && echo 1 > "$BOOST"
        HW_MAX=$(cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq)   # full range now that boost is on
        for f in $MIN_GLOB; do echo "$HW_MIN" > "$f"; done
        for f in $MAX_GLOB; do echo "$HW_MAX" > "$f"; done
        echo "Restored (boost on, full range):"; status ;;
    status) status ;;
    *) echo "usage: $0 {pin <MHz>|lock|restore|status}" >&2; exit 2 ;;
esac
