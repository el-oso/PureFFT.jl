# Stage 2 — AVX-512 benefit map. For each compute-core component, measure per-complex throughput at
# W=4 (Vec{4}, AVX2) vs W=8 (Vec{8}, AVX-512), both L1-resident (compute-bound) and memory-bound.
# The column butterflies are width-generic, so one @noinline pass serves both widths (dispatch on the
# buffer/twiddle element type). Run: taskset -c 2 julia -O3 --project=bench port/avx512_bench.jl
include(joinpath(@__DIR__, "..", "src", "avxradix", "avxport.jl"))
using SIMD: Vec
using Printf, Statistics

# width-generic passes: load R rows, apply cbR, store R rows; C chunks. (twiddle consts passed as args)
@noinline function pass_cb4!(b, C, rot)
    @inbounds for c in 0:(C - 1); i = 4c
        o = avx_column_butterfly4(b[i+1], b[i+2], b[i+3], b[i+4], rot)
        b[i+1] = o[1]; b[i+2] = o[2]; b[i+3] = o[3]; b[i+4] = o[4]
    end
end
@noinline function pass_cb8!(b, C, rot)
    @inbounds for c in 0:(C - 1); i = 8c
        o = avx_column_butterfly8(b[i+1], b[i+2], b[i+3], b[i+4], b[i+5], b[i+6], b[i+7], b[i+8], rot)
        for k in 1:8; b[i+k] = o[k]; end
    end
end
@noinline function pass_cb12!(b, C, bf3, rot)
    @inbounds for c in 0:(C - 1); i = 12c
        o = avx_column_butterfly12(b[i+1], b[i+2], b[i+3], b[i+4], b[i+5], b[i+6], b[i+7], b[i+8], b[i+9], b[i+10], b[i+11], b[i+12], bf3, rot)
        for k in 1:12; b[i+k] = o[k]; end
    end
end
@noinline function pass_cb9!(b, C, t1, t2, t3, bf3)
    @inbounds for c in 0:(C - 1); i = 9c
        o = avx_column_butterfly9(b[i+1], b[i+2], b[i+3], b[i+4], b[i+5], b[i+6], b[i+7], b[i+8], b[i+9], t1, t2, t3, bf3)
        for k in 1:9; b[i+k] = o[k]; end
    end
end
@noinline function pass_mulc!(b, C, tw)
    @inbounds for i in 1:C; b[i] = avx_mul_complex(tw, b[i]); end
end

mkv(::Type{V}, n) where {V} = [V(ntuple(_ -> randn(), Val(length(V)))) for _ in 1:n]
function med(f, args...)
    for _ in 1:50; f(args...); end
    ts = Float64[]; for _ in 1:151; t = time_ns(); for _ in 1:20; f(args...); end; push!(ts, (time_ns() - t) / 20); end
    median(ts)
end
# report: per-complex throughput (Gc/s) at each width + the W8/W4 ratio. NV = vectors per pass.
function row(label, R, NV, mk4, mk8, run4, run8)
    C = NV ÷ R
    b4 = mkv(V4f, R * C); b8 = mkv(V8f, R * C)
    m4 = med(run4, b4, C, mk4...); m8 = med(run8, b8, C, mk8...)
    g4 = 2 * R * C / m4; g8 = 4 * R * C / m8        # complex/ns: V4f=2/vec, V8f=4/vec
    @printf("%-10s NV=%-6d  W4 %5.2f Gc/s   W8 %5.2f Gc/s   ratio %.2f×  %s\n", label, NV, g4, g8, g8 / g4, g8/g4 ≥ 1.10 ? "CONVERT" : "")
end

F = true
a4 = (avx_broadcast_twiddle(1,3,F),); a8 = (avx_broadcast_twiddle8(1,3,F),)     # bf3 for cb6/12 (cb9 uses 4)
cb9_4 = (avx_broadcast_twiddle(1,9,F), avx_broadcast_twiddle(2,9,F), avx_broadcast_twiddle(4,9,F), avx_broadcast_twiddle(1,3,F))
cb9_8 = (avx_broadcast_twiddle8(1,9,F), avx_broadcast_twiddle8(2,9,F), avx_broadcast_twiddle8(4,9,F), avx_broadcast_twiddle8(1,3,F))
println("=== AVX-512 benefit map: per-complex throughput, W=8 vs W=4 ===")
println("--- compute-bound (L1-resident, ~256 vectors) ---")
row("cb4",  4, 256, (_ROT90_FWD,), (_ROT90_FWD8,), pass_cb4!, pass_cb4!)
row("cb8",  8, 256, (_ROT90_FWD,), (_ROT90_FWD8,), pass_cb8!, pass_cb8!)
row("cb12", 12, 252, (a4[1],_ROT90_FWD), (a8[1],_ROT90_FWD8), pass_cb12!, pass_cb12!)
row("cb9",  9, 252, cb9_4, cb9_8, pass_cb9!, pass_cb9!)
row("mulc", 1, 256, (avx_broadcast_twiddle(1,7,F),), (avx_broadcast_twiddle8(1,7,F),), pass_mulc!, pass_mulc!)
println("--- memory-bound (~32768 vectors, > L2) ---")
row("cb8",  8, 32768, (_ROT90_FWD,), (_ROT90_FWD8,), pass_cb8!, pass_cb8!)
row("mulc", 1, 32768, (avx_broadcast_twiddle(1,7,F),), (avx_broadcast_twiddle8(1,7,F),), pass_mulc!, pass_mulc!)
