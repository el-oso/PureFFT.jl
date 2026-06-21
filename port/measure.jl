# Rigorous parity measurement: rust (via ccall to librustfft_bench.so) vs the Julia keystone, measured
# in the SAME process, INTERLEAVED per block (same thermal/frequency), warmed up, median+σ, core-pinned.
# Run: taskset -c 2 julia -O3 --project=bench port/measure.jl
include(joinpath(@__DIR__, "recursive.jl"))
using Printf, Statistics

const LIB = joinpath(@__DIR__, "..", "bench", "rustfft_compare", "rust", "target", "release", "librustfft_bench.so")
rplan(n) = ccall((:rfft_plan, LIB), Ptr{Cvoid}, (Csize_t,), n)
rproc(h, d::Vector{ComplexF64}, n) = ccall((:rfft_process, LIB), Cvoid, (Ptr{Cvoid}, Ptr{ComplexF64}, Csize_t), h, d, n)
rfree(h) = ccall((:rfft_free, LIB), Cvoid, (Ptr{Cvoid},), h)

# measure one size: julia plan `jp` (applyplan!), rust handle `h`; interleave blocks.
function compare(n, jp)
    h = rplan(n)
    jb = copy(seeded(n)); rb = copy(seeded(n))
    kit = max(200, round(Int, 3.0e8 / (n * log2(n))))
    for _ in 1:15; for _ in 1:kit; applyplan!(jp, jb); end; for _ in 1:kit; rproc(h, rb, n); end; end  # warm to steady freq
    rt = Float64[]; jt = Float64[]
    GC.@preserve rb begin
        for _ in 1:201
            t = time_ns(); for _ in 1:kit; applyplan!(jp, jb); end; push!(jt, (time_ns() - t) / kit)
            t = time_ns(); for _ in 1:kit; rproc(h, rb, n); end;     push!(rt, (time_ns() - t) / kit)
        end
    end
    rfree(h)
    jm, js = median(jt), std(jt); rm, rs = median(rt), std(rt)
    @printf("n=%-6d  julia %8.1f ns (σ%.1f%%)  rust %8.1f ns (σ%.1f%%)  ratio=%.3f  %s\n",
        n, jm, 100js / jm, rm, 100rs / rm, rm / jm, rm / jm ≥ 0.96 ? "✓≥0.96" : "✗")
    return rm / jm
end

# keystone plans (all B36-based): 36, 144, 720, 2880, 11520
p36 = RPlan(B36(true))
p144 = RPlan(MR4(B36(true), true))
p720 = RPlan(MR5(MR4(B36(true), true), true))
p2880 = RPlan(MR5(MR4(MR4(B36(true), true), true), true))
p11520 = RPlan(MR4(MR5(MR4(MR4(B36(true), true), true), true), true))
# correctness sanity
import FFTW
for (n, p) in ((36, p36), (144, p144), (720, p720), (2880, p2880), (11520, p11520))
    y = copy(seeded(n)); applyplan!(p, y); rel = maximum(abs.(y .- FFTW.fft(seeded(n)))) / maximum(abs.(FFTW.fft(seeded(n))))
    rel < 1e-10 || @printf("  WARN n=%d rel-err %.0e\n", n, rel)
end
println("--- interleaved in-process parity (median, core-pinned) ---")
compare(36, p36); compare(144, p144); compare(720, p720); compare(2880, p2880); compare(11520, p11520)
