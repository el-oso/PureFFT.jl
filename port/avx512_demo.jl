# Stage 5 demo: a compact recursive W=8 (AVX-512) keystone for MR12 + B64, to show the non-pow2 512-bit
# win in a plot. Verifies vs FFTW and benchmarks W=8 vs the V4f keystone vs RustFFT on B64×12^k sizes.
# Run: taskset -c 2 julia -O3 --project=bench port/avx512_demo.jl
include(joinpath(@__DIR__, "..", "src", "avxradix", "recursive.jl"))   # V4f keystone + W=8 primitives/transposes
using SIMD: Vec
import FFTW
using Printf, Statistics, Plots

seeded(n) = [Complex(((k * 2 + 1) % 17) / 17 - 0.5, ((k * 3 + 2) % 19) / 19 - 0.5) for k in 0:(n - 1)]
L8(b, i) = avx_load_complex8(b, i); S8(b, i, v) = avx_store_complex8!(b, i, v)
bf64tw8(fwd) = [avx_mixedradix_twiddle_chunk8(cs * 4, r, 64, fwd) for cs in 0:1 for r in 1:7]
mrtw8(R, M, n, fwd) = [avx_mixedradix_twiddle_chunk8(c * 4, y, n, fwd) for c in 0:(M ÷ 4 - 1) for y in 1:(R - 1)]

function bf64w8!(out, inp, scr, base, tw, rot)
    @inbounds for cs in 0:1; b = base + cs * 4
        m = avx_column_butterfly8(L8(inp, b), L8(inp, b+8), L8(inp, b+16), L8(inp, b+24), L8(inp, b+32), L8(inp, b+40), L8(inp, b+48), L8(inp, b+56), rot)
        t = avx_transpose8_packed(m[1], avx_mul_complex(tw[7cs+1], m[2]), avx_mul_complex(tw[7cs+2], m[3]), avx_mul_complex(tw[7cs+3], m[4]), avx_mul_complex(tw[7cs+4], m[5]), avx_mul_complex(tw[7cs+5], m[6]), avx_mul_complex(tw[7cs+6], m[7]), avx_mul_complex(tw[7cs+7], m[8]))
        ob = base + cs * 32; for k in 1:8; S8(scr, ob + 4(k-1), t[k]); end; end
    @inbounds for cs in 0:1; b = base + cs * 4
        m = avx_column_butterfly8(L8(scr, b), L8(scr, b+8), L8(scr, b+16), L8(scr, b+24), L8(scr, b+32), L8(scr, b+40), L8(scr, b+48), L8(scr, b+56), rot)
        for r in 0:7; S8(out, b + 8r, m[r+1]); end; end
end
@inline function colbf12w8!(buf, o, M, tw, bf3, rot)
    @inbounds for c in 0:(M ÷ 4 - 1); ib = o + 4c
        r = avx_column_butterfly12(L8(buf,ib), L8(buf,ib+M), L8(buf,ib+2M), L8(buf,ib+3M), L8(buf,ib+4M), L8(buf,ib+5M), L8(buf,ib+6M), L8(buf,ib+7M), L8(buf,ib+8M), L8(buf,ib+9M), L8(buf,ib+10M), L8(buf,ib+11M), bf3, rot)
        S8(buf, ib, r[1]); for j in 1:11; S8(buf, ib + j*M, avx_mul_complex(tw[c*11+j], r[j+1])); end; end
end
@inline function trans12w8!(out, oo, buf, o, M)
    @inbounds for c in 0:(M ÷ 4 - 1); ib = o + 4c; ob = oo + 48c
        t = avx_transpose12_packed(L8(buf,ib), L8(buf,ib+M), L8(buf,ib+2M), L8(buf,ib+3M), L8(buf,ib+4M), L8(buf,ib+5M), L8(buf,ib+6M), L8(buf,ib+7M), L8(buf,ib+8M), L8(buf,ib+9M), L8(buf,ib+10M), L8(buf,ib+11M))
        for k in 1:12; S8(out, ob + 4(k-1), t[k]); end; end
end
struct B64w8; tw::Vector{V8f}; rot::V8f; end
struct MR12w8{I}; n::Int; M::Int; inner::I; tw::Vector{V8f}; bf3::V8f; rot::V8f; end
klen8(::B64w8) = 64; klen8(k::MR12w8) = k.n
B64w8(fwd) = B64w8(bf64tw8(fwd), fwd ? _ROT90_FWD8 : _ROT90_INV8)
MR12w8(inner, fwd) = (M = klen8(inner); MR12w8(12M, M, inner, mrtw8(12, M, 12M, fwd), avx_broadcast_twiddle8(1, 3, fwd), fwd ? _ROT90_FWD8 : _ROT90_INV8))
ip8!(k::B64w8, buf, scr) = (@inbounds for f in 0:(length(buf)÷64-1); bf64w8!(buf, buf, scr, 64f, k.tw, k.rot); end)
oop8!(k::B64w8, out, inp, scr) = (@inbounds for f in 0:(length(inp)÷64-1); bf64w8!(out, inp, out, 64f, k.tw, k.rot); end)
function ip8!(k::MR12w8, buf, scr); n = k.n; cnt = length(buf) ÷ n
    @inbounds for f in 0:cnt-1; colbf12w8!(buf, f*n, k.M, k.tw, k.bf3, k.rot); end
    oop8!(k.inner, scr, buf, scr); @inbounds for f in 0:cnt-1; trans12w8!(buf, f*n, scr, f*n, k.M); end; end
function oop8!(k::MR12w8, out, inp, scr); n = k.n; cnt = length(inp) ÷ n
    @inbounds for f in 0:cnt-1; colbf12w8!(inp, f*n, k.M, k.tw, k.bf3, k.rot); end
    ip8!(k.inner, inp, scr); @inbounds for f in 0:cnt-1; trans12w8!(out, f*n, inp, f*n, k.M); end; end
mkw8(n) = n == 768 ? MR12w8(B64w8(true), true) : n == 9216 ? MR12w8(MR12w8(B64w8(true), true), true) : MR12w8(MR12w8(MR12w8(B64w8(true), true), true), true)
mkw4(n) = n == 768 ? RPlan(MR12(B64(true), true)) : n == 9216 ? RPlan(MR12(MR12(B64(true), true), true)) : RPlan(MR12(MR12(MR12(B64(true), true), true), true))

const LIB = joinpath(@__DIR__, "..", "bench", "rustfft_compare", "rust", "target", "release", "librustfft_bench.so")
rpl(n) = ccall((:rfft_plan, LIB), Ptr{Cvoid}, (Csize_t,), n); rpr(h, d, n) = ccall((:rfft_process, LIB), Cvoid, (Ptr{Cvoid}, Ptr{ComplexF64}, Csize_t), h, d, n)
med(f) = (for _ in 1:20; f(); end; ts = Float64[]; for _ in 1:121; t = time_ns(); for _ in 1:30; f(); end; push!(ts, (time_ns()-t)/30); end; median(ts))
gf(n, t) = 5 * n * log2(n) / t

sizes = [768, 9216, 110592]
g8 = Float64[]; g4 = Float64[]; gr = Float64[]
for n in sizes
    k8 = mkw8(n); x = seeded(n); ref = FFTW.fft(x)
    y = copy(x); s = zeros(ComplexF64, n); ip8!(k8, y, s); rel = maximum(abs.(y .- ref)) / maximum(abs.(ref))
    p4 = mkw4(n); b8 = copy(x); s8 = zeros(ComplexF64, n); b4 = copy(x); rb = copy(x); h = rpl(n)
    m8 = med(() -> ip8!(k8, b8, s8)); m4 = med(() -> applyplan!(p4, b4)); GC.@preserve rb (mr = med(() -> rpr(h, rb, n)))
    push!(g8, gf(n, m8)); push!(g4, gf(n, m4)); push!(gr, gf(n, mr))
    @printf("n=%-7d rel=%.0e  W8 %.1f GF  W4 %.1f GF  rust %.1f GF  W8/W4=%.2f rust:W8=%.2f %s\n", n, rel, gf(n,m8), gf(n,m4), gf(n,mr), m4/m8, mr/m8, rel<1e-10 ? "✓" : "WRONG")
end
p = plot(; xlabel="N (B64 × 12^k, non-power-of-two)", ylabel="GFLOP/s", title="AVX-512 vs AVX2 on non-pow2 (PureFFT W=8 vs W=4 vs RustFFT)\n(Zen 5, single-thread, ComplexF64)", xscale=:log2, xticks=(sizes, string.(sizes)), legend=:topright, size=(800,500), dpi=150, margin=5Plots.mm, ylims=(0, 1.1*maximum(vcat(g8,g4,gr))))
plot!(p, sizes, gr; label="RustFFT (AVX2)", color=:tomato, lw=2, marker=:circle, ms=5)
plot!(p, sizes, g4; label="PureFFT W=4 (AVX2)", color=:steelblue, lw=2, marker=:circle, ms=5)
plot!(p, sizes, g8; label="PureFFT W=8 (AVX-512)", color=:seagreen, lw=2, marker=:circle, ms=5)
out = joinpath(@__DIR__, "..", "docs", "src", "assets", "avx512_nonpow2.png"); savefig(p, out); println("Saved: ", out)
