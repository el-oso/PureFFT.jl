# Step 6A keystone: recursive inner-FFT composition (mirrors MixedRadix + inner Fft).
# Each kernel implements proc_ip!(k,buf,scr) (in-place) and proc_oop!(k,out,inp,scr) (out-of-place),
# processing count = length(buf)/len(k) consecutive FFTs. MixedRadix.proc_ip! uses inner.proc_oop! and
# vice-versa (the in-place/out-of-place alternation), so buf↔scr ping-pong; leaf butterflies
# need no scratch. Even len_per_row only here (no partial column) — odd handled in a later step.
include(joinpath(@__DIR__, "kernels.jl"))
using SIMD: Vec

abstract type Kernel end
klen(k::Kernel) = k.n::Int
# Element type of the buffers a kernel processes (drives the RPlan scratch type). Float64 for the AVX2
# W=4 kernels and the Float64 W=8 kernels; the parameterized W=8 kernels override it (width8.jl).
keltype(::Kernel) = Float64

# ---- leaf: Butterfly36 ----
struct B36 <: Kernel
    n::Int
    tw::NTuple{15, V4f}
    tw3::V4f
end
B36(fwd::Bool) = B36(36, bf36_twiddles(fwd), avx_broadcast_twiddle(1, 3, fwd))
@inline function proc_ip!(k::B36, buf, scr)
    @inbounds for f in 0:(length(buf) ÷ 36 - 1)
        butterfly36!(buf, 36f, k.tw, k.tw3)     # base offset, no SubArray alloc
    end
end
@inline function proc_oop!(k::B36, out, inp, scr)
    @inbounds for f in 0:(length(inp) ÷ 36 - 1)
        butterfly36!(out, inp, 36f, k.tw, k.tw3)    # true out-of-place: load inp, store out (no copy)
    end
end

# ---- leaf: Butterfly9 (3x3, dual-width packed; in-register, no scratch) ----
struct B9 <: Kernel
    n::Int; tw::NTuple{2, V4f}; bf3::V4f; bf3lo::V2f
end
function B9(fwd::Bool)
    bf3 = avx_broadcast_twiddle(1, 3, fwd)
    B9(9, bf9_twiddles(fwd), bf3, avx_lo(bf3))
end
@inline function proc_ip!(k::B9, buf, scr)
    @inbounds for f in 0:(length(buf) ÷ 9 - 1); butterfly9!(buf, buf, 9f, k.tw, k.bf3, k.bf3lo); end
end
@inline function proc_oop!(k::B9, out, inp, scr)
    @inbounds for f in 0:(length(inp) ÷ 9 - 1); butterfly9!(out, inp, 9f, k.tw, k.bf3, k.bf3lo); end
end

# ---- leaf: Butterfly25 (5x5) / Butterfly49 (7x7) — direct prime-power DFT codelets, in-register, no
# scratch. The 5²/7² bases that root the radix-5/7 trees: 125=MR5(B25), 625=MR5²(B25), 343=MR7(B49). ----
struct B25 <: Kernel
    n::Int; tw1::NTuple{4, V4f}; tw2::NTuple{4, V4f}; t0::V4f; t1::V4f; t0lo::V2f; t1lo::V2f
end
function B25(fwd::Bool)
    t0 = avx_broadcast_twiddle(1, 5, fwd); t1 = avx_broadcast_twiddle(2, 5, fwd)
    tw1, tw2 = bf25_twiddles(fwd)
    B25(25, tw1, tw2, t0, t1, avx_lo(t0), avx_lo(t1))
end
@inline function proc_ip!(k::B25, buf, scr)
    @inbounds for f in 0:(length(buf) ÷ 25 - 1); butterfly25!(buf, buf, 25f, k.tw1, k.tw2, k.t0, k.t1, k.t0lo, k.t1lo); end
end
@inline function proc_oop!(k::B25, out, inp, scr)
    @inbounds for f in 0:(length(inp) ÷ 25 - 1); butterfly25!(out, inp, 25f, k.tw1, k.tw2, k.t0, k.t1, k.t0lo, k.t1lo); end
end

struct B49 <: Kernel
    n::Int; tw1::NTuple{6, V4f}; tw2::NTuple{6, V4f}; tw3::NTuple{6, V4f}; t0::V4f; t1::V4f; t2::V4f; t0lo::V2f; t1lo::V2f; t2lo::V2f
end
function B49(fwd::Bool)
    t0 = avx_broadcast_twiddle(1, 7, fwd); t1 = avx_broadcast_twiddle(2, 7, fwd); t2 = avx_broadcast_twiddle(3, 7, fwd)
    tw1, tw2, tw3 = bf49_twiddles(fwd)
    B49(49, tw1, tw2, tw3, t0, t1, t2, avx_lo(t0), avx_lo(t1), avx_lo(t2))
end
@inline function proc_ip!(k::B49, buf, scr)
    @inbounds for f in 0:(length(buf) ÷ 49 - 1); butterfly49!(buf, buf, 49f, k.tw1, k.tw2, k.tw3, k.t0, k.t1, k.t2, k.t0lo, k.t1lo, k.t2lo); end
end
@inline function proc_oop!(k::B49, out, inp, scr)
    @inbounds for f in 0:(length(inp) ÷ 49 - 1); butterfly49!(out, inp, 49f, k.tw1, k.tw2, k.tw3, k.t0, k.t1, k.t2, k.t0lo, k.t1lo, k.t2lo); end
end

# ---- leaf: BP{P} — direct size-P odd-prime DFT (one FFT per iter via the width-generic V2f
# avx_colbf_prime). The innermost base for odd composites carrying a prime with no SIMD radix pass
# (11/19/23/43…) and for pure prime powers (5ⁿ/7ⁿ via MR5/MR7). tws = W_P^1…W_P^{(P-1)/2} as V2f.
# (A V4f pack-2 variant was measured strictly slower — the merge/lo-hi shuffles cost more than the
# throughput gain on these leaf-bound pure-power sizes — so the straight V2f leaf is kept.) ----
@generated function _bfprime!(out, ob::Int, inp, ib::Int, tws, ::Val{P}) where {P}
    loads = [:($(Symbol(:x, j)) = avx_load_partial1(inp, ib + $j)) for j in 0:(P - 1)]
    rst = Expr(:tuple, [Symbol(:x, j) for j in 0:(P - 1)]...)
    stores = [:(avx_store_partial1!(out, ob + $k, r[$(k + 1)])) for k in 0:(P - 1)]
    quote
        @inbounds begin
            $(loads...)
            r = avx_colbf_prime($rst, tws)
            $(stores...)
        end
    end
end
struct BP{P, H} <: Kernel
    n::Int; tws::NTuple{H, V2f}
end
klen(::BP{P}) where {P} = P
function BP(P::Int, fwd::Bool)
    H = (P - 1) ÷ 2
    BP{P, H}(P, ntuple(a -> avx_lo(avx_broadcast_twiddle(a, P, fwd)), H))
end
@inline function proc_ip!(k::BP{P}, buf, scr) where {P}
    @inbounds for f in 0:(length(buf) ÷ P - 1); _bfprime!(buf, P * f, buf, P * f, k.tws, Val(P)); end
end
@inline function proc_oop!(k::BP{P}, out, inp, scr) where {P}
    @inbounds for f in 0:(length(inp) ÷ P - 1); _bfprime!(out, P * f, inp, P * f, k.tws, Val(P)); end
end

# ---- leaf: Butterfly18 (3x6, faithful rustfft port; col 0 partial V2f, cols 1-2 V4f; no scratch) ----
struct B18 <: Kernel
    n::Int; tw::NTuple{5, V4f}; bf3::V4f; bf3lo::V2f
end
B18(fwd::Bool) = (bf3 = avx_broadcast_twiddle(1, 3, fwd); B18(18, bf18_twiddles(fwd), bf3, avx_lo(bf3)))
@inline function proc_ip!(k::B18, buf, scr)
    @inbounds for f in 0:(length(buf) ÷ 18 - 1); butterfly18!(buf, buf, 18f, k.tw, k.bf3, k.bf3lo); end
end
@inline function proc_oop!(k::B18, out, inp, scr)
    @inbounds for f in 0:(length(inp) ÷ 18 - 1); butterfly18!(out, inp, 18f, k.tw, k.bf3, k.bf3lo); end
end

# ---- leaf: Butterfly8 (2x4 single-FFT; register-only, no scratch). The 2^3 leaf the high-power-5
# route needs: 1000=MR5^3(B8), 5000=MR5^4(B8). Faithful port of rustfft Butterfly8Avx64. ----
struct B8 <: Kernel
    n::Int; tw::NTuple{2, V4f}; rot::V4f
end
B8(fwd::Bool) = B8(8, bf8_twiddles(fwd), fwd ? _ROT90_FWD : _ROT90_INV)
@inline function proc_ip!(k::B8, buf, scr)
    @inbounds for f in 0:(length(buf) ÷ 8 - 1); butterfly8!(buf, buf, 8f, k.tw, k.rot); end
end
@inline function proc_oop!(k::B8, out, inp, scr)
    @inbounds for f in 0:(length(inp) ÷ 8 - 1); butterfly8!(out, inp, 8f, k.tw, k.rot); end
end

# ---- leaf: Butterfly2 (length-2 DFT [a+b, a-b]; register-only, no scratch). The smallest pow2 leaf;
# the inner of MR13(B2) so a lone factor of 2 carries the 13 through the FAST avx_column_butterfly13
# instead of the generic BP13 prime leaf (26=MR13(B2), 52=MR2(MR13(B2)), 78=MR3(MR13(B2))). ----
struct B2 <: Kernel
    n::Int
end
B2(fwd::Bool) = B2(2)
@inline function proc_ip!(k::B2, buf, scr)
    @inbounds for f in 0:(length(buf) ÷ 2 - 1)
        s, d = avx_butterfly2(avx_load_partial1(buf, 2f), avx_load_partial1(buf, 2f + 1))
        avx_store_partial1!(buf, 2f, s); avx_store_partial1!(buf, 2f + 1, d)
    end
end
@inline function proc_oop!(k::B2, out, inp, scr)
    @inbounds for f in 0:(length(inp) ÷ 2 - 1)
        s, d = avx_butterfly2(avx_load_partial1(inp, 2f), avx_load_partial1(inp, 2f + 1))
        avx_store_partial1!(out, 2f, s); avx_store_partial1!(out, 2f + 1, d)
    end
end

# ---- leaf: Butterfly16 (4x4, two-phase; needs scratch ≥ its length) ----
struct B16 <: Kernel
    n::Int; tw::Vector{V4f}; rot::V4f
end
B16(fwd::Bool) = B16(16, bf16_twiddles(fwd), fwd ? _ROT90_FWD : _ROT90_INV)
@inline function proc_ip!(k::B16, buf, scr)
    @inbounds for f in 0:(length(buf) ÷ 16 - 1); butterfly16!(buf, buf, scr, 16f, k.tw, k.rot); end
end
@inline function proc_oop!(k::B16, out, inp, scr)
    @inbounds for f in 0:(length(inp) ÷ 16 - 1); butterfly16!(out, inp, out, 16f, k.tw, k.rot); end  # out = workspace
end

# ---- leaf: Butterfly32 (4×8 two-phase; needs scratch ≥ its length). The 2^5 leaf the 5/7 routes need
# (160=MR5(B32), 224=MR7(B32), 240=MR5(MR3(B16)) — and 96=MR3(B32) via the single-3 route). ----
struct B32 <: Kernel
    n::Int; tw::Vector{V4f}; rot::V4f
end
B32(fwd::Bool) = B32(32, bf32_twiddles(fwd), fwd ? _ROT90_FWD : _ROT90_INV)
@inline function proc_ip!(k::B32, buf, scr)
    @inbounds for f in 0:(length(buf) ÷ 32 - 1); butterfly32!(buf, buf, scr, 32f, k.tw, k.rot); end
end
@inline function proc_oop!(k::B32, out, inp, scr)
    @inbounds for f in 0:(length(inp) ÷ 32 - 1); butterfly32!(out, inp, out, 32f, k.tw, k.rot); end  # out = workspace
end

# ---- leaf: Butterfly64 (8x8, two-phase; needs scratch ≥ its length) ----
struct B64 <: Kernel
    n::Int; tw::Vector{V4f}; rot::V4f
end
B64(fwd::Bool) = B64(64, bf64_twiddles(fwd), fwd ? _ROT90_FWD : _ROT90_INV)
@inline function proc_ip!(k::B64, buf, scr)
    @inbounds for f in 0:(length(buf) ÷ 64 - 1); butterfly64!(buf, buf, scr, 64f, k.tw, k.rot); end
end
@inline function proc_oop!(k::B64, out, inp, scr)
    @inbounds for f in 0:(length(inp) ÷ 64 - 1); butterfly64!(out, inp, out, 64f, k.tw, k.rot); end  # out = workspace
end

# ---- leaf: Butterfly256 (32x8 two-phase, faithful rustfft port; needs scratch ≥ its length) ----
struct B256 <: Kernel
    n::Int; tw::Vector{V4f}; tw32::NTuple{6, V4f}; rot::V4f
end
B256(fwd::Bool) = B256(256, bf256_phase1_tw(fwd), bf256_bf32_tw(fwd), fwd ? _ROT90_FWD : _ROT90_INV)
@inline function proc_ip!(k::B256, buf, scr)
    @inbounds for f in 0:(length(buf) ÷ 256 - 1); butterfly256!(buf, buf, scr, 256f, k.tw, k.tw32, k.rot); end
end
@inline function proc_oop!(k::B256, out, inp, scr)
    @inbounds for f in 0:(length(inp) ÷ 256 - 1); butterfly256!(out, inp, out, 256f, k.tw, k.tw32, k.rot); end  # out = workspace
end

# ---- leaf: Butterfly512 (32x16 two-phase, faithful rustfft port; needs scratch ≥ its length) ----
struct B512 <: Kernel
    n::Int; tw::Vector{V4f}; tw16::NTuple{2, V4f}; tw32::NTuple{6, V4f}; rot::V4f
end
B512(fwd::Bool) = B512(512, bf512_phase1_tw(fwd), bf512_bf16_tw(fwd), bf256_bf32_tw(fwd), fwd ? _ROT90_FWD : _ROT90_INV)
@inline function proc_ip!(k::B512, buf, scr)
    @inbounds for f in 0:(length(buf) ÷ 512 - 1); butterfly512!(buf, buf, scr, 512f, k.tw, k.tw16, k.tw32, k.rot); end
end
@inline function proc_oop!(k::B512, out, inp, scr)
    @inbounds for f in 0:(length(inp) ÷ 512 - 1); butterfly512!(out, inp, out, 512f, k.tw, k.tw16, k.tw32, k.rot); end  # out = workspace
end

# ---- column-butterfly + transpose passes (R=3,4,5; even M; per-FFT at offset `o`) ----
@inline function _colbf3!(buf, o, ::Val{M}, tw, bf3) where {M}
    @inbounds for c in 0:(M ÷ 2 - 1)
        ib = o + 2c
        r = avx_column_butterfly3(avx_load_complex(buf, ib), avx_load_complex(buf, ib + M), avx_load_complex(buf, ib + 2M), bf3)
        avx_store_complex!(buf, ib, r[1])
        avx_store_complex!(buf, ib + M, avx_mul_complex(tw[c * 2 + 1], r[2]))
        avx_store_complex!(buf, ib + 2M, avx_mul_complex(tw[c * 2 + 2], r[3]))
    end
    if isodd(M)                                          # leftover column M-1 as a partial V2f (1 complex)
        @inbounds begin
            ib = o + (M - 1); tc = (M ÷ 2) * 2
            r = avx_column_butterfly3(avx_load_partial1(buf, ib), avx_load_partial1(buf, ib + M), avx_load_partial1(buf, ib + 2M), avx_lo(bf3))
            avx_store_partial1!(buf, ib, r[1])
            avx_store_partial1!(buf, ib + M, avx_mul_complex(avx_lo(tw[tc + 1]), r[2]))
            avx_store_partial1!(buf, ib + 2M, avx_mul_complex(avx_lo(tw[tc + 2]), r[3]))
        end
    end
end
@inline function _trans3!(out, oo, buf, o, ::Val{M}) where {M}
    @inbounds for c in 0:(M ÷ 2 - 1)
        ib = o + 2c; ob = oo + 6c
        t = avx_transpose3_packed(avx_load_complex(buf, ib), avx_load_complex(buf, ib + M), avx_load_complex(buf, ib + 2M))
        avx_store_complex!(out, ob, t[1]); avx_store_complex!(out, ob + 2, t[2]); avx_store_complex!(out, ob + 4, t[3])
    end
    if isodd(M)                                          # transpose of a 3x1 column = 3 contiguous complex at oo+3(M-1)
        @inbounds begin
            ib = o + (M - 1); ob = oo + 3 * (M - 1)
            avx_store_partial1!(out, ob, avx_load_partial1(buf, ib)); avx_store_partial1!(out, ob + 1, avx_load_partial1(buf, ib + M)); avx_store_partial1!(out, ob + 2, avx_load_partial1(buf, ib + 2M))
        end
    end
end
@inline function _colbf4!(buf, o, ::Val{M}, tw, rot) where {M}
    @inbounds for c in 0:(M ÷ 2 - 1)
        ib = o + 2c
        r = avx_column_butterfly4(avx_load_complex(buf, ib), avx_load_complex(buf, ib + M), avx_load_complex(buf, ib + 2M), avx_load_complex(buf, ib + 3M), rot)
        avx_store_complex!(buf, ib, r[1])
        avx_store_complex!(buf, ib + M, avx_mul_complex(tw[c * 3 + 1], r[2]))
        avx_store_complex!(buf, ib + 2M, avx_mul_complex(tw[c * 3 + 2], r[3]))
        avx_store_complex!(buf, ib + 3M, avx_mul_complex(tw[c * 3 + 3], r[4]))
    end
end
@inline function _trans4!(out, oo, buf, o, ::Val{M}) where {M}
    @inbounds for c in 0:(M ÷ 2 - 1)
        ib = o + 2c; ob = oo + 8c
        t = avx_transpose4_packed(avx_load_complex(buf, ib), avx_load_complex(buf, ib + M), avx_load_complex(buf, ib + 2M), avx_load_complex(buf, ib + 3M))
        avx_store_complex!(out, ob, t[1]); avx_store_complex!(out, ob + 2, t[2]); avx_store_complex!(out, ob + 4, t[3]); avx_store_complex!(out, ob + 6, t[4])
    end
end
# NOTE: the odd-M leftover-column tail lives in a SEPARATE @noinline helper, called behind a
# compile-time `isodd(M) &&` guard. For even M (the common 2ᵏ·5ᵐ chains — 1000/5000/10000) the guard
# folds to `false`, the call is DCE'd, and `_colbf5!`'s inlinable body is BYTE-IDENTICAL to the
# pre-tail (master) version. Keeping the tail inline here instead bloated the typed-IR cost the
# inliner sees and regressed even-M 1000 ~0.98→0.87 (measured) even though LLVM later DCE'd the dead
# branch — so the tail MUST stay out-of-line.
@inline function _colbf5!(buf, o, ::Val{M}, tw, t0, t1) where {M}
    @inbounds for c in 0:(M ÷ 2 - 1)
        ib = o + 2c
        r = avx_column_butterfly5(avx_load_complex(buf, ib), avx_load_complex(buf, ib + M), avx_load_complex(buf, ib + 2M), avx_load_complex(buf, ib + 3M), avx_load_complex(buf, ib + 4M), t0, t1)
        avx_store_complex!(buf, ib, r[1])
        avx_store_complex!(buf, ib + M, avx_mul_complex(tw[c * 4 + 1], r[2]))
        avx_store_complex!(buf, ib + 2M, avx_mul_complex(tw[c * 4 + 2], r[3]))
        avx_store_complex!(buf, ib + 3M, avx_mul_complex(tw[c * 4 + 3], r[4]))
        avx_store_complex!(buf, ib + 4M, avx_mul_complex(tw[c * 4 + 4], r[5]))
    end
    isodd(M) && _colbf5_oddtail!(buf, o, Val(M), tw, t0, t1)
end
@noinline function _colbf5_oddtail!(buf, o, ::Val{M}, tw, t0, t1) where {M}   # leftover column M-1, partial V2f
    @inbounds begin
        ib = o + (M - 1); tc = (M ÷ 2) * 4
        r = avx_column_butterfly5(avx_load_partial1(buf, ib), avx_load_partial1(buf, ib + M), avx_load_partial1(buf, ib + 2M), avx_load_partial1(buf, ib + 3M), avx_load_partial1(buf, ib + 4M), avx_lo(t0), avx_lo(t1))
        avx_store_partial1!(buf, ib, r[1])
        avx_store_partial1!(buf, ib + M, avx_mul_complex(avx_lo(tw[tc + 1]), r[2])); avx_store_partial1!(buf, ib + 2M, avx_mul_complex(avx_lo(tw[tc + 2]), r[3]))
        avx_store_partial1!(buf, ib + 3M, avx_mul_complex(avx_lo(tw[tc + 3]), r[4])); avx_store_partial1!(buf, ib + 4M, avx_mul_complex(avx_lo(tw[tc + 4]), r[5]))
    end
end
@inline function _trans5!(out, oo, buf, o, ::Val{M}) where {M}
    @inbounds for c in 0:(M ÷ 2 - 1)
        ib = o + 2c; ob = oo + 10c
        t = avx_transpose5_packed(avx_load_complex(buf, ib), avx_load_complex(buf, ib + M), avx_load_complex(buf, ib + 2M), avx_load_complex(buf, ib + 3M), avx_load_complex(buf, ib + 4M))
        avx_store_complex!(out, ob, t[1]); avx_store_complex!(out, ob + 2, t[2]); avx_store_complex!(out, ob + 4, t[3]); avx_store_complex!(out, ob + 6, t[4]); avx_store_complex!(out, ob + 8, t[5])
    end
    isodd(M) && _trans5_oddtail!(out, oo, buf, o, Val(M))
end
@noinline function _trans5_oddtail!(out, oo, buf, o, ::Val{M}) where {M}   # transpose of a 5x1 column → 5 contiguous complex
    @inbounds begin
        ib = o + (M - 1); ob = oo + 5 * (M - 1)
        for r in 0:4; avx_store_partial1!(out, ob + r, avx_load_partial1(buf, ib + r * M)); end
    end
end
# ---- radix-7 passes (reuses verified avx_column_butterfly7) ----
@inline function _colbf7!(buf, o, ::Val{M}, tw, t0, t1, t2) where {M}
    @inbounds for c in 0:(M ÷ 2 - 1)
        ib = o + 2c
        r = avx_column_butterfly7(avx_load_complex(buf, ib), avx_load_complex(buf, ib + M), avx_load_complex(buf, ib + 2M), avx_load_complex(buf, ib + 3M),
                                  avx_load_complex(buf, ib + 4M), avx_load_complex(buf, ib + 5M), avx_load_complex(buf, ib + 6M), t0, t1, t2)
        avx_store_complex!(buf, ib, r[1])
        avx_store_complex!(buf, ib + M, avx_mul_complex(tw[c * 6 + 1], r[2])); avx_store_complex!(buf, ib + 2M, avx_mul_complex(tw[c * 6 + 2], r[3]))
        avx_store_complex!(buf, ib + 3M, avx_mul_complex(tw[c * 6 + 3], r[4])); avx_store_complex!(buf, ib + 4M, avx_mul_complex(tw[c * 6 + 4], r[5]))
        avx_store_complex!(buf, ib + 5M, avx_mul_complex(tw[c * 6 + 5], r[6])); avx_store_complex!(buf, ib + 6M, avx_mul_complex(tw[c * 6 + 6], r[7]))
    end
    isodd(M) && _colbf7_oddtail!(buf, o, Val(M), tw, t0, t1, t2)
end
@noinline function _colbf7_oddtail!(buf, o, ::Val{M}, tw, t0, t1, t2) where {M}   # leftover column M-1, partial V2f
    @inbounds begin
        ib = o + (M - 1); tc = (M ÷ 2) * 6
        r = avx_column_butterfly7(avx_load_partial1(buf, ib), avx_load_partial1(buf, ib + M), avx_load_partial1(buf, ib + 2M), avx_load_partial1(buf, ib + 3M),
                                  avx_load_partial1(buf, ib + 4M), avx_load_partial1(buf, ib + 5M), avx_load_partial1(buf, ib + 6M), avx_lo(t0), avx_lo(t1), avx_lo(t2))
        avx_store_partial1!(buf, ib, r[1])
        avx_store_partial1!(buf, ib + M, avx_mul_complex(avx_lo(tw[tc + 1]), r[2])); avx_store_partial1!(buf, ib + 2M, avx_mul_complex(avx_lo(tw[tc + 2]), r[3]))
        avx_store_partial1!(buf, ib + 3M, avx_mul_complex(avx_lo(tw[tc + 3]), r[4])); avx_store_partial1!(buf, ib + 4M, avx_mul_complex(avx_lo(tw[tc + 4]), r[5]))
        avx_store_partial1!(buf, ib + 5M, avx_mul_complex(avx_lo(tw[tc + 5]), r[6])); avx_store_partial1!(buf, ib + 6M, avx_mul_complex(avx_lo(tw[tc + 6]), r[7]))
    end
end
@inline function _trans7!(out, oo, buf, o, ::Val{M}) where {M}
    @inbounds for c in 0:(M ÷ 2 - 1)
        ib = o + 2c; ob = oo + 14c
        t = avx_transpose7_packed(avx_load_complex(buf, ib), avx_load_complex(buf, ib + M), avx_load_complex(buf, ib + 2M), avx_load_complex(buf, ib + 3M),
                                  avx_load_complex(buf, ib + 4M), avx_load_complex(buf, ib + 5M), avx_load_complex(buf, ib + 6M))
        avx_store_complex!(out, ob, t[1]); avx_store_complex!(out, ob + 2, t[2]); avx_store_complex!(out, ob + 4, t[3]); avx_store_complex!(out, ob + 6, t[4])
        avx_store_complex!(out, ob + 8, t[5]); avx_store_complex!(out, ob + 10, t[6]); avx_store_complex!(out, ob + 12, t[7])
    end
    isodd(M) && _trans7_oddtail!(out, oo, buf, o, Val(M))
end
@noinline function _trans7_oddtail!(out, oo, buf, o, ::Val{M}) where {M}   # transpose of a 7x1 column → 7 contiguous complex
    @inbounds begin
        ib = o + (M - 1); ob = oo + 7 * (M - 1)
        for r in 0:6; avx_store_partial1!(out, ob + r, avx_load_partial1(buf, ib + r * M)); end
    end
end

# ---- radix-2 passes (the F64 analogue of MR2W8; carries a factor of 2 over an odd 5-power core, e.g.
# 250=MR2(MR5(B25))). Odd-M partial-V2f tail, same structure as radix-5/7. ----
@inline function _colbf2!(buf, o, ::Val{M}, tw) where {M}
    @inbounds for c in 0:(M ÷ 2 - 1)
        ib = o + 2c
        s, d = avx_butterfly2(avx_load_complex(buf, ib), avx_load_complex(buf, ib + M))
        avx_store_complex!(buf, ib, s)
        avx_store_complex!(buf, ib + M, avx_mul_complex(tw[c + 1], d))
    end
    isodd(M) && _colbf2_oddtail!(buf, o, Val(M), tw)
end
@noinline function _colbf2_oddtail!(buf, o, ::Val{M}, tw) where {M}   # leftover column M-1, partial V2f
    @inbounds begin
        ib = o + (M - 1); tc = M ÷ 2
        s, d = avx_butterfly2(avx_load_partial1(buf, ib), avx_load_partial1(buf, ib + M))
        avx_store_partial1!(buf, ib, s)
        avx_store_partial1!(buf, ib + M, avx_mul_complex(avx_lo(tw[tc + 1]), d))
    end
end
@inline function _trans2!(out, oo, buf, o, ::Val{M}) where {M}
    @inbounds for c in 0:(M ÷ 2 - 1)
        ib = o + 2c; ob = oo + 4c
        t = avx_transpose_2x2(avx_load_complex(buf, ib), avx_load_complex(buf, ib + M))
        avx_store_complex!(out, ob, t[1]); avx_store_complex!(out, ob + 2, t[2])
    end
    isodd(M) && _trans2_oddtail!(out, oo, buf, o, Val(M))
end
@noinline function _trans2_oddtail!(out, oo, buf, o, ::Val{M}) where {M}   # transpose of a 2x1 column → 2 contiguous complex
    @inbounds begin
        ib = o + (M - 1); ob = oo + 2 * (M - 1)
        for r in 0:1; avx_store_partial1!(out, ob + r, avx_load_partial1(buf, ib + r * M)); end
    end
end

# ---- radix-8 passes ("blazing fast" 8xn) ----
@inline function _colbf8!(buf, o, ::Val{M}, tw, rot) where {M}
    @inbounds for c in 0:(M ÷ 2 - 1)
        ib = o + 2c
        r = avx_column_butterfly8(avx_load_complex(buf, ib), avx_load_complex(buf, ib + M), avx_load_complex(buf, ib + 2M), avx_load_complex(buf, ib + 3M),
                                  avx_load_complex(buf, ib + 4M), avx_load_complex(buf, ib + 5M), avx_load_complex(buf, ib + 6M), avx_load_complex(buf, ib + 7M), rot)
        avx_store_complex!(buf, ib, r[1])
        avx_store_complex!(buf, ib + M, avx_mul_complex(tw[c * 7 + 1], r[2])); avx_store_complex!(buf, ib + 2M, avx_mul_complex(tw[c * 7 + 2], r[3]))
        avx_store_complex!(buf, ib + 3M, avx_mul_complex(tw[c * 7 + 3], r[4])); avx_store_complex!(buf, ib + 4M, avx_mul_complex(tw[c * 7 + 4], r[5]))
        avx_store_complex!(buf, ib + 5M, avx_mul_complex(tw[c * 7 + 5], r[6])); avx_store_complex!(buf, ib + 6M, avx_mul_complex(tw[c * 7 + 6], r[7]))
        avx_store_complex!(buf, ib + 7M, avx_mul_complex(tw[c * 7 + 7], r[8]))
    end
end
@inline function _trans8!(out, oo, buf, o, ::Val{M}) where {M}
    @inbounds for c in 0:(M ÷ 2 - 1)
        ib = o + 2c; ob = oo + 16c
        t = avx_transpose8_packed(avx_load_complex(buf, ib), avx_load_complex(buf, ib + M), avx_load_complex(buf, ib + 2M), avx_load_complex(buf, ib + 3M),
                                  avx_load_complex(buf, ib + 4M), avx_load_complex(buf, ib + 5M), avx_load_complex(buf, ib + 6M), avx_load_complex(buf, ib + 7M))
        avx_store_complex!(out, ob, t[1]); avx_store_complex!(out, ob + 2, t[2]); avx_store_complex!(out, ob + 4, t[3]); avx_store_complex!(out, ob + 6, t[4])
        avx_store_complex!(out, ob + 8, t[5]); avx_store_complex!(out, ob + 10, t[6]); avx_store_complex!(out, ob + 12, t[7]); avx_store_complex!(out, ob + 14, t[8])
    end
end

# ---- radix-6 passes ----
@inline function _colbf6!(buf, o, ::Val{M}, tw, bf3) where {M}
    @inbounds for c in 0:(M ÷ 2 - 1)
        ib = o + 2c
        r = avx_column_butterfly6((avx_load_complex(buf, ib), avx_load_complex(buf, ib + M), avx_load_complex(buf, ib + 2M),
                                   avx_load_complex(buf, ib + 3M), avx_load_complex(buf, ib + 4M), avx_load_complex(buf, ib + 5M)), bf3)
        avx_store_complex!(buf, ib, r[1])
        avx_store_complex!(buf, ib + M, avx_mul_complex(tw[c * 5 + 1], r[2])); avx_store_complex!(buf, ib + 2M, avx_mul_complex(tw[c * 5 + 2], r[3]))
        avx_store_complex!(buf, ib + 3M, avx_mul_complex(tw[c * 5 + 3], r[4])); avx_store_complex!(buf, ib + 4M, avx_mul_complex(tw[c * 5 + 4], r[5]))
        avx_store_complex!(buf, ib + 5M, avx_mul_complex(tw[c * 5 + 5], r[6]))
    end
end
@inline function _trans6!(out, oo, buf, o, ::Val{M}) where {M}
    @inbounds for c in 0:(M ÷ 2 - 1)
        ib = o + 2c; ob = oo + 12c
        t = avx_transpose6_packed(avx_load_complex(buf, ib), avx_load_complex(buf, ib + M), avx_load_complex(buf, ib + 2M), avx_load_complex(buf, ib + 3M), avx_load_complex(buf, ib + 4M), avx_load_complex(buf, ib + 5M))
        avx_store_complex!(out, ob, t[1]); avx_store_complex!(out, ob + 2, t[2]); avx_store_complex!(out, ob + 4, t[3]); avx_store_complex!(out, ob + 6, t[4]); avx_store_complex!(out, ob + 8, t[5]); avx_store_complex!(out, ob + 10, t[6])
    end
end
# ---- radix-9 passes ----
@inline function _colbf9!(buf, o, ::Val{M}, tw, tw1, tw2, tw3, bf3) where {M}
    @inbounds for c in 0:(M ÷ 2 - 1)
        ib = o + 2c
        r = avx_column_butterfly9(avx_load_complex(buf, ib), avx_load_complex(buf, ib + M), avx_load_complex(buf, ib + 2M), avx_load_complex(buf, ib + 3M), avx_load_complex(buf, ib + 4M),
                                  avx_load_complex(buf, ib + 5M), avx_load_complex(buf, ib + 6M), avx_load_complex(buf, ib + 7M), avx_load_complex(buf, ib + 8M), tw1, tw2, tw3, bf3)
        avx_store_complex!(buf, ib, r[1])
        avx_store_complex!(buf, ib + M, avx_mul_complex(tw[c * 8 + 1], r[2])); avx_store_complex!(buf, ib + 2M, avx_mul_complex(tw[c * 8 + 2], r[3]))
        avx_store_complex!(buf, ib + 3M, avx_mul_complex(tw[c * 8 + 3], r[4])); avx_store_complex!(buf, ib + 4M, avx_mul_complex(tw[c * 8 + 4], r[5]))
        avx_store_complex!(buf, ib + 5M, avx_mul_complex(tw[c * 8 + 5], r[6])); avx_store_complex!(buf, ib + 6M, avx_mul_complex(tw[c * 8 + 6], r[7]))
        avx_store_complex!(buf, ib + 7M, avx_mul_complex(tw[c * 8 + 7], r[8])); avx_store_complex!(buf, ib + 8M, avx_mul_complex(tw[c * 8 + 8], r[9]))
    end
    if isodd(M)                                          # leftover column M-1 as a partial V2f (1 complex)
        @inbounds begin
            ib = o + (M - 1); tc = (M ÷ 2) * 8
            r = avx_column_butterfly9(avx_load_partial1(buf, ib), avx_load_partial1(buf, ib + M), avx_load_partial1(buf, ib + 2M), avx_load_partial1(buf, ib + 3M), avx_load_partial1(buf, ib + 4M),
                                      avx_load_partial1(buf, ib + 5M), avx_load_partial1(buf, ib + 6M), avx_load_partial1(buf, ib + 7M), avx_load_partial1(buf, ib + 8M), avx_lo(tw1), avx_lo(tw2), avx_lo(tw3), avx_lo(bf3))
            avx_store_partial1!(buf, ib, r[1])
            avx_store_partial1!(buf, ib + M, avx_mul_complex(avx_lo(tw[tc + 1]), r[2])); avx_store_partial1!(buf, ib + 2M, avx_mul_complex(avx_lo(tw[tc + 2]), r[3]))
            avx_store_partial1!(buf, ib + 3M, avx_mul_complex(avx_lo(tw[tc + 3]), r[4])); avx_store_partial1!(buf, ib + 4M, avx_mul_complex(avx_lo(tw[tc + 4]), r[5]))
            avx_store_partial1!(buf, ib + 5M, avx_mul_complex(avx_lo(tw[tc + 5]), r[6])); avx_store_partial1!(buf, ib + 6M, avx_mul_complex(avx_lo(tw[tc + 6]), r[7]))
            avx_store_partial1!(buf, ib + 7M, avx_mul_complex(avx_lo(tw[tc + 7]), r[8])); avx_store_partial1!(buf, ib + 8M, avx_mul_complex(avx_lo(tw[tc + 8]), r[9]))
        end
    end
end
@inline function _trans9!(out, oo, buf, o, ::Val{M}) where {M}
    @inbounds for c in 0:(M ÷ 2 - 1)
        ib = o + 2c; ob = oo + 18c
        t = avx_transpose9_packed(avx_load_complex(buf, ib), avx_load_complex(buf, ib + M), avx_load_complex(buf, ib + 2M), avx_load_complex(buf, ib + 3M), avx_load_complex(buf, ib + 4M),
                                  avx_load_complex(buf, ib + 5M), avx_load_complex(buf, ib + 6M), avx_load_complex(buf, ib + 7M), avx_load_complex(buf, ib + 8M))
        avx_store_complex!(out, ob, t[1]); avx_store_complex!(out, ob + 2, t[2]); avx_store_complex!(out, ob + 4, t[3]); avx_store_complex!(out, ob + 6, t[4]); avx_store_complex!(out, ob + 8, t[5])
        avx_store_complex!(out, ob + 10, t[6]); avx_store_complex!(out, ob + 12, t[7]); avx_store_complex!(out, ob + 14, t[8]); avx_store_complex!(out, ob + 16, t[9])
    end
    if isodd(M)                                          # transpose of a 9x1 column = 9 contiguous complex at oo+9(M-1)
        @inbounds begin
            ib = o + (M - 1); ob = oo + 9 * (M - 1)
            for r in 0:8; avx_store_partial1!(out, ob + r, avx_load_partial1(buf, ib + r * M)); end
        end
    end
end

# ---- radix-12 passes (preferred fast radix) ----
@inline function _colbf12!(buf, o, ::Val{M}, tw, bf3, rot) where {M}
    @inbounds for c in 0:(M ÷ 2 - 1)
        ib = o + 2c
        r = avx_column_butterfly12(avx_load_complex(buf, ib), avx_load_complex(buf, ib + M), avx_load_complex(buf, ib + 2M), avx_load_complex(buf, ib + 3M),
                                   avx_load_complex(buf, ib + 4M), avx_load_complex(buf, ib + 5M), avx_load_complex(buf, ib + 6M), avx_load_complex(buf, ib + 7M),
                                   avx_load_complex(buf, ib + 8M), avx_load_complex(buf, ib + 9M), avx_load_complex(buf, ib + 10M), avx_load_complex(buf, ib + 11M), bf3, rot)
        avx_store_complex!(buf, ib, r[1])
        avx_store_complex!(buf, ib + M, avx_mul_complex(tw[c * 11 + 1], r[2])); avx_store_complex!(buf, ib + 2M, avx_mul_complex(tw[c * 11 + 2], r[3]))
        avx_store_complex!(buf, ib + 3M, avx_mul_complex(tw[c * 11 + 3], r[4])); avx_store_complex!(buf, ib + 4M, avx_mul_complex(tw[c * 11 + 4], r[5]))
        avx_store_complex!(buf, ib + 5M, avx_mul_complex(tw[c * 11 + 5], r[6])); avx_store_complex!(buf, ib + 6M, avx_mul_complex(tw[c * 11 + 6], r[7]))
        avx_store_complex!(buf, ib + 7M, avx_mul_complex(tw[c * 11 + 7], r[8])); avx_store_complex!(buf, ib + 8M, avx_mul_complex(tw[c * 11 + 8], r[9]))
        avx_store_complex!(buf, ib + 9M, avx_mul_complex(tw[c * 11 + 9], r[10])); avx_store_complex!(buf, ib + 10M, avx_mul_complex(tw[c * 11 + 10], r[11]))
        avx_store_complex!(buf, ib + 11M, avx_mul_complex(tw[c * 11 + 11], r[12]))
    end
end
@inline function _trans12!(out, oo, buf, o, ::Val{M}) where {M}
    @inbounds for c in 0:(M ÷ 2 - 1)
        ib = o + 2c; ob = oo + 24c
        t = avx_transpose12_packed(avx_load_complex(buf, ib), avx_load_complex(buf, ib + M), avx_load_complex(buf, ib + 2M), avx_load_complex(buf, ib + 3M),
                                   avx_load_complex(buf, ib + 4M), avx_load_complex(buf, ib + 5M), avx_load_complex(buf, ib + 6M), avx_load_complex(buf, ib + 7M),
                                   avx_load_complex(buf, ib + 8M), avx_load_complex(buf, ib + 9M), avx_load_complex(buf, ib + 10M), avx_load_complex(buf, ib + 11M))
        avx_store_complex!(out, ob, t[1]); avx_store_complex!(out, ob + 2, t[2]); avx_store_complex!(out, ob + 4, t[3]); avx_store_complex!(out, ob + 6, t[4])
        avx_store_complex!(out, ob + 8, t[5]); avx_store_complex!(out, ob + 10, t[6]); avx_store_complex!(out, ob + 12, t[7]); avx_store_complex!(out, ob + 14, t[8])
        avx_store_complex!(out, ob + 16, t[9]); avx_store_complex!(out, ob + 18, t[10]); avx_store_complex!(out, ob + 20, t[11]); avx_store_complex!(out, ob + 22, t[12])
    end
end

# mixedradix twiddles: make_mixedradix_twiddle_chunk(c*2, y, n) for c in 0:M/2-1, y in 1:R-1 → [c*(R-1)+y]
# Pad to cld(M,2) chunks: for ODD M this adds one trailing chunk whose LO lane is the twiddle for the
# leftover column M-1 (HI lane = column M, a discarded dummy). The even-M V4f loop never reads it; the
# odd-M partial-V2f tail reads avx_lo(tw[(M÷2)*(R-1) + y]). Even M ⇒ cld(M,2)==M÷2, unchanged.
function mr_twiddles(R, M, n, fwd)
    [avx_mixedradix_twiddle_chunk(c * 2, y, n, fwd) for c in 0:(cld(M, 2) - 1) for y in 1:(R - 1)]
end

# ---- MixedRadix3 (R=3) ----
struct MR3{M, I <: Kernel} <: Kernel
    inner::I; tw::Vector{V4f}; bf3::V4f
end
klen(::MR3{M}) where {M} = 3M
function MR3(inner::Kernel, fwd::Bool)
    M = klen(inner)
    MR3{M, typeof(inner)}(inner, mr_twiddles(3, M, 3M, fwd), avx_broadcast_twiddle(1, 3, fwd))
end
@inline function proc_ip!(k::MR3{M}, buf, scr) where {M}
    n = 3M; cnt = length(buf) ÷ n
    @inbounds for f in 0:(cnt - 1); _colbf3!(buf, f * n, Val(M), k.tw, k.bf3); end
    proc_oop!(k.inner, scr, buf, scr)
    @inbounds for f in 0:(cnt - 1); _trans3!(buf, f * n, scr, f * n, Val(M)); end
end
@inline function proc_oop!(k::MR3{M}, out, inp, scr) where {M}
    n = 3M; cnt = length(inp) ÷ n
    @inbounds for f in 0:(cnt - 1); _colbf3!(inp, f * n, Val(M), k.tw, k.bf3); end
    proc_ip!(k.inner, inp, scr)
    @inbounds for f in 0:(cnt - 1); _trans3!(out, f * n, inp, f * n, Val(M)); end
end

# ---- MixedRadix6 (R=6) ----
struct MR6{M, I <: Kernel} <: Kernel
    inner::I; tw::Vector{V4f}; bf3::V4f
end
klen(::MR6{M}) where {M} = 6M
function MR6(inner::Kernel, fwd::Bool)
    M = klen(inner)
    MR6{M, typeof(inner)}(inner, mr_twiddles(6, M, 6M, fwd), avx_broadcast_twiddle(1, 3, fwd))
end
@inline function proc_ip!(k::MR6{M}, buf, scr) where {M}
    n = 6M; cnt = length(buf) ÷ n
    @inbounds for f in 0:(cnt - 1); _colbf6!(buf, f * n, Val(M), k.tw, k.bf3); end
    proc_oop!(k.inner, scr, buf, scr)
    @inbounds for f in 0:(cnt - 1); _trans6!(buf, f * n, scr, f * n, Val(M)); end
end
@inline function proc_oop!(k::MR6{M}, out, inp, scr) where {M}
    n = 6M; cnt = length(inp) ÷ n
    @inbounds for f in 0:(cnt - 1); _colbf6!(inp, f * n, Val(M), k.tw, k.bf3); end
    proc_ip!(k.inner, inp, scr)
    @inbounds for f in 0:(cnt - 1); _trans6!(out, f * n, inp, f * n, Val(M)); end
end

# ---- MixedRadix9 (R=9) ----
struct MR9{M, I <: Kernel} <: Kernel
    inner::I; tw::Vector{V4f}; tw1::V4f; tw2::V4f; tw3::V4f; bf3::V4f
end
klen(::MR9{M}) where {M} = 9M
function MR9(inner::Kernel, fwd::Bool)
    M = klen(inner)
    MR9{M, typeof(inner)}(inner, mr_twiddles(9, M, 9M, fwd), avx_broadcast_twiddle(1, 9, fwd), avx_broadcast_twiddle(2, 9, fwd), avx_broadcast_twiddle(4, 9, fwd), avx_broadcast_twiddle(1, 3, fwd))
end
@inline function proc_ip!(k::MR9{M}, buf, scr) where {M}
    n = 9M; cnt = length(buf) ÷ n
    @inbounds for f in 0:(cnt - 1); _colbf9!(buf, f * n, Val(M), k.tw, k.tw1, k.tw2, k.tw3, k.bf3); end
    proc_oop!(k.inner, scr, buf, scr)
    @inbounds for f in 0:(cnt - 1); _trans9!(buf, f * n, scr, f * n, Val(M)); end
end
@inline function proc_oop!(k::MR9{M}, out, inp, scr) where {M}
    n = 9M; cnt = length(inp) ÷ n
    @inbounds for f in 0:(cnt - 1); _colbf9!(inp, f * n, Val(M), k.tw, k.tw1, k.tw2, k.tw3, k.bf3); end
    proc_ip!(k.inner, inp, scr)
    @inbounds for f in 0:(cnt - 1); _trans9!(out, f * n, inp, f * n, Val(M)); end
end

# ---- MixedRadix12 (R=12) — preferred fast radix (good-thomas cb12) ----
struct MR12{M, I <: Kernel} <: Kernel
    inner::I; tw::Vector{V4f}; bf3::V4f; rot::V4f
end
klen(::MR12{M}) where {M} = 12M
function MR12(inner::Kernel, fwd::Bool)
    M = klen(inner)
    MR12{M, typeof(inner)}(inner, mr_twiddles(12, M, 12M, fwd), avx_broadcast_twiddle(1, 3, fwd), fwd ? _ROT90_FWD : _ROT90_INV)
end
@inline function proc_ip!(k::MR12{M}, buf, scr) where {M}
    n = 12M; cnt = length(buf) ÷ n
    @inbounds for f in 0:(cnt - 1); _colbf12!(buf, f * n, Val(M), k.tw, k.bf3, k.rot); end
    proc_oop!(k.inner, scr, buf, scr)
    @inbounds for f in 0:(cnt - 1); _trans12!(buf, f * n, scr, f * n, Val(M)); end
end
@inline function proc_oop!(k::MR12{M}, out, inp, scr) where {M}
    n = 12M; cnt = length(inp) ÷ n
    @inbounds for f in 0:(cnt - 1); _colbf12!(inp, f * n, Val(M), k.tw, k.bf3, k.rot); end
    proc_ip!(k.inner, inp, scr)
    @inbounds for f in 0:(cnt - 1); _trans12!(out, f * n, inp, f * n, Val(M)); end
end

# ---- MixedRadix8 (R=8) — preferred fast radix ----
struct MR8{M, I <: Kernel} <: Kernel
    inner::I; tw::Vector{V4f}; rot::V4f
end
klen(::MR8{M}) where {M} = 8M
function MR8(inner::Kernel, fwd::Bool)
    M = klen(inner)
    MR8{M, typeof(inner)}(inner, mr_twiddles(8, M, 8M, fwd), fwd ? _ROT90_FWD : _ROT90_INV)
end
@inline function proc_ip!(k::MR8{M}, buf, scr) where {M}
    n = 8M; cnt = length(buf) ÷ n
    @inbounds for f in 0:(cnt - 1); _colbf8!(buf, f * n, Val(M), k.tw, k.rot); end
    proc_oop!(k.inner, scr, buf, scr)
    @inbounds for f in 0:(cnt - 1); _trans8!(buf, f * n, scr, f * n, Val(M)); end
end
@inline function proc_oop!(k::MR8{M}, out, inp, scr) where {M}
    n = 8M; cnt = length(inp) ÷ n
    @inbounds for f in 0:(cnt - 1); _colbf8!(inp, f * n, Val(M), k.tw, k.rot); end
    proc_ip!(k.inner, inp, scr)
    @inbounds for f in 0:(cnt - 1); _trans8!(out, f * n, inp, f * n, Val(M)); end
end

# ---- MixedRadix4 (R=4) — M (len_per_row) is a TYPE PARAMETER so the passes const-fold ----
struct MR4{M, I <: Kernel} <: Kernel
    inner::I; tw::Vector{V4f}; rot::V4f
end
klen(::MR4{M}) where {M} = 4M
function MR4(inner::Kernel, fwd::Bool)
    M = klen(inner)
    MR4{M, typeof(inner)}(inner, mr_twiddles(4, M, 4M, fwd), fwd ? _ROT90_FWD : _ROT90_INV)
end
@inline function proc_ip!(k::MR4{M}, buf, scr) where {M}                # ONE scratch reused as workspace
    n = 4M; cnt = length(buf) ÷ n
    @inbounds for f in 0:(cnt - 1); _colbf4!(buf, f * n, Val(M), k.tw, k.rot); end
    proc_oop!(k.inner, scr, buf, scr)                                   # inner: buf→scr, scr also its scratch
    @inbounds for f in 0:(cnt - 1); _trans4!(buf, f * n, scr, f * n, Val(M)); end
end
@inline function proc_oop!(k::MR4{M}, out, inp, scr) where {M}
    n = 4M; cnt = length(inp) ÷ n
    @inbounds for f in 0:(cnt - 1); _colbf4!(inp, f * n, Val(M), k.tw, k.rot); end
    proc_ip!(k.inner, inp, scr)
    @inbounds for f in 0:(cnt - 1); _trans4!(out, f * n, inp, f * n, Val(M)); end
end

# ---- MixedRadix5 (R=5) ----
struct MR5{M, I <: Kernel} <: Kernel
    inner::I; tw::Vector{V4f}; t0::V4f; t1::V4f
end
klen(::MR5{M}) where {M} = 5M
function MR5(inner::Kernel, fwd::Bool)
    M = klen(inner)
    MR5{M, typeof(inner)}(inner, mr_twiddles(5, M, 5M, fwd), avx_broadcast_twiddle(1, 5, fwd), avx_broadcast_twiddle(2, 5, fwd))
end
@inline function proc_ip!(k::MR5{M}, buf, scr) where {M}
    n = 5M; cnt = length(buf) ÷ n
    @inbounds for f in 0:(cnt - 1); _colbf5!(buf, f * n, Val(M), k.tw, k.t0, k.t1); end
    proc_oop!(k.inner, scr, buf, scr)
    @inbounds for f in 0:(cnt - 1); _trans5!(buf, f * n, scr, f * n, Val(M)); end
end
@inline function proc_oop!(k::MR5{M}, out, inp, scr) where {M}
    n = 5M; cnt = length(inp) ÷ n
    @inbounds for f in 0:(cnt - 1); _colbf5!(inp, f * n, Val(M), k.tw, k.t0, k.t1); end
    proc_ip!(k.inner, inp, scr)
    @inbounds for f in 0:(cnt - 1); _trans5!(out, f * n, inp, f * n, Val(M)); end
end

# ---- MixedRadix2 (R=2) — carries a lone factor of 2 over an odd 5-power core (250/500/1000/2000) ----
struct MR2{M, I <: Kernel} <: Kernel
    inner::I; tw::Vector{V4f}
end
klen(::MR2{M}) where {M} = 2M
function MR2(inner::Kernel, fwd::Bool)
    M = klen(inner)
    MR2{M, typeof(inner)}(inner, mr_twiddles(2, M, 2M, fwd))
end
@inline function proc_ip!(k::MR2{M}, buf, scr) where {M}
    n = 2M; cnt = length(buf) ÷ n
    @inbounds for f in 0:(cnt - 1); _colbf2!(buf, f * n, Val(M), k.tw); end
    proc_oop!(k.inner, scr, buf, scr)
    @inbounds for f in 0:(cnt - 1); _trans2!(buf, f * n, scr, f * n, Val(M)); end
end
@inline function proc_oop!(k::MR2{M}, out, inp, scr) where {M}
    n = 2M; cnt = length(inp) ÷ n
    @inbounds for f in 0:(cnt - 1); _colbf2!(inp, f * n, Val(M), k.tw); end
    proc_ip!(k.inner, inp, scr)
    @inbounds for f in 0:(cnt - 1); _trans2!(out, f * n, inp, f * n, Val(M)); end
end

# ---- MixedRadix7 (R=7) ----
struct MR7{M, I <: Kernel} <: Kernel
    inner::I; tw::Vector{V4f}; t0::V4f; t1::V4f; t2::V4f
end
klen(::MR7{M}) where {M} = 7M
function MR7(inner::Kernel, fwd::Bool)
    M = klen(inner)
    MR7{M, typeof(inner)}(inner, mr_twiddles(7, M, 7M, fwd),
        avx_broadcast_twiddle(1, 7, fwd), avx_broadcast_twiddle(2, 7, fwd), avx_broadcast_twiddle(3, 7, fwd))
end
@inline function proc_ip!(k::MR7{M}, buf, scr) where {M}
    n = 7M; cnt = length(buf) ÷ n
    @inbounds for f in 0:(cnt - 1); _colbf7!(buf, f * n, Val(M), k.tw, k.t0, k.t1, k.t2); end
    proc_oop!(k.inner, scr, buf, scr)
    @inbounds for f in 0:(cnt - 1); _trans7!(buf, f * n, scr, f * n, Val(M)); end
end
@inline function proc_oop!(k::MR7{M}, out, inp, scr) where {M}
    n = 7M; cnt = length(inp) ÷ n
    @inbounds for f in 0:(cnt - 1); _colbf7!(inp, f * n, Val(M), k.tw, k.t0, k.t1, k.t2); end
    proc_ip!(k.inner, inp, scr)
    @inbounds for f in 0:(cnt - 1); _trans7!(out, f * n, inp, f * n, Val(M)); end
end

# ---- radix-13 passes (reuses verified avx_column_butterfly13) ----
@inline function _colbf13!(buf, o, ::Val{M}, tw, t0, t1, t2, t3, t4, t5) where {M}
    @inbounds for c in 0:(M ÷ 2 - 1)
        ib = o + 2c
        r = avx_column_butterfly13(avx_load_complex(buf, ib), avx_load_complex(buf, ib + M), avx_load_complex(buf, ib + 2M), avx_load_complex(buf, ib + 3M),
                                   avx_load_complex(buf, ib + 4M), avx_load_complex(buf, ib + 5M), avx_load_complex(buf, ib + 6M), avx_load_complex(buf, ib + 7M),
                                   avx_load_complex(buf, ib + 8M), avx_load_complex(buf, ib + 9M), avx_load_complex(buf, ib + 10M), avx_load_complex(buf, ib + 11M),
                                   avx_load_complex(buf, ib + 12M), t0, t1, t2, t3, t4, t5)
        avx_store_complex!(buf, ib, r[1])
        avx_store_complex!(buf, ib + M, avx_mul_complex(tw[c * 12 + 1], r[2])); avx_store_complex!(buf, ib + 2M, avx_mul_complex(tw[c * 12 + 2], r[3]))
        avx_store_complex!(buf, ib + 3M, avx_mul_complex(tw[c * 12 + 3], r[4])); avx_store_complex!(buf, ib + 4M, avx_mul_complex(tw[c * 12 + 4], r[5]))
        avx_store_complex!(buf, ib + 5M, avx_mul_complex(tw[c * 12 + 5], r[6])); avx_store_complex!(buf, ib + 6M, avx_mul_complex(tw[c * 12 + 6], r[7]))
        avx_store_complex!(buf, ib + 7M, avx_mul_complex(tw[c * 12 + 7], r[8])); avx_store_complex!(buf, ib + 8M, avx_mul_complex(tw[c * 12 + 8], r[9]))
        avx_store_complex!(buf, ib + 9M, avx_mul_complex(tw[c * 12 + 9], r[10])); avx_store_complex!(buf, ib + 10M, avx_mul_complex(tw[c * 12 + 10], r[11]))
        avx_store_complex!(buf, ib + 11M, avx_mul_complex(tw[c * 12 + 11], r[12])); avx_store_complex!(buf, ib + 12M, avx_mul_complex(tw[c * 12 + 12], r[13]))
    end
    isodd(M) && _colbf13_oddtail!(buf, o, Val(M), tw, t0, t1, t2, t3, t4, t5)
end
@noinline function _colbf13_oddtail!(buf, o, ::Val{M}, tw, t0, t1, t2, t3, t4, t5) where {M}   # leftover column M-1, partial V2f
    @inbounds begin
        ib = o + (M - 1); tc = (M ÷ 2) * 12
        r = avx_column_butterfly13(avx_load_partial1(buf, ib), avx_load_partial1(buf, ib + M), avx_load_partial1(buf, ib + 2M), avx_load_partial1(buf, ib + 3M),
                                   avx_load_partial1(buf, ib + 4M), avx_load_partial1(buf, ib + 5M), avx_load_partial1(buf, ib + 6M), avx_load_partial1(buf, ib + 7M),
                                   avx_load_partial1(buf, ib + 8M), avx_load_partial1(buf, ib + 9M), avx_load_partial1(buf, ib + 10M), avx_load_partial1(buf, ib + 11M),
                                   avx_load_partial1(buf, ib + 12M), avx_lo(t0), avx_lo(t1), avx_lo(t2), avx_lo(t3), avx_lo(t4), avx_lo(t5))
        avx_store_partial1!(buf, ib, r[1])
        avx_store_partial1!(buf, ib + M, avx_mul_complex(avx_lo(tw[tc + 1]), r[2])); avx_store_partial1!(buf, ib + 2M, avx_mul_complex(avx_lo(tw[tc + 2]), r[3]))
        avx_store_partial1!(buf, ib + 3M, avx_mul_complex(avx_lo(tw[tc + 3]), r[4])); avx_store_partial1!(buf, ib + 4M, avx_mul_complex(avx_lo(tw[tc + 4]), r[5]))
        avx_store_partial1!(buf, ib + 5M, avx_mul_complex(avx_lo(tw[tc + 5]), r[6])); avx_store_partial1!(buf, ib + 6M, avx_mul_complex(avx_lo(tw[tc + 6]), r[7]))
        avx_store_partial1!(buf, ib + 7M, avx_mul_complex(avx_lo(tw[tc + 7]), r[8])); avx_store_partial1!(buf, ib + 8M, avx_mul_complex(avx_lo(tw[tc + 8]), r[9]))
        avx_store_partial1!(buf, ib + 9M, avx_mul_complex(avx_lo(tw[tc + 9]), r[10])); avx_store_partial1!(buf, ib + 10M, avx_mul_complex(avx_lo(tw[tc + 10]), r[11]))
        avx_store_partial1!(buf, ib + 11M, avx_mul_complex(avx_lo(tw[tc + 11]), r[12])); avx_store_partial1!(buf, ib + 12M, avx_mul_complex(avx_lo(tw[tc + 12]), r[13]))
    end
end
@inline function _trans13!(out, oo, buf, o, ::Val{M}) where {M}
    @inbounds for c in 0:(M ÷ 2 - 1)
        ib = o + 2c; ob = oo + 26c
        t = avx_transpose13_packed(avx_load_complex(buf, ib), avx_load_complex(buf, ib + M), avx_load_complex(buf, ib + 2M), avx_load_complex(buf, ib + 3M),
                                   avx_load_complex(buf, ib + 4M), avx_load_complex(buf, ib + 5M), avx_load_complex(buf, ib + 6M), avx_load_complex(buf, ib + 7M),
                                   avx_load_complex(buf, ib + 8M), avx_load_complex(buf, ib + 9M), avx_load_complex(buf, ib + 10M), avx_load_complex(buf, ib + 11M),
                                   avx_load_complex(buf, ib + 12M))
        avx_store_complex!(out, ob, t[1]); avx_store_complex!(out, ob + 2, t[2]); avx_store_complex!(out, ob + 4, t[3]); avx_store_complex!(out, ob + 6, t[4])
        avx_store_complex!(out, ob + 8, t[5]); avx_store_complex!(out, ob + 10, t[6]); avx_store_complex!(out, ob + 12, t[7]); avx_store_complex!(out, ob + 14, t[8])
        avx_store_complex!(out, ob + 16, t[9]); avx_store_complex!(out, ob + 18, t[10]); avx_store_complex!(out, ob + 20, t[11]); avx_store_complex!(out, ob + 22, t[12]); avx_store_complex!(out, ob + 24, t[13])
    end
    isodd(M) && _trans13_oddtail!(out, oo, buf, o, Val(M))
end
@noinline function _trans13_oddtail!(out, oo, buf, o, ::Val{M}) where {M}   # transpose of a 13x1 column → 13 contiguous complex
    @inbounds begin
        ib = o + (M - 1); ob = oo + 13 * (M - 1)
        for r in 0:12; avx_store_partial1!(out, ob + r, avx_load_partial1(buf, ib + r * M)); end
    end
end

# ---- MixedRadix13 (R=13) ----
struct MR13{M, I <: Kernel} <: Kernel
    inner::I; tw::Vector{V4f}; t0::V4f; t1::V4f; t2::V4f; t3::V4f; t4::V4f; t5::V4f
end
klen(::MR13{M}) where {M} = 13M
function MR13(inner::Kernel, fwd::Bool)
    M = klen(inner)
    MR13{M, typeof(inner)}(inner, mr_twiddles(13, M, 13M, fwd),
        avx_broadcast_twiddle(1, 13, fwd), avx_broadcast_twiddle(2, 13, fwd), avx_broadcast_twiddle(3, 13, fwd),
        avx_broadcast_twiddle(4, 13, fwd), avx_broadcast_twiddle(5, 13, fwd), avx_broadcast_twiddle(6, 13, fwd))
end
@inline function proc_ip!(k::MR13{M}, buf, scr) where {M}
    n = 13M; cnt = length(buf) ÷ n
    @inbounds for f in 0:(cnt - 1); _colbf13!(buf, f * n, Val(M), k.tw, k.t0, k.t1, k.t2, k.t3, k.t4, k.t5); end
    proc_oop!(k.inner, scr, buf, scr)
    @inbounds for f in 0:(cnt - 1); _trans13!(buf, f * n, scr, f * n, Val(M)); end
end
@inline function proc_oop!(k::MR13{M}, out, inp, scr) where {M}
    n = 13M; cnt = length(inp) ÷ n
    @inbounds for f in 0:(cnt - 1); _colbf13!(inp, f * n, Val(M), k.tw, k.t0, k.t1, k.t2, k.t3, k.t4, k.t5); end
    proc_ip!(k.inner, inp, scr)
    @inbounds for f in 0:(cnt - 1); _trans13!(out, f * n, inp, f * n, Val(M)); end
end

# ---- top-level: FFT(x) in place. ONE scratch buffer of size n (inplace_scratch_len). ----
struct RPlan{K <: Kernel, T}; k::K; scr::Vector{Complex{T}}; end
RPlan(k::Kernel) = RPlan(k, Vector{Complex{keltype(k)}}(undef, klen(k)))
function applyplan!(p::RPlan, x)
    proc_ip!(p.k, x, p.scr); x
end
