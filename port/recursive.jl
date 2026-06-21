# Step 6A keystone: recursive inner-FFT composition (mirrors rustfft MixedRadix + inner Fft).
# Each kernel implements proc_ip!(k,buf,scr) (in-place) and proc_oop!(k,out,inp,scr) (out-of-place),
# processing count = length(buf)/len(k) consecutive FFTs. MixedRadix.proc_ip! uses inner.proc_oop! and
# vice-versa (the rustfft in-place/out-of-place alternation), so buf↔scr ping-pong; leaf butterflies
# need no scratch. Even len_per_row only here (no partial column) — odd handled in a later step.
include(joinpath(@__DIR__, "kernels.jl"))
using SIMD: Vec

abstract type Kernel end
klen(k::Kernel) = k.n::Int

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

# ---- column-butterfly + transpose passes (R=4,5; even M; per-FFT at offset `o`) ----
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
end
@inline function _trans5!(out, oo, buf, o, ::Val{M}) where {M}
    @inbounds for c in 0:(M ÷ 2 - 1)
        ib = o + 2c; ob = oo + 10c
        t = avx_transpose5_packed(avx_load_complex(buf, ib), avx_load_complex(buf, ib + M), avx_load_complex(buf, ib + 2M), avx_load_complex(buf, ib + 3M), avx_load_complex(buf, ib + 4M))
        avx_store_complex!(out, ob, t[1]); avx_store_complex!(out, ob + 2, t[2]); avx_store_complex!(out, ob + 4, t[3]); avx_store_complex!(out, ob + 6, t[4]); avx_store_complex!(out, ob + 8, t[5])
    end
end

# mixedradix twiddles: make_mixedradix_twiddle_chunk(c*2, y, n) for c in 0:M/2-1, y in 1:R-1 → [c*(R-1)+y]
function mr_twiddles(R, M, n, fwd)
    [avx_mixedradix_twiddle_chunk(c * 2, y, n, fwd) for c in 0:(M ÷ 2 - 1) for y in 1:(R - 1)]
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
@inline function proc_ip!(k::MR4{M}, buf, scr) where {M}                # ONE scratch (rustfft): reuse buf as workspace
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

# ---- top-level: FFT(x) in place. ONE scratch buffer of size n (rustfft inplace_scratch_len). ----
struct RPlan{K <: Kernel}; k::K; scr::Vector{ComplexF64}; end
RPlan(k::Kernel) = RPlan(k, Vector{ComplexF64}(undef, klen(k)))
function applyplan!(p::RPlan, x)
    proc_ip!(p.k, x, p.scr); x
end
