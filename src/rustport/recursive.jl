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

# ---- column-butterfly + transpose passes (R=3,4,5; even M; per-FFT at offset `o`) ----
@inline function _colbf3!(buf, o, ::Val{M}, tw, bf3) where {M}
    @inbounds for c in 0:(M ÷ 2 - 1)
        ib = o + 2c
        r = avx_column_butterfly3(avx_load_complex(buf, ib), avx_load_complex(buf, ib + M), avx_load_complex(buf, ib + 2M), bf3)
        avx_store_complex!(buf, ib, r[1])
        avx_store_complex!(buf, ib + M, avx_mul_complex(tw[c * 2 + 1], r[2]))
        avx_store_complex!(buf, ib + 2M, avx_mul_complex(tw[c * 2 + 2], r[3]))
    end
end
@inline function _trans3!(out, oo, buf, o, ::Val{M}) where {M}
    @inbounds for c in 0:(M ÷ 2 - 1)
        ib = o + 2c; ob = oo + 6c
        t = avx_transpose3_packed(avx_load_complex(buf, ib), avx_load_complex(buf, ib + M), avx_load_complex(buf, ib + 2M))
        avx_store_complex!(out, ob, t[1]); avx_store_complex!(out, ob + 2, t[2]); avx_store_complex!(out, ob + 4, t[3])
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

# ---- radix-8 passes (rust's "blazing fast" 8xn) ----
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
end
@inline function _trans9!(out, oo, buf, o, ::Val{M}) where {M}
    @inbounds for c in 0:(M ÷ 2 - 1)
        ib = o + 2c; ob = oo + 18c
        t = avx_transpose9_packed(avx_load_complex(buf, ib), avx_load_complex(buf, ib + M), avx_load_complex(buf, ib + 2M), avx_load_complex(buf, ib + 3M), avx_load_complex(buf, ib + 4M),
                                  avx_load_complex(buf, ib + 5M), avx_load_complex(buf, ib + 6M), avx_load_complex(buf, ib + 7M), avx_load_complex(buf, ib + 8M))
        avx_store_complex!(out, ob, t[1]); avx_store_complex!(out, ob + 2, t[2]); avx_store_complex!(out, ob + 4, t[3]); avx_store_complex!(out, ob + 6, t[4]); avx_store_complex!(out, ob + 8, t[5])
        avx_store_complex!(out, ob + 10, t[6]); avx_store_complex!(out, ob + 12, t[7]); avx_store_complex!(out, ob + 14, t[8]); avx_store_complex!(out, ob + 16, t[9])
    end
end

# ---- radix-12 passes (rust's preferred fast radix) ----
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
function mr_twiddles(R, M, n, fwd)
    [avx_mixedradix_twiddle_chunk(c * 2, y, n, fwd) for c in 0:(M ÷ 2 - 1) for y in 1:(R - 1)]
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

# ---- MixedRadix12 (R=12) — rust's preferred fast radix (good-thomas cb12) ----
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

# ---- MixedRadix8 (R=8) — rust's preferred fast radix ----
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
