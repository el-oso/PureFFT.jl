# AVX-512 (W=8, Vec{8,Float64} = 4 complex/vector) kernel set — the differentiator vs RustFFT (AVX2-only).
# Targeted: B64 base + radix-8/12 levels (all W=8-clean: every len_per_row is divisible by CPV=4). These
# are Kernels reusing the keystone's RPlan/applyplan!/proc alternation; only the passes are W=8. W=8 only
# helps small COMPUTE-bound non-pow2 sizes — autoplan times it vs W=4 and keeps it only when it wins.
# (Larger/memory-bound sizes regress at W=8 — see the roadmap.)

# ---- W=8 passes (load/store via V8f; transposes from avxport: avx_transpose8/12_packed(::V8f)) ----
@inline _L8(b, i) = avx_load_complex8(b, i)
@inline _S8(b, i, v) = avx_store_complex8!(b, i, v)
bf64_tw_w8(fwd) = [avx_mixedradix_twiddle_chunk8(cs * 4, r, 64, fwd) for cs in 0:1 for r in 1:7]
mr_twiddles_w8(R, M, n, fwd) = [avx_mixedradix_twiddle_chunk8(c * 4, y, n, fwd) for c in 0:(M ÷ 4 - 1) for y in 1:(R - 1)]

# Butterfly64 at W=8 (out/inp/scr; out===inp ⇒ in-place via scr). Verified bit-exact vs V4f.
function butterfly64_w8!(out, inp, scr, base::Int, tw, rot)
    @inbounds for cs in 0:1; b = base + cs * 4
        m = avx_column_butterfly8(_L8(inp, b), _L8(inp, b+8), _L8(inp, b+16), _L8(inp, b+24), _L8(inp, b+32), _L8(inp, b+40), _L8(inp, b+48), _L8(inp, b+56), rot)
        t = avx_transpose8_packed(m[1], avx_mul_complex(tw[7cs+1], m[2]), avx_mul_complex(tw[7cs+2], m[3]), avx_mul_complex(tw[7cs+3], m[4]), avx_mul_complex(tw[7cs+4], m[5]), avx_mul_complex(tw[7cs+5], m[6]), avx_mul_complex(tw[7cs+6], m[7]), avx_mul_complex(tw[7cs+7], m[8]))
        ob = base + cs * 32; for k in 1:8; _S8(scr, ob + 4(k-1), t[k]); end; end
    @inbounds for cs in 0:1; b = base + cs * 4
        m = avx_column_butterfly8(_L8(scr, b), _L8(scr, b+8), _L8(scr, b+16), _L8(scr, b+24), _L8(scr, b+32), _L8(scr, b+40), _L8(scr, b+48), _L8(scr, b+56), rot)
        for r in 0:7; _S8(out, b + 8r, m[r+1]); end; end
end
# Store loops use Base.Cartesian.@nexprs so the tuple is indexed with LITERAL k/j (no runtime tuple index
# → no bounds-check throw path; matches the V4f kernels + CLAUDE.md rule #1). r[1]/t[k] then constant-fold.
@inline function _colbf8_w8!(buf, o, ::Val{M}, tw, rot) where {M}
    @inbounds for c in 0:(M ÷ 4 - 1); ib = o + 4c
        r = avx_column_butterfly8(_L8(buf,ib), _L8(buf,ib+M), _L8(buf,ib+2M), _L8(buf,ib+3M), _L8(buf,ib+4M), _L8(buf,ib+5M), _L8(buf,ib+6M), _L8(buf,ib+7M), rot)
        _S8(buf, ib, r[1]); Base.Cartesian.@nexprs 7 j -> _S8(buf, ib + j*M, avx_mul_complex(tw[c*7+j], r[j+1])); end
end
@inline function _trans8_w8!(out, oo, buf, o, ::Val{M}) where {M}
    @inbounds for c in 0:(M ÷ 4 - 1); ib = o + 4c; ob = oo + 32c
        t = avx_transpose8_packed(_L8(buf,ib), _L8(buf,ib+M), _L8(buf,ib+2M), _L8(buf,ib+3M), _L8(buf,ib+4M), _L8(buf,ib+5M), _L8(buf,ib+6M), _L8(buf,ib+7M))
        Base.Cartesian.@nexprs 8 k -> _S8(out, ob + 4(k-1), t[k]); end
end
@inline function _colbf12_w8!(buf, o, ::Val{M}, tw, bf3, rot) where {M}
    @inbounds for c in 0:(M ÷ 4 - 1); ib = o + 4c
        r = avx_column_butterfly12(_L8(buf,ib), _L8(buf,ib+M), _L8(buf,ib+2M), _L8(buf,ib+3M), _L8(buf,ib+4M), _L8(buf,ib+5M), _L8(buf,ib+6M), _L8(buf,ib+7M), _L8(buf,ib+8M), _L8(buf,ib+9M), _L8(buf,ib+10M), _L8(buf,ib+11M), bf3, rot)
        _S8(buf, ib, r[1]); Base.Cartesian.@nexprs 11 j -> _S8(buf, ib + j*M, avx_mul_complex(tw[c*11+j], r[j+1])); end
end
@inline function _trans12_w8!(out, oo, buf, o, ::Val{M}) where {M}
    @inbounds for c in 0:(M ÷ 4 - 1); ib = o + 4c; ob = oo + 48c
        t = avx_transpose12_packed(_L8(buf,ib), _L8(buf,ib+M), _L8(buf,ib+2M), _L8(buf,ib+3M), _L8(buf,ib+4M), _L8(buf,ib+5M), _L8(buf,ib+6M), _L8(buf,ib+7M), _L8(buf,ib+8M), _L8(buf,ib+9M), _L8(buf,ib+10M), _L8(buf,ib+11M))
        Base.Cartesian.@nexprs 12 k -> _S8(out, ob + 4(k-1), t[k]); end
end
@inline function _colbf9_w8!(buf, o, ::Val{M}, tw, tw1, tw2, tw3, bf3) where {M}
    @inbounds for c in 0:(M ÷ 4 - 1); ib = o + 4c
        r = avx_column_butterfly9(_L8(buf,ib), _L8(buf,ib+M), _L8(buf,ib+2M), _L8(buf,ib+3M), _L8(buf,ib+4M), _L8(buf,ib+5M), _L8(buf,ib+6M), _L8(buf,ib+7M), _L8(buf,ib+8M), tw1, tw2, tw3, bf3)
        _S8(buf, ib, r[1]); Base.Cartesian.@nexprs 8 j -> _S8(buf, ib + j*M, avx_mul_complex(tw[c*8+j], r[j+1])); end
end
@inline function _trans9_w8!(out, oo, buf, o, ::Val{M}) where {M}
    @inbounds for c in 0:(M ÷ 4 - 1); ib = o + 4c; ob = oo + 36c
        t = avx_transpose9_packed(_L8(buf,ib), _L8(buf,ib+M), _L8(buf,ib+2M), _L8(buf,ib+3M), _L8(buf,ib+4M), _L8(buf,ib+5M), _L8(buf,ib+6M), _L8(buf,ib+7M), _L8(buf,ib+8M))
        Base.Cartesian.@nexprs 9 k -> _S8(out, ob + 4(k-1), t[k]); end
end
@inline function _colbf5_w8!(buf, o, ::Val{M}, tw, t0, t1) where {M}
    @inbounds for c in 0:(M ÷ 4 - 1); ib = o + 4c
        r = avx_column_butterfly5(_L8(buf,ib), _L8(buf,ib+M), _L8(buf,ib+2M), _L8(buf,ib+3M), _L8(buf,ib+4M), t0, t1)
        _S8(buf, ib, r[1]); Base.Cartesian.@nexprs 4 j -> _S8(buf, ib + j*M, avx_mul_complex(tw[c*4+j], r[j+1])); end
end
@inline function _trans5_w8!(out, oo, buf, o, ::Val{M}) where {M}
    @inbounds for c in 0:(M ÷ 4 - 1); ib = o + 4c; ob = oo + 20c
        t = avx_transpose5_packed(_L8(buf,ib), _L8(buf,ib+M), _L8(buf,ib+2M), _L8(buf,ib+3M), _L8(buf,ib+4M))
        Base.Cartesian.@nexprs 5 k -> _S8(out, ob + 4(k-1), t[k]); end
end

# ---- W=8 kernel types (reuse Kernel/RPlan/applyplan! + the proc_ip!/proc_oop! alternation) ----
struct B64W8 <: Kernel
    n::Int; tw::Vector{V8f}; rot::V8f
end
B64W8(fwd::Bool) = B64W8(64, bf64_tw_w8(fwd), fwd ? _ROT90_FWD8 : _ROT90_INV8)
@inline proc_ip!(k::B64W8, buf, scr) = (@inbounds for f in 0:(length(buf) ÷ 64 - 1); butterfly64_w8!(buf, buf, scr, 64f, k.tw, k.rot); end)
@inline proc_oop!(k::B64W8, out, inp, scr) = (@inbounds for f in 0:(length(inp) ÷ 64 - 1); butterfly64_w8!(out, inp, out, 64f, k.tw, k.rot); end)

struct MR8W8{M, I <: Kernel} <: Kernel
    inner::I; tw::Vector{V8f}; rot::V8f
end
klen(::MR8W8{M}) where {M} = 8M
MR8W8(inner::Kernel, fwd::Bool) = (M = klen(inner); MR8W8{M, typeof(inner)}(inner, mr_twiddles_w8(8, M, 8M, fwd), fwd ? _ROT90_FWD8 : _ROT90_INV8))
@inline function proc_ip!(k::MR8W8{M}, buf, scr) where {M}
    n = 8M; cnt = length(buf) ÷ n
    @inbounds for f in 0:(cnt-1); _colbf8_w8!(buf, f*n, Val(M), k.tw, k.rot); end
    proc_oop!(k.inner, scr, buf, scr)
    @inbounds for f in 0:(cnt-1); _trans8_w8!(buf, f*n, scr, f*n, Val(M)); end
end
@inline function proc_oop!(k::MR8W8{M}, out, inp, scr) where {M}
    n = 8M; cnt = length(inp) ÷ n
    @inbounds for f in 0:(cnt-1); _colbf8_w8!(inp, f*n, Val(M), k.tw, k.rot); end
    proc_ip!(k.inner, inp, scr)
    @inbounds for f in 0:(cnt-1); _trans8_w8!(out, f*n, inp, f*n, Val(M)); end
end

struct MR12W8{M, I <: Kernel} <: Kernel
    inner::I; tw::Vector{V8f}; bf3::V8f; rot::V8f
end
klen(::MR12W8{M}) where {M} = 12M
MR12W8(inner::Kernel, fwd::Bool) = (M = klen(inner); MR12W8{M, typeof(inner)}(inner, mr_twiddles_w8(12, M, 12M, fwd), avx_broadcast_twiddle8(1, 3, fwd), fwd ? _ROT90_FWD8 : _ROT90_INV8))
@inline function proc_ip!(k::MR12W8{M}, buf, scr) where {M}
    n = 12M; cnt = length(buf) ÷ n
    @inbounds for f in 0:(cnt-1); _colbf12_w8!(buf, f*n, Val(M), k.tw, k.bf3, k.rot); end
    proc_oop!(k.inner, scr, buf, scr)
    @inbounds for f in 0:(cnt-1); _trans12_w8!(buf, f*n, scr, f*n, Val(M)); end
end
@inline function proc_oop!(k::MR12W8{M}, out, inp, scr) where {M}
    n = 12M; cnt = length(inp) ÷ n
    @inbounds for f in 0:(cnt-1); _colbf12_w8!(inp, f*n, Val(M), k.tw, k.bf3, k.rot); end
    proc_ip!(k.inner, inp, scr)
    @inbounds for f in 0:(cnt-1); _trans12_w8!(out, f*n, inp, f*n, Val(M)); end
end

struct MR9W8{M, I <: Kernel} <: Kernel
    inner::I; tw::Vector{V8f}; tw1::V8f; tw2::V8f; tw3::V8f; bf3::V8f
end
klen(::MR9W8{M}) where {M} = 9M
MR9W8(inner::Kernel, fwd::Bool) = (M = klen(inner); MR9W8{M, typeof(inner)}(inner, mr_twiddles_w8(9, M, 9M, fwd), avx_broadcast_twiddle8(1, 9, fwd), avx_broadcast_twiddle8(2, 9, fwd), avx_broadcast_twiddle8(4, 9, fwd), avx_broadcast_twiddle8(1, 3, fwd)))
@inline function proc_ip!(k::MR9W8{M}, buf, scr) where {M}
    n = 9M; cnt = length(buf) ÷ n
    @inbounds for f in 0:(cnt-1); _colbf9_w8!(buf, f*n, Val(M), k.tw, k.tw1, k.tw2, k.tw3, k.bf3); end
    proc_oop!(k.inner, scr, buf, scr)
    @inbounds for f in 0:(cnt-1); _trans9_w8!(buf, f*n, scr, f*n, Val(M)); end
end
@inline function proc_oop!(k::MR9W8{M}, out, inp, scr) where {M}
    n = 9M; cnt = length(inp) ÷ n
    @inbounds for f in 0:(cnt-1); _colbf9_w8!(inp, f*n, Val(M), k.tw, k.tw1, k.tw2, k.tw3, k.bf3); end
    proc_ip!(k.inner, inp, scr)
    @inbounds for f in 0:(cnt-1); _trans9_w8!(out, f*n, inp, f*n, Val(M)); end
end

struct MR5W8{M, I <: Kernel} <: Kernel
    inner::I; tw::Vector{V8f}; t0::V8f; t1::V8f
end
klen(::MR5W8{M}) where {M} = 5M
MR5W8(inner::Kernel, fwd::Bool) = (M = klen(inner); MR5W8{M, typeof(inner)}(inner, mr_twiddles_w8(5, M, 5M, fwd), avx_broadcast_twiddle8(1, 5, fwd), avx_broadcast_twiddle8(2, 5, fwd)))
@inline function proc_ip!(k::MR5W8{M}, buf, scr) where {M}
    n = 5M; cnt = length(buf) ÷ n
    @inbounds for f in 0:(cnt-1); _colbf5_w8!(buf, f*n, Val(M), k.tw, k.t0, k.t1); end
    proc_oop!(k.inner, scr, buf, scr)
    @inbounds for f in 0:(cnt-1); _trans5_w8!(buf, f*n, scr, f*n, Val(M)); end
end
@inline function proc_oop!(k::MR5W8{M}, out, inp, scr) where {M}
    n = 5M; cnt = length(inp) ÷ n
    @inbounds for f in 0:(cnt-1); _colbf5_w8!(inp, f*n, Val(M), k.tw, k.t0, k.t1); end
    proc_ip!(k.inner, inp, scr)
    @inbounds for f in 0:(cnt-1); _trans5_w8!(out, f*n, inp, f*n, Val(M)); end
end

# W=8-clean tree for n = 2^(6+3a+2b)·3^b·5^v5 = Butterfly64 · radix-8^a · radix-12^b · radix-9^b9 · radix-5^v5
# (every len_per_row divisible by CPV=4). Returns nothing for any other size.
function plan_tree_w8(n::Int, fwd::Bool = true)
    _HAS_AVX512 || return nothing                           # no real AVX-512 ⇒ don't build/time a W=8 tree
    v2 = 0; t = n; while t % 2 == 0; t ÷= 2; v2 += 1; end
    v3 = 0; while t % 3 == 0; t ÷= 3; v3 += 1; end
    v5 = 0; while t % 5 == 0; t ÷= 5; v5 += 1; end
    t == 1 || return nothing                                # not 2·3·5-smooth
    # Consume the 3s with b9 radix-9 (3² each, no 2s) + b12 radix-12 (3·2² each), the leftover 2s with `a`
    # radix-8 (2³) over the Butterfly64 base (2⁶), and the 5s with radix-5 (no 2s/3s):  2·b9 + b12 = v3,
    # 6 + 2·b12 + 3·a = v2.  Prefer MORE radix-9 (gains most from 512-bit), falling back to radix-12 for the
    # 2-count — so every size the old radix-12-only planner handled still resolves (with b9 = 0).
    b9 = -1; b12 = 0; a = 0
    for cand9 in (v3 ÷ 2):-1:0
        c12 = v3 - 2cand9
        rem2 = v2 - 6 - 2c12
        if rem2 >= 0 && rem2 % 3 == 0
            b9 = cand9; b12 = c12; a = rem2 ÷ 3; break
        end
    end
    b9 < 0 && return nothing
    k::Kernel = B64W8(fwd)
    for _ in 1:b9;  k = MR9W8(k, fwd);  end
    for _ in 1:b12; k = MR12W8(k, fwd); end
    for _ in 1:a;   k = MR8W8(k, fwd);  end
    for _ in 1:v5;  k = MR5W8(k, fwd);  end
    RPlan(k)
end
