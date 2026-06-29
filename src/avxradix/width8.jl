# AVX-512 (W=8, Vec{8,Float64} = 4 complex/vector) kernel set — the differentiator vs RustFFT (AVX2-only).
# Targeted: B64 base + radix-8/12 levels (all W=8-clean: every len_per_row is divisible by CPV=4). These
# are Kernels reusing the keystone's RPlan/applyplan!/proc alternation; only the passes are W=8. W=8 only
# helps small COMPUTE-bound non-pow2 sizes — autoplan times it vs W=4 and keeps it only when it wins.
# (Larger/memory-bound sizes regress at W=8 — see the roadmap.)

# ---- W=8 passes (load/store via V8f; transposes from avxport: avx_transpose8/12_packed(::V8f)) ----
@inline _L8(b, i) = avx_load_complex8(b, i)
@inline _S8(b, i, v) = avx_store_complex8!(b, i, v)
@inline _LP(b, i) = avx_load_partial2(b, i)        # 2-complex (Vec4) load, zero-padded to Vec8
@inline _SP(b, i, v) = avx_store_partial2!(b, i, v) # store low 2 complex of a Vec8
@inline _LP1(b, i) = avx_load_partial1(b, i)       # 1-complex load (v2=0 odd-M rem=1)
@inline _SP1(b, i, v) = avx_store_partial1!(b, i, v)
@inline _LP3(b, i) = avx_load_partial3(b, i)       # 3-complex load (v2=0 odd-M rem=3)
@inline _SP3(b, i, v) = avx_store_partial3!(b, i, v)
bf64_tw_w8(::Type{T}, fwd) where {T} = Vec{8, T}[avx_mixedradix_twiddle_chunk8(T, cs * 4, r, 64, fwd) for cs in 0:1 for r in 1:7]
function mr_twiddles_w8(::Type{T}, R, M, n, fwd) where {T}
    tw = Vec{8, T}[avx_mixedradix_twiddle_chunk8(T, c * 4, y, n, fwd) for c in 0:(M ÷ 4 - 1) for y in 1:(R - 1)]
    # Partial-column sizes leave rem = M mod 4 ∈ {1,2,3} leftover columns per pass (v2=1 ⇒ rem=2; v2=0 odd
    # ⇒ rem∈{1,3}): append ONE extra twiddle chunk per harmonic covering columns [M-rem .. M-1]. No-op for
    # the ÷4-clean kernels (M%4==0) ⇒ byte-identical twiddle tables, no F64/green-size regression.
    rem = M % 4
    if rem != 0
        for y in 1:(R - 1); push!(tw, avx_mixedradix_twiddle_chunk8(T, M - rem, y, n, fwd)); end
    end
    tw
end

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
    if M % 4 == 2; @inbounds begin; ib = o + (M-2); tc = (M ÷ 4) * 8   # rem=2 tail: 2 leftover columns (v2=1)
        r = avx_column_butterfly9(_LP(buf,ib), _LP(buf,ib+M), _LP(buf,ib+2M), _LP(buf,ib+3M), _LP(buf,ib+4M), _LP(buf,ib+5M), _LP(buf,ib+6M), _LP(buf,ib+7M), _LP(buf,ib+8M), tw1, tw2, tw3, bf3)
        _SP(buf, ib, r[1]); Base.Cartesian.@nexprs 8 j -> _SP(buf, ib + j*M, avx_mul_complex(tw[tc+j], r[j+1])); end; end
    if M % 4 == 1; @inbounds begin; ib = o + (M-1); tc = (M ÷ 4) * 8   # rem=1 tail (v2=0 odd-M)
        r = avx_column_butterfly9(_LP1(buf,ib), _LP1(buf,ib+M), _LP1(buf,ib+2M), _LP1(buf,ib+3M), _LP1(buf,ib+4M), _LP1(buf,ib+5M), _LP1(buf,ib+6M), _LP1(buf,ib+7M), _LP1(buf,ib+8M), tw1, tw2, tw3, bf3)
        _SP1(buf, ib, r[1]); Base.Cartesian.@nexprs 8 j -> _SP1(buf, ib + j*M, avx_mul_complex(tw[tc+j], r[j+1])); end; end
    if M % 4 == 3; @inbounds begin; ib = o + (M-3); tc = (M ÷ 4) * 8   # rem=3 tail (v2=0 odd-M)
        r = avx_column_butterfly9(_LP3(buf,ib), _LP3(buf,ib+M), _LP3(buf,ib+2M), _LP3(buf,ib+3M), _LP3(buf,ib+4M), _LP3(buf,ib+5M), _LP3(buf,ib+6M), _LP3(buf,ib+7M), _LP3(buf,ib+8M), tw1, tw2, tw3, bf3)
        _SP3(buf, ib, r[1]); Base.Cartesian.@nexprs 8 j -> _SP3(buf, ib + j*M, avx_mul_complex(tw[tc+j], r[j+1])); end; end
end
@inline function _trans9_w8!(out, oo, buf, o, ::Val{M}) where {M}
    @inbounds for c in 0:(M ÷ 4 - 1); ib = o + 4c; ob = oo + 36c
        t = avx_transpose9_packed(_L8(buf,ib), _L8(buf,ib+M), _L8(buf,ib+2M), _L8(buf,ib+3M), _L8(buf,ib+4M), _L8(buf,ib+5M), _L8(buf,ib+6M), _L8(buf,ib+7M), _L8(buf,ib+8M))
        Base.Cartesian.@nexprs 9 k -> _S8(out, ob + 4(k-1), t[k]); end
    if M % 4 == 1; @inbounds begin; ib = o + (M-1); ob = oo + (M-1)*9   # R=9 valid: 2 full Vec8 + 1 Vec1
        t = avx_transpose9_packed(_LP1(buf,ib), _LP1(buf,ib+M), _LP1(buf,ib+2M), _LP1(buf,ib+3M), _LP1(buf,ib+4M), _LP1(buf,ib+5M), _LP1(buf,ib+6M), _LP1(buf,ib+7M), _LP1(buf,ib+8M))
        _S8(out, ob, t[1]); _S8(out, ob+4, t[2]); _SP1(out, ob+8, t[3]); end; end
    if M % 4 == 3; @inbounds begin; ib = o + (M-3); ob = oo + (M-3)*9   # 3R=27 valid: 6 full Vec8 + 1 Vec3
        t = avx_transpose9_packed(_LP3(buf,ib), _LP3(buf,ib+M), _LP3(buf,ib+2M), _LP3(buf,ib+3M), _LP3(buf,ib+4M), _LP3(buf,ib+5M), _LP3(buf,ib+6M), _LP3(buf,ib+7M), _LP3(buf,ib+8M))
        _S8(out, ob, t[1]); _S8(out, ob+4, t[2]); _S8(out, ob+8, t[3]); _S8(out, ob+12, t[4]); _S8(out, ob+16, t[5]); _S8(out, ob+20, t[6]); _SP3(out, ob+24, t[7]); end; end
    if M % 4 == 2; @inbounds begin; ib = o + (M-2); ob = oo + (M-2)*9   # 2R=18 valid complex: 4 full Vec8 + 1 Vec4
        t = avx_transpose9_packed(_LP(buf,ib), _LP(buf,ib+M), _LP(buf,ib+2M), _LP(buf,ib+3M), _LP(buf,ib+4M), _LP(buf,ib+5M), _LP(buf,ib+6M), _LP(buf,ib+7M), _LP(buf,ib+8M))
        _S8(out, ob, t[1]); _S8(out, ob+4, t[2]); _S8(out, ob+8, t[3]); _S8(out, ob+12, t[4]); _SP(out, ob+16, t[5]); end; end
end
@inline function _colbf5_w8!(buf, o, ::Val{M}, tw, t0, t1) where {M}
    @inbounds for c in 0:(M ÷ 4 - 1); ib = o + 4c
        r = avx_column_butterfly5(_L8(buf,ib), _L8(buf,ib+M), _L8(buf,ib+2M), _L8(buf,ib+3M), _L8(buf,ib+4M), t0, t1)
        _S8(buf, ib, r[1]); Base.Cartesian.@nexprs 4 j -> _S8(buf, ib + j*M, avx_mul_complex(tw[c*4+j], r[j+1])); end
    if M % 4 == 2; @inbounds begin; ib = o + (M-2); tc = (M ÷ 4) * 4   # rem=2 tail: 2 leftover columns (v2=1)
        r = avx_column_butterfly5(_LP(buf,ib), _LP(buf,ib+M), _LP(buf,ib+2M), _LP(buf,ib+3M), _LP(buf,ib+4M), t0, t1)
        _SP(buf, ib, r[1]); Base.Cartesian.@nexprs 4 j -> _SP(buf, ib + j*M, avx_mul_complex(tw[tc+j], r[j+1])); end; end
    if M % 4 == 1; @inbounds begin; ib = o + (M-1); tc = (M ÷ 4) * 4   # rem=1 tail (v2=0 odd-M)
        r = avx_column_butterfly5(_LP1(buf,ib), _LP1(buf,ib+M), _LP1(buf,ib+2M), _LP1(buf,ib+3M), _LP1(buf,ib+4M), t0, t1)
        _SP1(buf, ib, r[1]); Base.Cartesian.@nexprs 4 j -> _SP1(buf, ib + j*M, avx_mul_complex(tw[tc+j], r[j+1])); end; end
    if M % 4 == 3; @inbounds begin; ib = o + (M-3); tc = (M ÷ 4) * 4   # rem=3 tail (v2=0 odd-M)
        r = avx_column_butterfly5(_LP3(buf,ib), _LP3(buf,ib+M), _LP3(buf,ib+2M), _LP3(buf,ib+3M), _LP3(buf,ib+4M), t0, t1)
        _SP3(buf, ib, r[1]); Base.Cartesian.@nexprs 4 j -> _SP3(buf, ib + j*M, avx_mul_complex(tw[tc+j], r[j+1])); end; end
end
@inline function _trans5_w8!(out, oo, buf, o, ::Val{M}) where {M}
    @inbounds for c in 0:(M ÷ 4 - 1); ib = o + 4c; ob = oo + 20c
        t = avx_transpose5_packed(_L8(buf,ib), _L8(buf,ib+M), _L8(buf,ib+2M), _L8(buf,ib+3M), _L8(buf,ib+4M))
        Base.Cartesian.@nexprs 5 k -> _S8(out, ob + 4(k-1), t[k]); end
    if M % 4 == 2; @inbounds begin; ib = o + (M-2); ob = oo + (M-2)*5   # 2R=10 valid complex: 2 full Vec8 + 1 Vec4
        t = avx_transpose5_packed(_LP(buf,ib), _LP(buf,ib+M), _LP(buf,ib+2M), _LP(buf,ib+3M), _LP(buf,ib+4M))
        _S8(out, ob, t[1]); _S8(out, ob+4, t[2]); _SP(out, ob+8, t[3]); end; end
    if M % 4 == 1; @inbounds begin; ib = o + (M-1); ob = oo + (M-1)*5   # R=5 valid: 1 full Vec8 + 1 Vec1
        t = avx_transpose5_packed(_LP1(buf,ib), _LP1(buf,ib+M), _LP1(buf,ib+2M), _LP1(buf,ib+3M), _LP1(buf,ib+4M))
        _S8(out, ob, t[1]); _SP1(out, ob+4, t[2]); end; end
    if M % 4 == 3; @inbounds begin; ib = o + (M-3); ob = oo + (M-3)*5   # 3R=15 valid: 3 full Vec8 + 1 Vec3
        t = avx_transpose5_packed(_LP3(buf,ib), _LP3(buf,ib+M), _LP3(buf,ib+2M), _LP3(buf,ib+3M), _LP3(buf,ib+4M))
        _S8(out, ob, t[1]); _S8(out, ob+4, t[2]); _S8(out, ob+8, t[3]); _SP3(out, ob+12, t[4]); end; end
end
# radix-7 (the lone factor of 7: 2^k·7 sizes 112/224/448, base ÷4 so M stays ÷4 — no partial column),
# PLUS the v2=1 rem=2 tail so 2·7^k sizes (98=2·7², …) route here (DST-I n=48 wraps inner 98).
@inline function _colbf7_w8!(buf, o, ::Val{M}, tw, t0, t1, t2) where {M}
    @inbounds for c in 0:(M ÷ 4 - 1); ib = o + 4c
        r = avx_column_butterfly7(_L8(buf,ib), _L8(buf,ib+M), _L8(buf,ib+2M), _L8(buf,ib+3M), _L8(buf,ib+4M), _L8(buf,ib+5M), _L8(buf,ib+6M), t0, t1, t2)
        _S8(buf, ib, r[1]); Base.Cartesian.@nexprs 6 j -> _S8(buf, ib + j*M, avx_mul_complex(tw[c*6+j], r[j+1])); end
    if M % 4 == 2; @inbounds begin; ib = o + (M-2); tc = (M ÷ 4) * 6   # rem=2 tail: 2 leftover columns (v2=1)
        r = avx_column_butterfly7(_LP(buf,ib), _LP(buf,ib+M), _LP(buf,ib+2M), _LP(buf,ib+3M), _LP(buf,ib+4M), _LP(buf,ib+5M), _LP(buf,ib+6M), t0, t1, t2)
        _SP(buf, ib, r[1]); Base.Cartesian.@nexprs 6 j -> _SP(buf, ib + j*M, avx_mul_complex(tw[tc+j], r[j+1])); end; end
    if M % 4 == 1; @inbounds begin; ib = o + (M-1); tc = (M ÷ 4) * 6   # rem=1 tail: 1 leftover column (v2=0 odd-M)
        r = avx_column_butterfly7(_LP1(buf,ib), _LP1(buf,ib+M), _LP1(buf,ib+2M), _LP1(buf,ib+3M), _LP1(buf,ib+4M), _LP1(buf,ib+5M), _LP1(buf,ib+6M), t0, t1, t2)
        _SP1(buf, ib, r[1]); Base.Cartesian.@nexprs 6 j -> _SP1(buf, ib + j*M, avx_mul_complex(tw[tc+j], r[j+1])); end; end
    if M % 4 == 3; @inbounds begin; ib = o + (M-3); tc = (M ÷ 4) * 6   # rem=3 tail: 3 leftover columns (v2=0 odd-M)
        r = avx_column_butterfly7(_LP3(buf,ib), _LP3(buf,ib+M), _LP3(buf,ib+2M), _LP3(buf,ib+3M), _LP3(buf,ib+4M), _LP3(buf,ib+5M), _LP3(buf,ib+6M), t0, t1, t2)
        _SP3(buf, ib, r[1]); Base.Cartesian.@nexprs 6 j -> _SP3(buf, ib + j*M, avx_mul_complex(tw[tc+j], r[j+1])); end; end
end
@inline function _trans7_w8!(out, oo, buf, o, ::Val{M}) where {M}
    @inbounds for c in 0:(M ÷ 4 - 1); ib = o + 4c; ob = oo + 28c
        t = avx_transpose7_packed(_L8(buf,ib), _L8(buf,ib+M), _L8(buf,ib+2M), _L8(buf,ib+3M), _L8(buf,ib+4M), _L8(buf,ib+5M), _L8(buf,ib+6M))
        Base.Cartesian.@nexprs 7 k -> _S8(out, ob + 4(k-1), t[k]); end
    if M % 4 == 2; @inbounds begin; ib = o + (M-2); ob = oo + (M-2)*7   # 2R=14 valid complex: 3 full Vec8 + 1 Vec4
        t = avx_transpose7_packed(_LP(buf,ib), _LP(buf,ib+M), _LP(buf,ib+2M), _LP(buf,ib+3M), _LP(buf,ib+4M), _LP(buf,ib+5M), _LP(buf,ib+6M))
        _S8(out, ob, t[1]); _S8(out, ob+4, t[2]); _S8(out, ob+8, t[3]); _SP(out, ob+12, t[4]); end; end
    if M % 4 == 1; @inbounds begin; ib = o + (M-1); ob = oo + (M-1)*7   # R=7 valid complex: 1 full Vec8 + 1 Vec(3)
        t = avx_transpose7_packed(_LP1(buf,ib), _LP1(buf,ib+M), _LP1(buf,ib+2M), _LP1(buf,ib+3M), _LP1(buf,ib+4M), _LP1(buf,ib+5M), _LP1(buf,ib+6M))
        _S8(out, ob, t[1]); _SP3(out, ob+4, t[2]); end; end
    if M % 4 == 3; @inbounds begin; ib = o + (M-3); ob = oo + (M-3)*7   # 3R=21 valid complex: 5 full Vec8 + 1 Vec(1)
        t = avx_transpose7_packed(_LP3(buf,ib), _LP3(buf,ib+M), _LP3(buf,ib+2M), _LP3(buf,ib+3M), _LP3(buf,ib+4M), _LP3(buf,ib+5M), _LP3(buf,ib+6M))
        _S8(out, ob, t[1]); _S8(out, ob+4, t[2]); _S8(out, ob+8, t[3]); _S8(out, ob+12, t[4]); _S8(out, ob+16, t[5]); _SP1(out, ob+20, t[6]); end; end
end
# radix-4 (covers the leftover 2s that radix-8/12 can't, so the W=8 tree spans ALL pow2 — needed for the
# F32 pow2 path, where the W4 monolith is unavailable). 3 twiddles/column; transpose4 = one 4×4 reg block.
@inline function _colbf4_w8!(buf, o, ::Val{M}, tw, rot) where {M}
    @inbounds for c in 0:(M ÷ 4 - 1); ib = o + 4c
        r = avx_column_butterfly4(_L8(buf,ib), _L8(buf,ib+M), _L8(buf,ib+2M), _L8(buf,ib+3M), rot)
        _S8(buf, ib, r[1]); Base.Cartesian.@nexprs 3 j -> _S8(buf, ib + j*M, avx_mul_complex(tw[c*3+j], r[j+1])); end
end
@inline function _trans4_w8!(out, oo, buf, o, ::Val{M}) where {M}
    @inbounds for c in 0:(M ÷ 4 - 1); ib = o + 4c; ob = oo + 16c
        t = avx_transpose4_packed(_L8(buf,ib), _L8(buf,ib+M), _L8(buf,ib+2M), _L8(buf,ib+3M))
        Base.Cartesian.@nexprs 4 k -> _S8(out, ob + 4(k-1), t[k]); end
end
# radix-2 (the lone factor of 2 above a bare-prime BP-W8 leaf: 2·P sizes 22/26/190). 1 twiddle/column;
# transpose2 = a zero-padded 4×4 register transpose keeping rows 0,1. The inner M is always ODD here (it
# wraps a prime/prime·5/7 leaf) ⇒ rem ∈ {1,3}; rem=2 included for completeness (DCE'd when M%4≠2).
@inline function _colbf2_w8!(buf, o, ::Val{M}, tw) where {M}
    @inbounds for c in 0:(M ÷ 4 - 1); ib = o + 4c
        a = _L8(buf, ib); b = _L8(buf, ib + M)
        _S8(buf, ib, a + b); _S8(buf, ib + M, avx_mul_complex(tw[c + 1], a - b)); end
    if M % 4 == 2; @inbounds begin; ib = o + (M-2); tc = M ÷ 4
        a = _LP(buf, ib); b = _LP(buf, ib + M)
        _SP(buf, ib, a + b); _SP(buf, ib + M, avx_mul_complex(tw[tc + 1], a - b)); end; end
    if M % 4 == 1; @inbounds begin; ib = o + (M-1); tc = M ÷ 4
        a = _LP1(buf, ib); b = _LP1(buf, ib + M)
        _SP1(buf, ib, a + b); _SP1(buf, ib + M, avx_mul_complex(tw[tc + 1], a - b)); end; end
    if M % 4 == 3; @inbounds begin; ib = o + (M-3); tc = M ÷ 4
        a = _LP3(buf, ib); b = _LP3(buf, ib + M)
        _SP3(buf, ib, a + b); _SP3(buf, ib + M, avx_mul_complex(tw[tc + 1], a - b)); end; end
end
@inline function _trans2_w8!(out, oo, buf, o, ::Val{M}) where {M}
    @inbounds for c in 0:(M ÷ 4 - 1); ib = o + 4c; ob = oo + 8c
        t = avx_transpose2_packed(_L8(buf,ib), _L8(buf,ib+M))
        _S8(out, ob, t[1]); _S8(out, ob+4, t[2]); end
    if M % 4 == 2; @inbounds begin; ib = o + (M-2); ob = oo + (M-2)*2   # 2R=4 valid complex: 1 full Vec8
        t = avx_transpose2_packed(_LP(buf,ib), _LP(buf,ib+M)); _S8(out, ob, t[1]); end; end
    if M % 4 == 1; @inbounds begin; ib = o + (M-1); ob = oo + (M-1)*2   # R=2 valid: 1 Vec(2) partial
        t = avx_transpose2_packed(_LP1(buf,ib), _LP1(buf,ib+M)); _SP(out, ob, t[1]); end; end
    if M % 4 == 3; @inbounds begin; ib = o + (M-3); ob = oo + (M-3)*2   # 3R=6 valid: 1 full Vec8 + 1 Vec(2)
        t = avx_transpose2_packed(_LP3(buf,ib), _LP3(buf,ib+M)); _S8(out, ob, t[1]); _SP(out, ob+4, t[2]); end; end
end
# radix-3 (the lone factor of 3 that radix-9/12 can't consume: 2^k·3 sizes — 48/96/192/384). bf3 twiddle;
# transpose3 = a 4×4 register transpose (zero-padded) compacted to 3 vectors. 2 twiddles/column.
@inline function _colbf3_w8!(buf, o, ::Val{M}, tw, bf3) where {M}
    @inbounds for c in 0:(M ÷ 4 - 1); ib = o + 4c
        r = avx_column_butterfly3(_L8(buf,ib), _L8(buf,ib+M), _L8(buf,ib+2M), bf3)
        _S8(buf, ib, r[1]); Base.Cartesian.@nexprs 2 j -> _S8(buf, ib + j*M, avx_mul_complex(tw[c*2+j], r[j+1])); end
    if M % 4 == 2; @inbounds begin; ib = o + (M-2); tc = (M ÷ 4) * 2   # rem=2 tail: 2 leftover columns (v2=1)
        r = avx_column_butterfly3(_LP(buf,ib), _LP(buf,ib+M), _LP(buf,ib+2M), bf3)
        _SP(buf, ib, r[1]); Base.Cartesian.@nexprs 2 j -> _SP(buf, ib + j*M, avx_mul_complex(tw[tc+j], r[j+1])); end; end
    if M % 4 == 1; @inbounds begin; ib = o + (M-1); tc = (M ÷ 4) * 2   # rem=1 tail (v2=0 odd-M)
        r = avx_column_butterfly3(_LP1(buf,ib), _LP1(buf,ib+M), _LP1(buf,ib+2M), bf3)
        _SP1(buf, ib, r[1]); Base.Cartesian.@nexprs 2 j -> _SP1(buf, ib + j*M, avx_mul_complex(tw[tc+j], r[j+1])); end; end
    if M % 4 == 3; @inbounds begin; ib = o + (M-3); tc = (M ÷ 4) * 2   # rem=3 tail (v2=0 odd-M)
        r = avx_column_butterfly3(_LP3(buf,ib), _LP3(buf,ib+M), _LP3(buf,ib+2M), bf3)
        _SP3(buf, ib, r[1]); Base.Cartesian.@nexprs 2 j -> _SP3(buf, ib + j*M, avx_mul_complex(tw[tc+j], r[j+1])); end; end
end
@inline function _trans3_w8!(out, oo, buf, o, ::Val{M}) where {M}
    @inbounds for c in 0:(M ÷ 4 - 1); ib = o + 4c; ob = oo + 12c
        t = avx_transpose3_packed(_L8(buf,ib), _L8(buf,ib+M), _L8(buf,ib+2M))
        Base.Cartesian.@nexprs 3 k -> _S8(out, ob + 4(k-1), t[k]); end
    if M % 4 == 2; @inbounds begin; ib = o + (M-2); ob = oo + (M-2)*3   # 2R=6 valid complex: 1 full Vec8 + 1 Vec4
        t = avx_transpose3_packed(_LP(buf,ib), _LP(buf,ib+M), _LP(buf,ib+2M))
        _S8(out, ob, t[1]); _SP(out, ob+4, t[2]); end; end
    if M % 4 == 1; @inbounds begin; ib = o + (M-1); ob = oo + (M-1)*3   # R=3 valid: 1 Vec3
        t = avx_transpose3_packed(_LP1(buf,ib), _LP1(buf,ib+M), _LP1(buf,ib+2M))
        _SP3(out, ob, t[1]); end; end
    if M % 4 == 3; @inbounds begin; ib = o + (M-3); ob = oo + (M-3)*3   # 3R=9 valid: 2 full Vec8 + 1 Vec1
        t = avx_transpose3_packed(_LP3(buf,ib), _LP3(buf,ib+M), _LP3(buf,ib+2M))
        _S8(out, ob, t[1]); _S8(out, ob+4, t[2]); _SP1(out, ob+8, t[3]); end; end
end

# ---- W=8 kernel types (reuse Kernel/RPlan/applyplan! + the proc_ip!/proc_oop! alternation) ----
struct B64W8{T} <: Kernel
    n::Int; tw::Vector{Vec{8, T}}; rot::Vec{8, T}
end
keltype(::B64W8{T}) where {T} = T
B64W8(fwd::Bool) = B64W8(Float64, fwd)
B64W8(::Type{T}, fwd::Bool) where {T} = B64W8{T}(64, bf64_tw_w8(T, fwd), fwd ? _rot90_fwd8(T) : _rot90_inv8(T))
@inline proc_ip!(k::B64W8, buf, scr) = (@inbounds for f in 0:(length(buf) ÷ 64 - 1); butterfly64_w8!(buf, buf, scr, 64f, k.tw, k.rot); end)
@inline proc_oop!(k::B64W8, out, inp, scr) = (@inbounds for f in 0:(length(inp) ÷ 64 - 1); butterfly64_w8!(out, inp, out, 64f, k.tw, k.rot); end)

# Butterfly1 at W8 — identity leaf (size-1 DFT = no-op). The base for v2=0 odd prime-power sizes: a size-P
# prime DFT is built as MR{P}W8 with M=1 (the rem=1 partial-column tail does the single P-point butterfly),
# so 49=7²=MR7(MR7(B1)), 9=MR9(B1), 25=MR5(MR5(B1)), etc. — no strided-gather leaf needed.
struct B1W8{T} <: Kernel
    n::Int
end
keltype(::B1W8{T}) where {T} = T
B1W8(::Type{T}, fwd::Bool) where {T} = B1W8{T}(1)
@inline proc_ip!(k::B1W8, buf, scr) = nothing
@inline proc_oop!(k::B1W8, out, inp, scr) = (out === inp || copyto!(out, inp); nothing)

# BPW8{P} — direct size-P odd-prime DFT leaf at W=8 (the bare primes 11/13/19 the padding trick CANNOT
# reach: there is no avx_column_butterfly{11,13,19}). One FFT per iter via the width-generic
# `avx_colbf_prime` over zero-padded V8f32 partial-1 loads (complex in lane 0; upper 3 lanes are zeroed
# garbage discarded by the partial-1 store) — the SAME idiom as the D1 rem=1 padding tail, reusing
# `_bfprime!` (recursive.jl) verbatim. Only the twiddle WIDTH differs from the V2f `BP` leaf: V8f32
# broadcast (avx_colbf_prime is width-generic). keltype = T so the W=8 RPlan allocates T-typed scratch.
struct BPW8{P, H, T} <: Kernel
    n::Int; tws::NTuple{H, Vec{8, T}}
end
klen(::BPW8{P}) where {P} = P
keltype(::BPW8{P, H, T}) where {P, H, T} = T
BPW8(::Type{T}, P::Int, fwd::Bool) where {T} = (H = (P - 1) ÷ 2; BPW8{P, H, T}(P, ntuple(a -> avx_broadcast_twiddle8(T, a, P, fwd), H)))
@inline proc_ip!(k::BPW8{P}, buf, scr) where {P} = (@inbounds for f in 0:(length(buf) ÷ P - 1); _bfprime!(buf, P * f, buf, P * f, k.tws, Val(P)); end)
@inline proc_oop!(k::BPW8{P}, out, inp, scr) where {P} = (@inbounds for f in 0:(length(inp) ÷ P - 1); _bfprime!(out, P * f, inp, P * f, k.tws, Val(P)); end)

# Butterfly2 at W8 — the v2=1 (2·odd) base: a size-2 DFT [a,b] → [a+b, a-b] (no twiddles, fwd≡inv). Packs
# 2 instances (4 complex) per Vec{8}; one leftover instance (count = n/2 = odd for 2·odd sizes) handled by a
# Vec{4} partial. Float32-only (the partial-column path is F32-only). The radix-3/5 passes above it carry
# M ≡ 2 mod 4 (uniform rem=2) and use the partial-column tails. Verified bit-exact (n=2 DFT is exact).
const _SGN2_F32 = V8f32((1f0, 1f0, -1f0, -1f0, 1f0, 1f0, -1f0, -1f0))
@inline function _dft2_pair(v::Vec{8})
    A = shufflevector(v, Val((0, 1, 0, 1, 4, 5, 4, 5)))   # [a,a,a',a']
    B = shufflevector(v, Val((2, 3, 2, 3, 6, 7, 6, 7)))   # [b,b,b',b']
    muladd(_SGN2_F32, B, A)                                # [a+b, a-b, a'+b', a'-b']
end
struct B2W8{T} <: Kernel
    n::Int
end
keltype(::B2W8{T}) where {T} = T
B2W8(::Type{T}, fwd::Bool) where {T} = B2W8{T}(2)
@inline function proc_ip!(k::B2W8, buf, scr)
    cnt = length(buf) ÷ 2; np = cnt ÷ 2
    @inbounds for p in 0:(np-1); _S8(buf, 4p, _dft2_pair(_L8(buf, 4p))); end
    isodd(cnt) && (@inbounds _SP(buf, 2(cnt-1), _dft2_pair(_LP(buf, 2(cnt-1)))))
end
@inline function proc_oop!(k::B2W8, out, inp, scr)
    cnt = length(inp) ÷ 2; np = cnt ÷ 2
    @inbounds for p in 0:(np-1); _S8(out, 4p, _dft2_pair(_L8(inp, 4p))); end
    isodd(cnt) && (@inbounds _SP(out, 2(cnt-1), _dft2_pair(_LP(inp, 2(cnt-1)))))
end

# Butterfly4 at W8 (register-only — 4 complex = 1 V8 vector; a single size-4 DFT in lanes via _dft4_lane).
# The 2² base for v2=2 sizes (12=MR3(B4), 36=MR9(B4), …). No twiddles, no scratch.
struct B4W8{T} <: Kernel
    n::Int; rot::Vec{8, T}
end
keltype(::B4W8{T}) where {T} = T
B4W8(fwd::Bool) = B4W8(Float64, fwd)
B4W8(::Type{T}, fwd::Bool) where {T} = B4W8{T}(4, fwd ? _rot90_fwd8(T) : _rot90_inv8(T))
@inline proc_ip!(k::B4W8, buf, scr) = (@inbounds for f in 0:(length(buf) ÷ 4 - 1); _S8(buf, 4f, _dft4_lane(_L8(buf, 4f), k.rot)); end)
@inline proc_oop!(k::B4W8, out, inp, scr) = (@inbounds for f in 0:(length(inp) ÷ 4 - 1); _S8(out, 4f, _dft4_lane(_L8(inp, 4f), k.rot)); end)

# Butterfly8 at W8 (register-only — 8 complex = 2 V8 vectors; 2×4 split: a size-4 DFT in lanes per row,
# twiddle row 1 by [W8⁰…W8³], radix-2 across the 2 rows, interleave to natural order). The 2³ base for
# v2=3 sizes (24=MR3(B8), 360=MR5(MR9(B8)), 3000=MR5³(MR3(B8))). Verified bit-exact vs FFTW.
function butterfly8_w8!(out, inp, base::Int, tw8, rot)
    @inbounds begin
        v0 = _L8(inp, base); v1 = _L8(inp, base + 4)
        E = _dft4_lane(v0 + v1, rot)                       # even outputs X[0,2,4,6] = DFT4(row0+row1)
        O = _dft4_lane(avx_mul_complex(tw8, v0 - v1), rot) # odd  outputs X[1,3,5,7] = DFT4(tw8·(row0-row1))
        _S8(out, base,     shufflevector(E, O, Val((0, 1, 8, 9, 2, 3, 10, 11))))
        _S8(out, base + 4, shufflevector(E, O, Val((4, 5, 12, 13, 6, 7, 14, 15))))
    end
end
struct B8W8{T} <: Kernel
    n::Int; tw8::Vec{8, T}; rot::Vec{8, T}
end
keltype(::B8W8{T}) where {T} = T
B8W8(fwd::Bool) = B8W8(Float64, fwd)
B8W8(::Type{T}, fwd::Bool) where {T} = B8W8{T}(8, avx_mixedradix_twiddle_chunk8(T, 0, 1, 8, fwd), fwd ? _rot90_fwd8(T) : _rot90_inv8(T))
@inline proc_ip!(k::B8W8, buf, scr) = (@inbounds for f in 0:(length(buf) ÷ 8 - 1); butterfly8_w8!(buf, buf, 8f, k.tw8, k.rot); end)
@inline proc_oop!(k::B8W8, out, inp, scr) = (@inbounds for f in 0:(length(inp) ÷ 8 - 1); butterfly8_w8!(out, inp, 8f, k.tw8, k.rot); end)

# Butterfly16 at W8 (4×4, register-only — 16 complex = 4 V8 vectors, 1 columnset; the 4 cols pack into a
# vector). phase1: col bf4 + twiddle + transpose4; phase2: col bf4 across. Verified bit-exact vs FFTW.
bf16_tw_w8(::Type{T}, fwd) where {T} = ntuple(r -> avx_mixedradix_twiddle_chunk8(T, 0, r, 16, fwd), 3)
function butterfly16_w8!(out, inp, base::Int, tw, rot)
    @inbounds begin
        m = avx_column_butterfly4(_L8(inp, base), _L8(inp, base+4), _L8(inp, base+8), _L8(inp, base+12), rot)
        t = avx_transpose4_packed(m[1], avx_mul_complex(tw[1], m[2]), avx_mul_complex(tw[2], m[3]), avx_mul_complex(tw[3], m[4]))
        o = avx_column_butterfly4(t[1], t[2], t[3], t[4], rot)
        _S8(out, base, o[1]); _S8(out, base+4, o[2]); _S8(out, base+8, o[3]); _S8(out, base+12, o[4])
    end
end
struct B16W8{T} <: Kernel
    n::Int; tw::NTuple{3, Vec{8, T}}; rot::Vec{8, T}
end
keltype(::B16W8{T}) where {T} = T
B16W8(fwd::Bool) = B16W8(Float64, fwd)
B16W8(::Type{T}, fwd::Bool) where {T} = B16W8{T}(16, bf16_tw_w8(T, fwd), fwd ? _rot90_fwd8(T) : _rot90_inv8(T))
@inline proc_ip!(k::B16W8, buf, scr) = (@inbounds for f in 0:(length(buf) ÷ 16 - 1); butterfly16_w8!(buf, buf, 16f, k.tw, k.rot); end)
@inline proc_oop!(k::B16W8, out, inp, scr) = (@inbounds for f in 0:(length(inp) ÷ 16 - 1); butterfly16_w8!(out, inp, 16f, k.tw, k.rot); end)

# Butterfly32 at W8 (4×8 two-phase; needs scratch ≥ length). 32 complex = 8 V8 vectors. phase1: 2
# columnsets — col bf4 (4 rows, stride 8) + twiddle + transpose4 → scr; phase2: 1 columnset — col bf8
# across 8 rows (stride 4) → out. Faithful W8 scale of butterfly32!. Verified bit-exact vs FFTW.
bf32_tw_w8(::Type{T}, fwd) where {T} = Vec{8, T}[avx_mixedradix_twiddle_chunk8(T, cs * 4, r, 32, fwd) for cs in 0:1 for r in 1:3]   # 6, index 3cs+r
function butterfly32_w8!(out, inp, scr, base::Int, tw, rot)
    @inbounds for cs in 0:1
        b = base + cs * 4
        m = avx_column_butterfly4(_L8(inp, b), _L8(inp, b+8), _L8(inp, b+16), _L8(inp, b+24), rot)
        t = avx_transpose4_packed(m[1], avx_mul_complex(tw[3cs+1], m[2]), avx_mul_complex(tw[3cs+2], m[3]), avx_mul_complex(tw[3cs+3], m[4]))
        ob = base + cs * 16
        _S8(scr, ob, t[1]); _S8(scr, ob+4, t[2]); _S8(scr, ob+8, t[3]); _S8(scr, ob+12, t[4])
    end
    @inbounds begin
        m = avx_column_butterfly8(_L8(scr, base), _L8(scr, base+4), _L8(scr, base+8), _L8(scr, base+12), _L8(scr, base+16), _L8(scr, base+20), _L8(scr, base+24), _L8(scr, base+28), rot)
        for r in 0:7; _S8(out, base + 4r, m[r+1]); end
    end
end
struct B32W8{T} <: Kernel
    n::Int; tw::Vector{Vec{8, T}}; rot::Vec{8, T}
end
keltype(::B32W8{T}) where {T} = T
B32W8(fwd::Bool) = B32W8(Float64, fwd)
B32W8(::Type{T}, fwd::Bool) where {T} = B32W8{T}(32, bf32_tw_w8(T, fwd), fwd ? _rot90_fwd8(T) : _rot90_inv8(T))
@inline proc_ip!(k::B32W8, buf, scr) = (@inbounds for f in 0:(length(buf) ÷ 32 - 1); butterfly32_w8!(buf, buf, scr, 32f, k.tw, k.rot); end)
@inline proc_oop!(k::B32W8, out, inp, scr) = (@inbounds for f in 0:(length(inp) ÷ 32 - 1); butterfly32_w8!(out, inp, out, 32f, k.tw, k.rot); end)

# Butterfly256 at W8 (faithful port of rustfft Butterfly256Avx<f32>, 4 complex/vec): 32×8 two-phase.
# phase 1: 8 columnsets — col bf8 + twiddle + transpose8 → scr; phase 2: 2 columnsets — col bf32 → out.
# (The V4f Butterfly256 (kernels.jl) is the same at 2 complex/vec — 16 columnsets, half strides.)
bf256_tw_w8(::Type{T}, fwd) where {T} = Vec{8, T}[avx_mixedradix_twiddle_chunk8(T, cs * 4, r, 256, fwd) for cs in 0:7 for r in 1:7]   # 56, index 7cs+r
bf256_bf32_tw_w8(::Type{T}, fwd) where {T} = (avx_broadcast_twiddle8(T, 1, 32, fwd), avx_broadcast_twiddle8(T, 2, 32, fwd), avx_broadcast_twiddle8(T, 3, 32, fwd),
    avx_broadcast_twiddle8(T, 5, 32, fwd), avx_broadcast_twiddle8(T, 6, 32, fwd), avx_broadcast_twiddle8(T, 7, 32, fwd))
@inline _bf256_ld8_w8(buf, b) = (_L8(buf, b), _L8(buf, b + 32), _L8(buf, b + 64), _L8(buf, b + 96), _L8(buf, b + 128), _L8(buf, b + 160), _L8(buf, b + 192), _L8(buf, b + 224))
function butterfly256_w8!(out, inp, scr, base::Int, tw, tw32, rot)
    @inbounds for cs in 0:7
        b = base + cs * 4
        m = avx_column_butterfly8(_bf256_ld8_w8(inp, b)..., rot)
        t = avx_transpose8_packed(m[1], avx_mul_complex(tw[7cs + 1], m[2]), avx_mul_complex(tw[7cs + 2], m[3]), avx_mul_complex(tw[7cs + 3], m[4]),
            avx_mul_complex(tw[7cs + 4], m[5]), avx_mul_complex(tw[7cs + 5], m[6]), avx_mul_complex(tw[7cs + 6], m[7]), avx_mul_complex(tw[7cs + 7], m[8]))
        ob = base + cs * 32
        _S8(scr, ob, t[1]); _S8(scr, ob + 4, t[2]); _S8(scr, ob + 8, t[3]); _S8(scr, ob + 12, t[4])
        _S8(scr, ob + 16, t[5]); _S8(scr, ob + 20, t[6]); _S8(scr, ob + 24, t[7]); _S8(scr, ob + 28, t[8])
    end
    @inbounds for cs in 0:1
        b = base + cs * 4
        avx_column_butterfly32(scr, b, 8, out, b, 8, tw32, rot)
    end
end
struct B256W8{T} <: Kernel
    n::Int; tw::Vector{Vec{8, T}}; tw32::NTuple{6, Vec{8, T}}; rot::Vec{8, T}
end
keltype(::B256W8{T}) where {T} = T
B256W8(fwd::Bool) = B256W8(Float64, fwd)
B256W8(::Type{T}, fwd::Bool) where {T} = B256W8{T}(256, bf256_tw_w8(T, fwd), bf256_bf32_tw_w8(T, fwd), fwd ? _rot90_fwd8(T) : _rot90_inv8(T))
@inline proc_ip!(k::B256W8, buf, scr) = (@inbounds for f in 0:(length(buf) ÷ 256 - 1); butterfly256_w8!(buf, buf, scr, 256f, k.tw, k.tw32, k.rot); end)
@inline proc_oop!(k::B256W8, out, inp, scr) = (@inbounds for f in 0:(length(inp) ÷ 256 - 1); butterfly256_w8!(out, inp, out, 256f, k.tw, k.tw32, k.rot); end)

# Butterfly512 at W8 (faithful port of rustfft Butterfly512Avx<f32>, 4 complex/vec): 32×16 two-phase.
# phase 1: 8 columnsets — col bf16 + chunked twiddle + transpose4 → scr; phase 2: 4 columnsets — col bf32.
bf512_tw_w8(::Type{T}, fwd) where {T} = Vec{8, T}[avx_mixedradix_twiddle_chunk8(T, cs * 4, r, 512, fwd) for cs in 0:7 for r in 1:15]   # 120, chunks of 15
bf512_bf16_tw_w8(::Type{T}, fwd) where {T} = (avx_broadcast_twiddle8(T, 1, 16, fwd), avx_broadcast_twiddle8(T, 3, 16, fwd))
function butterfly512_w8!(out, inp, scr, base::Int, tw, tw16, tw32, rot)
    @inbounds for cs in 0:7
        b = base + cs * 4
        mid = avx_column_butterfly16(inp, b, 32, tw16, rot)
        tc = 15 * cs; ob = base + cs * 64
        # chunk 0 (t0 untwiddled), 1, 2, 3 — literal indices into mid (no runtime tuple index)
        let tr = avx_transpose4_packed(mid[1], avx_mul_complex(mid[2], tw[tc+1]), avx_mul_complex(mid[3], tw[tc+2]), avx_mul_complex(mid[4], tw[tc+3]))
            _S8(scr, ob, tr[1]); _S8(scr, ob+16, tr[2]); _S8(scr, ob+32, tr[3]); _S8(scr, ob+48, tr[4]); end
        let tr = avx_transpose4_packed(avx_mul_complex(mid[5], tw[tc+4]), avx_mul_complex(mid[6], tw[tc+5]), avx_mul_complex(mid[7], tw[tc+6]), avx_mul_complex(mid[8], tw[tc+7]))
            _S8(scr, ob+4, tr[1]); _S8(scr, ob+20, tr[2]); _S8(scr, ob+36, tr[3]); _S8(scr, ob+52, tr[4]); end
        let tr = avx_transpose4_packed(avx_mul_complex(mid[9], tw[tc+8]), avx_mul_complex(mid[10], tw[tc+9]), avx_mul_complex(mid[11], tw[tc+10]), avx_mul_complex(mid[12], tw[tc+11]))
            _S8(scr, ob+8, tr[1]); _S8(scr, ob+24, tr[2]); _S8(scr, ob+40, tr[3]); _S8(scr, ob+56, tr[4]); end
        let tr = avx_transpose4_packed(avx_mul_complex(mid[13], tw[tc+12]), avx_mul_complex(mid[14], tw[tc+13]), avx_mul_complex(mid[15], tw[tc+14]), avx_mul_complex(mid[16], tw[tc+15]))
            _S8(scr, ob+12, tr[1]); _S8(scr, ob+28, tr[2]); _S8(scr, ob+44, tr[3]); _S8(scr, ob+60, tr[4]); end
    end
    @inbounds for cs in 0:3
        b = base + cs * 4
        avx_column_butterfly32(scr, b, 16, out, b, 16, tw32, rot)
    end
end
struct B512W8{T} <: Kernel
    n::Int; tw::Vector{Vec{8, T}}; tw16::NTuple{2, Vec{8, T}}; tw32::NTuple{6, Vec{8, T}}; rot::Vec{8, T}
end
keltype(::B512W8{T}) where {T} = T
B512W8(fwd::Bool) = B512W8(Float64, fwd)
B512W8(::Type{T}, fwd::Bool) where {T} = B512W8{T}(512, bf512_tw_w8(T, fwd), bf512_bf16_tw_w8(T, fwd), bf256_bf32_tw_w8(T, fwd), fwd ? _rot90_fwd8(T) : _rot90_inv8(T))
@inline proc_ip!(k::B512W8, buf, scr) = (@inbounds for f in 0:(length(buf) ÷ 512 - 1); butterfly512_w8!(buf, buf, scr, 512f, k.tw, k.tw16, k.tw32, k.rot); end)
@inline proc_oop!(k::B512W8, out, inp, scr) = (@inbounds for f in 0:(length(inp) ÷ 512 - 1); butterfly512_w8!(out, inp, out, 512f, k.tw, k.tw16, k.tw32, k.rot); end)

struct MR8W8{M, I <: Kernel, T} <: Kernel
    inner::I; tw::Vector{Vec{8, T}}; rot::Vec{8, T}
end
klen(::MR8W8{M}) where {M} = 8M
keltype(::MR8W8{M, I, T}) where {M, I, T} = T
MR8W8(inner::Kernel, fwd::Bool) = (T = keltype(inner); M = klen(inner); MR8W8{M, typeof(inner), T}(inner, mr_twiddles_w8(T, 8, M, 8M, fwd), fwd ? _rot90_fwd8(T) : _rot90_inv8(T)))
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

struct MR4W8{M, I <: Kernel, T} <: Kernel
    inner::I; tw::Vector{Vec{8, T}}; rot::Vec{8, T}
end
klen(::MR4W8{M}) where {M} = 4M
keltype(::MR4W8{M, I, T}) where {M, I, T} = T
MR4W8(inner::Kernel, fwd::Bool) = (T = keltype(inner); M = klen(inner); MR4W8{M, typeof(inner), T}(inner, mr_twiddles_w8(T, 4, M, 4M, fwd), fwd ? _rot90_fwd8(T) : _rot90_inv8(T)))
@inline function proc_ip!(k::MR4W8{M}, buf, scr) where {M}
    n = 4M; cnt = length(buf) ÷ n
    @inbounds for f in 0:(cnt-1); _colbf4_w8!(buf, f*n, Val(M), k.tw, k.rot); end
    proc_oop!(k.inner, scr, buf, scr)
    @inbounds for f in 0:(cnt-1); _trans4_w8!(buf, f*n, scr, f*n, Val(M)); end
end
@inline function proc_oop!(k::MR4W8{M}, out, inp, scr) where {M}
    n = 4M; cnt = length(inp) ÷ n
    @inbounds for f in 0:(cnt-1); _colbf4_w8!(inp, f*n, Val(M), k.tw, k.rot); end
    proc_ip!(k.inner, inp, scr)
    @inbounds for f in 0:(cnt-1); _trans4_w8!(out, f*n, inp, f*n, Val(M)); end
end

struct MR2W8{M, I <: Kernel, T} <: Kernel
    inner::I; tw::Vector{Vec{8, T}}
end
klen(::MR2W8{M}) where {M} = 2M
keltype(::MR2W8{M, I, T}) where {M, I, T} = T
MR2W8(inner::Kernel, fwd::Bool) = (T = keltype(inner); M = klen(inner); MR2W8{M, typeof(inner), T}(inner, mr_twiddles_w8(T, 2, M, 2M, fwd)))
@inline function proc_ip!(k::MR2W8{M}, buf, scr) where {M}
    n = 2M; cnt = length(buf) ÷ n
    @inbounds for f in 0:(cnt-1); _colbf2_w8!(buf, f*n, Val(M), k.tw); end
    proc_oop!(k.inner, scr, buf, scr)
    @inbounds for f in 0:(cnt-1); _trans2_w8!(buf, f*n, scr, f*n, Val(M)); end
end
@inline function proc_oop!(k::MR2W8{M}, out, inp, scr) where {M}
    n = 2M; cnt = length(inp) ÷ n
    @inbounds for f in 0:(cnt-1); _colbf2_w8!(inp, f*n, Val(M), k.tw); end
    proc_ip!(k.inner, inp, scr)
    @inbounds for f in 0:(cnt-1); _trans2_w8!(out, f*n, inp, f*n, Val(M)); end
end

struct MR3W8{M, I <: Kernel, T} <: Kernel
    inner::I; tw::Vector{Vec{8, T}}; bf3::Vec{8, T}
end
klen(::MR3W8{M}) where {M} = 3M
keltype(::MR3W8{M, I, T}) where {M, I, T} = T
MR3W8(inner::Kernel, fwd::Bool) = (T = keltype(inner); M = klen(inner); MR3W8{M, typeof(inner), T}(inner, mr_twiddles_w8(T, 3, M, 3M, fwd), avx_broadcast_twiddle8(T, 1, 3, fwd)))
@inline function proc_ip!(k::MR3W8{M}, buf, scr) where {M}
    n = 3M; cnt = length(buf) ÷ n
    @inbounds for f in 0:(cnt-1); _colbf3_w8!(buf, f*n, Val(M), k.tw, k.bf3); end
    proc_oop!(k.inner, scr, buf, scr)
    @inbounds for f in 0:(cnt-1); _trans3_w8!(buf, f*n, scr, f*n, Val(M)); end
end
@inline function proc_oop!(k::MR3W8{M}, out, inp, scr) where {M}
    n = 3M; cnt = length(inp) ÷ n
    @inbounds for f in 0:(cnt-1); _colbf3_w8!(inp, f*n, Val(M), k.tw, k.bf3); end
    proc_ip!(k.inner, inp, scr)
    @inbounds for f in 0:(cnt-1); _trans3_w8!(out, f*n, inp, f*n, Val(M)); end
end

struct MR12W8{M, I <: Kernel, T} <: Kernel
    inner::I; tw::Vector{Vec{8, T}}; bf3::Vec{8, T}; rot::Vec{8, T}
end
klen(::MR12W8{M}) where {M} = 12M
keltype(::MR12W8{M, I, T}) where {M, I, T} = T
MR12W8(inner::Kernel, fwd::Bool) = (T = keltype(inner); M = klen(inner); MR12W8{M, typeof(inner), T}(inner, mr_twiddles_w8(T, 12, M, 12M, fwd), avx_broadcast_twiddle8(T, 1, 3, fwd), fwd ? _rot90_fwd8(T) : _rot90_inv8(T)))
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

struct MR9W8{M, I <: Kernel, T} <: Kernel
    inner::I; tw::Vector{Vec{8, T}}; tw1::Vec{8, T}; tw2::Vec{8, T}; tw3::Vec{8, T}; bf3::Vec{8, T}
end
klen(::MR9W8{M}) where {M} = 9M
keltype(::MR9W8{M, I, T}) where {M, I, T} = T
MR9W8(inner::Kernel, fwd::Bool) = (T = keltype(inner); M = klen(inner); MR9W8{M, typeof(inner), T}(inner, mr_twiddles_w8(T, 9, M, 9M, fwd), avx_broadcast_twiddle8(T, 1, 9, fwd), avx_broadcast_twiddle8(T, 2, 9, fwd), avx_broadcast_twiddle8(T, 4, 9, fwd), avx_broadcast_twiddle8(T, 1, 3, fwd)))
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

struct MR5W8{M, I <: Kernel, T} <: Kernel
    inner::I; tw::Vector{Vec{8, T}}; t0::Vec{8, T}; t1::Vec{8, T}
end
klen(::MR5W8{M}) where {M} = 5M
keltype(::MR5W8{M, I, T}) where {M, I, T} = T
MR5W8(inner::Kernel, fwd::Bool) = (T = keltype(inner); M = klen(inner); MR5W8{M, typeof(inner), T}(inner, mr_twiddles_w8(T, 5, M, 5M, fwd), avx_broadcast_twiddle8(T, 1, 5, fwd), avx_broadcast_twiddle8(T, 2, 5, fwd)))
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

struct MR7W8{M, I <: Kernel, T} <: Kernel
    inner::I; tw::Vector{Vec{8, T}}; t0::Vec{8, T}; t1::Vec{8, T}; t2::Vec{8, T}
end
klen(::MR7W8{M}) where {M} = 7M
keltype(::MR7W8{M, I, T}) where {M, I, T} = T
MR7W8(inner::Kernel, fwd::Bool) = (T = keltype(inner); M = klen(inner); MR7W8{M, typeof(inner), T}(inner, mr_twiddles_w8(T, 7, M, 7M, fwd), avx_broadcast_twiddle8(T, 1, 7, fwd), avx_broadcast_twiddle8(T, 2, 7, fwd), avx_broadcast_twiddle8(T, 3, 7, fwd)))
@inline function proc_ip!(k::MR7W8{M}, buf, scr) where {M}
    n = 7M; cnt = length(buf) ÷ n
    @inbounds for f in 0:(cnt-1); _colbf7_w8!(buf, f*n, Val(M), k.tw, k.t0, k.t1, k.t2); end
    proc_oop!(k.inner, scr, buf, scr)
    @inbounds for f in 0:(cnt-1); _trans7_w8!(buf, f*n, scr, f*n, Val(M)); end
end
@inline function proc_oop!(k::MR7W8{M}, out, inp, scr) where {M}
    n = 7M; cnt = length(inp) ÷ n
    @inbounds for f in 0:(cnt-1); _colbf7_w8!(inp, f*n, Val(M), k.tw, k.t0, k.t1, k.t2); end
    proc_ip!(k.inner, inp, scr)
    @inbounds for f in 0:(cnt-1); _trans7_w8!(out, f*n, inp, f*n, Val(M)); end
end

# W=8-clean tree for n = 2^(6+3a+2b)·3^b·5^v5 = Butterfly64 · radix-8^a · radix-12^b · radix-9^b9 · radix-5^v5
# (every len_per_row divisible by CPV=4). Returns nothing for any other size.
function _plan_tree_w8_main(::Type{T}, n::Int, fwd::Bool = true) where {T}
    v2 = 0; t = n; while t % 2 == 0; t ÷= 2; v2 += 1; end
    v3 = 0; while t % 3 == 0; t ÷= 3; v3 += 1; end
    v5 = 0; while t % 5 == 0; t ÷= 5; v5 += 1; end
    t == 1 || return nothing                                # not 2·3·5-smooth
    # Pure power-of-two ≥ 256: a B256/B512 monolith base + radix-8/4 chain (rustfft's "8xn" scheme — the F32
    # equivalent of the F64 B256/B512 path in avxradix/planner.jl). Gives F32 the monolith bases that close
    # the odd-power gap. Base exp 9 (B512) or 8 (B256) chosen so the leftover rem = 3a + 2·c4 is never 1.
    if v3 == 0 && v5 == 0 && v2 >= 8
        if v2 - 9 >= 0 && v2 - 9 != 1
            kb::Kernel = B512W8(T, fwd); rem = v2 - 9
        else
            kb = B256W8(T, fwd); rem = v2 - 8
        end
        r3 = rem % 3
        aa = r3 == 0 ? rem ÷ 3 : (r3 == 2 ? (rem - 2) ÷ 3 : (rem - 4) ÷ 3)
        cc = r3 == 0 ? 0 : (r3 == 2 ? 1 : 2)
        for _ in 1:aa; kb = MR8W8(kb, fwd); end
        for _ in 1:cc; kb = MR4W8(kb, fwd); end
        return RPlan(kb)
    end
    # Consume the 3s with b9 radix-9 (3² each, no 2s) + b12 radix-12 (3·2² each), the 5s with radix-5, and
    # the leftover 2s over the Butterfly64 base (2⁶) with `a` radix-8 (2³ each) + `c4` radix-4 (2² each):
    #   2·b9 + b12 = v3,   6 + 2·b12 + 3·a + 2·c4 = v2.
    # Prefer MORE radix-9 (gains most from 512-bit), then MORE radix-8, using radix-4 only for the leftover
    # rem2 = 3a + 2·c4 (c4 ∈ {0,1,2}). This spans ALL pow2 (rem2 ≠ 1; 2^(6+1) alone would need a radix-2).
    b9 = -1; b12 = 0; a = 0; c4 = 0
    for cand9 in (v3 ÷ 2):-1:0
        c12 = v3 - 2cand9
        rem2 = v2 - 6 - 2c12
        rem2 < 0 && continue
        r3 = rem2 % 3
        if r3 == 0
            aa, cc = rem2 ÷ 3, 0
        elseif r3 == 2
            aa, cc = (rem2 - 2) ÷ 3, 1
        else                                # r3 == 1: 3a+2·2 = rem2 needs rem2 ≥ 4 (rem2 == 1 ⇒ radix-2)
            rem2 < 4 && continue
            aa, cc = (rem2 - 4) ÷ 3, 2
        end
        b9 = cand9; b12 = c12; a = aa; c4 = cc; break
    end
    b9 < 0 && return nothing
    k::Kernel = B64W8(T, fwd)
    for _ in 1:b9;  k = MR9W8(k, fwd);  end
    for _ in 1:b12; k = MR12W8(k, fwd); end
    for _ in 1:a;   k = MR8W8(k, fwd);  end
    for _ in 1:c4;  k = MR4W8(k, fwd);  end
    for _ in 1:v5;  k = MR5W8(k, fwd);  end
    RPlan(k)
end

# Pure-pow2 W=8 kernel of exponent e (no partial columns needed: every base/radix is ÷4-clean). Small
# bases B16/B32/B64 (2⁴/2⁵/2⁶), e==7 = B16·radix-8, e≥8 the B256/B512 monolith + radix-8/4 chain. e<4
# (2³ and below) is unsupported here (no B8W8) — those low-v2 sizes stay on the fallback. nothing if none.
function _pow2_kernel_w8(::Type{T}, e::Int, fwd::Bool) where {T}
    e == 2 && return B4W8(T, fwd)
    e == 3 && return B8W8(T, fwd)
    e == 4 && return B16W8(T, fwd)
    e == 5 && return B32W8(T, fwd)
    e == 6 && return B64W8(T, fwd)
    e == 7 && return MR8W8(B16W8(T, fwd), fwd)              # 2⁴·2³
    e >= 8 || return nothing
    base, m = e % 3 == 0 ? (B512W8(T, fwd), e - 9) : (B256W8(T, fwd), e - 8)
    m % 3 == 1 && return nothing
    k::Kernel = base
    for _ in 1:(m ÷ 3); k = MR8W8(k, fwd); end
    m % 3 == 2 && (k = MR4W8(k, fwd))
    return k
end

# Small-base smooth tree for 2·3·5-smooth sizes the main W=8 solver rejects (its base is B64W8 = 2⁶, with
# no radix-3 for a lone factor of 3). Here a pow2 W=8 base consumes the 2s (v2≥4 ⇒ B16/B32/B64…, so every
# inner length stays a multiple of 4 — no partial-column path), then radix-9 (pairs of 3) + radix-3 (lone
# 3) + radix-5. Unlocks 2^k·{3,5,3·5} F32 non-pow2 sizes (48/80/96/160/192/240/384/480/720…). Purely
# additive (only reached when the main solver returns nothing), so it can't change any existing W=8 tree.
function _plan_tree_w8_small(::Type{T}, n::Int, fwd::Bool) where {T}
    n >= 1 || return nothing
    v2 = 0; t = n; while t % 2 == 0; t ÷= 2; v2 += 1; end
    v3 = 0; while t % 3 == 0; t ÷= 3; v3 += 1; end
    v5 = 0; while t % 5 == 0; t ÷= 5; v5 += 1; end
    v7 = 0; while t % 7 == 0; t ÷= 7; v7 += 1; end
    if t != 1                                               # residual bare prime 11/13/17/19 → BP-W8 leaf (D2)
        # The padding trick can't reach these (no avx_column_butterfly{11,13,19}); use the direct size-P
        # BPW8 leaf as the innermost base, wrapped by radix-9/3/5/7 for the smooth part and one radix-2
        # (MR2W8) for the lone factor of 2 (2·P composites 22/26/190). v2≤1 only (the targets); v2≥2 with a
        # residual prime falls back. Float32-only path ⇒ no F64 concern.
        (11 <= t <= 19 && _isprime_odd(t) && v2 <= 1) || return nothing
        kp::Kernel = BPW8(T, t, fwd)
        for _ in 1:(v3 ÷ 2); kp = MR9W8(kp, fwd); end
        isodd(v3) && (kp = MR3W8(kp, fwd))
        for _ in 1:v5; kp = MR5W8(kp, fwd); end
        for _ in 1:v7; kp = MR7W8(kp, fwd); end
        v2 == 1 && (kp = MR2W8(kp, fwd))
        return RPlan(kp)
    end
    if v2 == 0                                              # pure odd: padding-trick (B1 base, M=1 ⇒ rem∈{1,3})
        # Odd {3,5,7}-smooth via the B1 identity leaf + odd-M rem∈{1,3} tails on MR9/MR3/MR5/MR7 — every
        # inner length is odd ⇒ rem ∈ {1,3} at every pass. Clears the odd prime-powers (9/25/49/27/45/63/
        # 75/81/343…) and the keystone 49=7² (= rfft-98 inner → r2r DST-I n=48). n=1 unsupported here.
        n == 1 && return nothing
        k0::Kernel = B1W8(T, fwd)
        for _ in 1:(v3 ÷ 2); k0 = MR9W8(k0, fwd); end       # radix-9 (pairs of 3), then lone 3, then 5/7 —
        isodd(v3) && (k0 = MR3W8(k0, fwd))                  # same innermost→outermost order as the v2=1 path
        for _ in 1:v5; k0 = MR5W8(k0, fwd); end             # (measured faster than pushing the lone 3 outermost)
        for _ in 1:v7; k0 = MR7W8(k0, fwd); end
        return RPlan(k0)
    end
    if v2 == 1                                              # 2·odd: rem = M mod 4 = 2 uniformly (partial cols)
        k1::Kernel = B2W8(T, fwd)                           # B2 base (the lone factor of 2)
        for _ in 1:(v3 ÷ 2); k1 = MR9W8(k1, fwd); end       # radix-9 per pair of 3s (preferred; fewer passes)
        isodd(v3) && (k1 = MR3W8(k1, fwd))                  # lone factor of 3 (M ≡ 2 mod 4 ⇒ rem-2 tail)
        for _ in 1:v5; k1 = MR5W8(k1, fwd); end             # then radix-5
        for _ in 1:v7; k1 = MR7W8(k1, fwd); end             # radix-7 per factor of 7 (98=2·7² → DST-I n=48)
        return RPlan(k1)
    end
    v7 <= 1 || return nothing                               # v2≥2 path: radix-7 only as the lone factor
    v2 >= 2 || return nothing                               # need a ÷4-clean base ≥ B4W8 (v2≤1 → partial cols)
    base = _pow2_kernel_w8(T, v2, fwd)
    isnothing(base) && return nothing
    k::Kernel = base
    for _ in 1:(v3 ÷ 2); k = MR9W8(k, fwd); end
    isodd(v3) && (k = MR3W8(k, fwd))
    for _ in 1:v5; k = MR5W8(k, fwd); end
    v7 == 1 && (k = MR7W8(k, fwd))
    return RPlan(k)
end

# W=8-clean tree (main solver), else the additive small-base tree (above). The Float64 W=8 path = 512-bit
# ⇒ needs real AVX-512; Float32 W=8 = 256-bit (plain AVX2) is always buildable — so the gate is F64-only.
plan_tree_w8(n::Int, fwd::Bool = true) = plan_tree_w8(Float64, n, fwd)
function plan_tree_w8(::Type{T}, n::Int, fwd::Bool = true) where {T}
    T === Float64 && !_HAS_AVX512 && return nothing
    r = _plan_tree_w8_main(T, n, fwd)
    isnothing(r) || return r
    # The small-base path (B16/B32/B64W8 + radix-3) is Float32-ONLY: it exists to give ComplexF32 a fast
    # non-64-multiple non-pow2 kernel (no W=4 Float64 port exists for it). ComplexF64 already has the tuned
    # W=4 `AvxMixedRadixPlan` (B16/B18/B36 + radix passes) for these sizes — offering a slower W=8 variant
    # only lets autoplan's noisy plan-time timing mis-rank it and REGRESS F64 (measured). So gate it off.
    T === Float32 || return nothing
    return _plan_tree_w8_small(T, n, fwd)
end
