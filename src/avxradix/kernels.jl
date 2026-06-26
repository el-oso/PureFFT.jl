# Canonical faithful-port FFT kernels (no verify/bench blocks). Built on avxport.jl.
# These are the reusable pieces; verify scripts and (eventually) PureFFT integration include this.
include(joinpath(@__DIR__, "avxport.jl"))
using SIMD: Vec

# ===== Butterfly7Avx64 =====
function bf7_twiddles(forward::Bool)
    t1r, t1i = compute_twiddle(1, 7, forward); t2r, t2i = compute_twiddle(2, 7, forward); t3r, t3i = compute_twiddle(3, 7, forward)
    (V4f((t1r, t1r, t1i, t1i)), V4f((t2r, t2r, t2i, t2i)), V4f((t3r, t3r, t3i, t3i)), V4f((t3r, t3r, -t3i, -t3i)), V4f((t1r, t1r, -t1i, -t1i)))
end
@inline butterfly7!(buf, tw, invm::V4f, invm_lo::V2f) = butterfly7!(buf, 0, tw, invm, invm_lo)
function butterfly7!(buf, base::Int, tw, invm::V4f, invm_lo::V2f)
    c0 = avx_load_partial1(buf, base); input0 = avx_merge(c0, c0)
    input12 = avx_load_complex(buf, base + 1); input3 = avx_load_partial1(buf, base + 3); input4 = avx_load_partial1(buf, base + 4); input56 = avx_load_complex(buf, base + 5)
    input65 = avx_reverse_complex(input56)
    sum12, diff65 = avx_butterfly2(input12, input65); sum3, diff4 = avx_butterfly2(input3, input4)
    rotated65 = avx_rotate90(diff65, invm); rotated4 = avx_rotate90(diff4, invm_lo)
    mid16, mid25 = avx_transpose_2x2(sum12, rotated65); mid34 = avx_merge(sum3, rotated4)
    o0 = avx_add(avx_add(avx_lo(mid16), avx_lo(mid25)), avx_add(avx_lo(input0), avx_lo(mid34)))
    a = avx_mul(mid16, tw[1]); b = avx_mul(mid16, tw[2]); c = avx_mul(mid16, tw[3])
    a = avx_fmadd(mid25, tw[2], a); b = avx_fmadd(mid25, tw[4], b); c = avx_fmadd(mid25, tw[5], c)
    tw16 = avx_fmadd(mid34, tw[3], a); tw25 = avx_fmadd(mid34, tw[5], b); tw34 = avx_fmadd(mid34, tw[2], c)
    tw12, tw65 = avx_transpose_2x2(tw16, tw25); tw03 = avx_add(avx_lo(tw34), avx_lo(input0))
    out12, out65 = avx_butterfly2(tw12, tw65); final12 = avx_add(out12, input0)
    out56 = avx_reverse_complex(out65); final56 = avx_add(out56, input0); final3, final4 = avx_butterfly2(tw03, avx_hi(tw34))
    avx_store_partial1!(buf, base, o0); avx_store_complex!(buf, base + 1, final12); avx_store_partial1!(buf, base + 3, final3); avx_store_partial1!(buf, base + 4, final4); avx_store_complex!(buf, base + 5, final56)
end

# ===== Butterfly36Avx64 (6x6) =====
function bf36_twiddles(forward::Bool)
    ntuple(15) do idx0
        idx = idx0 - 1; y = (idx % 5) + 1; x = (idx ÷ 5) * 2
        avx_mixedradix_twiddle_chunk(x, y, 36, forward)
    end
end
@inline function avx_transpose_6x6(r0::NTuple{6}, r1::NTuple{6}, r2::NTuple{6})
    t(a, b) = avx_transpose_2x2(a, b)
    o00 = t(r0[1], r0[2]); o01 = t(r1[1], r1[2]); o02 = t(r2[1], r2[2])
    o10 = t(r0[3], r0[4]); o11 = t(r1[3], r1[4]); o12 = t(r2[3], r2[4])
    o20 = t(r0[5], r0[6]); o21 = t(r1[5], r1[6]); o22 = t(r2[5], r2[6])
    ((o00[1], o00[2], o01[1], o01[2], o02[1], o02[2]),
     (o10[1], o10[2], o11[1], o11[2], o12[1], o12[2]),
     (o20[1], o20[2], o21[1], o21[2], o22[1], o22[2]))
end
@inline _bf36_ld(buf, off) = (avx_load_complex(buf, off), avx_load_complex(buf, off + 6), avx_load_complex(buf, off + 12),
                              avx_load_complex(buf, off + 18), avx_load_complex(buf, off + 24), avx_load_complex(buf, off + 30))
@inline function _bf36_twmul(m, t1, t2, t3, t4, t5)
    (m[1], avx_mul_complex(m[2], t1), avx_mul_complex(m[3], t2), avx_mul_complex(m[4], t3), avx_mul_complex(m[5], t4), avx_mul_complex(m[6], t5))
end
@inline butterfly36!(buf, tw::NTuple{15, V4f}, tw3::V4f) = butterfly36!(buf, buf, 0, tw, tw3)
@inline butterfly36!(buf, base::Int, tw::NTuple{15, V4f}, tw3::V4f) = butterfly36!(buf, buf, base, tw, tw3)
function butterfly36!(out, inp, base::Int, tw::NTuple{15, V4f}, tw3::V4f)  # out-of-place: load inp, store out (out===inp ⇒ in-place)
    mid0 = _bf36_twmul(avx_column_butterfly6(_bf36_ld(inp, base + 0), tw3), tw[1], tw[2], tw[3], tw[4], tw[5])
    mid1 = _bf36_twmul(avx_column_butterfly6(_bf36_ld(inp, base + 2), tw3), tw[6], tw[7], tw[8], tw[9], tw[10])
    mid2 = _bf36_twmul(avx_column_butterfly6(_bf36_ld(inp, base + 4), tw3), tw[11], tw[12], tw[13], tw[14], tw[15])
    t0, t1, t2 = avx_transpose_6x6(mid0, mid1, mid2)
    o0 = avx_column_butterfly6(t0, tw3)
    avx_store_complex!(out, base + 0, o0[1]); avx_store_complex!(out, base + 6, o0[2]); avx_store_complex!(out, base + 12, o0[3]); avx_store_complex!(out, base + 18, o0[4]); avx_store_complex!(out, base + 24, o0[5]); avx_store_complex!(out, base + 30, o0[6])
    o1 = avx_column_butterfly6(t1, tw3)
    avx_store_complex!(out, base + 2, o1[1]); avx_store_complex!(out, base + 8, o1[2]); avx_store_complex!(out, base + 14, o1[3]); avx_store_complex!(out, base + 20, o1[4]); avx_store_complex!(out, base + 26, o1[5]); avx_store_complex!(out, base + 32, o1[6])
    o2 = avx_column_butterfly6(t2, tw3)
    avx_store_complex!(out, base + 4, o2[1]); avx_store_complex!(out, base + 10, o2[2]); avx_store_complex!(out, base + 16, o2[3]); avx_store_complex!(out, base + 22, o2[4]); avx_store_complex!(out, base + 28, o2[5]); avx_store_complex!(out, base + 34, o2[6])
end

# ===== Butterfly64: 8x8, two-phase (col+twiddle+transpose, then row) =====
# twiddles: gen_butterfly_twiddles_separated_columns!(8,8) = 28 = [mixedradix_twiddle_chunk(cs*2, r, 64)
# for cs in 0:3, r in 1:7], index [7cs + r].
bf64_twiddles(fwd) = [avx_mixedradix_twiddle_chunk(cs * 2, r, 64, fwd) for cs in 0:3 for r in 1:7]
@inline _bf64_ld8(buf, b) = (avx_load_complex(buf, b), avx_load_complex(buf, b + 8), avx_load_complex(buf, b + 16), avx_load_complex(buf, b + 24),
                             avx_load_complex(buf, b + 32), avx_load_complex(buf, b + 40), avx_load_complex(buf, b + 48), avx_load_complex(buf, b + 56))
# out-of-place size-64 FFT: load inp, store out; scr (size ≥64 at base) is phase-1 workspace (out===scr ⇒ in-place ok)
function butterfly64!(out, inp, scr, base::Int, tw::Vector{V4f}, rot::V4f)
    @inbounds for cs in 0:3                                  # phase 1: col bf8 + twiddle + transpose → scr
        b = base + cs * 2
        m = avx_column_butterfly8(_bf64_ld8(inp, b)..., rot)
        t = avx_transpose8_packed(m[1], avx_mul_complex(tw[7cs + 1], m[2]), avx_mul_complex(tw[7cs + 2], m[3]), avx_mul_complex(tw[7cs + 3], m[4]),
                                  avx_mul_complex(tw[7cs + 4], m[5]), avx_mul_complex(tw[7cs + 5], m[6]), avx_mul_complex(tw[7cs + 6], m[7]), avx_mul_complex(tw[7cs + 7], m[8]))
        ob = base + cs * 16
        avx_store_complex!(scr, ob, t[1]); avx_store_complex!(scr, ob + 2, t[2]); avx_store_complex!(scr, ob + 4, t[3]); avx_store_complex!(scr, ob + 6, t[4])
        avx_store_complex!(scr, ob + 8, t[5]); avx_store_complex!(scr, ob + 10, t[6]); avx_store_complex!(scr, ob + 12, t[7]); avx_store_complex!(scr, ob + 14, t[8])
    end
    @inbounds for cs in 0:3                                  # phase 2: row bf8 (scr → out)
        b = base + cs * 2
        m = avx_column_butterfly8(_bf64_ld8(scr, b)..., rot)
        avx_store_complex!(out, b, m[1]); avx_store_complex!(out, b + 8, m[2]); avx_store_complex!(out, b + 16, m[3]); avx_store_complex!(out, b + 24, m[4])
        avx_store_complex!(out, b + 32, m[5]); avx_store_complex!(out, b + 40, m[6]); avx_store_complex!(out, b + 48, m[7]); avx_store_complex!(out, b + 56, m[8])
    end
end

# ===== Butterfly256: 32x8 two-phase (faithful port of rustfft Butterfly256Avx64) =====
# phase 1: col bf8 + twiddle + transpose8 (16 columnsets) → scr;  phase 2: col bf32 (4 columnsets) scr → out.
bf256_phase1_tw(fwd) = [avx_mixedradix_twiddle_chunk(cs * 2, r, 256, fwd) for cs in 0:15 for r in 1:7]   # 112, index 7cs+r
bf256_bf32_tw(fwd) = (avx_broadcast_twiddle(1, 32, fwd), avx_broadcast_twiddle(2, 32, fwd), avx_broadcast_twiddle(3, 32, fwd),
    avx_broadcast_twiddle(5, 32, fwd), avx_broadcast_twiddle(6, 32, fwd), avx_broadcast_twiddle(7, 32, fwd))
@inline _bf256_ld8(buf, b) = (avx_load_complex(buf, b), avx_load_complex(buf, b + 32), avx_load_complex(buf, b + 64), avx_load_complex(buf, b + 96),
    avx_load_complex(buf, b + 128), avx_load_complex(buf, b + 160), avx_load_complex(buf, b + 192), avx_load_complex(buf, b + 224))
function butterfly256!(out, inp, scr, base::Int, tw::Vector{V4f}, tw32::NTuple{6, V4f}, rot::V4f)
    @inbounds for cs in 0:15                                 # phase 1: 32×8 col bf8 + twiddle + transpose → scr
        b = base + cs * 2
        m = avx_column_butterfly8(_bf256_ld8(inp, b)..., rot)
        t = avx_transpose8_packed(m[1], avx_mul_complex(tw[7cs + 1], m[2]), avx_mul_complex(tw[7cs + 2], m[3]), avx_mul_complex(tw[7cs + 3], m[4]),
            avx_mul_complex(tw[7cs + 4], m[5]), avx_mul_complex(tw[7cs + 5], m[6]), avx_mul_complex(tw[7cs + 6], m[7]), avx_mul_complex(tw[7cs + 7], m[8]))
        ob = base + cs * 16
        avx_store_complex!(scr, ob, t[1]); avx_store_complex!(scr, ob + 2, t[2]); avx_store_complex!(scr, ob + 4, t[3]); avx_store_complex!(scr, ob + 6, t[4])
        avx_store_complex!(scr, ob + 8, t[5]); avx_store_complex!(scr, ob + 10, t[6]); avx_store_complex!(scr, ob + 12, t[7]); avx_store_complex!(scr, ob + 14, t[8])
    end
    @inbounds for cs in 0:3                                  # phase 2: col bf32 (scr → out)
        b = base + cs * 2
        avx_column_butterfly32(scr, b, 8, out, b, 8, tw32, rot)
    end
end

# ===== Butterfly512: 32x16 two-phase (faithful port of rustfft Butterfly512Avx64) =====
# phase 1: col bf16 + chunked twiddle + transpose4 (16 columnsets) → scr;  phase 2: col bf32 (8 columnsets) scr → out.
bf512_phase1_tw(fwd) = [avx_mixedradix_twiddle_chunk(cs * 2, r, 512, fwd) for cs in 0:15 for r in 1:15]   # 240, chunks of 15
bf512_bf16_tw(fwd) = (avx_broadcast_twiddle(1, 16, fwd), avx_broadcast_twiddle(3, 16, fwd))
function butterfly512!(out, inp, scr, base::Int, tw::Vector{V4f}, tw16::NTuple{2, V4f}, tw32::NTuple{6, V4f}, rot::V4f)
    @inbounds for cs in 0:15                                 # phase 1: 32×16 col bf16 + chunked twiddle + transpose4 → scr
        b = base + cs * 2
        mid = avx_column_butterfly16(inp, b, 32, tw16, rot)
        tc = 15 * cs
        for chunk in 0:3
            j = 4 * chunk
            t0 = chunk == 0 ? mid[1] : avx_mul_complex(mid[j + 1], tw[tc + j])
            t1 = avx_mul_complex(mid[j + 2], tw[tc + j + 1])
            t2 = avx_mul_complex(mid[j + 3], tw[tc + j + 2])
            t3 = avx_mul_complex(mid[j + 4], tw[tc + j + 3])
            tr = avx_transpose4_packed(t0, t1, t2, t3)
            ob = base + cs * 32 + 4 * chunk
            avx_store_complex!(scr, ob, tr[1]); avx_store_complex!(scr, ob + 2, tr[2]); avx_store_complex!(scr, ob + 16, tr[3]); avx_store_complex!(scr, ob + 18, tr[4])
        end
    end
    @inbounds for cs in 0:7                                  # phase 2: col bf32 (scr → out)
        b = base + cs * 2
        avx_column_butterfly32(scr, b, 16, out, b, 16, tw32, rot)
    end
end

# ===== Butterfly9: 3x3, dual-width packed (col 0 partial V2f, cols 1-2 V4f) =====
bf9_twiddles(fwd) = (avx_mixedradix_twiddle_chunk(1, 1, 9, fwd), avx_mixedradix_twiddle_chunk(1, 2, 9, fwd))
function butterfly9!(out, inp, base::Int, tw::NTuple{2, V4f}, bf3::V4f, bf3lo::V2f)
    a1 = avx_load_partial1(inp, base + 0); a2 = avx_load_partial1(inp, base + 3); a3 = avx_load_partial1(inp, base + 6)
    b1 = avx_load_complex(inp, base + 1); b2 = avx_load_complex(inp, base + 4); b3 = avx_load_complex(inp, base + 7)
    mid0 = avx_column_butterfly3(a1, a2, a3, bf3lo)          # V2f cb3 (column 0)
    mid1 = avx_column_butterfly3(b1, b2, b3, bf3)            # V4f cb3 (columns 1,2)
    m2 = avx_mul_complex(mid1[2], tw[1]); m3 = avx_mul_complex(mid1[3], tw[2])
    t0, t1 = avx_transpose_3x3(mid0[1], mid0[2], mid0[3], mid1[1], m2, m3)
    o0 = avx_column_butterfly3(t0[1], t0[2], t0[3], bf3lo)
    o1 = avx_column_butterfly3(t1[1], t1[2], t1[3], bf3)
    avx_store_partial1!(out, base + 0, o0[1]); avx_store_partial1!(out, base + 3, o0[2]); avx_store_partial1!(out, base + 6, o0[3])
    avx_store_complex!(out, base + 1, o1[1]); avx_store_complex!(out, base + 4, o1[2]); avx_store_complex!(out, base + 7, o1[3])
end

# ===== Butterfly18: 3x6 (faithful port of rustfft Butterfly18Avx64) — col 0 partial V2f, cols 1-2 V4f =====
# twiddles: gen_butterfly_twiddles_interleaved_columns!(6,3,1) = chunk(1, y, 18) for y in 1:5.
bf18_twiddles(fwd) = ntuple(y -> avx_mixedradix_twiddle_chunk(1, y, 18, fwd), 5)
# transpose_3x6_to_6x3_f64 (avx64_utils): partial col 0 merged pairwise, full cols 1-2 transposed 2x2.
@inline function avx_transpose_3x6_to_6x3(r0::NTuple{6, V2f}, r1::NTuple{6, V4f})
    t0 = avx_transpose_2x2(r1[1], r1[2]); t1 = avx_transpose_2x2(r1[3], r1[4]); t2 = avx_transpose_2x2(r1[5], r1[6])
    ((avx_merge(r0[1], r0[2]), t0[1], t0[2]),
     (avx_merge(r0[3], r0[4]), t1[1], t1[2]),
     (avx_merge(r0[5], r0[6]), t2[1], t2[2]))
end
@inline butterfly18!(buf, base::Int, tw::NTuple{5, V4f}, bf3::V4f, bf3lo::V2f) = butterfly18!(buf, buf, base, tw, bf3, bf3lo)
function butterfly18!(out, inp, base::Int, tw::NTuple{5, V4f}, bf3::V4f, bf3lo::V2f)  # out===inp ⇒ in-place
    rows0 = ntuple(n -> avx_load_partial1(inp, base + (n - 1) * 3), 6)       # col 0 (1 complex each)
    rows1 = ntuple(n -> avx_load_complex(inp, base + (n - 1) * 3 + 1), 6)    # cols 1,2 (2 complex each)
    mid0 = avx_column_butterfly6(rows0, bf3lo)                               # butterfly6 down the 6 rows
    mid1 = avx_column_butterfly6(rows1, bf3)
    mid1 = (mid1[1], avx_mul_complex(mid1[2], tw[1]), avx_mul_complex(mid1[3], tw[2]),
        avx_mul_complex(mid1[4], tw[3]), avx_mul_complex(mid1[5], tw[4]), avx_mul_complex(mid1[6], tw[5]))
    t0, t1, t2 = avx_transpose_3x6_to_6x3(mid0, mid1)                        # 3x6 → 6x3
    o0 = avx_column_butterfly3(t0[1], t0[2], t0[3], bf3)                     # butterfly3 down the 3 columns
    o1 = avx_column_butterfly3(t1[1], t1[2], t1[3], bf3)
    o2 = avx_column_butterfly3(t2[1], t2[2], t2[3], bf3)
    @inbounds for r in 0:2
        avx_store_complex!(out, base + 6r + 0, o0[r + 1])
        avx_store_complex!(out, base + 6r + 2, o1[r + 1])
        avx_store_complex!(out, base + 6r + 4, o2[r + 1])
    end
end
