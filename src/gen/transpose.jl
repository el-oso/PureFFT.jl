# Generated packed in-register transpose (WIRED — P0.2: avx_transpose{3,5,7,13}_packed forward to this).
#
# Generalizes the hand-written avx_transpose{3,5,7,9,13}_packed (avxradix/avxport.jl) to ANY odd N.
# These do a 2×N → N×2 packed transpose: N input V4f registers, each holding 2 complex
# [lo_i, hi_i] (lanes [0,1]=lo complex, [2,3]=hi complex). The 2N complex are re-packed in
# column-major order — first all N lo-complex (a_1..a_N), then all N hi-complex (b_1..b_N) —
# 2-per-register into N output V4f.
#
# THE NETWORK (regular, no recursion needed) for N = 2H+1:
#   out[1..H]      = avx_unpacklo_complex(rs[2j-1], rs[2j])   j=1..H  → [a_{2j-1}, a_{2j}]
#                    (covers a_1..a_{N-1}; a_N is the bridge's first half)
#   out[H+1]       = _blend03(rs[1], rs[N])                            → [a_N, b_1]   (bridge)
#   out[H+2..N]    = avx_unpackhi_complex(rs[2j], rs[2j+1])   j=1..H  → [b_{2j}, b_{2j+1}]
#                    (covers b_2..b_N; b_1 is the bridge's second half)
#
# Each output is exactly ONE 128-bit shuffle (vperm2f128 / vunpck / vblend), so the network is
# N shuffles for N outputs — identical op-count to the hand-written codelets.
#
# Generate-time only: the @generated body emits straight-line code with LITERAL tuple indices
# (CLAUDE.md rule 1 — never index a tuple with a runtime variable). Emitted code is concrete /
# trim-safe (no Vector{Any} at runtime). Needs avx_unpacklo_complex / avx_unpackhi_complex /
# _blend03 from avxport.jl in scope.

@inline @generated function gen_transpose_packed(rs::NTuple{N, V4f}) where {N}   # @inline: large-N (e.g. 13) else stays a non-inlined call in the _trans pass loop
    isodd(N) || error("gen_transpose_packed: N must be odd, got $N")
    H = (N - 1) ÷ 2
    outs = Any[]
    for j in 1:H                                    # lo-complex pairs: a_1..a_{N-1}
        push!(outs, :(avx_unpacklo_complex(rs[$(2j - 1)], rs[$(2j)])))
    end
    push!(outs, :(_blend03(rs[1], rs[$N])))         # bridge: [a_N, b_1]
    for j in 1:H                                    # hi-complex pairs: b_2..b_N
        push!(outs, :(avx_unpackhi_complex(rs[$(2j)], rs[$(2j + 1)])))
    end
    Expr(:tuple, outs...)
end
