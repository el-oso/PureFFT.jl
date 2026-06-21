# Verify the Julia primitive port (avxport.jl) bit-exactly against the Rust golden file.
# Run: julia -O3 --project=. port/verify_primitives.jl
include(joinpath(@__DIR__, "avxport.jl"))
using SIMD: Vec

const GOLD = joinpath(@__DIR__, "..", "bench", "rustfft_compare", "golden.txt")

# parse the "P <name> b0 b1 b2 b3" lines into name => Vec{4,Float64}
gold = Dict{String, V4f}()
for ln in eachline(GOLD)
    startswith(ln, "P ") || continue
    parts = split(ln)
    name = parts[2]
    bs = (parse(UInt64, parts[3]; base = 16), parse(UInt64, parts[4]; base = 16),
          parse(UInt64, parts[5]; base = 16), parse(UInt64, parts[6]; base = 16))
    gold[name] = V4f(reinterpret.(Float64, bs))
end

bits(v::V4f) = ntuple(i -> reinterpret(UInt64, v[i]), 4)
A = gold["A"]; B = gold["B"]; C = gold["C"]

checks = [
    ("swap_A",        avx_swap_complex(A)),
    ("dupre_A",       avx_dup_re(A)),
    ("dupim_A",       avx_dup_im(A)),
    ("reverse_A",     avx_reverse_complex(A)),
    ("unpacklo_AB",   avx_unpacklo_complex(A, B)),
    ("unpackhi_AB",   avx_unpackhi_complex(A, B)),
    ("fmadd_ABC",     avx_fmadd(A, B, C)),
    ("fnmadd_ABC",    avx_fnmadd(A, B, C)),
    ("fmaddsub_ABC",  avx_fmaddsub(A, B, C)),
    ("fmsubadd_ABC",  avx_fmsubadd(A, B, C)),
    ("mulcomplex_AB", avx_mul_complex(A, B)),
    ("rotate90fwd_A", avx_rotate90(A, _ROT90_FWD)),
    ("rotate90inv_A", avx_rotate90(A, _ROT90_INV)),
]

function run_checks(checks, gold)
    allok = true
    for (name, got) in checks
        want = gold[name]
        ok = bits(got) == bits(want)
        allok &= ok
        println(rpad(name, 16), ok ? "  ✓ bit-exact" : "  ✗ MISMATCH\n    got  $(bits(got))\n    want $(bits(want))")
    end
    return allok
end
allok = run_checks(checks, gold)
println(allok ? "\nALL PRIMITIVES BIT-EXACT ✓" : "\nSOME PRIMITIVES MISMATCH ✗")
exit(allok ? 0 : 1)
