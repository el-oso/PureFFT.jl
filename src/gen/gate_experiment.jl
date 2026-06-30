# GATE experiment: does an explicit CSE pass reduce the COMPILED (post-LLVM) arithmetic
# instruction count below what LLVM already extracts from the naive split-real emission?
#
# Emits the size-25 split-real DFT body two ways and compares:
#   naive  = current `_gen_dft_soa_mixed!(25)` emission (PureFFT internal).
#   cse    = IR(25) -> CSE -> emit  (this directory's generator).
# (a) bit-exact FIRST (naive ≡ cse ≡ reference DFT, ≤1e-13), then
# (b) @code_native debuginfo=:none on both, count vfmadd*/vmul*/vadd*/vsub*/... + note spills.
#
# Run:  julia --project=. src/gen/gate_experiment.jl
import PureFFT
using InteractiveUtils: code_native
include(joinpath(@__DIR__, "ir.jl"))

const R = 25
const S = -1

# --- naive: PureFFT's `_gen_dft_soa_mixed!`, scalar Float64, single transform ---------------------
@generated function naive_dft!(outr, outi, xr, xi, ::Val{Rv}, ::Val{Sv}) where {Rv, Sv}
    T = Float64
    sb = Any[]
    sr = Vector{Any}(undef, Rv); si = Vector{Any}(undef, Rv)
    for j in 0:(Rv - 1)
        a = Symbol("sinr", j); b = Symbol("sini", j)
        sr[j + 1] = a; si[j + 1] = b
        push!(sb, :(@inbounds $a = xr[$j + 1]))
        push!(sb, :(@inbounds $b = xi[$j + 1]))
    end
    sor, soi = PureFFT._gen_dft_soa_mixed!(sb, sr, si, Rv, Sv, T, Ref(0), "s")
    for k in 0:(Rv - 1)
        push!(sb, :(@inbounds outr[$(k + 1)] = $(sor[k + 1])))
        push!(sb, :(@inbounds outi[$(k + 1)] = $(soi[k + 1])))
    end
    push!(sb, :(return nothing))
    return Expr(:block, sb...)
end

# --- cse: IR -> CSE -> emit ----------------------------------------------------------------------
@generated function cse_dft!(outr, outi, xr, xi, ::Val{Rv}, ::Val{Sv}) where {Rv, Sv}
    nodes, oR, oI = build_dft_ir(Rv, Sv)
    cnodes, remap = cse(nodes)
    return emit_block(cnodes, Int[remap[i] for i in oR], Int[remap[i] for i in oI])
end

# --- ir-naive: IR -> emit (NO CSE) — bridge to confirm the IR reproduces the naive arithmetic ----
@generated function irnaive_dft!(outr, outi, xr, xi, ::Val{Rv}, ::Val{Sv}) where {Rv, Sv}
    nodes, oR, oI = build_dft_ir(Rv, Sv)
    return emit_block(nodes, oR, oI)
end

# --- reference DFT (Complex, direct) -------------------------------------------------------------
function refdft(x::Vector{ComplexF64}, s::Int)
    N = length(x)
    [sum(x[j + 1] * cispi(s * 2 * j * k / N) for j in 0:(N - 1)) for k in 0:(N - 1)]
end

relerr(a, b) = maximum(abs.(a .- b)) / max(maximum(abs.(b)), eps())

function run_correctness()
    x = randn(ComplexF64, R)
    xr = real.(x); xi = imag.(x)
    nr = zeros(R); ni = zeros(R); naive_dft!(nr, ni, xr, xi, Val(R), Val(S))
    cr = zeros(R); ci = zeros(R); cse_dft!(cr, ci, xr, xi, Val(R), Val(S))
    ir = zeros(R); ii = zeros(R); irnaive_dft!(ir, ii, xr, xi, Val(R), Val(S))
    ref = refdft(x, S)
    nv = complex.(nr, ni); cv = complex.(cr, ci); iv = complex.(ir, ii)
    e_nc  = relerr(nv, cv)       # naive vs cse
    e_ni  = relerr(nv, iv)       # naive vs ir-naive
    e_nr  = relerr(nv, ref)      # naive vs reference
    e_cr  = relerr(cv, ref)      # cse   vs reference
    println("bit-exact / accuracy (rel-err):")
    println("  naive vs cse        : ", e_nc)
    println("  naive vs ir-naive   : ", e_ni)
    println("  naive vs reference  : ", e_nr)
    println("  cse   vs reference  : ", e_cr)
    ok = e_nc <= 1e-13 && e_nr <= 1e-13 && e_cr <= 1e-13
    println("  PASS (≤1e-13)       : ", ok)
    return ok
end

# --- compiled instruction counting ---------------------------------------------------------------
const ARITH = r"\b(vfmadd|vfmsub|vfnmadd|vfnmsub)[0-9a-z]*\b|\bv(mul|add|sub)(pd|sd)\b"
# stack spills: a vmov to/from a [rsp/rbp ± off] memory operand
const SPILL = r"\bvmov[a-z]*\b.*\b(rsp|rbp)\b"

function native_str(f)
    io = IOBuffer()
    code_native(io, f, Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64},
                             Val{R}, Val{S}}; debuginfo = :none, dump_module = false)
    return String(take!(io))
end

function count_mnemonics(asm::String)
    arith = 0; spill = 0
    detail = Dict{String, Int}()
    for ln in split(asm, '\n')
        code = strip(first(split(ln, '#')))            # drop trailing comments
        isempty(code) && continue
        for m in eachmatch(ARITH, code)
            arith += 1
            mn = first(split(strip(m.match)))
            detail[mn] = get(detail, mn, 0) + 1
        end
        occursin(SPILL, code) && (spill += 1)
    end
    return arith, spill, detail
end

function run_counts()
    for (name, f) in (("naive", naive_dft!), ("cse", cse_dft!), ("ir-naive", irnaive_dft!))
        asm = native_str(f)
        a, s, d = count_mnemonics(asm)
        println("\n=== $name ===")
        println("  arithmetic insns : ", a)
        println("  stack spills     : ", s)
        println("  breakdown        : ", sort(collect(d); by = first))
    end
end

# IR-level stats (how much CSE actually removed)
function run_ir_stats()
    nodes, oR, oI = build_dft_ir(R, S)
    cnodes, _ = cse(nodes)
    na = count(n -> n.op in (ADD, SUB), nodes)
    nm = count(n -> n.op == MULC, nodes)
    nf = count(n -> n.op in (FMA, FMS), nodes)
    ca = count(n -> n.op in (ADD, SUB), cnodes)
    cm = count(n -> n.op == MULC, cnodes)
    cf = count(n -> n.op in (FMA, FMS), cnodes)
    println("\nIR node counts (generate-time, before LLVM):")
    println("  naive : total=$(length(nodes))  add/sub=$na  mulc=$nm  fma/fms=$nf")
    println("  cse   : total=$(length(cnodes))  add/sub=$ca  mulc=$cm  fma/fms=$cf")
    println("  CSE removed $(length(nodes) - length(cnodes)) nodes "
            * "($(round(100*(1-length(cnodes)/length(nodes)); digits=1))%)")
end

println("PureFFT codelet-generator GATE — size-$R split-real DFT, CSE vs LLVM\n")
ok = run_correctness()
run_ir_stats()
run_counts()
println("\ncorrectness gate passed: ", ok)
