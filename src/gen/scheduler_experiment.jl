# SCHEDULER GATE: does genfft-style register-pressure scheduling reduce the COMPILED stack-spill
# count below what LLVM's own scheduler + register allocator achieves on our codelets?
#
# Compares, per size R, two emissions of the SAME split-real DFT DAG (ir.jl):
#   naive     = IR in natural build order (all 2R loads front-loaded, then compute).
#   scheduled = schedule_ir() reorder (Sethi–Ullman, heavier-subtree-first, demand-driven loads).
# Both are pure SSA reorders ⇒ bit-identical values. The reorder only changes live ranges, so the
# question is whether the IR-level peak-liveness reduction survives LLVM as fewer stack spills.
#
# (a) bit-exact FIRST (scheduled ≡ naive, and both ≈ reference DFT ≤1e-13), then
# (b) @code_native debuginfo=:none: count stack spills (vmov to/from rsp/rbp) + arith insns,
#     plus the IR-level peak-liveness before/after, for each size.
#
# Run:  taskset -c 2 julia --project=. src/gen/scheduler_experiment.jl
using InteractiveUtils: code_native
include(joinpath(@__DIR__, "ir.jl"))
include(joinpath(@__DIR__, "scheduler.jl"))

const SIZES = (25, 49, 64)
const S = -1

# --- naive: IR in natural build order ------------------------------------------------------------
@generated function naive_dft!(outr, outi, xr, xi, ::Val{Rv}, ::Val{Sv}) where {Rv, Sv}
    nodes, oR, oI = build_dft_ir(Rv, Sv)
    return emit_block(nodes, oR, oI)
end

# --- scheduled (greedy register-pressure): IR -> schedule_ir -> emit -----------------------------
@generated function sched_dft!(outr, outi, xr, xi, ::Val{Rv}, ::Val{Sv}) where {Rv, Sv}
    nodes, oR, oI = build_dft_ir(Rv, Sv)
    snodes, sR, sI = schedule_ir(nodes, oR, oI)
    return emit_block(snodes, sR, sI)
end

# --- scheduled (Sethi–Ullman DFS contrast): IR -> schedule_ir_dfs -> emit ------------------------
@generated function dfs_dft!(outr, outi, xr, xi, ::Val{Rv}, ::Val{Sv}) where {Rv, Sv}
    nodes, oR, oI = build_dft_ir(Rv, Sv)
    snodes, sR, sI = schedule_ir_dfs(nodes, oR, oI)
    return emit_block(snodes, sR, sI)
end

# --- reference DFT (Complex, direct) -------------------------------------------------------------
refdft(x, s) = (N = length(x); [sum(x[j+1] * cispi(s*2*j*k/N) for j in 0:N-1) for k in 0:N-1])
relerr(a, b) = maximum(abs.(a .- b)) / max(maximum(abs.(b)), eps())

# --- compiled instruction counting (same regexes as the CSE gate) --------------------------------
const ARITH = r"\b(vfmadd|vfmsub|vfnmadd|vfnmsub)[0-9a-z]*\b|\bv(mul|add|sub)(pd|sd)\b"
const SPILL = r"\bvmov[a-z]*\b.*\b(rsp|rbp)\b"

function native_str(f, R)
    io = IOBuffer()
    code_native(io, f, Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64},
                             Val{R}, Val{S}}; debuginfo = :none, dump_module = false)
    return String(take!(io))
end

function count_mnemonics(asm)
    arith = 0; spill = 0
    for ln in split(asm, '\n')
        code = strip(first(split(ln, '#')))
        isempty(code) && continue
        for _ in eachmatch(ARITH, code); arith += 1; end
        occursin(SPILL, code) && (spill += 1)
    end
    return arith, spill
end

function run_size(R)
    println("\n", "="^64, "\nSIZE $R  (split-real DFT, s=$S)")

    # (a) bit-exact
    x = randn(ComplexF64, R); xr = real.(x); xi = imag.(x)
    nr = zeros(R); ni = zeros(R); naive_dft!(nr, ni, xr, xi, Val(R), Val(S))
    sr = zeros(R); si = zeros(R); sched_dft!(sr, si, xr, xi, Val(R), Val(S))
    dr = zeros(R); di = zeros(R); dfs_dft!(dr, di, xr, xi, Val(R), Val(S))
    nv = complex.(nr, ni); sv = complex.(sr, si); dv = complex.(dr, di); ref = refdft(x, S)
    e_ns = relerr(nv, sv); e_nd = relerr(nv, dv); e_nr = relerr(nv, ref); e_sr = relerr(sv, ref)
    ok = e_ns <= 1e-13 && e_nd <= 1e-13 && e_nr <= 1e-13 && e_sr <= 1e-13
    println("  bit-exact: greedy vs naive = $e_ns | dfs vs naive = $e_nd | naive vs ref = $e_nr")
    println("  PASS (≤1e-13): $ok")

    # IR peak-liveness before/after
    nodes, oR, oI = build_dft_ir(R, S)
    gnodes, gR, gI = schedule_ir(nodes, oR, oI)
    fnodes, fR, fI = schedule_ir_dfs(nodes, oR, oI)
    pl_n = peak_liveness(nodes, oR, oI)
    pl_g = peak_liveness(gnodes, gR, gI)
    pl_f = peak_liveness(fnodes, fR, fI)
    println("  IR nodes: $(length(nodes))")
    println("  IR peak-liveness: naive=$pl_n  greedy=$pl_g  dfs=$pl_f")

    # (b) compiled spills + arith
    an, spn = count_mnemonics(native_str(naive_dft!, R))
    ag, spg = count_mnemonics(native_str(sched_dft!, R))
    af, spf = count_mnemonics(native_str(dfs_dft!, R))
    println("  compiled | naive:  arith=$an  spills=$spn")
    println("  compiled | greedy: arith=$ag  spills=$spg   (Δspill=$(spg-spn), Δarith=$(ag-an))")
    println("  compiled | dfs:    arith=$af  spills=$spf   (Δspill=$(spf-spn), Δarith=$(af-an))")
    return (R, ok, pl_n, pl_g, pl_f, an, spn, ag, spg, af, spf)
end

println("PureFFT codelet-generator SCHEDULER GATE — register-pressure scheduling vs LLVM")
results = [run_size(R) for R in SIZES]

println("\n", "="^64, "\nSUMMARY  (live=IR peak-liveness, sp=compiled stack spills)")
println(rpad("size",5), rpad("PASS",6),
        rpad("liveN",6), rpad("liveG",6), rpad("liveF",6),
        rpad("spN",6), rpad("spG",6), rpad("spF",6), rpad("ΔspG",6), "ΔspF")
for (R, ok, pln, plg, plf, an, spn, ag, spg, af, spf) in results
    println(rpad(R,5), rpad(ok,6),
            rpad(pln,6), rpad(plg,6), rpad(plf,6),
            rpad(spn,6), rpad(spg,6), rpad(spf,6), rpad(spg-spn,6), spf-spn)
end
