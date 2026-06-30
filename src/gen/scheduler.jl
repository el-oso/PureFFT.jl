# Generate-time register-pressure scheduler for the Node IR (genfft phase-3 analogue).
#
# GENERATE-TIME ONLY — operates on the flat `Vector{Node}` IR from ir.jl, before the emitter.
# Reorders the (SSA, side-effect-free) nodes into a topological order chosen to keep the number of
# simultaneously-live values (register pressure) small. The values are unchanged (pure SSA reorder),
# so the emitted code is bit-identical to the naive order — only the live-range structure differs.
#
# Algorithm: greedy register-pressure list scheduling (Sethi–Ullman-style, generalised to a DAG).
#   * "ready" = a node all of whose operands are already scheduled (loads are ready from the start).
#   * At each step pick the ready node that minimises the *change in live-value count*:
#         delta = creates - frees
#         creates = 1 if the node still has a future consumer (or is an output) else 0
#         frees   = number of operands whose LAST remaining use this node is (they go dead)
#     Picking min-delta (tie: free more, then lower index) greedily holds peak liveness down and is
#     demand-driven: a load is scheduled only when its consumer is about to fire, not all up front.
# This directly targets the metric the experiment reports (peak simultaneous liveness), unlike a
# pure heavier-subtree-first DFS which ignores DAG sharing and can raise the global peak.

_arity(n::Node) = (n.op == LOADR || n.op == LOADI) ? 0 : (n.op == MULC ? 1 : 2)

# operands as a small tuple (only the real node-references; loads have none)
function _operand_list(n::Node)
    a = _arity(n)
    a == 0 ? Int[] : a == 1 ? Int[n.a] : Int[n.a, n.b]
end

# Greedy register-pressure schedule. Returns (newnodes, newoutr, newouti) with operands remapped.
# Dead nodes (unreachable from any output) are naturally dropped — they are never made ready.
function schedule_ir(nodes::Vector{Node}, outr::Vector{Int}, outi::Vector{Int})
    N = length(nodes)
    consumers = [Int[] for _ in 1:N]
    pending   = zeros(Int, N)                 # # operands not yet scheduled
    remaining = zeros(Int, N)                 # # consumers not yet scheduled
    for i in 1:N
        for o in _operand_list(nodes[i])
            push!(consumers[o], i)
            pending[i] += 1
            remaining[o] += 1
        end
    end
    is_out = falses(N)
    for i in vcat(outr, outi); is_out[i] = true; end

    ready = Int[i for i in 1:N if pending[i] == 0 && (remaining[i] > 0 || is_out[i])]
    pos = zeros(Int, N)                        # old index -> new position
    order = Int[]

    while !isempty(ready)
        bestk = 0; bestdelta = typemax(Int); bestfrees = -1; besti = typemax(Int)
        for (k, i) in enumerate(ready)
            ops = _operand_list(nodes[i])
            frees = 0
            for o in ops
                # last use of o iff o has exactly this many remaining consumers among `ops`
                if !is_out[o] && remaining[o] == count(==(o), ops)
                    frees += 1
                end
            end
            # distinct operands only (x+x frees one register, not two)
            frees = length(ops) == 2 && ops[1] == ops[2] ?
                    ((!is_out[ops[1]] && remaining[ops[1]] == 2) ? 1 : 0) : frees
            creates = (remaining[i] > 0 || is_out[i]) ? 1 : 0
            delta = creates - frees
            if delta < bestdelta || (delta == bestdelta && frees > bestfrees) ||
               (delta == bestdelta && frees == bestfrees && i < besti)
                bestk = k; bestdelta = delta; bestfrees = frees; besti = i
            end
        end
        i = ready[bestk]
        deleteat!(ready, bestk)
        push!(order, i); pos[i] = length(order)
        for o in _operand_list(nodes[i]); remaining[o] -= 1; end
        for c in consumers[i]
            pending[c] -= 1
            pending[c] == 0 && (remaining[c] > 0 || is_out[c]) && push!(ready, c)
        end
    end

    newnodes = Vector{Node}(undef, length(order))
    for (newpos, old) in enumerate(order)
        n = nodes[old]
        newnodes[newpos] = _arity(n) == 0 ? n :
            _arity(n) == 1 ? Node(n.op, pos[n.a], 0, n.w) :
                             Node(n.op, pos[n.a], pos[n.b], n.w)
    end
    return newnodes, Int[pos[i] for i in outr], Int[pos[i] for i in outi]
end

# --- secondary scheduler: Sethi–Ullman heavier-subtree-first DFS (demand-driven) ----------------
# Kept for the experiment as a *contrast*: it produces a DIFFERENT valid topological order at (very
# nearly) the SAME peak liveness as greedy/naive, yet hands LLVM a markedly different instruction
# stream. Used to show that any spill movement is order-perturbation, not register-pressure change.
function _su_labels(nodes::Vector{Node})
    L = Vector{Int}(undef, length(nodes))
    for i in eachindex(nodes)
        a = _arity(nodes[i])
        if a == 0
            L[i] = 1
        elseif a == 1
            L[i] = L[nodes[i].a]
        else
            l1, l2 = L[nodes[i].a], L[nodes[i].b]
            l1 < l2 && ((l1, l2) = (l2, l1))
            L[i] = l1 == l2 ? l1 + 1 : l1
        end
    end
    return L
end

function schedule_ir_dfs(nodes::Vector{Node}, outr::Vector{Int}, outi::Vector{Int})
    L = _su_labels(nodes)
    pos = zeros(Int, length(nodes)); order = Int[]
    stack = Tuple{Int, Int}[]
    for r in vcat(outr, outi)
        pos[r] != 0 && continue
        push!(stack, (r, 0))
        while !isempty(stack)
            i, stage = pop!(stack)
            if stage == 1
                pos[i] == 0 && (push!(order, i); pos[i] = length(order)); continue
            end
            pos[i] == 0 || continue
            push!(stack, (i, 1))
            ops = _operand_list(nodes[i])
            if length(ops) == 2
                a, b = ops
                lo, hi = L[a] >= L[b] ? (b, a) : (a, b)   # push smaller first → larger popped first
                pos[lo] == 0 && push!(stack, (lo, 0))
                pos[hi] == 0 && push!(stack, (hi, 0))
            elseif length(ops) == 1
                pos[ops[1]] == 0 && push!(stack, (ops[1], 0))
            end
        end
    end
    newnodes = Vector{Node}(undef, length(order))
    for (newpos, old) in enumerate(order)
        n = nodes[old]
        newnodes[newpos] = _arity(n) == 0 ? n :
            _arity(n) == 1 ? Node(n.op, pos[n.a], 0, n.w) :
                             Node(n.op, pos[n.a], pos[n.b], n.w)
    end
    return newnodes, Int[pos[i] for i in outr], Int[pos[i] for i in outi]
end

# Peak simultaneous liveness for a Vector{Node} in evaluation (vector) order.
# A node is live from its definition until its last use (or to the end if it is an output).
function peak_liveness(nodes::Vector{Node}, outr::Vector{Int}, outi::Vector{Int})
    N = length(nodes)
    lastuse = zeros(Int, N)
    for i in 1:N, o in _operand_list(nodes[i])
        lastuse[o] = max(lastuse[o], i)
    end
    for i in vcat(outr, outi); lastuse[i] = N + 1; end   # outputs live to the end
    peak = 0
    for p in 1:N
        live = 0
        for i in 1:p
            lastuse[i] > p && (live += 1)
        end
        peak = max(peak, live)
    end
    return peak
end
