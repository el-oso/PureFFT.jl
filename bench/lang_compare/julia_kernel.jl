# Controlled language experiment — JULIA side.
#
# The SAME radix-2 DIT algorithm as rust/src/main.rs: split-layout (separate re/im Float64),
# precomputed twiddles indexed with a stride, in-place bit-reversal, iterative butterfly stages,
# FMA-fused complex multiply via `muladd`, unchecked indexing via `@inbounds`. Same memory
# layout, same operations, same numerics. Both compile through LLVM. Run with `julia -O3`.
#
# Timing is identical to the Rust harness: per-transform = (time(copy+fft) - time(copy)) / K,
# minimum over trials. A checksum is printed to defeat dead-code elimination.

function radix2_dit!(xr, xi, twr, twi, n)
    @inbounds begin
        # bit-reversal permutation (applied to both re and im)
        j = 0
        for i in 0:(n - 2)
            if i < j
                xr[i + 1], xr[j + 1] = xr[j + 1], xr[i + 1]
                xi[i + 1], xi[j + 1] = xi[j + 1], xi[i + 1]
            end
            m = n >> 1
            while m >= 1 && j >= m
                j -= m
                m >>= 1
            end
            j += m
        end
        # iterative radix-2 decimation-in-time
        len = 2
        while len <= n
            half = len >> 1
            stride = n ÷ len
            base = 0
            while base < n
                ti = 0
                for jj in 0:(half - 1)
                    wr = twr[ti + 1]; wi = twi[ti + 1]
                    pr = xr[base + jj + half + 1]; pii = xi[base + jj + half + 1]
                    tr = muladd(pr, wr, -(pii * wi))
                    tii = muladd(pr, wi, pii * wr)
                    ar = xr[base + jj + 1]; ai = xi[base + jj + 1]
                    xr[base + jj + 1] = ar + tr
                    xi[base + jj + 1] = ai + tii
                    xr[base + jj + half + 1] = ar - tr
                    xi[base + jj + half + 1] = ai - tii
                    ti += stride
                end
                base += len
            end
            len <<= 1
        end
    end
    return
end

function main()
    sizes = (64, 256, 1024, 4096, 16384, 65536, 262144)
    println("# julia  -O$(Base.JLOptions().opt_level)  $(Sys.CPU_NAME)")
    println("# n\tns_per_transform\tGFLOPS\tchecksum")
    for n in sizes
        twr = [cos(-2pi * k / n) for k in 0:(n ÷ 2 - 1)]
        twi = [sin(-2pi * k / n) for k in 0:(n ÷ 2 - 1)]
        # deterministic input, identical to the Rust side → matching checksums verify the
        # two kernels compute exactly the same transform.
        srcr = [((k * 2 + 1) % 17) / 17 - 0.5 for k in 0:(n - 1)]
        srci = [((k * 3 + 2) % 19) / 19 - 0.5 for k in 0:(n - 1)]
        workr = similar(srcr); worki = similar(srci)
        # target ~0.05 s of transforms per trial
        K = max(1, Int(round(2.0e8 / (n * log2(n)))))
        # warm up
        copyto!(workr, srcr); copyto!(worki, srci)
        radix2_dit!(workr, worki, twr, twi, n)
        best = Inf
        chk = 0.0
        for _ in 1:25
            t1 = time_ns()
            for _ in 1:K
                copyto!(workr, srcr); copyto!(worki, srci)
                radix2_dit!(workr, worki, twr, twi, n)
            end
            t2 = time_ns()
            chk += workr[1] + worki[2]   # work holds the FFT result here (also defeats DCE)
            for _ in 1:K
                copyto!(workr, srcr); copyto!(worki, srci)
            end
            t3 = time_ns()
            per = ((t2 - t1) - (t3 - t2)) / K
            best = min(best, per)
        end
        gf = 5 * n * log2(n) / best
        println("$n\t$(round(best; digits = 1))\t$(round(gf; digits = 1))\t$(round(chk; digits = 3))")
    end
    return
end

main()
