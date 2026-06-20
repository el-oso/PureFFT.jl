# PureFFT side of the rustfft head-to-head. Same copy-subtract harness, same deterministic input,
# same checksum convention as rust/src/main.rs → checksums must match the rustfft output.
#
# Run:  julia -O3 --project=.. bench/rustfft_compare/julia_radix4.jl
# (activate the bench env so PureFFT is available)

import PureFFT

function bench(variant)
    sizes = (64, 256, 1024, 4096, 16384, 65536, 262144)
    println("# PureFFT $variant  -O$(Base.JLOptions().opt_level)  $(Sys.CPU_NAME)")
    println("# n\tns_per_transform\tGFLOPS\tchecksum")
    for n in sizes
        src = [Complex(((k * 2 + 1) % 17) / 17 - 0.5, ((k * 3 + 2) % 19) / 19 - 0.5) for k in 0:(n - 1)]
        work = copy(src)
        p = PureFFT.plan_pfft(src; variant)
        K = max(1, Int(round(2.0e8 / (n * log2(n)))))
        copyto!(work, src); PureFFT.pfft!(work, p)   # warm up
        best = Inf
        chk = 0.0
        for _ in 1:25
            t1 = time_ns()
            for _ in 1:K
                copyto!(work, src); PureFFT.pfft!(work, p)
            end
            t2 = time_ns()
            chk += real(work[1]) + imag(work[2])     # FFT result here (matches Rust checksum)
            for _ in 1:K
                copyto!(work, src)
            end
            t3 = time_ns()
            best = min(best, ((t2 - t1) - (t3 - t2)) / K)
        end
        gf = 5 * n * log2(n) / best
        println("$n\t$(round(best; digits = 1))\t$(round(gf; digits = 1))\t$(round(chk; digits = 3))")
    end
    return
end

bench(:radix4)
bench(:fast)
