# Generates the illustrative figures used in docs/src/guide.md (the tutorial).
# Uses the standard AbstractFFTs interface (fft/ifft/fftshift) — PureFFT is the provider
# (no FFTW loaded). Run:  julia --project=bench bench/plot_guide_examples.jl
using PureFFT, AbstractFFTs, Plots
gr()
const OUT = joinpath(@__DIR__, "..", "docs", "src", "assets")
isdir(OUT) || mkpath(OUT)

# ── 1) 1-D complex FFT: a two-tone signal and its magnitude spectrum ──────────────────────────────
let n = 256
    t = (0:n-1) ./ n
    sig = sin.(2π .* 8 .* t) .+ 0.5 .* sin.(2π .* 20 .* t)          # 8 + 20 cycles
    X = fft(ComplexF64.(sig))                                       # forward FFT (AbstractFFTs → PureFFT)
    mag = abs.(X)[1:(n ÷ 2)]
    p1 = plot(t, sig; title = "signal: 8 + ½·20 cycles", xlabel = "time", lw = 1.6, legend = false, color = :steelblue)
    p2 = plot(0:(n ÷ 2 - 1), mag; title = "fft magnitude |X[k]|", xlabel = "frequency bin k",
              lw = 1.6, legend = false, color = :firebrick, marker = (:circle, 2))
    plot(p1, p2; layout = (1, 2), size = (820, 300), plot_title = "1-D FFT")
    savefig(joinpath(OUT, "guide_fft_1d.png"))
end

# ── 2) N-D (2-D) FFT: a tilted sinusoidal grating and its 2-D spectrum ─────────────────────────────
let n = 64
    A = [cos(2π * (3i + 5j) / n) for i in 0:n-1, j in 0:n-1]        # a 2-D grating (freq (3,5))
    F = fft(ComplexF64.(A))                                         # 2-D FFT
    S = fftshift(log10.(abs.(F) .+ 1e-6))                           # log-magnitude, zero-freq centred
    h1 = heatmap(A; title = "2-D grating (3,5)", aspect_ratio = 1, axis = false, colorbar = false, c = :viridis)
    h2 = heatmap(S; title = "fft 2-D log-magnitude", aspect_ratio = 1, axis = false, colorbar = false, c = :magma)
    plot(h1, h2; layout = (1, 2), size = (720, 340), plot_title = "N-dimensional FFT")
    savefig(joinpath(OUT, "guide_fft_2d.png"))
end

# ── 3) DCT: energy compaction — a smooth signal vs its DCT coefficients ────────────────────────────
let n = 128
    v = [2.0 * exp(-((k - 30) / 25)^2) + 0.4 * (k / n) for k in 1:n]   # a smooth bump + ramp
    c = PureFFT.dct(v)                                                 # orthonormal DCT-II
    p1 = plot(1:n, v; title = "smooth signal", xlabel = "index", lw = 1.6, legend = false, color = :steelblue)
    p2 = bar(0:(n - 1), c; title = "dct coefficients (energy in low k)", xlabel = "coefficient k",
             legend = false, color = :seagreen, linecolor = :seagreen)
    plot(p1, p2; layout = (1, 2), size = (820, 300), plot_title = "DCT — energy compaction")
    savefig(joinpath(OUT, "guide_dct.png"))
end

# ── 4) Worked 2-D example: low-pass filtering an image via the 2-D FFT ─────────────────────────────
let n = 96
    xs = range(-3, 3; length = n)
    base = [exp(-(x^2 + y^2) / 4) + 0.5 * cos(2x) * cos(3y) for x in xs, y in xs]   # smooth structure
    img  = base .+ 0.6 .* randn(n, n)                                                # + high-freq noise
    F = fft(ComplexF64.(img))                                                        # 2-D FFT
    cen = fftshift(F)                                                                # zero freq to centre
    c = (n ÷ 2) + 1; r = n ÷ 6
    mask = [hypot(i - c, j - c) <= r for i in 1:n, j in 1:n]                         # circular low-pass mask
    filt = real.(ifft(ifftshift(cen .* mask)))                                       # mask → inverse 2-D FFT
    cl = (minimum(base), maximum(base))
    h1 = heatmap(img;  title = "noisy image", aspect_ratio = 1, axis = false, colorbar = false, c = :viridis)
    h2 = heatmap(log10.(abs.(cen) .+ 1e-6); title = "2-D spectrum + LP mask", aspect_ratio = 1, axis = false, colorbar = false, c = :magma)
    h3 = heatmap(filt; title = "low-pass filtered (FFT⁻¹)", aspect_ratio = 1, axis = false, colorbar = false, c = :viridis, clims = cl)
    plot(h1, h2, h3; layout = (1, 3), size = (980, 320), plot_title = "2-D low-pass filter via the FFT")
    savefig(joinpath(OUT, "guide_2d_filter.png"))
end

println("wrote guide_fft_1d.png, guide_fft_2d.png, guide_dct.png, guide_2d_filter.png → $OUT")
