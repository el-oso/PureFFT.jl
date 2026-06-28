using PureFFT, JET, LinearAlgebra, ErrorTypes
for T in (Float64, Float32), n in (8, 9, 256)
    x = randn(T, n); y = similar(x)
    p = unwrap(PureFFT.tryplan_r2r(x, REDFT11))
    mul!(y, p, x)  # warm
    a = @allocated mul!(y, p, x)
    println("T=$T n=$n alloc=$a")
    @assert a == 0 "alloc!"
    @test_opt target_modules=(PureFFT,) mul!(y, p, x)
end
println("OK zero-alloc + @test_opt clean")
