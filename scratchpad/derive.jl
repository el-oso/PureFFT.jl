naive_dct4(x) = [2*sum(x[j+1]*cos(pi*(2j+1)*(2k+1)/(4length(x))) for j in 0:length(x)-1) for k in 0:length(x)-1]
function mydct4(x)
    n=length(x); T=Float64
    pre  = [cispi(-T(p)/T(n))      for p in 0:n-1]
    post = [cispi(-T(2k+1)/T(4n))  for k in 0:n-1]
    c = zeros(ComplexF64, n)
    for m in 0:((n+1)÷2 - 1); c[m+1] = pre[m+1]*x[2m+1]; end
    for m in 0:(n÷2 - 1);     c[n-m] = pre[n-m]*(-x[2m+2]); end
    # naive DFT to avoid FFTW dep in scratch
    C = [sum(c[j+1]*cispi(-2*T(j*k)/T(n)) for j in 0:n-1) for k in 0:n-1]
    [2*real(post[k+1]*C[k+1]) for k in 0:n-1]
end
for n in (1,2,3,4,5,8,9,16,17,32)
    x=randn(n)
    a=mydct4(x); c=naive_dct4(x)
    e2=maximum(abs.(a.-c))
    si=maximum(abs.(mydct4(a) .- 2n.*x))
    println("n=$n  vsNaive=$(round(e2,sigdigits=3))  selfinv=$(round(si,sigdigits=3))")
end
