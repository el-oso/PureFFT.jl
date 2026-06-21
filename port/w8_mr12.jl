include(joinpath(pwd(),"src","avxradix","recursive.jl"))   # V4f keystone (MR12/B64/RPlan) + W=8 primitives
using SIMD: Vec, shufflevector; import FFTW; using Printf, Statistics
seeded(n)=[Complex(((k*2+1)%17)/17-0.5,((k*3+2)%19)/19-0.5) for k in 0:(n-1)]
@inline function _t4w8(C0::V8f,C1::V8f,C2::V8f,C3::V8f)
    P0=shufflevector(C0,C1,Val((0,1,8,9,2,3,10,11))); P1=shufflevector(C0,C1,Val((4,5,12,13,6,7,14,15)))
    P2=shufflevector(C2,C3,Val((0,1,8,9,2,3,10,11))); P3=shufflevector(C2,C3,Val((4,5,12,13,6,7,14,15)))
    (shufflevector(P0,P2,Val((0,1,2,3,8,9,10,11))),shufflevector(P0,P2,Val((4,5,6,7,12,13,14,15))),shufflevector(P1,P3,Val((0,1,2,3,8,9,10,11))),shufflevector(P1,P3,Val((4,5,6,7,12,13,14,15))))
end
@inline function t8w8(r1,r2,r3,r4,r5,r6,r7,r8); a=_t4w8(r1,r2,r3,r4); b=_t4w8(r5,r6,r7,r8); (a[1],b[1],a[2],b[2],a[3],b[3],a[4],b[4]); end
@inline function t12w8(r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12)
    a=_t4w8(r1,r2,r3,r4); b=_t4w8(r5,r6,r7,r8); c=_t4w8(r9,r10,r11,r12)
    (a[1],b[1],c[1],a[2],b[2],c[2],a[3],b[3],c[3],a[4],b[4],c[4])
end
L8(b,i)=avx_load_complex8(b,i); S8(b,i,v)=avx_store_complex8!(b,i,v)
bf64tw8(fwd)=[avx_mixedradix_twiddle_chunk8(cs*4,r,64,fwd) for cs in 0:1 for r in 1:7]
function bf64_w8!(out,inp,scr,base,tw,rot)
    @inbounds for cs in 0:1; b=base+cs*4
        m=avx_column_butterfly8(L8(inp,b),L8(inp,b+8),L8(inp,b+16),L8(inp,b+24),L8(inp,b+32),L8(inp,b+40),L8(inp,b+48),L8(inp,b+56),rot)
        t=t8w8(m[1],avx_mul_complex(tw[7cs+1],m[2]),avx_mul_complex(tw[7cs+2],m[3]),avx_mul_complex(tw[7cs+3],m[4]),avx_mul_complex(tw[7cs+4],m[5]),avx_mul_complex(tw[7cs+5],m[6]),avx_mul_complex(tw[7cs+6],m[7]),avx_mul_complex(tw[7cs+7],m[8]))
        ob=base+cs*32; for k in 1:8; S8(scr,ob+4(k-1),t[k]); end; end
    @inbounds for cs in 0:1; b=base+cs*4
        m=avx_column_butterfly8(L8(scr,b),L8(scr,b+8),L8(scr,b+16),L8(scr,b+24),L8(scr,b+32),L8(scr,b+40),L8(scr,b+48),L8(scr,b+56),rot)
        for r in 0:7; S8(out,b+8r,m[r+1]); end; end
end
mrtw8(R,M,n,fwd)=[avx_mixedradix_twiddle_chunk8(c*4,y,n,fwd) for c in 0:(M÷4-1) for y in 1:(R-1)]
function mr12b64_w8!(x,scr,tw,bf3,rot,b64tw)
    @inbounds for c in 0:15; ib=4c
        r=avx_column_butterfly12(L8(x,ib),L8(x,ib+64),L8(x,ib+128),L8(x,ib+192),L8(x,ib+256),L8(x,ib+320),L8(x,ib+384),L8(x,ib+448),L8(x,ib+512),L8(x,ib+576),L8(x,ib+640),L8(x,ib+704),bf3,rot)
        S8(x,ib,r[1]); for j in 1:11; S8(x,ib+j*64,avx_mul_complex(tw[c*11+j],r[j+1])); end; end
    @inbounds for f in 0:11; bf64_w8!(scr,x,scr,64f,b64tw,rot); end
    @inbounds for c in 0:15; ib=4c; ob=48c
        t=t12w8(L8(scr,ib),L8(scr,ib+64),L8(scr,ib+128),L8(scr,ib+192),L8(scr,ib+256),L8(scr,ib+320),L8(scr,ib+384),L8(scr,ib+448),L8(scr,ib+512),L8(scr,ib+576),L8(scr,ib+640),L8(scr,ib+704))
        for k in 1:12; S8(x,ob+4(k-1),t[k]); end; end
end
n=768; x=seeded(n); ref=FFTW.fft(x)
tw=mrtw8(12,64,n,true); bf3=avx_broadcast_twiddle8(1,3,true); b64tw=bf64tw8(true)
y=copy(x); scr=zeros(ComplexF64,n); mr12b64_w8!(y,scr,tw,bf3,_ROT90_FWD8,b64tw)
@printf("768=MR12(B64) W=8 vs FFTW: rel-err %.2e %s\n", maximum(abs.(y.-ref))/maximum(abs.(ref)), maximum(abs.(y.-ref))/maximum(abs.(ref))<1e-10 ? "✓" : "WRONG")

# benchmark: W=8 vs V4f (existing keystone) vs rust, n=768
const LIB=joinpath(pwd(),"bench","rustfft_compare","rust","target","release","librustfft_bench.so")
rpl(n)=ccall((:rfft_plan,LIB),Ptr{Cvoid},(Csize_t,),n); rpr(h,d,n)=ccall((:rfft_process,LIB),Cvoid,(Ptr{Cvoid},Ptr{ComplexF64},Csize_t),h,d,n)
rp4=RPlan(MR12(B64(true),true))   # V4f reference
y4=copy(x); applyplan!(rp4,y4); @printf("V4f keystone vs FFTW: %.2e\n", maximum(abs.(y4.-ref))/maximum(abs.(ref)))
function med(f)
    for _ in 1:30; f(); end; ts=Float64[]; for _ in 1:151; t=time_ns(); for _ in 1:50; f(); end; push!(ts,(time_ns()-t)/50); end; median(ts)
end
b8=copy(x); s8=zeros(ComplexF64,n); b4=copy(x); rb=copy(x); h=rpl(n)
m8=med(()->mr12b64_w8!(b8,s8,tw,bf3,_ROT90_FWD8,b64tw))
m4=med(()->applyplan!(rp4,b4))
GC.@preserve rb (mr=med(()->rpr(h,rb,n)))
@printf("768  W8 %.1f ns   W4 %.1f ns   rust %.1f ns   W8/W4=%.2f×  rust/W8=%.2f\n", m8,m4,mr, m4/m8, mr/m8)
