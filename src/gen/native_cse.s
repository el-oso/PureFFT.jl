	.text
	push	rbp
	mov	rbp, rsp
	push	r15
	push	r14
	push	rbx
	sub	rsp, 1816
	mov	qword ptr [rbp - 1960], rsi
	mov	rax, qword ptr fs:[0]
	movabs	r11, 136143485006896
	movabs	r10, 136143485006904
	movabs	r9, 136143485006912
	movabs	r15, 136143485007080
	movabs	r14, 136143485007072
	movabs	rbx, 136143485007064
	mov	rdx, qword ptr [rax - 8]
	mov	rdi, qword ptr [rsi + 16]
	mov	rcx, qword ptr [rsi]
	mov	rax, qword ptr [rsi + 8]
	mov	r8, qword ptr [rsi + 24]
	vmovsd	xmm2, qword ptr [r11]
	vmovsd	xmm25, qword ptr [r10]
	vmovddup	xmm21, qword ptr [r11]          # xmm21 = mem[0,0]
	movabs	r11, 136143485006920
	vmovddup	xmm27, qword ptr [r10]          # xmm27 = mem[0,0]
	movabs	r10, 136143485007128
	mov	rdx, qword ptr [rdx + 16]
	vmovapd	xmm20, xmm2
	vmovapd	xmm4, xmm25
	mov	rdx, qword ptr [rdx + 16]
	mov	rdx, qword ptr [rdx]
	mov	rsi, qword ptr [rdi]
	mov	rdx, qword ptr [r8]
	movabs	rdi, offset .rodata.cst8
	movabs	r8, 136143485006888
	vmovapd	xmmword ptr [rbp - 1056], xmm21
	vmovsd	qword ptr [rbp - 1672], xmm25
	mov	rcx, qword ptr [rcx]
	vmovsd	xmm28, qword ptr [rdi]
	vmovsd	xmm15, qword ptr [r8]
	vmovsd	xmm1, qword ptr [rsi + 16]
	vmovsd	xmm8, qword ptr [rsi + 56]
	vmovupd	xmm29, xmmword ptr [rdx + 128]
	vmovupd	xmm11, xmmword ptr [rdx + 48]
	vmovupd	xmm7, xmmword ptr [rsi + 128]
	vmovupd	xmm30, xmmword ptr [rsi + 48]
	vmovupd	xmm5, xmmword ptr [rdx + 88]
	vmovupd	xmm26, xmmword ptr [rsi + 88]
	vmovupd	xmm9, xmmword ptr [rsi + 8]
	vmovupd	xmm24, xmmword ptr [rsi + 168]
	vmovapd	xmm16, xmm28
	vmovsd	qword ptr [rbp - 192], xmm1
	vaddsd	xmm1, xmm8, xmm1
	vmulpd	xmm22, xmm29, xmm21
	vmulsd	xmm0, xmm11, xmm28
	vmulpd	xmm12, xmm5, xmm21
	vmulsd	xmm19, xmm11, xmm20
	vmulsd	xmm31, xmm30, xmm20
	vmovapd	xmmword ptr [rbp - 736], xmm9
	vmovapd	xmmword ptr [rbp - 832], xmm11
	vmovapd	xmmword ptr [rbp - 1120], xmm29
	vmovapd	xmmword ptr [rbp - 1072], xmm24
	vmovsd	qword ptr [rbp - 152], xmm1
	vmovsd	xmm1, qword ptr [rdx + 56]
	vfmsub213sd	xmm4, xmm7, xmm22       # xmm4 = (xmm7 * xmm4) - xmm22
	vfmadd231sd	xmm0, xmm30, xmm15      # xmm0 = (xmm30 * xmm15) + xmm0
	vmovapd	xmmword ptr [rbp - 720], xmm12
	vmovapd	xmmword ptr [rbp - 704], xmm22
	vaddsd	xmm0, xmm9, xmm0
	vmulsd	xmm3, xmm1, xmm28
	vmovsd	qword ptr [rbp - 72], xmm1
	vfmadd231sd	xmm3, xmm8, xmm15       # xmm3 = (xmm8 * xmm15) + xmm3
	vmovsd	qword ptr [rbp - 672], xmm3
	vmulsd	xmm3, xmm15, xmm1
	vmulsd	xmm1, xmm1, xmm2
	vfnmadd231sd	xmm3, xmm8, xmm28       # xmm3 = -(xmm8 * xmm28) + xmm3
	vmovsd	qword ptr [rbp - 688], xmm3
	vmovapd	xmm3, xmm25
	vfmadd213sd	xmm3, xmm8, xmm1        # xmm3 = (xmm8 * xmm3) + xmm1
	vmovsd	qword ptr [rbp - 56], xmm3
	vmulsd	xmm3, xmm8, xmm2
	vfmsub213sd	xmm8, xmm25, xmm1       # xmm8 = (xmm25 * xmm8) - xmm1
	vmovsd	xmm1, qword ptr [rsi + 32]
	vmovsd	qword ptr [rbp - 64], xmm3
	vmovsd	xmm3, qword ptr [rsi + 72]
	vmovsd	qword ptr [rbp - 48], xmm8
	vmovapd	xmm8, xmm7
	vmovupd	xmm7, xmmword ptr [rdx + 168]
	vmovsd	qword ptr [rbp - 160], xmm1
	vaddsd	xmm1, xmm1, xmm3
	vmovapd	xmmword ptr [rbp - 816], xmm7
	vmovsd	qword ptr [rbp - 616], xmm1
	vmovsd	xmm1, qword ptr [rdx + 72]
	vmulsd	xmm2, xmm1, xmm28
	vmovsd	qword ptr [rbp - 168], xmm1
	vfmadd231sd	xmm2, xmm3, xmm15       # xmm2 = (xmm3 * xmm15) + xmm2
	vmovsd	qword ptr [rbp - 32], xmm2
	vmulsd	xmm2, xmm15, xmm1
	vfnmadd231sd	xmm2, xmm3, xmm28       # xmm2 = -(xmm3 * xmm28) + xmm2
	vmovsd	qword ptr [rbp - 40], xmm2
	vmulsd	xmm2, xmm1, xmm20
	vmovapd	xmm1, xmm25
	vfmadd213sd	xmm1, xmm3, xmm2        # xmm1 = (xmm3 * xmm1) + xmm2
	vmovsd	qword ptr [rbp - 416], xmm1
	vmulsd	xmm1, xmm3, xmm20
	vfmsub213sd	xmm3, xmm25, xmm2       # xmm3 = (xmm25 * xmm3) - xmm2
	vmovapd	xmm2, xmm25
	vfmadd213sd	xmm2, xmm26, xmm12      # xmm2 = (xmm26 * xmm2) + xmm12
	vmovsd	qword ptr [rbp - 424], xmm1
	vmovsd	xmm1, qword ptr [r9]
	vmovsd	qword ptr [rbp - 400], xmm3
	vmovsd	xmm3, qword ptr [r11]
	vaddsd	xmm2, xmm0, xmm2
	vmulpd	xmm0, xmm5, xmm27
	vaddsd	xmm2, xmm2, xmm4
	vmovupd	xmm4, xmmword ptr [rdx + 8]
	vmovapd	xmmword ptr [rbp - 752], xmm0
	vmulsd	xmm6, xmm7, xmm1
	vmulsd	xmm23, xmm5, xmm1
	vmulsd	xmm18, xmm7, xmm3
	vmulsd	xmm10, xmm5, xmm3
	vfmsub231sd	xmm6, xmm24, xmm3       # xmm6 = (xmm24 * xmm3) - xmm6
	vfmsub231sd	xmm23, xmm26, xmm3      # xmm23 = (xmm26 * xmm3) - xmm23
	vfmadd231sd	xmm18, xmm24, xmm1      # xmm18 = (xmm24 * xmm1) + xmm18
	vfmadd231sd	xmm10, xmm26, xmm1      # xmm10 = (xmm26 * xmm1) + xmm10
	vaddsd	xmm17, xmm2, xmm6
	vmulsd	xmm2, xmm11, xmm15
	vmulpd	xmm6, xmm26, xmm21
	vfnmadd231sd	xmm2, xmm30, xmm28      # xmm2 = -(xmm30 * xmm28) + xmm2
	vsubpd	xmm14, xmm0, xmm6
	vmulpd	xmm0, xmm29, xmm27
	vmovapd	xmmword ptr [rbp - 768], xmm6
	vmovapd	xmm6, xmm4
	vmovapd	xmmword ptr [rbp - 1104], xmm6
	vmovapd	xmmword ptr [rbp - 640], xmm0
	vaddsd	xmm2, xmm4, xmm2
	vaddsd	xmm13, xmm14, xmm2
	vmulpd	xmm2, xmm8, xmm21
	vaddpd	xmm21, xmm2, xmm0
	vmovapd	xmm0, xmm9
	vmovapd	xmmword ptr [rbp - 656], xmm2
	vmovapd	xmm2, xmm5
	vmovapd	xmm5, xmm7
	vmulsd	xmm7, xmm7, xmm20
	vmovapd	xmmword ptr [rbp - 1136], xmm2
	vaddsd	xmm13, xmm13, xmm21
	vaddsd	xmm18, xmm13, xmm18
	vmovapd	xmm13, xmm25
	vfmadd213sd	xmm13, xmm30, xmm19     # xmm13 = (xmm30 * xmm13) + xmm19
	vfmsub231sd	xmm19, xmm30, xmm25     # xmm19 = (xmm30 * xmm25) - xmm19
	vaddsd	xmm13, xmm9, xmm13
	vaddsd	xmm13, xmm13, xmm23
	vmulsd	xmm23, xmm11, xmm25
	vmovapd	xmm11, xmm8
	vsubsd	xmm9, xmm23, xmm31
	vaddsd	xmm9, xmm9, xmm4
	vaddsd	xmm9, xmm9, xmm10
	vmulsd	xmm10, xmm29, xmm28
	vfmadd231sd	xmm10, xmm8, xmm15      # xmm10 = (xmm8 * xmm15) + xmm10
	vaddsd	xmm10, xmm13, xmm10
	vmulsd	xmm13, xmm29, xmm15
	vfnmadd231sd	xmm13, xmm8, xmm28      # xmm13 = -(xmm8 * xmm28) + xmm13
	vmovapd	xmm28, xmm20
	vaddsd	xmm9, xmm9, xmm13
	vmulsd	xmm13, xmm24, xmm25
	vsubsd	xmm4, xmm13, xmm7
	vaddsd	xmm4, xmm10, xmm4
	vmulsd	xmm10, xmm24, xmm20
	vmovapd	xmm20, xmm1
	vmovsd	qword ptr [rbp - 368], xmm4
	vmulsd	xmm4, xmm5, xmm25
	vaddsd	xmm5, xmm10, xmm4
	vaddsd	xmm5, xmm9, xmm5
	vaddsd	xmm9, xmm0, xmm19
	vmulsd	xmm19, xmm2, xmm16
	vaddsd	xmm0, xmm13, xmm7
	vmovsd	xmm13, qword ptr [rsi + 96]
	vmovsd	xmm7, qword ptr [rsi + 176]
	vfmadd231sd	xmm19, xmm26, xmm15     # xmm19 = (xmm26 * xmm15) + xmm19
	vmovsd	qword ptr [rbp - 336], xmm5
	vaddsd	xmm5, xmm31, xmm23
	vmovapd	xmm31, xmm6
	vmovapd	xmm23, xmm3
	vaddsd	xmm5, xmm6, xmm5
	vmovsd	xmm6, qword ptr [rbp - 192]
	vaddsd	xmm9, xmm9, xmm19
	vmulsd	xmm19, xmm2, xmm15
	vmovsd	xmm2, qword ptr [rdx + 16]
	vmovsd	qword ptr [rbp - 448], xmm7
	vmovsd	qword ptr [rbp - 104], xmm13
	vfnmadd231sd	xmm19, xmm26, xmm16     # xmm19 = -(xmm26 * xmm16) + xmm19
	vaddsd	xmm5, xmm5, xmm19
	vmulsd	xmm19, xmm29, xmm1
	vfmsub231sd	xmm19, xmm8, xmm3       # xmm19 = (xmm8 * xmm3) - xmm19
	vaddsd	xmm9, xmm9, xmm19
	vmulsd	xmm19, xmm29, xmm3
	vfmadd231sd	xmm19, xmm8, xmm1       # xmm19 = (xmm8 * xmm1) + xmm19
	vaddsd	xmm0, xmm9, xmm0
	vmovsd	xmm8, qword ptr [rdx + 176]
	vmovapd	xmm9, xmm2
	vmovsd	qword ptr [rbp - 112], xmm9
	vmovsd	qword ptr [rbp - 360], xmm0
	vsubsd	xmm0, xmm4, xmm10
	vaddsd	xmm4, xmm6, qword ptr [rbp - 672]
	vaddsd	xmm5, xmm5, xmm19
	vmovsd	xmm19, qword ptr [rsi + 136]
	vaddsd	xmm0, xmm5, xmm0
	vshufpd	xmm5, xmm14, xmm14, 1           # xmm5 = xmm14[1,0]
	vmovapd	xmm14, xmm9
	vmovsd	qword ptr [rbp - 456], xmm8
	vmovsd	qword ptr [rbp - 344], xmm0
	vshufpd	xmm0, xmm12, xmm12, 1           # xmm0 = xmm12[1,0]
	vmovsd	xmm12, qword ptr [rdx + 96]
	vfmadd231sd	xmm0, xmm13, xmm25      # xmm0 = (xmm13 * xmm25) + xmm0
	vmovsd	qword ptr [rbp - 88], xmm19
	vaddsd	xmm0, xmm4, xmm0
	vaddsd	xmm4, xmm2, qword ptr [rbp - 688]
	vshufpd	xmm2, xmm21, xmm21, 1           # xmm2 = xmm21[1,0]
	vmovapd	xmm21, xmm6
	vmovsd	qword ptr [rbp - 408], xmm12
	vaddsd	xmm4, xmm4, xmm5
	vshufpd	xmm5, xmm22, xmm22, 1           # xmm5 = xmm22[1,0]
	vmovapd	xmm22, xmm16
	vmovapd	xmm29, xmm22
	vfmsub231sd	xmm5, xmm19, xmm25      # xmm5 = (xmm19 * xmm25) - xmm5
	vaddsd	xmm2, xmm4, xmm2
	vmulsd	xmm4, xmm8, xmm1
	vaddsd	xmm0, xmm0, xmm5
	vmulsd	xmm5, xmm8, xmm28
	vfmsub231sd	xmm4, xmm7, xmm3        # xmm4 = (xmm7 * xmm3) - xmm4
	vaddsd	xmm0, xmm0, xmm4
	vmulsd	xmm4, xmm12, xmm23
	vmovsd	qword ptr [rbp - 320], xmm0
	vmulsd	xmm0, xmm8, xmm3
	vfmadd231sd	xmm4, xmm13, xmm1       # xmm4 = (xmm13 * xmm1) + xmm4
	vfmadd231sd	xmm0, xmm7, xmm1        # xmm0 = (xmm7 * xmm1) + xmm0
	vaddsd	xmm0, xmm2, xmm0
	vmulsd	xmm2, xmm12, xmm1
	vmovsd	xmm1, qword ptr [rdx + 136]
	vmovsd	qword ptr [rbp - 312], xmm0
	vaddsd	xmm0, xmm6, qword ptr [rbp - 56]
	vfmsub231sd	xmm2, xmm13, xmm3       # xmm2 = (xmm13 * xmm3) - xmm2
	vmulsd	xmm3, xmm25, qword ptr [rbp - 72]
	vaddsd	xmm6, xmm0, xmm2
	vmovsd	xmm0, qword ptr [rbp - 64]
	vsubsd	xmm2, xmm3, xmm0
	vaddsd	xmm3, xmm0, xmm3
	vaddsd	xmm2, xmm9, xmm2
	vaddsd	xmm3, xmm14, xmm3
	vmovapd	xmm14, xmm20
	vaddsd	xmm2, xmm2, xmm4
	vmulsd	xmm4, xmm1, xmm16
	vmovapd	xmm16, xmm1
	vmovsd	qword ptr [rbp - 96], xmm16
	vfmadd231sd	xmm4, xmm19, xmm15      # xmm4 = (xmm19 * xmm15) + xmm4
	vaddsd	xmm9, xmm6, xmm4
	vmulsd	xmm4, xmm15, xmm1
	vmovapd	xmm1, xmm7
	vmovapd	xmm6, xmm15
	vmulsd	xmm15, xmm8, xmm25
	vaddsd	xmm8, xmm21, qword ptr [rbp - 48]
	vmovapd	xmm21, xmmword ptr [rbp - 816]
	vfnmadd231sd	xmm4, xmm19, xmm22      # xmm4 = -(xmm19 * xmm22) + xmm4
	vmovapd	xmm22, xmm28
	vmovsd	qword ptr [rbp - 792], xmm22
	vaddsd	xmm2, xmm2, xmm4
	vmulsd	xmm4, xmm7, xmm25
	vsubsd	xmm7, xmm4, xmm5
	vaddsd	xmm4, xmm5, xmm4
	vmovddup	xmm5, qword ptr [r9]            # xmm5 = mem[0,0]
	movabs	r9, 136143485006928
	vaddsd	xmm7, xmm9, xmm7
	vmulsd	xmm9, xmm1, xmm28
	vmovapd	xmm1, xmmword ptr [rbp - 832]
	vmovapd	xmm28, xmm6
	vmovsd	qword ptr [rbp - 784], xmm28
	vaddsd	xmm10, xmm9, xmm15
	vsubsd	xmm0, xmm15, xmm9
	vmovapd	xmmword ptr [rbp - 1008], xmm5
	vaddsd	xmm2, xmm10, xmm2
	vmulsd	xmm10, xmm12, xmm29
	vfmadd231sd	xmm10, xmm13, xmm6      # xmm10 = (xmm13 * xmm6) + xmm10
	vaddsd	xmm8, xmm8, xmm10
	vmulsd	xmm10, xmm12, xmm6
	vfnmadd231sd	xmm10, xmm29, xmm13     # xmm10 = -(xmm29 * xmm13) + xmm10
	vmovapd	xmm13, xmm23
	vaddsd	xmm3, xmm10, xmm3
	vmulsd	xmm10, xmm16, xmm20
	vfmsub231sd	xmm10, xmm19, xmm23     # xmm10 = (xmm19 * xmm23) - xmm10
	vaddsd	xmm8, xmm8, xmm10
	vmulsd	xmm10, xmm16, xmm23
	vfmadd231sd	xmm10, xmm20, xmm19     # xmm10 = (xmm20 * xmm19) + xmm10
	vaddsd	xmm8, xmm8, xmm4
	vaddsd	xmm3, xmm10, xmm3
	vaddsd	xmm15, xmm3, xmm0
	vmovddup	xmm3, qword ptr [r11]           # xmm3 = mem[0,0]
	vmulpd	xmm0, xmm1, xmm5
	movabs	r11, 136143485007136
	vmovapd	xmmword ptr [rbp - 1040], xmm3
	vfmsub231pd	xmm0, xmm30, xmm3       # xmm0 = (xmm30 * xmm3) - xmm0
	vmulpd	xmm3, xmm1, xmm3
	vmovapd	xmm1, xmmword ptr [rbp - 736]
	vfmadd231pd	xmm3, xmm5, xmm30       # xmm3 = (xmm5 * xmm30) + xmm3
	vaddpd	xmm3, xmm31, xmm3
	vmovupd	xmm31, xmmword ptr [rdx + 104]
	vaddsd	xmm4, xmm1, xmm30
	vaddpd	xmm0, xmm1, xmm0
	vmovapd	xmm1, xmmword ptr [rbp - 752]
	vaddpd	xmm5, xmm1, xmmword ptr [rbp - 768]
	vaddsd	xmm4, xmm4, xmm26
	vfmsub213pd	xmm26, xmm27, xmmword ptr [rbp - 720] # xmm26 = (xmm27 * xmm26) - mem
	vaddsd	xmm1, xmm11, xmm4
	vmovddup	xmm4, qword ptr [rdi]           # xmm4 = mem[0,0]
	movabs	rdi, 136143485006936
	vmovapd	xmmword ptr [rbp - 1088], xmm31
	vmovsd	qword ptr [rbp - 80], xmm1
	vmovddup	xmm1, qword ptr [r9]            # xmm1 = mem[0,0]
	movabs	r9, 136143485007120
	vfnmadd213pd	xmm11, xmm1, xmmword ptr [rbp - 704] # xmm11 = -(xmm1 * xmm11) + mem
	vaddpd	xmm3, xmm3, xmm5
	vmovddup	xmm5, qword ptr [r8]            # xmm5 = mem[0,0]
	movabs	r8, 136143485006944
	vmovsd	xmm23, qword ptr [r8]
	movabs	r8, 136143485007008
	vmovsd	xmm20, qword ptr [r8]
	movabs	r8, 136143485007112
	vaddpd	xmm0, xmm0, xmm26
	vmovsd	xmm19, qword ptr [r8]
	vmovapd	xmmword ptr [rbp - 976], xmm4
	movabs	r8, 136143485007184
	vmovapd	xmmword ptr [rbp - 1024], xmm1
	vmovapd	xmm1, xmmword ptr [rbp - 640]
	vsubpd	xmm1, xmm1, xmmword ptr [rbp - 656]
	vmovapd	xmmword ptr [rbp - 992], xmm5
	vaddpd	xmm0, xmm11, xmm0
	vaddpd	xmm1, xmm3, xmm1
	vmulpd	xmm3, xmm21, xmm4
	vfmadd231pd	xmm3, xmm24, xmm5       # xmm3 = (xmm24 * xmm5) + xmm3
	vaddpd	xmm0, xmm0, xmm3
	vmovsd	xmm3, qword ptr [r10]
	movabs	r10, 136143485007144
	vmovapd	xmmword ptr [rbp - 880], xmm0
	vmulpd	xmm0, xmm21, xmm5
	vmovsd	xmm5, qword ptr [r8]
	movabs	r8, 136143485006976
	vmovsd	xmm6, qword ptr [r8]
	movabs	r8, 136143485007096
	vfnmadd231pd	xmm0, xmm4, xmm24       # xmm0 = -(xmm4 * xmm24) + xmm0
	vmovsd	xmm4, qword ptr [r9]
	vmovupd	xmm24, xmmword ptr [rdx + 64]
	movabs	r9, 136143485007016
	vmovapd	xmm9, xmm3
	vmovsd	qword ptr [rbp - 384], xmm3
	vaddpd	xmm0, xmm1, xmm0
	vmovsd	xmm1, qword ptr [rdi]
	movabs	rdi, 136143485007000
	vmovsd	xmm21, qword ptr [rdi]
	movabs	rdi, 136143485007104
	vmovapd	xmm16, xmm5
	vmovsd	qword ptr [rbp - 296], xmm16
	vmovsd	qword ptr [rbp - 256], xmm6
	vmovapd	xmmword ptr [rbp - 864], xmm0
	vmovsd	qword ptr [rbp - 280], xmm4
	vmovapd	xmmword ptr [rbp - 656], xmm24
	vmulsd	xmm0, xmm18, xmm1
	vmovsd	qword ptr [rbp - 264], xmm1
	vfmadd231sd	xmm0, xmm17, xmm23      # xmm0 = (xmm17 * xmm23) + xmm0
	vmovsd	qword ptr [rbp - 504], xmm0
	vmulsd	xmm0, xmm18, xmm23
	vfnmadd231sd	xmm0, xmm17, xmm1       # xmm0 = -(xmm17 * xmm1) + xmm0
	vmovsd	qword ptr [rbp - 496], xmm0
	vmulsd	xmm0, xmm18, xmm21
	vfmadd231sd	xmm0, xmm17, xmm20      # xmm0 = (xmm17 * xmm20) + xmm0
	vmovsd	qword ptr [rbp - 568], xmm0
	vmulsd	xmm0, xmm18, xmm20
	vfnmadd231sd	xmm0, xmm17, xmm21      # xmm0 = -(xmm17 * xmm21) + xmm0
	vmovsd	qword ptr [rbp - 560], xmm0
	vmulsd	xmm0, xmm18, xmm3
	vmovsd	xmm3, qword ptr [r11]
	movabs	r11, 136143485007056
	vfnmadd231sd	xmm0, xmm17, xmm3       # xmm0 = -(xmm17 * xmm3) + xmm0
	vmovapd	xmm11, xmm3
	vmovsd	qword ptr [rbp - 352], xmm11
	vmovsd	qword ptr [rbp - 600], xmm0
	vmulsd	xmm0, xmm18, xmm3
	vmovsd	xmm3, qword ptr [r10]
	movabs	r10, 136143485007024
	vfmsub231sd	xmm0, xmm17, xmm3       # xmm0 = (xmm17 * xmm3) - xmm0
	vmovapd	xmm10, xmm3
	vmovsd	qword ptr [rbp - 376], xmm3
	vmovsd	xmm3, qword ptr [rdi]
	movabs	rdi, 136143485007176
	vmovsd	xmm12, qword ptr [rdi]
	movabs	rdi, 136143485006968
	vmovsd	qword ptr [rbp - 592], xmm0
	vmulsd	xmm0, xmm18, xmm3
	vmovsd	qword ptr [rbp - 288], xmm3
	vmovsd	qword ptr [rbp - 392], xmm12
	vfmadd231sd	xmm0, xmm17, xmm19      # xmm0 = (xmm17 * xmm19) + xmm0
	vmovsd	qword ptr [rbp - 736], xmm0
	vmulsd	xmm0, xmm18, xmm19
	vfmadd231sd	xmm0, xmm17, xmm4       # xmm0 = (xmm17 * xmm4) + xmm0
	vmovsd	qword ptr [rbp - 720], xmm0
	vmulsd	xmm0, xmm18, xmm12
	vfmsub231sd	xmm0, xmm17, xmm5       # xmm0 = (xmm17 * xmm5) - xmm0
	vmovsd	qword ptr [rbp - 768], xmm0
	vmulsd	xmm0, xmm18, xmm5
	vmovsd	xmm5, qword ptr [rdi]
	movabs	rdi, 136143485007088
	vfmadd231sd	xmm0, xmm12, xmm17      # xmm0 = (xmm12 * xmm17) + xmm0
	vmovupd	xmm17, xmmword ptr [rsi + 64]
	vmovsd	qword ptr [rbp - 752], xmm0
	vmulsd	xmm0, xmm2, xmm5
	vmovsd	qword ptr [rbp - 248], xmm5
	vfmadd231sd	xmm0, xmm7, xmm6        # xmm0 = (xmm7 * xmm6) + xmm0
	vmovapd	xmmword ptr [rbp - 640], xmm17
	vmovsd	qword ptr [rbp - 120], xmm0
	vmulsd	xmm0, xmm2, xmm6
	vmovsd	xmm6, qword ptr [r8]
	movabs	r8, 136143485007232
	vfnmadd231sd	xmm0, xmm7, xmm5        # xmm0 = -(xmm7 * xmm5) + xmm0
	vmovsd	xmm5, qword ptr [rdi]
	movabs	rdi, 136143485007152
	vmovsd	qword ptr [rbp - 464], xmm0
	vmovsd	qword ptr [rbp - 328], xmm6
	vmulsd	xmm0, xmm2, xmm5
	vmovsd	qword ptr [rbp - 304], xmm5
	vfmsub231sd	xmm0, xmm7, xmm6        # xmm0 = (xmm7 * xmm6) - xmm0
	vmovsd	qword ptr [rbp - 472], xmm0
	vmulsd	xmm0, xmm2, xmm6
	vfmadd231sd	xmm0, xmm7, xmm5        # xmm0 = (xmm7 * xmm5) + xmm0
	vmovsd	xmm5, qword ptr [rdi]
	movabs	rdi, offset .rodata.cst16
	vmovsd	qword ptr [rbp - 144], xmm0
	vmulsd	xmm0, xmm2, xmm5
	vmovsd	qword ptr [rbp - 272], xmm5
	vfmsub231sd	xmm0, xmm7, xmm23       # xmm0 = (xmm7 * xmm23) - xmm0
	vmovsd	qword ptr [rbp - 520], xmm0
	vmulsd	xmm0, xmm2, xmm23
	vfmadd231sd	xmm0, xmm7, xmm5        # xmm0 = (xmm7 * xmm5) + xmm0
	vmovupd	xmm5, xmmword ptr [rdx + 184]
	vmovsd	qword ptr [rbp - 512], xmm0
	vmulsd	xmm0, xmm2, xmm4
	vfmadd231sd	xmm0, xmm7, xmm19       # xmm0 = (xmm7 * xmm19) + xmm0
	vmulsd	xmm6, xmm5, xmm22
	vmovapd	xmmword ptr [rbp - 192], xmm5
	vmulsd	xmm5, xmm5, xmm25
	vmovsd	qword ptr [rbp - 552], xmm0
	vmulsd	xmm0, xmm2, xmm19
	vfmadd231sd	xmm0, xmm7, xmm3        # xmm0 = (xmm7 * xmm3) + xmm0
	vmovsd	qword ptr [rbp - 544], xmm0
	vmulsd	xmm0, xmm2, xmm21
	vfmsub231sd	xmm0, xmm7, xmm20       # xmm0 = (xmm7 * xmm20) - xmm0
	vmovsd	qword ptr [rbp - 704], xmm0
	vmulsd	xmm0, xmm2, xmm20
	vmulsd	xmm2, xmm24, xmm22
	vfmadd231sd	xmm0, xmm21, xmm7       # xmm0 = (xmm21 * xmm7) + xmm0
	vmulsd	xmm7, xmm31, xmm14
	vmovsd	qword ptr [rbp - 608], xmm0
	vmulsd	xmm0, xmm15, xmm21
	vfmadd231sd	xmm0, xmm8, xmm20       # xmm0 = (xmm8 * xmm20) + xmm0
	vmovsd	qword ptr [rbp - 440], xmm0
	vmulsd	xmm0, xmm15, xmm20
	vfnmadd231sd	xmm0, xmm8, xmm21       # xmm0 = -(xmm8 * xmm21) + xmm0
	vmovsd	qword ptr [rbp - 432], xmm0
	vmulsd	xmm0, xmm15, xmm3
	vfmadd231sd	xmm0, xmm8, xmm19       # xmm0 = (xmm8 * xmm19) + xmm0
	vmovsd	qword ptr [rbp - 136], xmm0
	vmulsd	xmm0, xmm15, xmm19
	vfmadd231sd	xmm0, xmm8, xmm4        # xmm0 = (xmm8 * xmm4) + xmm0
	vmovupd	xmm4, xmmword ptr [rsi + 104]
	vmovsd	qword ptr [rbp - 128], xmm0
	vmulsd	xmm0, xmm15, xmm1
	vfmadd231sd	xmm0, xmm8, xmm23       # xmm0 = (xmm8 * xmm23) + xmm0
	vfmsub231sd	xmm7, xmm4, xmm13       # xmm7 = (xmm4 * xmm13) - xmm7
	vmovsd	qword ptr [rbp - 488], xmm0
	vmulsd	xmm0, xmm15, xmm23
	vfnmadd231sd	xmm0, xmm8, xmm1        # xmm0 = -(xmm8 * xmm1) + xmm0
	vmovapd	xmm1, xmm25
	vfmadd213sd	xmm1, xmm17, xmm2       # xmm1 = (xmm17 * xmm1) + xmm2
	vfmsub231sd	xmm2, xmm17, xmm25      # xmm2 = (xmm17 * xmm25) - xmm2
	vmovsd	qword ptr [rbp - 480], xmm0
	vmulsd	xmm0, xmm15, xmm9
	vmovupd	xmm9, xmmword ptr [rdx + 24]
	vfnmadd231sd	xmm0, xmm8, xmm11       # xmm0 = -(xmm8 * xmm11) + xmm0
	vmovsd	qword ptr [rbp - 536], xmm0
	vmulsd	xmm0, xmm15, xmm11
	vmovapd	xmm11, xmm29
	vmovapd	xmmword ptr [rbp - 688], xmm9
	vfmsub231sd	xmm0, xmm8, xmm10       # xmm0 = (xmm8 * xmm10) - xmm0
	vmovupd	xmm10, xmmword ptr [rsi + 24]
	vmovsd	qword ptr [rbp - 528], xmm0
	vmulsd	xmm0, xmm15, xmm12
	vfmsub231sd	xmm0, xmm8, xmm16       # xmm0 = (xmm8 * xmm16) - xmm0
	vaddsd	xmm1, xmm10, xmm1
	vaddsd	xmm2, xmm10, xmm2
	vmovapd	xmmword ptr [rbp - 1888], xmm10
	vmovsd	qword ptr [rbp - 584], xmm0
	vmulsd	xmm0, xmm15, xmm16
	vmulsd	xmm15, xmm17, xmm22
	vaddsd	xmm3, xmm1, xmm7
	vmovupd	xmm1, xmmword ptr [rsi + 144]
	vmovapd	xmm16, xmm13
	vmovsd	qword ptr [rbp - 776], xmm16
	vfmadd231sd	xmm0, xmm12, xmm8       # xmm0 = (xmm12 * xmm8) + xmm0
	vmovapd	xmm12, xmm14
	vmulsd	xmm14, xmm24, xmm25
	vmulsd	xmm8, xmm31, xmm13
	vmovupd	xmm13, xmmword ptr [rsi + 184]
	vmovsd	qword ptr [rbp - 1688], xmm12
	vsubsd	xmm7, xmm14, xmm15
	vfmadd231sd	xmm8, xmm4, xmm12       # xmm8 = (xmm4 * xmm12) + xmm8
	vaddsd	xmm14, xmm15, xmm14
	vmulsd	xmm15, xmm31, xmm11
	vmovsd	qword ptr [rbp - 576], xmm0
	vmovupd	xmm0, xmmword ptr [rdx + 144]
	vaddsd	xmm7, xmm9, xmm7
	vfmadd231sd	xmm15, xmm4, xmm28      # xmm15 = (xmm4 * xmm28) + xmm15
	vaddsd	xmm14, xmm9, xmm14
	vmulsd	xmm30, xmm13, xmm25
	vaddsd	xmm7, xmm8, xmm7
	vaddsd	xmm2, xmm15, xmm2
	vmulsd	xmm15, xmm31, xmm28
	vmulsd	xmm8, xmm0, xmm29
	vfnmadd231sd	xmm15, xmm4, xmm11      # xmm15 = -(xmm4 * xmm11) + xmm15
	vfmadd231sd	xmm8, xmm1, xmm28       # xmm8 = (xmm1 * xmm28) + xmm8
	vaddsd	xmm14, xmm14, xmm15
	vmulsd	xmm15, xmm12, xmm0
	vaddsd	xmm3, xmm8, xmm3
	vmulsd	xmm8, xmm0, xmm28
	vmovapd	xmm28, xmm11
	vmovapd	xmm11, xmmword ptr [rbp - 1040]
	vfmsub231sd	xmm15, xmm1, xmm16      # xmm15 = (xmm1 * xmm16) - xmm15
	vfnmadd231sd	xmm8, xmm1, xmm29       # xmm8 = -(xmm1 * xmm29) + xmm8
	vaddsd	xmm2, xmm15, xmm2
	vmulsd	xmm15, xmm0, xmm16
	vaddsd	xmm7, xmm8, xmm7
	vsubsd	xmm8, xmm30, xmm6
	vfmadd231sd	xmm15, xmm1, xmm12      # xmm15 = (xmm1 * xmm12) + xmm15
	vaddsd	xmm26, xmm3, xmm8
	vmulsd	xmm3, xmm13, xmm22
	vmovsd	xmm22, qword ptr [rsi + 152]
	vmulpd	xmm8, xmm31, xmm27
	vaddsd	xmm14, xmm14, xmm15
	vaddsd	xmm29, xmm3, xmm5
	vaddsd	xmm18, xmm7, xmm29
	vmovapd	xmm29, xmm4
	vmovapd	xmm4, xmm0
	vsubsd	xmm0, xmm5, xmm3
	vmovapd	xmm5, xmmword ptr [rbp - 1056]
	vmovapd	xmm7, xmm1
	vaddsd	xmm1, xmm6, xmm30
	vmovapd	xmm6, xmm29
	vmovapd	xmmword ptr [rbp - 1952], xmm4
	vmovapd	xmmword ptr [rbp - 1904], xmm6
	vmovapd	xmmword ptr [rbp - 672], xmm7
	vmovsd	qword ptr [rbp - 48], xmm22
	vaddsd	xmm15, xmm2, xmm1
	vmovsd	xmm1, qword ptr [rsi + 112]
	vaddsd	xmm14, xmm14, xmm0
	vmulpd	xmm30, xmm31, xmm5
	vmulpd	xmm3, xmm29, xmm5
	vmulpd	xmm29, xmm4, xmm5
	vmulpd	xmm5, xmm7, xmm5
	vshufpd	xmm0, xmm30, xmm30, 1           # xmm0 = xmm30[1,0]
	vmovsd	qword ptr [rbp - 56], xmm1
	vsubpd	xmm2, xmm8, xmm3
	vaddpd	xmm3, xmm8, xmm3
	vmovsd	xmm8, qword ptr [rbp - 272]
	vmovapd	xmmword ptr [rbp - 928], xmm29
	vfmadd231sd	xmm0, xmm1, xmm25       # xmm0 = (xmm1 * xmm25) + xmm0
	vmovsd	xmm1, qword ptr [rbp - 32]
	vaddsd	xmm1, xmm1, qword ptr [rbp - 160]
	vmovapd	xmmword ptr [rbp - 960], xmm2
	vshufpd	xmm2, xmm2, xmm2, 1             # xmm2 = xmm2[1,0]
	vaddsd	xmm0, xmm1, xmm0
	vmovsd	xmm1, qword ptr [rdx + 32]
	vmovsd	qword ptr [rbp - 64], xmm1
	vaddsd	xmm1, xmm1, qword ptr [rbp - 40]
	vaddsd	xmm1, xmm1, xmm2
	vshufpd	xmm2, xmm29, xmm29, 1           # xmm2 = xmm29[1,0]
	vfmsub231sd	xmm2, xmm22, xmm25      # xmm2 = (xmm22 * xmm25) - xmm2
	vmovsd	xmm22, qword ptr [rsi + 192]
	vaddsd	xmm0, xmm0, xmm2
	vmulpd	xmm2, xmm4, xmm27
	vfmsub213pd	xmm27, xmm6, xmm30      # xmm27 = (xmm6 * xmm27) - xmm30
	vaddpd	xmm4, xmm5, xmm2
	vsubpd	xmm2, xmm2, xmm5
	vmovapd	xmm5, xmmword ptr [rbp - 192]
	vmovsd	qword ptr [rbp - 32], xmm22
	vshufpd	xmm31, xmm4, xmm4, 1            # xmm31 = xmm4[1,0]
	vmovapd	xmmword ptr [rbp - 944], xmm4
	vmovsd	xmm4, qword ptr [rdx + 192]
	vaddsd	xmm1, xmm1, xmm31
	vmulsd	xmm31, xmm4, xmm12
	vmovsd	qword ptr [rbp - 40], xmm4
	vfmsub231sd	xmm31, xmm22, xmm16     # xmm31 = (xmm22 * xmm16) - xmm31
	vaddsd	xmm0, xmm0, xmm31
	vmulsd	xmm31, xmm4, xmm16
	vmovapd	xmm4, xmmword ptr [rbp - 1008]
	vfmadd231sd	xmm31, xmm22, xmm12     # xmm31 = (xmm22 * xmm12) + xmm31
	vmulpd	xmm22, xmm24, xmm11
	vmovapd	xmm12, xmm10
	vaddsd	xmm1, xmm1, xmm31
	vfmadd231pd	xmm22, xmm17, xmm4      # xmm22 = (xmm17 * xmm4) + xmm22
	vmulpd	xmm31, xmm24, xmm4
	vmovapd	xmm24, xmm6
	vmovapd	xmm6, xmmword ptr [rbp - 976]
	vfmsub231pd	xmm31, xmm17, xmm11     # xmm31 = (xmm17 * xmm11) - xmm31
	vaddpd	xmm4, xmm9, xmm22
	vaddpd	xmm3, xmm4, xmm3
	vmovapd	xmm4, xmmword ptr [rbp - 1024]
	vaddpd	xmm25, xmm10, xmm31
	vaddpd	xmm2, xmm3, xmm2
	vmulpd	xmm3, xmm5, xmm6
	vaddpd	xmm16, xmm25, xmm27
	vmovapd	xmm27, xmm13
	vmovapd	xmmword ptr [rbp - 1872], xmm27
	vfnmadd213pd	xmm4, xmm7, xmm29       # xmm4 = -(xmm7 * xmm4) + xmm29
	vmovapd	xmm7, xmmword ptr [rbp - 992]
	vaddpd	xmm4, xmm16, xmm4
	vmovsd	xmm16, qword ptr [rbp - 32]
	vfmadd231pd	xmm3, xmm13, xmm7       # xmm3 = (xmm13 * xmm7) + xmm3
	vaddpd	xmm4, xmm4, xmm3
	vmulpd	xmm3, xmm5, xmm7
	vmovsd	xmm5, qword ptr [rbp - 392]
	vmovsd	xmm7, qword ptr [rbp - 280]
	vfnmadd231pd	xmm3, xmm13, xmm6       # xmm3 = -(xmm13 * xmm6) + xmm3
	vmovsd	xmm6, qword ptr [rbp - 296]
	vmovapd	xmmword ptr [rbp - 912], xmm4
	vshufpd	xmm10, xmm4, xmm4, 1            # xmm10 = xmm4[1,0]
	vmovsd	xmm4, qword ptr [rbp - 384]
	vaddpd	xmm3, xmm2, xmm3
	vmulsd	xmm2, xmm18, xmm5
	vmovapd	xmmword ptr [rbp - 896], xmm3
	vshufpd	xmm9, xmm3, xmm3, 1             # xmm9 = xmm3[1,0]
	vmovsd	xmm3, qword ptr [rbp - 352]
	vfmsub231sd	xmm2, xmm26, xmm6       # xmm2 = (xmm26 * xmm6) - xmm2
	vmovsd	qword ptr [rbp - 1056], xmm2
	vmulsd	xmm2, xmm18, xmm6
	vmovsd	xmm6, qword ptr [rbp - 328]
	vfmadd231sd	xmm2, xmm26, xmm5       # xmm2 = (xmm26 * xmm5) + xmm2
	vmovsd	xmm5, qword ptr [rbp - 376]
	vmovsd	qword ptr [rbp - 392], xmm2
	vmulsd	xmm2, xmm9, xmm4
	vfnmadd231sd	xmm2, xmm10, xmm3       # xmm2 = -(xmm10 * xmm3) + xmm2
	vmovsd	qword ptr [rbp - 1040], xmm2
	vmulsd	xmm2, xmm9, xmm3
	vfmsub231sd	xmm2, xmm10, xmm5       # xmm2 = (xmm10 * xmm5) - xmm2
	vmovsd	qword ptr [rbp - 1024], xmm2
	vmulsd	xmm2, xmm18, xmm4
	vmovapd	xmm4, xmmword ptr [rbp - 864]
	vfnmadd231sd	xmm2, xmm26, xmm3       # xmm2 = -(xmm26 * xmm3) + xmm2
	vmovsd	qword ptr [rbp - 384], xmm2
	vmulsd	xmm2, xmm18, xmm3
	vmovapd	xmm3, xmmword ptr [rbp - 880]
	vfmsub231sd	xmm2, xmm26, xmm5       # xmm2 = (xmm26 * xmm5) - xmm2
	vmovsd	xmm5, qword ptr [rbp - 304]
	vmovsd	qword ptr [rbp - 376], xmm2
	vmulsd	xmm2, xmm4, xmm5
	vfmsub231sd	xmm2, xmm3, xmm6        # xmm2 = (xmm3 * xmm6) - xmm2
	vmovsd	qword ptr [rbp - 1824], xmm2
	vmulsd	xmm2, xmm4, xmm6
	vfmadd231sd	xmm2, xmm3, xmm5        # xmm2 = (xmm3 * xmm5) + xmm2
	vmovsd	qword ptr [rbp - 1816], xmm2
	vmulsd	xmm2, xmm14, xmm5
	vfmsub231sd	xmm2, xmm15, xmm6       # xmm2 = (xmm15 * xmm6) - xmm2
	vmovsd	qword ptr [rbp - 352], xmm2
	vmulsd	xmm2, xmm14, xmm6
	vfmadd231sd	xmm2, xmm15, xmm5       # xmm2 = (xmm15 * xmm5) + xmm2
	vmovsd	qword ptr [rbp - 1008], xmm2
	vmulsd	xmm2, xmm1, xmm5
	vfmsub231sd	xmm2, xmm0, xmm6        # xmm2 = (xmm0 * xmm6) - xmm2
	vmovsd	qword ptr [rbp - 992], xmm2
	vmulsd	xmm2, xmm1, xmm6
	vmovsd	xmm6, qword ptr [rbp - 256]
	vfmadd231sd	xmm2, xmm0, xmm5        # xmm2 = (xmm0 * xmm5) + xmm2
	vmovsd	xmm5, qword ptr [rbp - 248]
	vmovsd	qword ptr [rbp - 328], xmm2
	vmulsd	xmm2, xmm1, xmm5
	vfmadd231sd	xmm2, xmm0, xmm6        # xmm2 = (xmm0 * xmm6) + xmm2
	vmovsd	qword ptr [rbp - 1808], xmm2
	vmulsd	xmm2, xmm1, xmm6
	vfnmadd231sd	xmm2, xmm0, xmm5        # xmm2 = -(xmm0 * xmm5) + xmm2
	vmovsd	qword ptr [rbp - 1800], xmm2
	vmulsd	xmm2, xmm8, xmm1
	vfmsub231sd	xmm2, xmm0, xmm23       # xmm2 = (xmm0 * xmm23) - xmm2
	vmovsd	qword ptr [rbp - 1840], xmm2
	vmulsd	xmm2, xmm1, xmm23
	vfmadd231sd	xmm2, xmm0, xmm8        # xmm2 = (xmm0 * xmm8) + xmm2
	vmovsd	qword ptr [rbp - 1832], xmm2
	vmulsd	xmm2, xmm1, xmm21
	vfmsub231sd	xmm2, xmm0, xmm20       # xmm2 = (xmm0 * xmm20) - xmm2
	vmovsd	qword ptr [rbp - 304], xmm2
	vmulsd	xmm2, xmm1, xmm20
	vfmadd231sd	xmm2, xmm0, xmm21       # xmm2 = (xmm0 * xmm21) + xmm2
	vmovsd	qword ptr [rbp - 976], xmm2
	vmulsd	xmm2, xmm1, xmm7
	vmulsd	xmm1, xmm1, xmm19
	vfmadd231sd	xmm2, xmm0, xmm19       # xmm2 = (xmm0 * xmm19) + xmm2
	vmovsd	qword ptr [rbp - 296], xmm2
	vmovsd	xmm2, qword ptr [rbp - 288]
	vfmadd231sd	xmm1, xmm2, xmm0        # xmm1 = (xmm2 * xmm0) + xmm1
	vmulsd	xmm0, xmm4, xmm7
	vfmadd231sd	xmm0, xmm3, xmm19       # xmm0 = (xmm3 * xmm19) + xmm0
	vmovsd	qword ptr [rbp - 1848], xmm1
	vmovsd	xmm1, qword ptr [rbp - 264]
	vmovsd	qword ptr [rbp - 1664], xmm0
	vmulsd	xmm0, xmm4, xmm19
	vfmadd231sd	xmm0, xmm3, xmm2        # xmm0 = (xmm3 * xmm2) + xmm0
	vmovsd	qword ptr [rbp - 1656], xmm0
	vmulsd	xmm0, xmm14, xmm7
	vfmadd231sd	xmm0, xmm15, xmm19      # xmm0 = (xmm15 * xmm19) + xmm0
	vmovsd	qword ptr [rbp - 1712], xmm0
	vmulsd	xmm0, xmm14, xmm19
	vfmadd231sd	xmm0, xmm15, xmm2       # xmm0 = (xmm15 * xmm2) + xmm0
	vmovsd	qword ptr [rbp - 1704], xmm0
	vmulsd	xmm0, xmm9, xmm2
	vfmadd231sd	xmm0, xmm10, xmm19      # xmm0 = (xmm10 * xmm19) + xmm0
	vmovsd	qword ptr [rbp - 1728], xmm0
	vmulsd	xmm0, xmm9, xmm19
	vfmadd231sd	xmm0, xmm10, xmm7       # xmm0 = (xmm10 * xmm7) + xmm0
	vmovsd	qword ptr [rbp - 1720], xmm0
	vmulsd	xmm0, xmm18, xmm2
	vfmadd231sd	xmm0, xmm26, xmm19      # xmm0 = (xmm26 * xmm19) + xmm0
	vmovsd	qword ptr [rbp - 288], xmm0
	vmulsd	xmm0, xmm18, xmm19
	vmovsd	xmm19, qword ptr [rbp - 1672]
	vfmadd231sd	xmm0, xmm26, xmm7       # xmm0 = (xmm26 * xmm7) + xmm0
	vmovsd	qword ptr [rbp - 280], xmm0
	vmulsd	xmm0, xmm18, xmm21
	vfmadd231sd	xmm0, xmm26, xmm20      # xmm0 = (xmm26 * xmm20) + xmm0
	vmovsd	qword ptr [rbp - 1696], xmm0
	vmulsd	xmm0, xmm18, xmm20
	vfnmadd231sd	xmm0, xmm26, xmm21      # xmm0 = -(xmm26 * xmm21) + xmm0
	vmovsd	qword ptr [rbp - 1680], xmm0
	vmulsd	xmm0, xmm18, xmm1
	vfmadd231sd	xmm0, xmm26, xmm23      # xmm0 = (xmm26 * xmm23) + xmm0
	vmovsd	qword ptr [rbp - 1792], xmm0
	vmulsd	xmm0, xmm18, xmm23
	vmovsd	xmm18, qword ptr [rdx + 152]
	vfnmadd231sd	xmm0, xmm26, xmm1       # xmm0 = -(xmm26 * xmm1) + xmm0
	vmovsd	xmm26, qword ptr [rbp - 1688]
	vmovsd	qword ptr [rbp - 1784], xmm0
	vmulsd	xmm0, xmm14, xmm23
	vmovsd	qword ptr [rbp - 1320], xmm18
	vfmadd231sd	xmm0, xmm15, xmm8       # xmm0 = (xmm15 * xmm8) + xmm0
	vmovsd	qword ptr [rbp - 1744], xmm0
	vmulsd	xmm0, xmm14, xmm8
	vfmsub231sd	xmm0, xmm15, xmm23      # xmm0 = (xmm15 * xmm23) - xmm0
	vmovsd	qword ptr [rbp - 1736], xmm0
	vmulsd	xmm0, xmm9, xmm1
	vfmadd231sd	xmm0, xmm10, xmm23      # xmm0 = (xmm10 * xmm23) + xmm0
	vmovsd	qword ptr [rbp - 272], xmm0
	vmulsd	xmm0, xmm9, xmm23
	vfnmadd231sd	xmm0, xmm10, xmm1       # xmm0 = -(xmm10 * xmm1) + xmm0
	vmovapd	xmm1, xmmword ptr [r8]
	movabs	r8, 136143485006992
	vmovsd	qword ptr [rbp - 264], xmm0
	vmulsd	xmm0, xmm9, xmm21
	vfmadd231sd	xmm0, xmm10, xmm20      # xmm0 = (xmm10 * xmm20) + xmm0
	vmovsd	qword ptr [rbp - 1760], xmm0
	vmulsd	xmm0, xmm9, xmm20
	vfnmadd231sd	xmm0, xmm10, xmm21      # xmm0 = -(xmm10 * xmm21) + xmm0
	vmovsd	xmm10, qword ptr [r8]
	movabs	r8, 136143485006960
	vmovsd	xmm17, qword ptr [r8]
	movabs	r8, 136143485007168
	vmovsd	xmm25, qword ptr [r8]
	movabs	r8, 136143485007200
	vmovsd	xmm29, qword ptr [r8]
	movabs	r8, 136143485007264
	vmovsd	qword ptr [rbp - 1752], xmm0
	vmulsd	xmm0, xmm4, xmm5
	vfmadd231sd	xmm0, xmm3, xmm6        # xmm0 = (xmm3 * xmm6) + xmm0
	vmovsd	qword ptr [rbp - 1624], xmm0
	vmulsd	xmm0, xmm4, xmm6
	vfnmadd231sd	xmm0, xmm3, xmm5        # xmm0 = -(xmm3 * xmm5) + xmm0
	vmovsd	qword ptr [rbp - 1632], xmm0
	vmulsd	xmm0, xmm14, xmm5
	vfmadd231sd	xmm0, xmm15, xmm6       # xmm0 = (xmm15 * xmm6) + xmm0
	vmovsd	qword ptr [rbp - 1776], xmm0
	vmulsd	xmm0, xmm14, xmm6
	vfnmadd231sd	xmm0, xmm15, xmm5       # xmm0 = -(xmm15 * xmm5) + xmm0
	vshufpd	xmm5, xmm3, xmm3, 1             # xmm5 = xmm3[1,0]
	vmovsd	qword ptr [rbp - 1768], xmm0
	vmulsd	xmm0, xmm14, xmm21
	vfmsub231sd	xmm0, xmm15, xmm20      # xmm0 = (xmm15 * xmm20) - xmm0
	vmovsd	qword ptr [rbp - 256], xmm0
	vmulsd	xmm0, xmm14, xmm20
	vfmadd231sd	xmm0, xmm15, xmm21      # xmm0 = (xmm15 * xmm21) + xmm0
	vmovsd	qword ptr [rbp - 248], xmm0
	vmulsd	xmm0, xmm4, xmm21
	vfmsub231sd	xmm0, xmm3, xmm20       # xmm0 = (xmm3 * xmm20) - xmm0
	vmovsd	qword ptr [rbp - 1648], xmm0
	vmulsd	xmm0, xmm4, xmm20
	vfmadd231sd	xmm0, xmm3, xmm21       # xmm0 = (xmm3 * xmm21) + xmm0
	vmovsd	xmm21, qword ptr [rbp - 160]
	vmovsd	qword ptr [rbp - 1640], xmm0
	vmovapd	xmm0, xmmword ptr [rdi]
	movabs	rdi, 136143485006984
	vmovsd	xmm7, qword ptr [rdi]
	movabs	rdi, 136143485006952
	vmovsd	xmm22, qword ptr [rdi]
	movabs	rdi, 136143485007160
	vmovsd	xmm23, qword ptr [rdi]
	movabs	rdi, 136143485007192
	vmovsd	xmm31, qword ptr [rdi]
	movabs	rdi, 136143485007248
	vmulpd	xmm2, xmm4, xmm0
	vfmsub231pd	xmm2, xmm3, xmm1        # xmm2 = (xmm3 * xmm1) - xmm2
	vmulpd	xmm1, xmm4, xmm1
	vfmadd231pd	xmm1, xmm3, xmm0        # xmm1 = (xmm3 * xmm0) + xmm1
	vmovsd	xmm3, qword ptr [r11]
	vmovapd	xmmword ptr [rbp - 1920], xmm2
	vmovsd	xmm2, qword ptr [r9]
	movabs	r9, 136143485007032
	vmovapd	xmmword ptr [rbp - 1936], xmm1
	vshufpd	xmm1, xmm4, xmm4, 1             # xmm1 = xmm4[1,0]
	vmovsd	xmm4, qword ptr [r14]
	vmovapd	xmm6, xmm3
	vmovsd	qword ptr [rbp - 240], xmm6
	vmulsd	xmm0, xmm1, xmm2
	vmovapd	xmm11, xmm2
	vmovsd	qword ptr [rbp - 216], xmm2
	vmovsd	xmm2, qword ptr [r10]
	movabs	r10, 136143485007048
	vfnmadd231sd	xmm0, xmm5, xmm2        # xmm0 = -(xmm5 * xmm2) + xmm0
	vmovapd	xmm15, xmm2
	vmovsd	qword ptr [rbp - 232], xmm15
	vmovsd	qword ptr [rbp - 1552], xmm0
	vmulsd	xmm0, xmm1, xmm2
	vmovsd	xmm2, qword ptr [r9]
	movabs	r9, 136143485007040
	vmovsd	xmm14, qword ptr [r9]
	vfmsub231sd	xmm0, xmm5, xmm2        # xmm0 = (xmm5 * xmm2) - xmm0
	vmovapd	xmm13, xmm2
	vmovsd	qword ptr [rbp - 224], xmm2
	vmovsd	xmm2, qword ptr [r15]
	vmovsd	qword ptr [rbp - 208], xmm14
	vmovsd	qword ptr [rbp - 1544], xmm0
	vmulsd	xmm0, xmm1, xmm2
	vmovapd	xmm8, xmm2
	vmovsd	xmm2, qword ptr [rbx]
	vmovsd	qword ptr [rbp - 848], xmm8
	vfmadd231sd	xmm0, xmm5, xmm4        # xmm0 = (xmm5 * xmm4) + xmm0
	vmovsd	qword ptr [rbp - 1568], xmm0
	vmulsd	xmm0, xmm1, xmm4
	vmovapd	xmm9, xmm2
	vmovsd	qword ptr [rbp - 200], xmm9
	vfmadd231sd	xmm0, xmm5, xmm2        # xmm0 = (xmm5 * xmm2) + xmm0
	vmovsd	xmm2, qword ptr [r10]
	vmovsd	qword ptr [rbp - 1560], xmm0
	vmulsd	xmm0, xmm1, xmm7
	vfmadd231sd	xmm0, xmm5, xmm10       # xmm0 = (xmm5 * xmm10) + xmm0
	vmovsd	qword ptr [rbp - 1608], xmm0
	vmulsd	xmm0, xmm10, xmm1
	vfnmadd231sd	xmm0, xmm5, xmm7        # xmm0 = -(xmm5 * xmm7) + xmm0
	vmovsd	qword ptr [rbp - 1600], xmm0
	vmulsd	xmm0, xmm1, xmm3
	vfmadd231sd	xmm0, xmm5, xmm2        # xmm0 = (xmm5 * xmm2) + xmm0
	vmovsd	qword ptr [rbp - 880], xmm0
	vmulsd	xmm0, xmm1, xmm2
	vmovsd	xmm1, qword ptr [rbp - 312]
	vfmadd231sd	xmm0, xmm14, xmm5       # xmm0 = (xmm14 * xmm5) + xmm0
	vmovsd	xmm5, qword ptr [rbp - 776]
	vmovsd	qword ptr [rbp - 864], xmm0
	vmovsd	xmm0, qword ptr [rbp - 320]
	vmulsd	xmm3, xmm1, xmm22
	vfmadd231sd	xmm3, xmm0, xmm17       # xmm3 = (xmm0 * xmm17) + xmm3
	vmovsd	qword ptr [rbp - 1288], xmm3
	vmulsd	xmm3, xmm1, xmm17
	vfnmadd231sd	xmm3, xmm0, xmm22       # xmm3 = -(xmm0 * xmm22) + xmm3
	vmovsd	qword ptr [rbp - 1280], xmm3
	vmulsd	xmm3, xmm14, xmm1
	vfmadd231sd	xmm3, xmm0, xmm2        # xmm3 = (xmm0 * xmm2) + xmm3
	vmovsd	qword ptr [rbp - 1400], xmm3
	vmulsd	xmm3, xmm1, xmm2
	vfmadd231sd	xmm3, xmm0, xmm6        # xmm3 = (xmm0 * xmm6) + xmm3
	vmovsd	qword ptr [rbp - 1392], xmm3
	vmulsd	xmm3, xmm1, xmm7
	vfmsub231sd	xmm3, xmm0, xmm10       # xmm3 = (xmm0 * xmm10) - xmm3
	vmovsd	qword ptr [rbp - 1536], xmm3
	vmulsd	xmm3, xmm10, xmm1
	vfmadd231sd	xmm3, xmm0, xmm7        # xmm3 = (xmm0 * xmm7) + xmm3
	vmovsd	qword ptr [rbp - 1528], xmm3
	vmulsd	xmm3, xmm9, xmm1
	vfmadd231sd	xmm3, xmm0, xmm4        # xmm3 = (xmm0 * xmm4) + xmm3
	vmovsd	qword ptr [rbp - 1584], xmm3
	vmulsd	xmm3, xmm1, xmm4
	vfmadd231sd	xmm3, xmm0, xmm8        # xmm3 = (xmm0 * xmm8) + xmm3
	vmovsd	qword ptr [rbp - 1592], xmm3
	vmulsd	xmm3, xmm1, xmm23
	vmulsd	xmm1, xmm1, xmm25
	vfmadd231sd	xmm1, xmm23, xmm0       # xmm1 = (xmm23 * xmm0) + xmm1
	vfmsub231sd	xmm3, xmm0, xmm25       # xmm3 = (xmm0 * xmm25) - xmm3
	vmovsd	xmm0, qword ptr [rbp - 368]
	vmovsd	qword ptr [rbp - 320], xmm1
	vmovsd	xmm1, qword ptr [rbp - 336]
	vmovsd	qword ptr [rbp - 1616], xmm3
	vmulsd	xmm3, xmm1, xmm22
	vfmadd231sd	xmm3, xmm0, xmm17       # xmm3 = (xmm0 * xmm17) + xmm3
	vmovsd	qword ptr [rbp - 1216], xmm3
	vmulsd	xmm3, xmm1, xmm17
	vfnmadd231sd	xmm3, xmm0, xmm22       # xmm3 = -(xmm0 * xmm22) + xmm3
	vmovsd	qword ptr [rbp - 1224], xmm3
	vmulsd	xmm3, xmm9, xmm1
	vfmadd231sd	xmm3, xmm0, xmm4        # xmm3 = (xmm0 * xmm4) + xmm3
	vmovsd	qword ptr [rbp - 1304], xmm3
	vmulsd	xmm3, xmm1, xmm4
	vfmadd231sd	xmm3, xmm0, xmm8        # xmm3 = (xmm0 * xmm8) + xmm3
	vmovsd	qword ptr [rbp - 1296], xmm3
	vmulsd	xmm3, xmm14, xmm1
	vfmadd231sd	xmm3, xmm0, xmm2        # xmm3 = (xmm0 * xmm2) + xmm3
	vmovsd	qword ptr [rbp - 1416], xmm3
	vmulsd	xmm3, xmm1, xmm2
	vfmadd231sd	xmm3, xmm0, xmm6        # xmm3 = (xmm0 * xmm6) + xmm3
	vmovsd	qword ptr [rbp - 1408], xmm3
	vmulsd	xmm3, xmm1, xmm23
	vfmsub231sd	xmm3, xmm0, xmm25       # xmm3 = (xmm0 * xmm25) - xmm3
	vmovsd	qword ptr [rbp - 1504], xmm3
	vmulsd	xmm3, xmm1, xmm25
	vfmadd231sd	xmm3, xmm0, xmm23       # xmm3 = (xmm0 * xmm23) + xmm3
	vmovsd	qword ptr [rbp - 1496], xmm3
	vmulsd	xmm3, xmm1, xmm7
	vmulsd	xmm1, xmm10, xmm1
	vfmadd231sd	xmm1, xmm7, xmm0        # xmm1 = (xmm7 * xmm0) + xmm1
	vfmsub231sd	xmm3, xmm0, xmm10       # xmm3 = (xmm0 * xmm10) - xmm3
	vmovsd	xmm0, qword ptr [rbp - 360]
	vmovsd	qword ptr [rbp - 1576], xmm1
	vmovsd	xmm1, qword ptr [rbp - 344]
	vmovsd	qword ptr [rbp - 312], xmm3
	vmulsd	xmm3, xmm1, xmm7
	vfmadd231sd	xmm3, xmm0, xmm10       # xmm3 = (xmm0 * xmm10) + xmm3
	vmovsd	qword ptr [rbp - 1184], xmm3
	vmulsd	xmm3, xmm10, xmm1
	vfnmadd231sd	xmm3, xmm0, xmm7        # xmm3 = -(xmm0 * xmm7) + xmm3
	vmovsd	qword ptr [rbp - 1192], xmm3
	vmulsd	xmm3, xmm11, xmm1
	vmovsd	xmm11, qword ptr [rbp - 56]
	vfnmadd231sd	xmm3, xmm0, xmm15       # xmm3 = -(xmm0 * xmm15) + xmm3
	vmovsd	qword ptr [rbp - 1272], xmm3
	vmulsd	xmm3, xmm15, xmm1
	vmovsd	xmm15, qword ptr [rbp - 784]
	vfmsub231sd	xmm3, xmm0, xmm13       # xmm3 = (xmm0 * xmm13) - xmm3
	vmovsd	qword ptr [rbp - 1264], xmm3
	vmulsd	xmm3, xmm1, xmm6
	vmovapd	xmm6, xmm19
	vfmadd213sd	xmm6, xmm24, xmm30      # xmm6 = (xmm24 * xmm6) + xmm30
	vmovsd	xmm24, qword ptr [rbp - 48]
	vmovsd	xmm30, qword ptr [rbp - 792]
	vfmadd231sd	xmm3, xmm0, xmm2        # xmm3 = (xmm0 * xmm2) + xmm3
	vmovsd	qword ptr [rbp - 1360], xmm3
	vmulsd	xmm3, xmm1, xmm2
	vfmadd231sd	xmm3, xmm0, xmm14       # xmm3 = (xmm0 * xmm14) + xmm3
	vmovsd	qword ptr [rbp - 1352], xmm3
	vmulsd	xmm3, xmm8, xmm1
	vmovsd	xmm8, qword ptr [rdx + 112]
	vfmadd231sd	xmm3, xmm0, xmm4        # xmm3 = (xmm0 * xmm4) + xmm3
	vmovsd	qword ptr [rbp - 1432], xmm3
	vmulsd	xmm3, xmm1, xmm4
	vmulsd	xmm14, xmm8, xmm5
	vmovsd	qword ptr [rbp - 1368], xmm8
	vfmadd231sd	xmm3, xmm0, xmm9        # xmm3 = (xmm0 * xmm9) + xmm3
	vfmadd231sd	xmm14, xmm11, xmm26     # xmm14 = (xmm11 * xmm26) + xmm14
	vmovsd	qword ptr [rbp - 1440], xmm3
	vmulsd	xmm3, xmm1, xmm31
	vmulsd	xmm1, xmm1, xmm29
	vfmadd231sd	xmm1, xmm31, xmm0       # xmm1 = (xmm31 * xmm0) + xmm1
	vfmsub231sd	xmm3, xmm0, xmm29       # xmm3 = (xmm0 * xmm29) - xmm3
	vmovapd	xmm0, xmmword ptr [rbp - 928]
	vmovsd	qword ptr [rbp - 1512], xmm1
	vmovapd	xmm1, xmmword ptr [rbp - 656]
	vmovsd	qword ptr [rbp - 1520], xmm3
	vmovapd	xmm3, xmmword ptr [rbp - 640]
	vmulsd	xmm9, xmm1, xmm28
	vfmadd231sd	xmm9, xmm3, xmm15       # xmm9 = (xmm3 * xmm15) + xmm9
	vaddsd	xmm9, xmm12, xmm9
	vmovapd	xmm12, xmm19
	vfmsub132sd	xmm12, xmm0, qword ptr [rbp - 672] # xmm12 = (xmm12 * mem) - xmm0
	vmovapd	xmm0, xmmword ptr [rbp - 192]
	vaddsd	xmm6, xmm9, xmm6
	vmulsd	xmm9, xmm15, xmm1
	vmovsd	xmm1, qword ptr [rbp - 40]
	vfnmadd231sd	xmm9, xmm3, xmm28       # xmm9 = -(xmm3 * xmm28) + xmm9
	vaddsd	xmm9, xmm9, qword ptr [rbp - 688]
	vmovsd	xmm3, qword ptr [rbp - 424]
	vaddsd	xmm9, xmm9, qword ptr [rbp - 960]
	vaddsd	xmm6, xmm12, xmm6
	vaddsd	xmm12, xmm9, qword ptr [rbp - 944]
	vmulsd	xmm9, xmm0, xmm26
	vfmsub231sd	xmm9, xmm27, xmm5       # xmm9 = (xmm27 * xmm5) - xmm9
	vaddsd	xmm9, xmm9, xmm6
	vmulsd	xmm6, xmm0, xmm5
	vmovsd	xmm0, qword ptr [rbp - 64]
	vfmadd231sd	xmm6, xmm27, xmm26      # xmm6 = (xmm27 * xmm26) + xmm6
	vmulsd	xmm27, xmm16, xmm19
	vaddsd	xmm20, xmm12, xmm6
	vaddsd	xmm6, xmm21, qword ptr [rbp - 416]
	vmulsd	xmm12, xmm8, xmm26
	vfmsub231sd	xmm12, xmm11, xmm5      # xmm12 = (xmm11 * xmm5) - xmm12
	vmovapd	xmm5, xmm28
	vaddsd	xmm6, xmm12, xmm6
	vmulsd	xmm12, xmm19, qword ptr [rbp - 168]
	vsubsd	xmm13, xmm12, xmm3
	vaddsd	xmm12, xmm12, xmm3
	vmovsd	xmm3, qword ptr [rbp - 216]
	vaddsd	xmm13, xmm13, xmm0
	vaddsd	xmm12, xmm12, xmm0
	vaddsd	xmm13, xmm13, xmm14
	vmulsd	xmm14, xmm18, xmm28
	vfmadd231sd	xmm14, xmm24, xmm15     # xmm14 = (xmm24 * xmm15) + xmm14
	vaddsd	xmm6, xmm14, xmm6
	vmulsd	xmm14, xmm18, xmm15
	vfnmadd231sd	xmm14, xmm24, xmm28     # xmm14 = -(xmm24 * xmm28) + xmm14
	vmulsd	xmm28, xmm1, xmm30
	vaddsd	xmm13, xmm13, xmm14
	vsubsd	xmm14, xmm27, xmm28
	vaddsd	xmm27, xmm28, xmm27
	vaddsd	xmm14, xmm14, xmm6
	vmulsd	xmm6, xmm1, xmm19
	vmulsd	xmm1, xmm16, xmm30
	vmulsd	xmm30, xmm8, xmm5
	vaddsd	xmm16, xmm1, xmm6
	vfmadd231sd	xmm30, xmm11, xmm15     # xmm30 = (xmm11 * xmm15) + xmm30
	vsubsd	xmm1, xmm6, xmm1
	vmovsd	xmm6, qword ptr [rbp - 224]
	vaddsd	xmm16, xmm13, xmm16
	vaddsd	xmm13, xmm21, qword ptr [rbp - 400]
	vmovapd	xmm21, xmm5
	vmulsd	xmm0, xmm16, xmm3
	vaddsd	xmm13, xmm13, xmm30
	vmulsd	xmm30, xmm8, xmm15
	vmovsd	xmm8, qword ptr [rbp - 848]
	vmovsd	xmm15, qword ptr [rsi + 40]
	vfnmadd231sd	xmm30, xmm5, xmm11      # xmm30 = -(xmm5 * xmm11) + xmm30
	vmovsd	xmm5, qword ptr [rbp - 232]
	vmovsd	xmm11, qword ptr [rbp - 776]
	vaddsd	xmm12, xmm12, xmm30
	vmulsd	xmm30, xmm18, xmm26
	vfnmadd231sd	xmm0, xmm14, xmm5       # xmm0 = -(xmm14 * xmm5) + xmm0
	vfmsub231sd	xmm30, xmm24, xmm11     # xmm30 = (xmm24 * xmm11) - xmm30
	vmovsd	qword ptr [rbp - 1328], xmm0
	vmulsd	xmm0, xmm16, xmm5
	vaddsd	xmm13, xmm13, xmm30
	vmulsd	xmm30, xmm18, xmm11
	vmovsd	xmm18, qword ptr [rsi]
	vfmsub231sd	xmm0, xmm14, xmm6       # xmm0 = (xmm14 * xmm6) - xmm0
	vfmadd231sd	xmm30, xmm26, xmm24     # xmm30 = (xmm26 * xmm24) + xmm30
	vaddsd	xmm13, xmm13, xmm27
	vmovsd	xmm24, qword ptr [rbp - 792]
	vmovapd	xmm27, xmm21
	vmovsd	qword ptr [rbp - 1312], xmm0
	vmulsd	xmm0, xmm20, xmm3
	vmovapd	xmm3, xmmword ptr [rbp - 912]
	vaddsd	xmm12, xmm12, xmm30
	vmovsd	xmm30, qword ptr [rdx + 120]
	vfnmadd231sd	xmm0, xmm9, xmm5        # xmm0 = -(xmm9 * xmm5) + xmm0
	vaddsd	xmm1, xmm12, xmm1
	vmovsd	xmm12, qword ptr [rbp - 208]
	vmovsd	qword ptr [rbp - 344], xmm0
	vmulsd	xmm0, xmm20, xmm5
	vfmsub231sd	xmm0, xmm9, xmm6        # xmm0 = (xmm9 * xmm6) - xmm0
	vmovapd	xmm6, xmmword ptr [rbp - 896]
	vmovsd	qword ptr [rbp - 336], xmm0
	vmulsd	xmm0, xmm6, xmm22
	vfmadd231sd	xmm0, xmm3, xmm17       # xmm0 = (xmm3 * xmm17) + xmm0
	vmovsd	qword ptr [rbp - 1480], xmm0
	vmulsd	xmm0, xmm6, xmm17
	vfnmadd231sd	xmm0, xmm3, xmm22       # xmm0 = -(xmm3 * xmm22) + xmm0
	vmovsd	qword ptr [rbp - 1472], xmm0
	vmulsd	xmm0, xmm1, xmm22
	vfmadd231sd	xmm0, xmm13, xmm17      # xmm0 = (xmm13 * xmm17) + xmm0
	vmovsd	qword ptr [rbp - 368], xmm0
	vmulsd	xmm0, xmm1, xmm17
	vfnmadd231sd	xmm0, xmm13, xmm22      # xmm0 = -(xmm13 * xmm22) + xmm0
	vmovsd	xmm22, qword ptr [rsi + 80]
	vmovsd	qword ptr [rbp - 360], xmm0
	vmulsd	xmm0, xmm20, xmm7
	vfmadd231sd	xmm0, xmm9, xmm10       # xmm0 = (xmm9 * xmm10) + xmm0
	vmovsd	qword ptr [rbp - 1256], xmm0
	vmulsd	xmm0, xmm20, xmm10
	vfnmadd231sd	xmm0, xmm9, xmm7        # xmm0 = -(xmm9 * xmm7) + xmm0
	vmovsd	qword ptr [rbp - 1248], xmm0
	vmulsd	xmm0, xmm16, xmm7
	vfmadd231sd	xmm0, xmm14, xmm10      # xmm0 = (xmm14 * xmm10) + xmm0
	vmovsd	qword ptr [rbp - 1384], xmm0
	vmulsd	xmm0, xmm16, xmm10
	vfnmadd231sd	xmm0, xmm14, xmm7       # xmm0 = -(xmm14 * xmm7) + xmm0
	vmovsd	qword ptr [rbp - 1376], xmm0
	vmulsd	xmm0, xmm1, xmm7
	vfmsub231sd	xmm0, xmm13, xmm10      # xmm0 = (xmm13 * xmm10) - xmm0
	vmovsd	qword ptr [rbp - 416], xmm0
	vmulsd	xmm0, xmm10, xmm1
	vmovsd	xmm10, qword ptr [rbp - 240]
	vfmadd231sd	xmm0, xmm13, xmm7       # xmm0 = (xmm13 * xmm7) + xmm0
	vmovsd	xmm7, qword ptr [rbp - 200]
	vmovsd	qword ptr [rbp - 400], xmm0
	vmulsd	xmm0, xmm12, xmm1
	vfmadd231sd	xmm0, xmm13, xmm2       # xmm0 = (xmm13 * xmm2) + xmm0
	vmovapd	xmm5, xmm7
	vmovsd	qword ptr [rbp - 1240], xmm0
	vmulsd	xmm0, xmm1, xmm2
	vfmadd231sd	xmm0, xmm13, xmm10      # xmm0 = (xmm13 * xmm10) + xmm0
	vmovsd	qword ptr [rbp - 1232], xmm0
	vmulsd	xmm0, xmm1, xmm7
	vfmadd231sd	xmm0, xmm13, xmm4       # xmm0 = (xmm13 * xmm4) + xmm0
	vmovsd	qword ptr [rbp - 1344], xmm0
	vmulsd	xmm0, xmm1, xmm4
	vfmadd231sd	xmm0, xmm13, xmm8       # xmm0 = (xmm13 * xmm8) + xmm0
	vmovsd	qword ptr [rbp - 1336], xmm0
	vmulsd	xmm0, xmm1, xmm23
	vfmsub231sd	xmm0, xmm13, xmm25      # xmm0 = (xmm13 * xmm25) - xmm0
	vmovsd	qword ptr [rbp - 160], xmm0
	vmulsd	xmm0, xmm1, xmm25
	vmovapd	xmm1, xmmword ptr [r8]
	vfmadd231sd	xmm0, xmm13, xmm23      # xmm0 = (xmm13 * xmm23) + xmm0
	vmovsd	xmm13, qword ptr [rdx]
	vmovsd	qword ptr [rbp - 424], xmm0
	vmulsd	xmm0, xmm6, xmm23
	vfmsub231sd	xmm0, xmm3, xmm25       # xmm0 = (xmm3 * xmm25) - xmm0
	vmovsd	qword ptr [rbp - 216], xmm0
	vmulsd	xmm0, xmm6, xmm25
	vmovsd	xmm25, qword ptr [rsi + 160]
	vfmadd231sd	xmm0, xmm3, xmm23       # xmm0 = (xmm3 * xmm23) + xmm0
	vmovsd	xmm23, qword ptr [rbp - 784]
	vmovsd	qword ptr [rbp - 1488], xmm0
	vmulsd	xmm0, xmm16, xmm31
	vfmsub231sd	xmm0, xmm14, xmm29      # xmm0 = (xmm14 * xmm29) - xmm0
	vmovsd	qword ptr [rbp - 1464], xmm0
	vmulsd	xmm0, xmm16, xmm29
	vfmadd231sd	xmm0, xmm14, xmm31      # xmm0 = (xmm14 * xmm31) + xmm0
	vmovsd	qword ptr [rbp - 1456], xmm0
	vmulsd	xmm0, xmm20, xmm31
	vfmsub231sd	xmm0, xmm9, xmm29       # xmm0 = (xmm9 * xmm29) - xmm0
	vmovsd	qword ptr [rbp - 232], xmm0
	vmulsd	xmm0, xmm20, xmm29
	vmovsd	xmm29, qword ptr [rdx + 40]
	vfmadd231sd	xmm0, xmm9, xmm31       # xmm0 = (xmm9 * xmm31) + xmm0
	vmovsd	qword ptr [rbp - 224], xmm0
	vmulsd	xmm0, xmm20, xmm8
	vfmadd231sd	xmm0, xmm9, xmm4        # xmm0 = (xmm9 * xmm4) + xmm0
	vmovsd	qword ptr [rbp - 1208], xmm0
	vmulsd	xmm0, xmm20, xmm4
	vfmadd231sd	xmm0, xmm9, xmm7        # xmm0 = (xmm9 * xmm7) + xmm0
	vmovsd	qword ptr [rbp - 1200], xmm0
	vmulsd	xmm0, xmm20, xmm10
	vfmadd231sd	xmm0, xmm9, xmm2        # xmm0 = (xmm9 * xmm2) + xmm0
	vmovsd	qword ptr [rbp - 960], xmm0
	vmulsd	xmm0, xmm20, xmm2
	vfmadd231sd	xmm0, xmm12, xmm9       # xmm0 = (xmm12 * xmm9) + xmm0
	vmovsd	qword ptr [rbp - 944], xmm0
	vmulsd	xmm0, xmm12, xmm6
	vfmadd231sd	xmm0, xmm3, xmm2        # xmm0 = (xmm3 * xmm2) + xmm0
	vmovsd	qword ptr [rbp - 1176], xmm0
	vmulsd	xmm0, xmm6, xmm2
	vfmadd231sd	xmm0, xmm3, xmm10       # xmm0 = (xmm3 * xmm10) + xmm0
	vmovsd	qword ptr [rbp - 1168], xmm0
	vmulsd	xmm0, xmm16, xmm10
	vmovsd	xmm10, qword ptr [rdx + 80]
	vfmadd231sd	xmm0, xmm14, xmm2       # xmm0 = (xmm14 * xmm2) + xmm0
	vmovsd	qword ptr [rbp - 928], xmm0
	vmulsd	xmm0, xmm16, xmm2
	vmulsd	xmm7, xmm10, xmm26
	vmulsd	xmm9, xmm10, xmm11
	vmulsd	xmm20, xmm10, xmm21
	vfmadd231sd	xmm0, xmm14, xmm12      # xmm0 = (xmm14 * xmm12) + xmm0
	vfmsub231sd	xmm7, xmm22, xmm11      # xmm7 = (xmm22 * xmm11) - xmm7
	vfmadd231sd	xmm9, xmm22, xmm26      # xmm9 = (xmm22 * xmm26) + xmm9
	vfmadd231sd	xmm20, xmm22, xmm23     # xmm20 = (xmm22 * xmm23) + xmm20
	vmovsd	qword ptr [rbp - 240], xmm0
	vmulsd	xmm0, xmm16, xmm8
	vfmadd231sd	xmm0, xmm14, xmm4       # xmm0 = (xmm14 * xmm4) + xmm0
	vmovsd	qword ptr [rbp - 1448], xmm0
	vmulsd	xmm0, xmm16, xmm4
	vmovapd	xmm16, xmm11
	vfmadd231sd	xmm0, xmm14, xmm5       # xmm0 = (xmm14 * xmm5) + xmm0
	vmovsd	qword ptr [rbp - 1424], xmm0
	vmulsd	xmm0, xmm6, xmm5
	vfmadd231sd	xmm0, xmm3, xmm4        # xmm0 = (xmm3 * xmm4) + xmm0
	vmovsd	qword ptr [rbp - 208], xmm0
	vmulsd	xmm0, xmm6, xmm4
	vmulsd	xmm4, xmm22, xmm24
	vfmadd231sd	xmm0, xmm3, xmm8        # xmm0 = (xmm3 * xmm8) + xmm0
	vmovsd	qword ptr [rbp - 1144], xmm4
	vmovsd	qword ptr [rbp - 200], xmm0
	vmovapd	xmm0, xmmword ptr [rdi]
	vmulpd	xmm2, xmm6, xmm0
	vfmsub231pd	xmm2, xmm3, xmm1        # xmm2 = (xmm3 * xmm1) - xmm2
	vmulpd	xmm1, xmm6, xmm1
	vmulsd	xmm6, xmm25, xmm19
	vfmadd231pd	xmm1, xmm3, xmm0        # xmm1 = (xmm3 * xmm0) + xmm1
	vaddsd	xmm0, xmm18, xmm15
	vmulsd	xmm3, xmm10, xmm24
	vmovapd	xmmword ptr [rbp - 848], xmm2
	vmulsd	xmm2, xmm10, xmm23
	vaddsd	xmm0, xmm0, xmm22
	vfnmadd231sd	xmm2, xmm22, xmm21      # xmm2 = -(xmm22 * xmm21) + xmm2
	vmovapd	xmmword ptr [rbp - 912], xmm1
	vmovapd	xmm1, xmm19
	vfmadd213sd	xmm1, xmm22, xmm3       # xmm1 = (xmm22 * xmm1) + xmm3
	vfmsub213sd	xmm22, xmm19, xmm3      # xmm22 = (xmm19 * xmm22) - xmm3
	vmovsd	qword ptr [rbp - 1160], xmm2
	vmovapd	xmm2, xmmword ptr [rbp - 1072]
	vaddsd	xmm28, xmm2, qword ptr [rbp - 80]
	vmovsd	xmm2, qword ptr [rbp - 112]
	vaddsd	xmm3, xmm2, qword ptr [rbp - 72]
	vmovsd	xmm2, qword ptr [rbp - 104]
	vaddsd	xmm8, xmm2, qword ptr [rbp - 152]
	vmovapd	xmm2, xmm19
	vaddsd	xmm3, xmm3, qword ptr [rbp - 408]
	vaddsd	xmm8, xmm8, qword ptr [rbp - 88]
	vaddsd	xmm11, xmm3, qword ptr [rbp - 96]
	vmulsd	xmm3, xmm25, xmm24
	vaddsd	xmm21, xmm8, qword ptr [rbp - 448]
	vmovsd	xmm8, qword ptr [rsi + 120]
	vaddsd	xmm17, xmm11, qword ptr [rbp - 456]
	vaddsd	xmm11, xmm13, xmm29
	vaddsd	xmm0, xmm8, xmm0
	vaddsd	xmm0, xmm0, xmm25
	vmovsd	qword ptr [rbp - 1072], xmm0
	vaddsd	xmm0, xmm0, xmm28
	vaddsd	xmm0, xmm0, xmm21
	vmovsd	qword ptr [rbp - 72], xmm0
	vmulsd	xmm0, xmm17, xmm24
	vfmadd213sd	xmm2, xmm21, xmm0       # xmm2 = (xmm21 * xmm2) + xmm0
	vmovsd	qword ptr [rbp - 112], xmm2
	vmulsd	xmm2, xmm17, xmm26
	vfmsub231sd	xmm2, xmm21, xmm16      # xmm2 = (xmm21 * xmm16) - xmm2
	vmovsd	qword ptr [rbp - 408], xmm2
	vmulsd	xmm2, xmm17, xmm16
	vfmadd231sd	xmm2, xmm21, xmm26      # xmm2 = (xmm21 * xmm26) + xmm2
	vmovsd	qword ptr [rbp - 448], xmm2
	vmulsd	xmm2, xmm17, xmm27
	vfmadd231sd	xmm2, xmm21, xmm23      # xmm2 = (xmm21 * xmm23) + xmm2
	vmovsd	qword ptr [rbp - 152], xmm2
	vmulsd	xmm2, xmm17, xmm23
	vfnmadd231sd	xmm2, xmm21, xmm27      # xmm2 = -(xmm21 * xmm27) + xmm2
	vmovsd	qword ptr [rbp - 456], xmm2
	vmulsd	xmm2, xmm21, xmm24
	vfmsub213sd	xmm21, xmm19, xmm0      # xmm21 = (xmm19 * xmm21) - xmm0
	vmulsd	xmm0, xmm29, xmm27
	vfmadd231sd	xmm0, xmm15, xmm23      # xmm0 = (xmm15 * xmm23) + xmm0
	vmovsd	qword ptr [rbp - 896], xmm2
	vmulsd	xmm2, xmm30, xmm24
	vmovsd	qword ptr [rbp - 96], xmm2
	vaddsd	xmm0, xmm18, xmm0
	vaddsd	xmm0, xmm0, xmm1
	vaddsd	xmm1, xmm11, xmm10
	vmulsd	xmm11, xmm30, xmm19
	vmovsd	qword ptr [rbp - 80], xmm1
	vmulsd	xmm1, xmm10, xmm19
	vmulsd	xmm10, xmm29, xmm23
	vmovsd	qword ptr [rbp - 104], xmm1
	vsubsd	xmm12, xmm1, xmm4
	vmulsd	xmm1, xmm8, xmm19
	vfnmadd231sd	xmm10, xmm15, xmm27     # xmm10 = -(xmm15 * xmm27) + xmm10
	vsubsd	xmm14, xmm1, xmm2
	vmovsd	qword ptr [rbp - 88], xmm1
	vmovsd	xmm1, qword ptr [rdx + 160]
	vmulsd	xmm2, xmm15, xmm24
	vaddsd	xmm10, xmm13, xmm10
	vaddsd	xmm14, xmm14, xmm0
	vaddsd	xmm12, xmm10, xmm12
	vmulsd	xmm10, xmm8, xmm24
	vaddsd	xmm0, xmm10, xmm11
	vmulsd	xmm31, xmm1, xmm26
	vmulsd	xmm5, xmm1, xmm24
	vmulsd	xmm4, xmm1, xmm19
	vaddsd	xmm12, xmm12, xmm0
	vfmsub231sd	xmm31, xmm25, xmm16     # xmm31 = (xmm25 * xmm16) - xmm31
	vaddsd	xmm0, xmm14, xmm31
	vmulsd	xmm31, xmm1, xmm16
	vfmadd231sd	xmm31, xmm25, xmm26     # xmm31 = (xmm25 * xmm26) + xmm31
	vmovsd	qword ptr [rbp - 1152], xmm0
	vmulsd	xmm0, xmm29, xmm24
	vaddsd	xmm14, xmm12, xmm31
	vmovapd	xmm31, xmm19
	vfmadd213sd	xmm31, xmm15, xmm0      # xmm31 = (xmm15 * xmm31) + xmm0
	vfmsub231sd	xmm0, xmm15, xmm19      # xmm0 = (xmm15 * xmm19) - xmm0
	vaddsd	xmm31, xmm18, xmm31
	vaddsd	xmm0, xmm18, xmm0
	vaddsd	xmm7, xmm31, xmm7
	vmulsd	xmm31, xmm29, xmm19
	vaddsd	xmm0, xmm0, xmm20
	vmulsd	xmm20, xmm30, xmm16
	vsubsd	xmm12, xmm31, xmm2
	vaddsd	xmm2, xmm2, xmm31
	vfmadd231sd	xmm20, xmm26, xmm8      # xmm20 = (xmm26 * xmm8) + xmm20
	vmovsd	xmm31, qword ptr [rbp - 1152]
	vaddsd	xmm12, xmm13, xmm12
	vaddsd	xmm2, xmm13, xmm2
	vaddsd	xmm2, xmm2, qword ptr [rbp - 1160]
	vaddsd	xmm9, xmm12, xmm9
	vmulsd	xmm12, xmm30, xmm27
	vfmadd231sd	xmm12, xmm8, xmm23      # xmm12 = (xmm8 * xmm23) + xmm12
	vaddsd	xmm7, xmm12, xmm7
	vmulsd	xmm12, xmm30, xmm23
	vaddsd	xmm2, xmm2, xmm20
	vmovapd	xmm20, xmm26
	vfnmadd231sd	xmm12, xmm8, xmm27      # xmm12 = -(xmm8 * xmm27) + xmm12
	vaddsd	xmm9, xmm9, xmm12
	vsubsd	xmm12, xmm6, xmm5
	vaddsd	xmm5, xmm5, xmm6
	vaddsd	xmm12, xmm12, xmm7
	vaddsd	xmm7, xmm3, xmm4
	vaddsd	xmm7, xmm9, xmm7
	vmulsd	xmm9, xmm30, xmm26
	vfmsub231sd	xmm9, xmm8, xmm16       # xmm9 = (xmm8 * xmm16) - xmm9
	vaddsd	xmm0, xmm9, xmm0
	vaddsd	xmm9, xmm30, qword ptr [rbp - 80]
	vmovapd	xmm30, xmm16
	vaddsd	xmm5, xmm0, xmm5
	vsubsd	xmm0, xmm4, xmm3
	vmovsd	xmm3, qword ptr [rbp - 1144]
	vaddsd	xmm4, xmm7, qword ptr [rbp - 1224]
	vaddsd	xmm3, xmm3, qword ptr [rbp - 104]
	vaddsd	xmm8, xmm2, xmm0
	vmulsd	xmm2, xmm29, xmm16
	vmulsd	xmm0, xmm29, xmm26
	vfmadd231sd	xmm2, xmm26, xmm15      # xmm2 = (xmm26 * xmm15) + xmm2
	vfmsub231sd	xmm0, xmm15, xmm16      # xmm0 = (xmm15 * xmm16) - xmm0
	vmovsd	xmm16, qword ptr [rbp - 1072]
	vmovsd	xmm26, qword ptr [rbp - 896]
	vaddsd	xmm2, xmm13, xmm2
	vaddsd	xmm0, xmm18, xmm0
	vaddsd	xmm2, xmm2, xmm3
	vmovsd	xmm3, qword ptr [rbp - 88]
	vaddsd	xmm0, xmm0, xmm22
	vaddsd	xmm3, xmm3, qword ptr [rbp - 96]
	vaddsd	xmm0, xmm0, xmm3
	vsubsd	xmm3, xmm11, xmm10
	vaddsd	xmm11, xmm9, xmm1
	vaddsd	xmm9, xmm4, qword ptr [rbp - 464]
	vaddsd	xmm4, xmm5, qword ptr [rbp - 1184]
	vaddsd	xmm2, xmm2, xmm3
	vmulsd	xmm3, xmm1, xmm27
	vaddsd	xmm4, xmm4, qword ptr [rbp - 440]
	vfmadd231sd	xmm3, xmm25, xmm23      # xmm3 = (xmm25 * xmm23) + xmm3
	vaddsd	xmm3, xmm0, xmm3
	vmulsd	xmm0, xmm1, xmm23
	vaddsd	xmm1, xmm31, qword ptr [rbp - 504]
	vfnmadd231sd	xmm0, xmm27, xmm25      # xmm0 = -(xmm27 * xmm25) + xmm0
	vmovsd	qword ptr [rbp - 88], xmm4
	vaddsd	xmm4, xmm8, qword ptr [rbp - 1192]
	vaddsd	xmm1, xmm1, qword ptr [rbp - 1288]
	vaddsd	xmm29, xmm4, qword ptr [rbp - 432]
	vaddsd	xmm4, xmm3, qword ptr [rbp - 1624]
	vaddsd	xmm2, xmm2, xmm0
	vmovapd	xmm0, xmmword ptr [rbp - 1104]
	vaddsd	xmm0, xmm0, qword ptr [rbp - 832]
	vaddsd	xmm4, xmm4, qword ptr [rbp - 1552]
	vaddsd	xmm0, xmm0, qword ptr [rbp - 1136]
	vmovsd	qword ptr [rbp - 96], xmm1
	vaddsd	xmm1, xmm14, qword ptr [rbp - 496]
	vaddsd	xmm0, xmm0, qword ptr [rbp - 1120]
	vaddsd	xmm22, xmm1, qword ptr [rbp - 1280]
	vaddsd	xmm1, xmm12, qword ptr [rbp - 1216]
	vaddsd	xmm0, xmm0, qword ptr [rbp - 816]
	vmovsd	qword ptr [rbp - 80], xmm4
	vaddsd	xmm4, xmm2, qword ptr [rbp - 1632]
	vaddsd	xmm1, xmm1, qword ptr [rbp - 120]
	vaddsd	xmm25, xmm4, qword ptr [rbp - 1544]
	vmulsd	xmm4, xmm0, xmm27
	vmulsd	xmm6, xmm0, xmm23
	vmulsd	xmm13, xmm0, xmm24
	vfmadd231sd	xmm4, xmm28, xmm23      # xmm4 = (xmm28 * xmm23) + xmm4
	vfnmadd231sd	xmm6, xmm28, xmm27      # xmm6 = -(xmm28 * xmm27) + xmm6
	vaddsd	xmm4, xmm16, xmm4
	vaddsd	xmm4, xmm4, qword ptr [rbp - 112]
	vaddsd	xmm6, xmm11, xmm6
	vmovsd	qword ptr [rbp - 440], xmm4
	vaddsd	xmm4, xmm11, xmm0
	vaddsd	xmm4, xmm4, xmm17
	vmovsd	qword ptr [rbp - 832], xmm4
	vmulsd	xmm4, xmm17, xmm19
	vmovapd	xmm17, xmm16
	vsubsd	xmm10, xmm4, xmm26
	vaddsd	xmm4, xmm26, xmm4
	vaddsd	xmm15, xmm10, xmm6
	vaddsd	xmm6, xmm31, qword ptr [rbp - 568]
	vaddsd	xmm6, xmm6, qword ptr [rbp - 1400]
	vmovsd	qword ptr [rbp - 120], xmm6
	vaddsd	xmm6, xmm14, qword ptr [rbp - 560]
	vaddsd	xmm6, xmm6, qword ptr [rbp - 1392]
	vmovsd	qword ptr [rbp - 432], xmm6
	vaddsd	xmm6, xmm12, qword ptr [rbp - 1304]
	vaddsd	xmm6, xmm6, qword ptr [rbp - 472]
	vmovsd	qword ptr [rbp - 464], xmm6
	vaddsd	xmm6, xmm7, qword ptr [rbp - 1296]
	vaddsd	xmm6, xmm6, qword ptr [rbp - 144]
	vmovsd	qword ptr [rbp - 112], xmm6
	vaddsd	xmm6, xmm5, qword ptr [rbp - 1272]
	vaddsd	xmm6, xmm6, qword ptr [rbp - 136]
	vmovsd	qword ptr [rbp - 144], xmm6
	vaddsd	xmm6, xmm8, qword ptr [rbp - 1264]
	vaddsd	xmm6, xmm6, qword ptr [rbp - 128]
	vmovsd	qword ptr [rbp - 128], xmm6
	vaddsd	xmm6, xmm3, qword ptr [rbp - 1664]
	vaddsd	xmm6, xmm6, qword ptr [rbp - 1568]
	vmovsd	qword ptr [rbp - 136], xmm6
	vaddsd	xmm6, xmm2, qword ptr [rbp - 1656]
	vaddsd	xmm6, xmm6, qword ptr [rbp - 1560]
	vmovsd	qword ptr [rbp - 104], xmm6
	vmovapd	xmm6, xmm19
	vfmadd213sd	xmm6, xmm28, xmm13      # xmm6 = (xmm28 * xmm6) + xmm13
	vfmsub231sd	xmm13, xmm28, xmm19     # xmm13 = (xmm28 * xmm19) - xmm13
	vaddsd	xmm6, xmm16, xmm6
	vaddsd	xmm10, xmm6, qword ptr [rbp - 408]
	vmulsd	xmm6, xmm0, xmm19
	vmulsd	xmm16, xmm28, xmm24
	vaddsd	xmm13, xmm17, xmm13
	vaddsd	xmm13, xmm13, qword ptr [rbp - 152]
	vsubsd	xmm18, xmm6, xmm16
	vaddsd	xmm6, xmm16, xmm6
	vaddsd	xmm18, xmm11, xmm18
	vaddsd	xmm18, xmm18, qword ptr [rbp - 448]
	vaddsd	xmm6, xmm11, xmm6
	vaddsd	xmm6, xmm6, qword ptr [rbp - 456]
	vmovsd	qword ptr [rbp - 472], xmm18
	vaddsd	xmm18, xmm31, qword ptr [rbp - 600]
	vmovsd	qword ptr [rbp - 504], xmm6
	vaddsd	xmm6, xmm31, qword ptr [rbp - 736]
	vaddsd	xmm18, xmm18, qword ptr [rbp - 1536]
	vaddsd	xmm6, xmm6, qword ptr [rbp - 1584]
	vmovsd	qword ptr [rbp - 816], xmm18
	vaddsd	xmm18, xmm14, qword ptr [rbp - 592]
	vmovsd	qword ptr [rbp - 152], xmm6
	vaddsd	xmm6, xmm14, qword ptr [rbp - 720]
	vaddsd	xmm18, xmm18, qword ptr [rbp - 1528]
	vaddsd	xmm6, xmm6, qword ptr [rbp - 1592]
	vmovsd	qword ptr [rbp - 568], xmm18
	vaddsd	xmm18, xmm12, qword ptr [rbp - 1416]
	vaddsd	xmm18, xmm18, qword ptr [rbp - 520]
	vmovsd	qword ptr [rbp - 520], xmm6
	vaddsd	xmm6, xmm12, qword ptr [rbp - 1504]
	vaddsd	xmm6, xmm6, qword ptr [rbp - 552]
	vmovsd	qword ptr [rbp - 1136], xmm18
	vaddsd	xmm18, xmm7, qword ptr [rbp - 1408]
	vmovsd	qword ptr [rbp - 736], xmm6
	vaddsd	xmm6, xmm7, qword ptr [rbp - 1496]
	vaddsd	xmm18, xmm18, qword ptr [rbp - 512]
	vmovsd	qword ptr [rbp - 512], xmm13
	vaddsd	xmm6, xmm6, qword ptr [rbp - 544]
	vmovsd	qword ptr [rbp - 560], xmm18
	vaddsd	xmm18, xmm5, qword ptr [rbp - 1360]
	vmovsd	qword ptr [rbp - 552], xmm6
	vaddsd	xmm6, xmm5, qword ptr [rbp - 1432]
	vaddsd	xmm18, xmm18, qword ptr [rbp - 488]
	vaddsd	xmm6, xmm6, qword ptr [rbp - 536]
	vmovsd	qword ptr [rbp - 1120], xmm18
	vaddsd	xmm18, xmm8, qword ptr [rbp - 1352]
	vmovsd	qword ptr [rbp - 720], xmm6
	vaddsd	xmm6, xmm8, qword ptr [rbp - 1440]
	vaddsd	xmm18, xmm18, qword ptr [rbp - 480]
	vaddsd	xmm6, xmm6, qword ptr [rbp - 528]
	vmovsd	qword ptr [rbp - 1104], xmm18
	vaddsd	xmm18, xmm3, qword ptr [rbp - 1824]
	vmovsd	qword ptr [rbp - 544], xmm6
	vaddsd	xmm6, xmm3, qword ptr [rbp - 1648]
	vaddsd	xmm18, xmm18, qword ptr [rbp - 1608]
	vaddsd	xmm6, xmm6, qword ptr [rbp - 880]
	vmovsd	qword ptr [rbp - 600], xmm18
	vaddsd	xmm18, xmm2, qword ptr [rbp - 1816]
	vmovsd	qword ptr [rbp - 592], xmm6
	vaddsd	xmm6, xmm2, qword ptr [rbp - 1640]
	vaddsd	xmm18, xmm18, qword ptr [rbp - 1600]
	vaddsd	xmm6, xmm6, qword ptr [rbp - 864]
	vmovsd	qword ptr [rbp - 536], xmm6
	vmulsd	xmm6, xmm0, xmm20
	vmulsd	xmm0, xmm0, xmm30
	vfmadd231sd	xmm0, xmm20, xmm28      # xmm0 = (xmm20 * xmm28) + xmm0
	vfmsub231sd	xmm6, xmm28, xmm30      # xmm6 = (xmm28 * xmm30) - xmm6
	vaddsd	xmm0, xmm11, xmm0
	vaddsd	xmm6, xmm17, xmm6
	vmovapd	xmm17, xmm24
	vaddsd	xmm0, xmm0, xmm4
	vaddsd	xmm4, xmm14, qword ptr [rbp - 752]
	vaddsd	xmm6, xmm6, xmm21
	vmovsd	qword ptr [rbp - 496], xmm0
	vaddsd	xmm0, xmm31, qword ptr [rbp - 768]
	vmovsd	qword ptr [rbp - 528], xmm6
	vaddsd	xmm31, xmm0, qword ptr [rbp - 1616]
	vaddsd	xmm0, xmm4, qword ptr [rbp - 320]
	vaddsd	xmm4, xmm7, qword ptr [rbp - 1576]
	vmovsd	qword ptr [rbp - 752], xmm0
	vaddsd	xmm0, xmm12, qword ptr [rbp - 312]
	vaddsd	xmm0, xmm0, qword ptr [rbp - 704]
	vmovsd	qword ptr [rbp - 704], xmm0
	vaddsd	xmm0, xmm4, qword ptr [rbp - 608]
	vaddsd	xmm4, xmm8, qword ptr [rbp - 1512]
	vmovsd	qword ptr [rbp - 608], xmm0
	vaddsd	xmm0, xmm5, qword ptr [rbp - 1520]
	vaddsd	xmm0, xmm0, qword ptr [rbp - 584]
	vmovsd	qword ptr [rbp - 584], xmm0
	vaddsd	xmm0, xmm4, qword ptr [rbp - 576]
	vmovapd	xmm4, xmmword ptr [rbp - 1920]
	vmovsd	qword ptr [rbp - 576], xmm0
	vaddsd	xmm0, xmm3, xmm4
	vshufpd	xmm3, xmm4, xmm4, 1             # xmm3 = xmm4[1,0]
	vaddsd	xmm0, xmm0, xmm3
	vmovapd	xmm3, xmmword ptr [rbp - 1936]
	vmovsd	qword ptr [rbp - 488], xmm0
	vaddsd	xmm0, xmm2, xmm3
	vshufpd	xmm2, xmm3, xmm3, 1             # xmm2 = xmm3[1,0]
	vaddsd	xmm0, xmm0, xmm2
	vmovsd	xmm2, qword ptr [rbp - 56]
	vaddsd	xmm2, xmm2, qword ptr [rbp - 616]
	vmovsd	qword ptr [rbp - 768], xmm0
	vmovapd	xmm0, xmmword ptr [rbp - 640]
	vaddsd	xmm0, xmm0, qword ptr [rbp - 1888]
	vaddsd	xmm2, xmm2, qword ptr [rbp - 48]
	vaddsd	xmm0, xmm0, qword ptr [rbp - 1904]
	vaddsd	xmm3, xmm2, qword ptr [rbp - 32]
	vaddsd	xmm2, xmm22, qword ptr [rbp - 1248]
	vaddsd	xmm0, xmm0, qword ptr [rbp - 672]
	vaddsd	xmm4, xmm0, qword ptr [rbp - 1872]
	vmovapd	xmm0, xmmword ptr [rbp - 656]
	vaddsd	xmm0, xmm0, qword ptr [rbp - 688]
	vaddsd	xmm0, xmm0, qword ptr [rbp - 1088]
	vaddsd	xmm0, xmm0, qword ptr [rbp - 1952]
	vaddsd	xmm7, xmm0, qword ptr [rbp - 192]
	vmovsd	xmm0, qword ptr [rbp - 64]
	vaddsd	xmm0, xmm0, qword ptr [rbp - 168]
	vaddsd	xmm12, xmm7, qword ptr [rbp - 832]
	vaddsd	xmm0, xmm0, qword ptr [rbp - 1368]
	vaddsd	xmm0, xmm0, qword ptr [rbp - 1320]
	vaddsd	xmm5, xmm0, qword ptr [rbp - 40]
	vmovsd	xmm0, qword ptr [rbp - 96]
	vmulsd	xmm6, xmm7, xmm24
	vaddsd	xmm0, xmm0, qword ptr [rbp - 1256]
	vaddsd	xmm0, xmm0, qword ptr [rbp - 1808]
	vmulsd	xmm11, xmm5, xmm20
	vaddsd	xmm16, xmm12, xmm5
	vmulsd	xmm13, xmm5, xmm24
	vfmsub231sd	xmm11, xmm3, xmm30      # xmm11 = (xmm3 * xmm30) - xmm11
	vmovsd	qword ptr [rbp - 192], xmm0
	vaddsd	xmm0, xmm2, qword ptr [rbp - 1800]
	vmulsd	xmm2, xmm4, xmm19
	vmovsd	qword ptr [rbp - 168], xmm0
	vaddsd	xmm0, xmm1, qword ptr [rbp - 1696]
	vaddsd	xmm1, xmm9, qword ptr [rbp - 1680]
	vaddsd	xmm0, xmm0, qword ptr [rbp - 1328]
	vmovsd	qword ptr [rbp - 688], xmm0
	vaddsd	xmm0, xmm1, qword ptr [rbp - 1312]
	vaddsd	xmm1, xmm29, qword ptr [rbp - 1704]
	vmovapd	xmm29, xmmword ptr [rbp - 848]
	vmovsd	qword ptr [rbp - 672], xmm0
	vmovsd	xmm0, qword ptr [rbp - 88]
	vaddsd	xmm0, xmm0, qword ptr [rbp - 1712]
	vaddsd	xmm0, xmm0, qword ptr [rbp - 1240]
	vmovsd	qword ptr [rbp - 64], xmm0
	vaddsd	xmm0, xmm1, qword ptr [rbp - 1232]
	vaddsd	xmm1, xmm25, qword ptr [rbp - 1168]
	vmovsd	qword ptr [rbp - 56], xmm0
	vmovsd	xmm0, qword ptr [rbp - 80]
	vaddsd	xmm0, xmm0, qword ptr [rbp - 1176]
	vaddsd	xmm0, xmm0, qword ptr [rbp - 1728]
	vmovsd	qword ptr [rbp - 48], xmm0
	vaddsd	xmm0, xmm1, qword ptr [rbp - 1720]
	vmulsd	xmm1, xmm7, xmm19
	vmovsd	qword ptr [rbp - 40], xmm0
	vsubsd	xmm0, xmm2, xmm6
	vaddsd	xmm8, xmm0, qword ptr [rbp - 440]
	vmulsd	xmm0, xmm4, xmm24
	vmulsd	xmm24, xmm5, xmm27
	vaddsd	xmm2, xmm6, xmm2
	vaddsd	xmm2, xmm2, qword ptr [rbp - 528]
	vaddsd	xmm9, xmm0, xmm1
	vfmadd231sd	xmm24, xmm3, xmm23      # xmm24 = (xmm3 * xmm23) + xmm24
	vsubsd	xmm0, xmm1, xmm0
	vaddsd	xmm0, xmm0, qword ptr [rbp - 496]
	vmovsd	xmm1, qword ptr [rbp - 576]
	vaddsd	xmm1, xmm1, qword ptr [rbp - 248]
	vaddsd	xmm9, xmm15, xmm9
	vaddsd	xmm1, xmm1, qword ptr [rbp - 424]
	vaddsd	xmm8, xmm8, xmm11
	vaddsd	xmm11, xmm4, qword ptr [rbp - 72]
	vmovsd	qword ptr [rbp - 32], xmm8
	vmulsd	xmm8, xmm5, xmm30
	vfmadd231sd	xmm8, xmm3, xmm20       # xmm8 = (xmm3 * xmm20) + xmm8
	vaddsd	xmm8, xmm9, xmm8
	vmovsd	xmm9, qword ptr [rbp - 432]
	vaddsd	xmm12, xmm11, xmm3
	vaddsd	xmm9, xmm9, qword ptr [rbp - 1200]
	vmovsd	qword ptr [rbp - 656], xmm8
	vmovsd	xmm8, qword ptr [rbp - 120]
	vmovsd	qword ptr [rcx], xmm12
	vmovsd	xmm12, qword ptr [rbp - 192]
	vaddsd	xmm8, xmm8, qword ptr [rbp - 1208]
	mov	rax, qword ptr [rax]
	vaddsd	xmm8, xmm8, qword ptr [rbp - 1840]
	vmovsd	qword ptr [rax], xmm16
	vmovsd	qword ptr [rcx + 8], xmm12
	vmovsd	xmm12, qword ptr [rbp - 168]
	vmovsd	qword ptr [rbp - 640], xmm8
	vaddsd	xmm8, xmm9, qword ptr [rbp - 1832]
	vmovsd	xmm9, qword ptr [rbp - 112]
	vaddsd	xmm9, xmm9, qword ptr [rbp - 392]
	vmovsd	qword ptr [rax + 8], xmm12
	vmovsd	xmm12, qword ptr [rbp - 688]
	vmovsd	qword ptr [rbp - 616], xmm8
	vmovsd	xmm8, qword ptr [rbp - 464]
	vmovsd	qword ptr [rcx + 16], xmm12
	vmovsd	xmm12, qword ptr [rbp - 672]
	vaddsd	xmm8, xmm8, qword ptr [rbp - 1056]
	vaddsd	xmm8, xmm8, qword ptr [rbp - 1384]
	vmovsd	qword ptr [rax + 16], xmm12
	vmovsd	xmm12, qword ptr [rbp - 64]
	vmovsd	qword ptr [rbp - 480], xmm8
	vaddsd	xmm8, xmm9, qword ptr [rbp - 1376]
	vmovsd	xmm9, qword ptr [rbp - 128]
	vaddsd	xmm9, xmm9, qword ptr [rbp - 1744]
	vmovsd	qword ptr [rcx + 24], xmm12
	vmovsd	xmm12, qword ptr [rbp - 56]
	vmovsd	qword ptr [rbp - 1088], xmm8
	vmovsd	xmm8, qword ptr [rbp - 144]
	vmovsd	qword ptr [rax + 24], xmm12
	vmovsd	xmm12, qword ptr [rbp - 48]
	vaddsd	xmm8, xmm8, qword ptr [rbp - 1736]
	vaddsd	xmm8, xmm8, qword ptr [rbp - 1344]
	vmovsd	qword ptr [rcx + 32], xmm12
	vmovsd	xmm12, qword ptr [rbp - 40]
	vmovsd	qword ptr [rbp - 144], xmm8
	vaddsd	xmm8, xmm9, qword ptr [rbp - 1336]
	vmovsd	xmm9, qword ptr [rbp - 104]
	vaddsd	xmm9, xmm9, qword ptr [rbp - 1472]
	vmovsd	qword ptr [rax + 32], xmm12
	vmovsd	xmm12, qword ptr [rbp - 32]
	vmovsd	qword ptr [rbp - 128], xmm8
	vmovsd	xmm8, qword ptr [rbp - 136]
	vmovsd	qword ptr [rcx + 40], xmm12
	vmovsd	xmm12, qword ptr [rbp - 656]
	vaddsd	xmm8, xmm8, qword ptr [rbp - 1480]
	vaddsd	xmm8, xmm8, qword ptr [rbp - 1040]
	vmovsd	qword ptr [rax + 40], xmm12
	vmovsd	xmm12, qword ptr [rbp - 640]
	vmovsd	qword ptr [rbp - 136], xmm8
	vaddsd	xmm8, xmm9, qword ptr [rbp - 1024]
	vmulsd	xmm9, xmm7, xmm20
	vmovsd	qword ptr [rcx + 48], xmm12
	vmovsd	xmm12, qword ptr [rbp - 616]
	vfmsub231sd	xmm9, xmm4, xmm30       # xmm9 = (xmm4 * xmm30) - xmm9
	vmovsd	qword ptr [rbp - 120], xmm8
	vmulsd	xmm8, xmm7, xmm27
	vmovsd	qword ptr [rax + 48], xmm12
	vmovsd	xmm12, qword ptr [rbp - 480]
	vfmadd231sd	xmm8, xmm4, xmm23       # xmm8 = (xmm4 * xmm23) + xmm8
	vaddsd	xmm8, xmm10, xmm8
	vmulsd	xmm10, xmm7, xmm30
	vmulsd	xmm7, xmm7, xmm23
	vmulsd	xmm30, xmm5, xmm23
	vmovsd	qword ptr [rcx + 56], xmm12
	vmovsd	xmm12, qword ptr [rbp - 1088]
	vfmadd231sd	xmm10, xmm4, xmm20      # xmm10 = (xmm4 * xmm20) + xmm10
	vfnmadd231sd	xmm7, xmm4, xmm27       # xmm7 = -(xmm4 * xmm27) + xmm7
	vaddsd	xmm4, xmm7, qword ptr [rbp - 472]
	vmulsd	xmm7, xmm3, xmm19
	vfnmadd231sd	xmm30, xmm3, xmm27      # xmm30 = -(xmm3 * xmm27) + xmm30
	vmulsd	xmm3, xmm3, xmm17
	vsubsd	xmm14, xmm7, xmm13
	vaddsd	xmm7, xmm13, xmm7
	vaddsd	xmm8, xmm8, xmm14
	vmovsd	qword ptr [rax + 56], xmm12
	vmovsd	xmm12, qword ptr [rbp - 144]
	vmovsd	qword ptr [rbp - 72], xmm8
	vmulsd	xmm8, xmm5, xmm19
	vmovsd	xmm5, qword ptr [rbp - 568]
	vaddsd	xmm11, xmm8, xmm3
	vsubsd	xmm3, xmm8, xmm3
	vmovsd	qword ptr [rcx + 64], xmm12
	vmovsd	xmm12, qword ptr [rbp - 128]
	vaddsd	xmm28, xmm4, xmm11
	vmovsd	xmm4, qword ptr [rbp - 816]
	vaddsd	xmm11, xmm5, qword ptr [rbp - 336]
	vmovsd	xmm5, qword ptr [rbp - 560]
	vaddsd	xmm4, xmm4, qword ptr [rbp - 344]
	vaddsd	xmm25, xmm11, qword ptr [rbp - 976]
	vaddsd	xmm11, xmm5, qword ptr [rbp - 376]
	vmovsd	xmm5, qword ptr [rbp - 1104]
	vaddsd	xmm23, xmm4, qword ptr [rbp - 304]
	vmovsd	xmm4, qword ptr [rbp - 1136]
	vaddsd	xmm4, xmm4, qword ptr [rbp - 384]
	vaddsd	xmm27, xmm11, qword ptr [rbp - 1456]
	vaddsd	xmm11, xmm5, qword ptr [rbp - 1008]
	vaddsd	xmm5, xmm0, xmm30
	vmovsd	xmm0, qword ptr [rbp - 752]
	vaddsd	xmm6, xmm0, qword ptr [rbp - 944]
	vmovsd	xmm0, qword ptr [rbp - 704]
	vmovsd	qword ptr [rax + 64], xmm12
	vmovsd	xmm12, qword ptr [rbp - 136]
	vaddsd	xmm26, xmm4, qword ptr [rbp - 1464]
	vmovsd	xmm4, qword ptr [rbp - 1120]
	vaddsd	xmm4, xmm4, qword ptr [rbp - 352]
	vaddsd	xmm20, xmm11, qword ptr [rbp - 360]
	vaddsd	xmm11, xmm18, qword ptr [rbp - 1488]
	vaddsd	xmm6, xmm6, qword ptr [rbp - 1848]
	vaddsd	xmm22, xmm4, qword ptr [rbp - 368]
	vmovsd	xmm4, qword ptr [rbp - 600]
	vaddsd	xmm18, xmm11, qword ptr [rbp - 1752]
	vaddsd	xmm4, xmm4, qword ptr [rbp - 216]
	vmovsd	qword ptr [rcx + 72], xmm12
	vmovsd	xmm12, qword ptr [rbp - 120]
	vaddsd	xmm19, xmm4, qword ptr [rbp - 1760]
	vaddsd	xmm4, xmm9, qword ptr [rbp - 512]
	vaddsd	xmm9, xmm10, qword ptr [rbp - 504]
	vmovsd	qword ptr [rax + 72], xmm12
	vmovsd	xmm12, qword ptr [rbp - 72]
	vaddsd	xmm17, xmm4, xmm7
	vmovsd	xmm4, qword ptr [rbp - 520]
	vaddsd	xmm15, xmm9, xmm3
	vmovsd	xmm3, qword ptr [rbp - 152]
	vaddsd	xmm4, xmm4, qword ptr [rbp - 224]
	vaddsd	xmm3, xmm3, qword ptr [rbp - 232]
	vaddsd	xmm14, xmm4, qword ptr [rbp - 328]
	vmovsd	xmm4, qword ptr [rbp - 552]
	vaddsd	xmm13, xmm3, qword ptr [rbp - 992]
	vmovsd	xmm3, qword ptr [rbp - 736]
	vaddsd	xmm4, xmm4, qword ptr [rbp - 1784]
	vaddsd	xmm3, xmm3, qword ptr [rbp - 1792]
	vmovsd	qword ptr [rcx + 80], xmm12
	vmovsd	qword ptr [rax + 80], xmm28
	vmovsd	qword ptr [rcx + 88], xmm23
	vmovsd	qword ptr [rax + 88], xmm25
	vmovsd	qword ptr [rcx + 96], xmm26
	vmovsd	qword ptr [rax + 96], xmm27
	vmovsd	qword ptr [rcx + 104], xmm22
	vmovsd	qword ptr [rax + 104], xmm20
	vmovsd	qword ptr [rcx + 112], xmm19
	vmovsd	qword ptr [rax + 112], xmm18
	vmovsd	qword ptr [rcx + 120], xmm17
	vmovsd	qword ptr [rax + 120], xmm15
	vaddsd	xmm11, xmm4, qword ptr [rbp - 1424]
	vmovsd	xmm4, qword ptr [rbp - 544]
	vaddsd	xmm10, xmm3, qword ptr [rbp - 1448]
	vmovsd	xmm3, qword ptr [rbp - 720]
	vaddsd	xmm4, xmm4, qword ptr [rbp - 1768]
	vaddsd	xmm3, xmm3, qword ptr [rbp - 1776]
	vaddsd	xmm9, xmm4, qword ptr [rbp - 400]
	vmovsd	xmm4, qword ptr [rbp - 536]
	vaddsd	xmm21, xmm3, qword ptr [rbp - 416]
	vmovsd	xmm3, qword ptr [rbp - 592]
	vaddsd	xmm4, xmm4, qword ptr [rbp - 200]
	vaddsd	xmm3, xmm3, qword ptr [rbp - 208]
	vmovsd	qword ptr [rcx + 128], xmm13
	vmovsd	qword ptr [rax + 128], xmm14
	vaddsd	xmm7, xmm4, qword ptr [rbp - 264]
	vaddsd	xmm4, xmm2, xmm24
	vaddsd	xmm24, xmm0, qword ptr [rbp - 288]
	vmovsd	xmm0, qword ptr [rbp - 608]
	vaddsd	xmm2, xmm31, qword ptr [rbp - 960]
	vaddsd	xmm8, xmm3, qword ptr [rbp - 272]
	vshufpd	xmm31, xmm29, xmm29, 1          # xmm31 = xmm29[1,0]
	vaddsd	xmm30, xmm0, qword ptr [rbp - 280]
	vmovsd	xmm0, qword ptr [rbp - 584]
	vaddsd	xmm0, xmm0, qword ptr [rbp - 256]
	vmovsd	qword ptr [rcx + 136], xmm10
	vmovsd	qword ptr [rax + 136], xmm11
	vaddsd	xmm3, xmm2, qword ptr [rbp - 296]
	vaddsd	xmm24, xmm24, qword ptr [rbp - 928]
	vaddsd	xmm30, xmm30, qword ptr [rbp - 240]
	vaddsd	xmm2, xmm0, qword ptr [rbp - 160]
	vaddsd	xmm0, xmm29, qword ptr [rbp - 488]
	vmovapd	xmm29, xmmword ptr [rbp - 912]
	vmovsd	qword ptr [rcx + 144], xmm21
	vmovsd	qword ptr [rax + 144], xmm9
	vmovsd	qword ptr [rcx + 152], xmm8
	vmovsd	qword ptr [rax + 152], xmm7
	vmovsd	qword ptr [rcx + 160], xmm4
	vmovsd	qword ptr [rax + 160], xmm5
	vmovsd	qword ptr [rcx + 168], xmm3
	vmovsd	qword ptr [rax + 168], xmm6
	vmovsd	qword ptr [rcx + 176], xmm24
	vmovsd	qword ptr [rax + 176], xmm30
	vaddsd	xmm0, xmm0, xmm31
	vaddsd	xmm31, xmm29, qword ptr [rbp - 768]
	vshufpd	xmm29, xmm29, xmm29, 1          # xmm29 = xmm29[1,0]
	vmovsd	qword ptr [rcx + 184], xmm2
	vmovsd	qword ptr [rax + 184], xmm1
	vmovsd	qword ptr [rcx + 192], xmm0
	movabs	rcx, offset jl_nothing
	vaddsd	xmm29, xmm31, xmm29
	vmovsd	qword ptr [rax + 192], xmm29
	mov	rax, qword ptr [rcx]
	add	rsp, 1816
	pop	rbx
	pop	r14
	pop	r15
	pop	rbp
	ret
