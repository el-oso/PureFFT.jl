	.text
	push	rbp
	mov	rbp, rsp
	push	r15
	push	r14
	push	rbx
	sub	rsp, 1736
	mov	qword ptr [rbp - 1880], rsi
	mov	rax, qword ptr fs:[0]
	movabs	r8, 136143485007368
	movabs	r15, 136143485007376
	movabs	r10, 136143485007384
	movabs	r14, 136143485007392
	movabs	r9, 136143485007400
	movabs	r11, 136143485007408
	movabs	rbx, 136143485007416
	mov	rdx, qword ptr [rax - 8]
	mov	rcx, qword ptr [rsi]
	mov	rax, qword ptr [rsi + 8]
	mov	rdi, qword ptr [rsi + 16]
	mov	rsi, qword ptr [rsi + 24]
	vmovsd	xmm22, qword ptr [r8]
	vmovsd	xmm7, qword ptr [r10]
	vmovddup	xmm17, qword ptr [r14]          # xmm17 = mem[0,0]
	vmovsd	xmm26, qword ptr [r9]
	vmovsd	xmm30, qword ptr [r15]
	vmovsd	xmm27, qword ptr [r11]
	vmovsd	xmm5, qword ptr [rbx]
	vmovsd	xmm19, qword ptr [r14]
	movabs	r14, 136143485007560
	mov	rdx, qword ptr [rdx + 16]
	vmovapd	xmm4, xmm7
	vmovapd	xmm8, xmm7
	mov	rdx, qword ptr [rdx + 16]
	mov	rdx, qword ptr [rdx]
	mov	rsi, qword ptr [rsi]
	mov	rdx, qword ptr [rdi]
	movabs	rdi, offset .rodata.cst8
	vmovapd	xmmword ptr [rbp - 704], xmm17
	vmovsd	qword ptr [rbp - 72], xmm30
	mov	rcx, qword ptr [rcx]
	vmovsd	xmm28, qword ptr [rdi]
	vmovupd	xmm9, xmmword ptr [rsi + 48]
	vmovupd	xmm21, xmmword ptr [rdx + 48]
	vmovupd	xmm31, xmmword ptr [rdx + 8]
	vmovupd	xmm12, xmmword ptr [rsi + 88]
	vmovupd	xmm6, xmmword ptr [rsi + 128]
	vmovupd	xmm18, xmmword ptr [rdx + 88]
	vmovupd	xmm3, xmmword ptr [rsi + 8]
	vmovupd	xmm20, xmmword ptr [rdx + 128]
	vmovupd	xmm11, xmmword ptr [rsi + 168]
	vmovupd	xmm13, xmmword ptr [rdx + 168]
	vmulsd	xmm0, xmm9, xmm28
	vmulsd	xmm1, xmm9, xmm22
	vmovapd	xmm24, xmm3
	vmovapd	xmmword ptr [rbp - 768], xmm9
	vmovapd	xmm25, xmm6
	vmulsd	xmm16, xmm13, xmm7
	vmovapd	xmm14, xmm13
	vmovapd	xmmword ptr [rbp - 752], xmm11
	vmovapd	xmmword ptr [rbp - 1104], xmm25
	vmovapd	xmmword ptr [rbp - 800], xmm12
	vmovapd	xmmword ptr [rbp - 736], xmm14
	vmovapd	xmmword ptr [rbp - 1088], xmm24
	vfmadd231sd	xmm0, xmm21, xmm22      # xmm0 = (xmm21 * xmm22) + xmm0
	vfnmadd231sd	xmm1, xmm21, xmm28      # xmm1 = -(xmm21 * xmm28) + xmm1
	vaddsd	xmm2, xmm31, xmm0
	vmovddup	xmm0, qword ptr [r15]           # xmm0 = mem[0,0]
	vaddsd	xmm1, xmm3, xmm1
	vmovapd	xmm3, xmm7
	movabs	r15, 136143485007568
	vmulpd	xmm29, xmm12, xmm0
	vmovapd	xmm10, xmm0
	vmulpd	xmm23, xmm6, xmm10
	vmovapd	xmmword ptr [rbp - 688], xmm0
	vmulpd	xmm0, xmm12, xmm17
	vfmadd213sd	xmm4, xmm18, xmm29      # xmm4 = (xmm18 * xmm4) + xmm29
	vfmsub213sd	xmm3, xmm20, xmm23      # xmm3 = (xmm20 * xmm3) - xmm23
	vmovapd	xmm15, xmm0
	vmovapd	xmmword ptr [rbp - 656], xmm0
	vaddsd	xmm2, xmm2, xmm4
	vmovapd	xmm4, xmm26
	vfmsub213sd	xmm4, xmm18, xmm0       # xmm4 = (xmm18 * xmm4) - xmm0
	vmulpd	xmm0, xmm6, xmm17
	vmovapd	xmm6, xmm19
	vmovapd	xmm17, xmm26
	vaddsd	xmm2, xmm2, xmm3
	vmovapd	xmm3, xmm30
	vfmsub213sd	xmm3, xmm20, xmm0       # xmm3 = (xmm20 * xmm3) - xmm0
	vmovapd	xmm10, xmm0
	vmovapd	xmmword ptr [rbp - 672], xmm0
	vaddsd	xmm1, xmm1, xmm4
	vaddsd	xmm1, xmm1, xmm3
	vmulsd	xmm3, xmm11, xmm27
	vfmsub231sd	xmm3, xmm13, xmm5       # xmm3 = (xmm13 * xmm5) - xmm3
	vaddsd	xmm0, xmm2, xmm3
	vmulsd	xmm2, xmm11, xmm5
	vmovapd	xmm3, xmm26
	vfmadd231sd	xmm2, xmm13, xmm27      # xmm2 = (xmm13 * xmm27) + xmm2
	vmovsd	qword ptr [rbp - 720], xmm0
	vmovapd	xmm13, xmm11
	vaddsd	xmm0, xmm1, xmm2
	vmulsd	xmm1, xmm9, xmm30
	vmulsd	xmm2, xmm9, xmm19
	vmulsd	xmm9, xmm12, xmm27
	vmovapd	xmm19, xmm30
	vfmadd213sd	xmm8, xmm21, xmm1       # xmm8 = (xmm21 * xmm8) + xmm1
	vfmsub231sd	xmm9, xmm18, xmm5       # xmm9 = (xmm18 * xmm5) - xmm9
	vfmsub213sd	xmm3, xmm21, xmm2       # xmm3 = (xmm21 * xmm3) - xmm2
	vfmsub231sd	xmm1, xmm21, xmm7       # xmm1 = (xmm21 * xmm7) - xmm1
	vfmsub231sd	xmm2, xmm21, xmm30      # xmm2 = (xmm21 * xmm30) - xmm2
	vmovsd	qword ptr [rbp - 32], xmm0
	vmovapd	xmm0, xmm5
	vaddsd	xmm8, xmm31, xmm8
	vaddsd	xmm3, xmm24, xmm3
	vaddsd	xmm1, xmm31, xmm1
	vaddsd	xmm2, xmm24, xmm2
	vaddsd	xmm8, xmm8, xmm9
	vmulsd	xmm9, xmm12, xmm5
	vmovapd	xmm5, xmm27
	vfmadd231sd	xmm9, xmm18, xmm27      # xmm9 = (xmm18 * xmm27) + xmm9
	vmovapd	xmm27, xmm6
	vmovsd	qword ptr [rbp - 56], xmm27
	vaddsd	xmm3, xmm9, xmm3
	vmulsd	xmm9, xmm25, xmm28
	vfmadd231sd	xmm9, xmm20, xmm22      # xmm9 = (xmm20 * xmm22) + xmm9
	vaddsd	xmm8, xmm8, xmm9
	vmulsd	xmm9, xmm25, xmm22
	vfnmadd231sd	xmm9, xmm20, xmm28      # xmm9 = -(xmm20 * xmm28) + xmm9
	vaddsd	xmm3, xmm9, xmm3
	vmulsd	xmm9, xmm11, xmm30
	vsubsd	xmm11, xmm16, xmm9
	vaddsd	xmm4, xmm8, xmm11
	vmulsd	xmm8, xmm13, xmm6
	vmovapd	xmm11, xmm30
	vmovapd	xmm6, xmm0
	vfmsub213sd	xmm11, xmm14, xmm8      # xmm11 = (xmm14 * xmm11) - xmm8
	vfmsub231sd	xmm8, xmm14, xmm26      # xmm8 = (xmm14 * xmm26) - xmm8
	vmovapd	xmm14, xmm7
	vmovsd	qword ptr [rbp - 544], xmm4
	vaddsd	xmm3, xmm11, xmm3
	vmovsd	xmm11, qword ptr [rdx + 16]
	vmovsd	qword ptr [rbp - 536], xmm3
	vmulsd	xmm3, xmm12, xmm28
	vfmadd231sd	xmm3, xmm18, xmm22      # xmm3 = (xmm18 * xmm22) + xmm3
	vaddsd	xmm1, xmm1, xmm3
	vmulsd	xmm3, xmm12, xmm22
	vfnmadd231sd	xmm3, xmm18, xmm28      # xmm3 = -(xmm18 * xmm28) + xmm3
	vaddsd	xmm2, xmm2, xmm3
	vmulsd	xmm3, xmm25, xmm5
	vfmsub231sd	xmm3, xmm20, xmm0       # xmm3 = (xmm20 * xmm0) - xmm3
	vaddsd	xmm1, xmm1, xmm3
	vmulsd	xmm3, xmm25, xmm0
	vaddsd	xmm0, xmm9, xmm16
	vshufpd	xmm9, xmm23, xmm23, 1           # xmm9 = xmm23[1,0]
	vmovapd	xmm16, xmm22
	vfnmadd231pd	xmm23, xmm20, xmmword ptr [rbp - 704] # xmm23 = -(xmm20 * mem) + xmm23
	vmovsd	qword ptr [rbp - 96], xmm16
	vfmadd231sd	xmm3, xmm20, xmm5       # xmm3 = (xmm20 * xmm5) + xmm3
	vaddsd	xmm0, xmm1, xmm0
	vshufpd	xmm1, xmm29, xmm29, 1           # xmm1 = xmm29[1,0]
	vmovsd	qword ptr [rbp - 528], xmm0
	vaddsd	xmm2, xmm2, xmm3
	vaddsd	xmm0, xmm8, xmm2
	vmovsd	xmm8, qword ptr [rsi + 56]
	vmovsd	xmm2, qword ptr [rdx + 56]
	vmovsd	qword ptr [rbp - 520], xmm0
	vmovsd	xmm0, qword ptr [rdx + 96]
	vmulsd	xmm3, xmm8, xmm28
	vmovapd	xmm25, xmm8
	vmovsd	qword ptr [rbp - 632], xmm8
	vmulsd	xmm13, xmm25, xmm27
	vfmadd231sd	xmm3, xmm2, xmm22       # xmm3 = (xmm2 * xmm22) + xmm3
	vfmadd231sd	xmm1, xmm0, xmm7        # xmm1 = (xmm0 * xmm7) + xmm1
	vaddsd	xmm3, xmm11, xmm3
	vaddsd	xmm1, xmm3, xmm1
	vmulsd	xmm3, xmm8, xmm22
	vmovsd	xmm8, qword ptr [rsi + 16]
	vfnmadd231sd	xmm3, xmm2, xmm28       # xmm3 = -(xmm2 * xmm28) + xmm3
	vmovapd	xmm26, xmm8
	vaddsd	xmm3, xmm8, xmm3
	vshufpd	xmm8, xmm15, xmm15, 1           # xmm8 = xmm15[1,0]
	vmovsd	xmm15, qword ptr [rsi + 176]
	vmovapd	xmm4, xmm26
	vmovsd	qword ptr [rbp - 600], xmm26
	vfmsub231sd	xmm8, xmm0, xmm17       # xmm8 = (xmm0 * xmm17) - xmm8
	vaddsd	xmm3, xmm8, xmm3
	vmovsd	xmm8, qword ptr [rdx + 136]
	vmulsd	xmm12, xmm15, xmm5
	vmovsd	qword ptr [rbp - 624], xmm15
	vfmsub231sd	xmm9, xmm8, xmm7        # xmm9 = (xmm8 * xmm7) - xmm9
	vmovapd	xmm7, xmm5
	vaddsd	xmm1, xmm9, xmm1
	vshufpd	xmm9, xmm10, xmm10, 1           # xmm9 = xmm10[1,0]
	vmovapd	xmm10, xmm6
	vfmsub231sd	xmm9, xmm8, xmm30       # xmm9 = (xmm8 * xmm30) - xmm9
	vaddsd	xmm3, xmm9, xmm3
	vmovsd	xmm9, qword ptr [rdx + 176]
	vfmsub231sd	xmm12, xmm9, xmm6       # xmm12 = (xmm9 * xmm6) - xmm12
	vaddsd	xmm1, xmm12, xmm1
	vmulsd	xmm12, xmm25, xmm30
	vmovsd	qword ptr [rbp - 48], xmm1
	vmulsd	xmm1, xmm15, xmm6
	vfmadd231sd	xmm1, xmm9, xmm5        # xmm1 = (xmm9 * xmm5) + xmm1
	vmovsd	xmm5, qword ptr [rsi + 96]
	vaddsd	xmm1, xmm3, xmm1
	vmovsd	qword ptr [rbp - 40], xmm1
	vmovapd	xmm1, xmm14
	vfmadd213sd	xmm1, xmm2, xmm12       # xmm1 = (xmm2 * xmm1) + xmm12
	vfmsub231sd	xmm12, xmm2, xmm14      # xmm12 = (xmm2 * xmm14) - xmm12
	vmulsd	xmm3, xmm5, xmm7
	vmulsd	xmm25, xmm5, xmm6
	vmovsd	qword ptr [rbp - 608], xmm5
	vfmsub231sd	xmm3, xmm0, xmm6        # xmm3 = (xmm0 * xmm6) - xmm3
	vaddsd	xmm1, xmm11, xmm1
	vmovsd	xmm6, qword ptr [rsi + 136]
	vfmadd231sd	xmm25, xmm0, xmm7       # xmm25 = (xmm0 * xmm7) + xmm25
	vaddsd	xmm1, xmm1, xmm3
	vmovapd	xmm3, xmm17
	vfmsub213sd	xmm3, xmm2, xmm13       # xmm3 = (xmm2 * xmm3) - xmm13
	vaddsd	xmm3, xmm26, xmm3
	vmulsd	xmm26, xmm15, xmm30
	vmovsd	qword ptr [rbp - 592], xmm6
	vaddsd	xmm3, xmm3, xmm25
	vmulsd	xmm25, xmm6, xmm28
	vfmadd231sd	xmm25, xmm8, xmm22      # xmm25 = (xmm8 * xmm22) + xmm25
	vaddsd	xmm1, xmm1, xmm25
	vmulsd	xmm25, xmm6, xmm22
	vmovapd	xmm22, xmm28
	vmovsd	qword ptr [rbp - 88], xmm22
	vfnmadd231sd	xmm25, xmm8, xmm28      # xmm25 = -(xmm8 * xmm28) + xmm25
	vaddsd	xmm3, xmm3, xmm25
	vmulsd	xmm25, xmm9, xmm14
	vsubsd	xmm28, xmm25, xmm26
	vaddsd	xmm1, xmm1, xmm28
	vmulsd	xmm28, xmm15, xmm27
	vmovapd	xmm27, xmmword ptr [rbp - 752]
	vmovapd	xmm15, xmm10
	vfmsub213sd	xmm30, xmm9, xmm28      # xmm30 = (xmm9 * xmm30) - xmm28
	vaddsd	xmm3, xmm3, xmm30
	vaddsd	xmm30, xmm11, xmm2
	vaddsd	xmm11, xmm11, xmm12
	vmulsd	xmm12, xmm5, xmm22
	vfmsub213sd	xmm2, xmm19, xmm13      # xmm2 = (xmm19 * xmm2) - xmm13
	vmulsd	xmm13, xmm5, xmm16
	vmovddup	xmm5, qword ptr [rbx]           # xmm5 = mem[0,0]
	vmovapd	xmm19, xmmword ptr [rbp - 768]
	movabs	rbx, 136143485007552
	vfmadd231sd	xmm12, xmm0, xmm16      # xmm12 = (xmm0 * xmm16) + xmm12
	vfnmadd231sd	xmm13, xmm22, xmm0      # xmm13 = -(xmm22 * xmm0) + xmm13
	vmovapd	xmm22, xmm7
	vmovsd	xmm16, qword ptr [rbp - 32]
	vaddsd	xmm11, xmm11, xmm12
	vaddsd	xmm12, xmm30, xmm0
	vaddsd	xmm0, xmm4, xmm2
	vmulsd	xmm2, xmm6, xmm7
	vmovddup	xmm4, qword ptr [r11]           # xmm4 = mem[0,0]
	movabs	r11, 136143485007624
	vmovapd	xmmword ptr [rbp - 1056], xmm5
	vfmsub231sd	xmm2, xmm8, xmm10       # xmm2 = (xmm8 * xmm10) - xmm2
	vaddsd	xmm0, xmm13, xmm0
	vaddsd	xmm2, xmm11, xmm2
	vaddsd	xmm11, xmm12, xmm8
	vmulsd	xmm12, xmm10, xmm6
	vmovddup	xmm6, qword ptr [rdi]           # xmm6 = mem[0,0]
	movabs	rdi, 136143485007424
	vmovapd	xmmword ptr [rbp - 1024], xmm4
	vfmadd231sd	xmm12, xmm7, xmm8       # xmm12 = (xmm7 * xmm8) + xmm12
	vaddsd	xmm8, xmm12, xmm0
	vaddsd	xmm0, xmm26, xmm25
	vmovsd	xmm12, qword ptr [rbp - 720]
	vmovapd	xmm25, xmm17
	vmovapd	xmmword ptr [rbp - 976], xmm6
	vaddsd	xmm0, xmm2, xmm0
	vaddsd	xmm2, xmm11, xmm9
	vfmsub213sd	xmm9, xmm17, xmm28      # xmm9 = (xmm17 * xmm9) - xmm28
	vaddsd	xmm11, xmm31, xmm21
	vmovupd	xmm28, xmmword ptr [rsi + 104]
	vmovsd	qword ptr [rbp - 616], xmm2
	vaddsd	xmm11, xmm11, xmm18
	vaddsd	xmm2, xmm8, xmm9
	vmulpd	xmm9, xmm19, xmm5
	vmulpd	xmm8, xmm19, xmm4
	vfmadd231pd	xmm9, xmm4, xmm21       # xmm9 = (xmm4 * xmm21) + xmm9
	vmovddup	xmm4, qword ptr [r10]           # xmm4 = mem[0,0]
	vfmsub231pd	xmm8, xmm21, xmm5       # xmm8 = (xmm21 * xmm5) - xmm8
	movabs	r10, 136143485007616
	vmovsd	xmm21, qword ptr [rbp - 96]
	vmovapd	xmmword ptr [rbp - 1872], xmm28
	vaddpd	xmm7, xmm24, xmm9
	vmovddup	xmm9, qword ptr [r8]            # xmm9 = mem[0,0]
	vaddpd	xmm8, xmm31, xmm8
	vmovapd	xmm31, xmmword ptr [rbp - 736]
	movabs	r8, 136143485007432
	vmovapd	xmm24, xmm22
	vfmsub231pd	xmm29, xmm18, xmm4      # xmm29 = (xmm18 * xmm4) - xmm29
	vmovapd	xmmword ptr [rbp - 1040], xmm4
	vmovapd	xmm4, xmmword ptr [rbp - 656]
	vfmsub132pd	xmm18, xmm4, xmmword ptr [rbp - 688] # xmm18 = (xmm18 * mem) - xmm4
	vaddsd	xmm4, xmm11, xmm20
	vmovsd	qword ptr [rbp - 584], xmm4
	vmovddup	xmm4, qword ptr [r9]            # xmm4 = mem[0,0]
	vaddpd	xmm8, xmm8, xmm29
	vmovapd	xmmword ptr [rbp - 992], xmm9
	movabs	r9, 136143485007600
	vfmsub213pd	xmm20, xmm4, xmmword ptr [rbp - 672] # xmm20 = (xmm4 * xmm20) - mem
	vmovsd	xmm30, qword ptr [r9]
	movabs	r9, 136143485007504
	vaddpd	xmm8, xmm8, xmm23
	vmovsd	xmm23, qword ptr [rbp - 72]
	vaddpd	xmm7, xmm7, xmm18
	vmovapd	xmmword ptr [rbp - 1008], xmm4
	vmovapd	xmm26, xmm23
	vaddpd	xmm5, xmm7, xmm20
	vmulpd	xmm7, xmm27, xmm6
	vmovsd	xmm20, qword ptr [rbp - 88]
	vfmadd231pd	xmm7, xmm31, xmm9       # xmm7 = (xmm31 * xmm9) + xmm7
	vaddpd	xmm4, xmm8, xmm7
	vmulpd	xmm7, xmm27, xmm9
	vmovsd	xmm8, qword ptr [r8]
	movabs	r8, 136143485007496
	vmovsd	xmm9, qword ptr [r11]
	vmovupd	xmm27, xmmword ptr [rdx + 64]
	movabs	r11, 136143485007544
	vmovsd	xmm17, qword ptr [r8]
	movabs	r8, 136143485007592
	vmovapd	xmm29, xmm20
	vfnmadd231pd	xmm7, xmm6, xmm31       # xmm7 = -(xmm6 * xmm31) + xmm7
	vmovapd	xmmword ptr [rbp - 784], xmm4
	vmovsd	xmm6, qword ptr [r10]
	movabs	r10, 136143485007632
	vmovupd	xmm31, xmmword ptr [rdx + 104]
	vaddpd	xmm4, xmm5, xmm7
	vmovsd	xmm5, qword ptr [rdi]
	movabs	rdi, 136143485007488
	vmovsd	xmm13, qword ptr [rdi]
	movabs	rdi, 136143485007608
	vmovsd	qword ptr [rbp - 288], xmm9
	vmovsd	qword ptr [rbp - 208], xmm8
	vmovapd	xmmword ptr [rbp - 848], xmm27
	vmovsd	xmm7, qword ptr [rdi]
	movabs	rdi, 136143485007664
	vmovsd	qword ptr [rbp - 960], xmm17
	vmovapd	xmmword ptr [rbp - 1072], xmm4
	vmovsd	xmm19, qword ptr [rdi]
	movabs	rdi, 136143485007456
	vmovapd	xmm10, xmm6
	vmovsd	qword ptr [rbp - 280], xmm6
	vmulsd	xmm4, xmm16, xmm5
	vmovsd	qword ptr [rbp - 240], xmm5
	vmovsd	qword ptr [rbp - 944], xmm13
	vmovsd	qword ptr [rbp - 256], xmm7
	vfmadd231sd	xmm4, xmm12, xmm8       # xmm4 = (xmm12 * xmm8) + xmm4
	vmovsd	qword ptr [rbp - 272], xmm19
	vmovsd	qword ptr [rbp - 576], xmm4
	vmulsd	xmm4, xmm16, xmm8
	vfnmadd231sd	xmm4, xmm12, xmm5       # xmm4 = -(xmm12 * xmm5) + xmm4
	vmovsd	qword ptr [rbp - 568], xmm4
	vmulsd	xmm4, xmm16, xmm13
	vfmadd231sd	xmm4, xmm12, xmm17      # xmm4 = (xmm12 * xmm17) + xmm4
	vmovsd	qword ptr [rbp - 560], xmm4
	vmulsd	xmm4, xmm16, xmm17
	vfnmadd231sd	xmm4, xmm12, xmm13      # xmm4 = -(xmm12 * xmm13) + xmm4
	vmovsd	qword ptr [rbp - 552], xmm4
	vmulsd	xmm4, xmm16, xmm6
	vmovsd	xmm6, qword ptr [r10]
	movabs	r10, 136143485007512
	vfnmadd231sd	xmm4, xmm12, xmm9       # xmm4 = -(xmm12 * xmm9) + xmm4
	vmovsd	qword ptr [rbp - 512], xmm4
	vmulsd	xmm4, xmm16, xmm9
	vmovapd	xmm11, xmm6
	vmovsd	qword ptr [rbp - 312], xmm6
	vfmsub231sd	xmm4, xmm12, xmm6       # xmm4 = (xmm12 * xmm6) - xmm4
	vmovsd	xmm6, qword ptr [r8]
	movabs	r8, 136143485007672
	vmovsd	xmm18, qword ptr [r8]
	movabs	r8, 136143485007464
	vmovsd	qword ptr [rbp - 504], xmm4
	vmulsd	xmm4, xmm16, xmm6
	vmovsd	qword ptr [rbp - 264], xmm6
	vfmadd231sd	xmm4, xmm12, xmm30      # xmm4 = (xmm12 * xmm30) + xmm4
	vmovsd	qword ptr [rbp - 496], xmm4
	vmulsd	xmm4, xmm16, xmm30
	vfmadd231sd	xmm4, xmm12, xmm7       # xmm4 = (xmm12 * xmm7) + xmm4
	vmovsd	qword ptr [rbp - 488], xmm4
	vmulsd	xmm4, xmm16, xmm19
	vfmsub231sd	xmm4, xmm12, xmm18      # xmm4 = (xmm12 * xmm18) - xmm4
	vmovsd	qword ptr [rbp - 480], xmm4
	vmulsd	xmm4, xmm16, xmm18
	vmovsd	xmm16, qword ptr [r8]
	movabs	r8, 136143485007584
	vfmadd231sd	xmm4, xmm19, xmm12      # xmm4 = (xmm19 * xmm12) + xmm4
	vmovsd	xmm12, qword ptr [rdi]
	movabs	rdi, 136143485007576
	vmovsd	qword ptr [rbp - 472], xmm4
	vmovsd	qword ptr [rbp - 232], xmm16
	vmulsd	xmm4, xmm12, xmm3
	vmovsd	qword ptr [rbp - 224], xmm12
	vfmadd231sd	xmm4, xmm1, xmm16       # xmm4 = (xmm1 * xmm16) + xmm4
	vmovsd	qword ptr [rbp - 360], xmm4
	vmulsd	xmm4, xmm3, xmm16
	vmovsd	xmm16, qword ptr [r8]
	movabs	r8, 136143485007712
	vfnmadd231sd	xmm4, xmm1, xmm12       # xmm4 = -(xmm1 * xmm12) + xmm4
	vmovsd	xmm12, qword ptr [rdi]
	movabs	rdi, 136143485007640
	vmovsd	qword ptr [rbp - 352], xmm4
	vmovsd	qword ptr [rbp - 304], xmm16
	vmulsd	xmm4, xmm12, xmm3
	vmovsd	qword ptr [rbp - 296], xmm12
	vfmsub231sd	xmm4, xmm1, xmm16       # xmm4 = (xmm1 * xmm16) - xmm4
	vmovsd	qword ptr [rbp - 392], xmm4
	vmulsd	xmm4, xmm3, xmm16
	vmovupd	xmm16, xmmword ptr [rdx + 24]
	vfmadd231sd	xmm4, xmm1, xmm12       # xmm4 = (xmm1 * xmm12) + xmm4
	vmovsd	xmm12, qword ptr [rdi]
	movabs	rdi, offset .rodata.cst16
	vmovsd	qword ptr [rbp - 384], xmm4
	vmovapd	xmmword ptr [rbp - 1824], xmm16
	vmulsd	xmm4, xmm12, xmm3
	vmovsd	qword ptr [rbp - 248], xmm12
	vfmsub231sd	xmm4, xmm1, xmm8        # xmm4 = (xmm1 * xmm8) - xmm4
	vmovsd	qword ptr [rbp - 432], xmm4
	vmulsd	xmm4, xmm8, xmm3
	vfmadd231sd	xmm4, xmm1, xmm12       # xmm4 = (xmm1 * xmm12) + xmm4
	vmovupd	xmm12, xmmword ptr [rsi + 24]
	vmovsd	qword ptr [rbp - 424], xmm4
	vmulsd	xmm4, xmm3, xmm7
	vfmadd231sd	xmm4, xmm1, xmm30       # xmm4 = (xmm1 * xmm30) + xmm4
	vmovapd	xmmword ptr [rbp - 720], xmm12
	vmovsd	qword ptr [rbp - 448], xmm4
	vmulsd	xmm4, xmm3, xmm30
	vfmadd231sd	xmm4, xmm1, xmm6        # xmm4 = (xmm1 * xmm6) + xmm4
	vmovsd	qword ptr [rbp - 440], xmm4
	vmulsd	xmm4, xmm13, xmm3
	vmulsd	xmm3, xmm3, xmm17
	vfmsub231sd	xmm4, xmm1, xmm17       # xmm4 = (xmm1 * xmm17) - xmm4
	vfmadd231sd	xmm3, xmm13, xmm1       # xmm3 = (xmm13 * xmm1) + xmm3
	vmulsd	xmm1, xmm13, xmm2
	vfmadd231sd	xmm1, xmm0, xmm17       # xmm1 = (xmm0 * xmm17) + xmm1
	vmovsd	qword ptr [rbp - 464], xmm4
	vmovsd	qword ptr [rbp - 456], xmm3
	vmulsd	xmm3, xmm28, xmm15
	vmovsd	qword ptr [rbp - 328], xmm1
	vmulsd	xmm1, xmm2, xmm17
	vmovupd	xmm17, xmmword ptr [rsi + 64]
	vfmadd231sd	xmm3, xmm31, xmm22      # xmm3 = (xmm31 * xmm22) + xmm3
	vfnmadd231sd	xmm1, xmm0, xmm13       # xmm1 = -(xmm0 * xmm13) + xmm1
	vmovsd	qword ptr [rbp - 320], xmm1
	vmulsd	xmm1, xmm2, xmm6
	vfmadd231sd	xmm1, xmm0, xmm30       # xmm1 = (xmm0 * xmm30) + xmm1
	vmovsd	qword ptr [rbp - 344], xmm1
	vmulsd	xmm1, xmm2, xmm30
	vfmadd231sd	xmm1, xmm0, xmm7        # xmm1 = (xmm0 * xmm7) + xmm1
	vmovapd	xmm7, xmm15
	vmovsd	qword ptr [rbp - 1632], xmm7
	vmovsd	qword ptr [rbp - 336], xmm1
	vmulsd	xmm1, xmm2, xmm5
	vfmadd231sd	xmm1, xmm0, xmm8        # xmm1 = (xmm0 * xmm8) + xmm1
	vmovsd	qword ptr [rbp - 376], xmm1
	vmulsd	xmm1, xmm8, xmm2
	vmovapd	xmm8, xmm25
	vmovsd	qword ptr [rbp - 80], xmm8
	vfnmadd231sd	xmm1, xmm0, xmm5        # xmm1 = -(xmm0 * xmm5) + xmm1
	vmovupd	xmm5, xmmword ptr [rsi + 144]
	vmovsd	qword ptr [rbp - 368], xmm1
	vmulsd	xmm1, xmm10, xmm2
	vmovupd	xmm10, xmmword ptr [rdx + 184]
	vfnmadd231sd	xmm1, xmm0, xmm9        # xmm1 = -(xmm0 * xmm9) + xmm1
	vmovsd	qword ptr [rbp - 408], xmm1
	vmulsd	xmm1, xmm9, xmm2
	vmovupd	xmm9, xmmword ptr [rdx + 144]
	vmovapd	xmmword ptr [rbp - 832], xmm10
	vfmsub231sd	xmm1, xmm0, xmm11       # xmm1 = (xmm0 * xmm11) - xmm1
	vmovupd	xmm11, xmmword ptr [rsi + 184]
	vmovsd	qword ptr [rbp - 400], xmm1
	vmulsd	xmm1, xmm2, xmm19
	vfmsub231sd	xmm1, xmm0, xmm18       # xmm1 = (xmm0 * xmm18) - xmm1
	vmovapd	xmmword ptr [rbp - 816], xmm11
	vmovsd	qword ptr [rbp - 416], xmm1
	vmulsd	xmm1, xmm2, xmm18
	vmulsd	xmm2, xmm28, xmm22
	vmovapd	xmm22, xmm14
	vfmadd231sd	xmm1, xmm19, xmm0       # xmm1 = (xmm19 * xmm0) + xmm1
	vmovsd	xmm19, qword ptr [rbp - 56]
	vmovapd	xmm0, xmm14
	vfmsub231sd	xmm2, xmm31, xmm15      # xmm2 = (xmm31 * xmm15) - xmm2
	vmovsd	qword ptr [rbp - 1776], xmm1
	vmulsd	xmm1, xmm17, xmm23
	vfmadd213sd	xmm0, xmm27, xmm1       # xmm0 = (xmm27 * xmm0) + xmm1
	vfmsub231sd	xmm1, xmm27, xmm22      # xmm1 = (xmm27 * xmm22) - xmm1
	vmulsd	xmm4, xmm17, xmm19
	vaddsd	xmm0, xmm16, xmm0
	vaddsd	xmm1, xmm16, xmm1
	vaddsd	xmm0, xmm0, xmm2
	vmovapd	xmm2, xmm25
	vfmsub213sd	xmm2, xmm27, xmm4       # xmm2 = (xmm27 * xmm2) - xmm4
	vmulsd	xmm25, xmm11, xmm23
	vfmsub231sd	xmm4, xmm27, xmm23      # xmm4 = (xmm27 * xmm23) - xmm4
	vaddsd	xmm2, xmm12, xmm2
	vaddsd	xmm4, xmm12, xmm4
	vaddsd	xmm2, xmm2, xmm3
	vmulsd	xmm3, xmm5, xmm20
	vfmadd231sd	xmm3, xmm9, xmm21       # xmm3 = (xmm9 * xmm21) + xmm3
	vaddsd	xmm0, xmm0, xmm3
	vmulsd	xmm3, xmm5, xmm21
	vfnmadd231sd	xmm3, xmm9, xmm20       # xmm3 = -(xmm9 * xmm20) + xmm3
	vmulsd	xmm20, xmm10, xmm14
	vaddsd	xmm2, xmm2, xmm3
	vsubsd	xmm3, xmm20, xmm25
	vaddsd	xmm14, xmm0, xmm3
	vmulsd	xmm0, xmm11, xmm19
	vmovapd	xmm19, xmm5
	vmovapd	xmm11, xmmword ptr [rbp - 688]
	vmovapd	xmm3, xmm29
	vmovapd	xmmword ptr [rbp - 1856], xmm19
	vfmsub213sd	xmm26, xmm10, xmm0      # xmm26 = (xmm10 * xmm26) - xmm0
	vfmsub231sd	xmm0, xmm10, xmm8       # xmm0 = (xmm10 * xmm8) - xmm0
	vaddsd	xmm13, xmm2, xmm26
	vmulsd	xmm26, xmm28, xmm29
	vmovapd	xmm2, xmm22
	vmovapd	xmm22, xmmword ptr [rbp - 704]
	vmovsd	qword ptr [rbp - 64], xmm2
	vmulpd	xmm6, xmm28, xmm11
	vfmadd231sd	xmm26, xmm31, xmm21     # xmm26 = (xmm31 * xmm21) + xmm26
	vaddsd	xmm1, xmm1, xmm26
	vmulsd	xmm26, xmm28, xmm21
	vfnmadd231sd	xmm26, xmm31, xmm29     # xmm26 = -(xmm31 * xmm29) + xmm26
	vmovapd	xmm29, xmm24
	vmovsd	qword ptr [rbp - 1624], xmm29
	vaddsd	xmm4, xmm4, xmm26
	vmulsd	xmm26, xmm5, xmm24
	vfmsub231sd	xmm26, xmm9, xmm15      # xmm26 = (xmm9 * xmm15) - xmm26
	vaddsd	xmm1, xmm1, xmm26
	vmulsd	xmm26, xmm5, xmm15
	vaddsd	xmm5, xmm25, xmm20
	vmovsd	xmm20, qword ptr [rsi + 72]
	vmovsd	xmm25, qword ptr [rdx + 32]
	vfmadd231sd	xmm26, xmm9, xmm24      # xmm26 = (xmm9 * xmm24) + xmm26
	vaddsd	xmm24, xmm1, xmm5
	vmovapd	xmm5, xmm8
	vmovsd	xmm8, qword ptr [rdx + 72]
	vaddsd	xmm4, xmm4, xmm26
	vmovapd	xmm26, xmm16
	vmulsd	xmm1, xmm20, xmm3
	vmovsd	qword ptr [rbp - 216], xmm25
	vmovsd	qword ptr [rbp - 672], xmm20
	vaddsd	xmm15, xmm4, xmm0
	vmovsd	xmm4, qword ptr [rdx + 112]
	vshufpd	xmm0, xmm6, xmm6, 1             # xmm0 = xmm6[1,0]
	vfmadd231sd	xmm1, xmm8, xmm21       # xmm1 = (xmm8 * xmm21) + xmm1
	vmovsd	qword ptr [rbp - 200], xmm8
	vaddsd	xmm1, xmm25, xmm1
	vfmadd231sd	xmm0, xmm4, xmm2        # xmm0 = (xmm4 * xmm2) + xmm0
	vmovsd	qword ptr [rbp - 192], xmm4
	vaddsd	xmm0, xmm1, xmm0
	vmulsd	xmm1, xmm20, xmm21
	vmulpd	xmm20, xmm19, xmm11
	vmovapd	xmm21, xmm22
	vfnmadd231sd	xmm1, xmm8, xmm3        # xmm1 = -(xmm8 * xmm3) + xmm1
	vmovsd	xmm3, qword ptr [rsi + 32]
	vmulpd	xmm8, xmm28, xmm22
	vshufpd	xmm25, xmm8, xmm8, 1            # xmm25 = xmm8[1,0]
	vfmsub213pd	xmm11, xmm31, xmm8      # xmm11 = (xmm31 * xmm11) - xmm8
	vmovapd	xmmword ptr [rbp - 912], xmm8
	vmovapd	xmm8, xmm9
	vmovapd	xmmword ptr [rbp - 1792], xmm8
	vfmsub231sd	xmm25, xmm4, xmm5       # xmm25 = (xmm4 * xmm5) - xmm25
	vmulpd	xmm5, xmm19, xmm22
	vmovsd	xmm4, qword ptr [rdx + 192]
	vmovapd	xmm22, xmmword ptr [rbp - 1024]
	vmovapd	xmm19, xmmword ptr [rbp - 1056]
	vmovapd	xmmword ptr [rbp - 928], xmm5
	vaddsd	xmm1, xmm3, xmm1
	vmovsd	qword ptr [rbp - 32], xmm3
	vmovsd	xmm3, qword ptr [rdx + 152]
	vaddsd	xmm1, xmm1, xmm25
	vshufpd	xmm25, xmm20, xmm20, 1          # xmm25 = xmm20[1,0]
	vmulpd	xmm28, xmm17, xmm19
	vmovsd	qword ptr [rbp - 176], xmm4
	vfmadd231pd	xmm28, xmm27, xmm22     # xmm28 = (xmm27 * xmm22) + xmm28
	vfmsub231sd	xmm25, xmm3, xmm2       # xmm25 = (xmm3 * xmm2) - xmm25
	vmovsd	qword ptr [rbp - 184], xmm3
	vaddpd	xmm28, xmm12, xmm28
	vaddsd	xmm0, xmm0, xmm25
	vshufpd	xmm25, xmm5, xmm5, 1            # xmm25 = xmm5[1,0]
	vfmsub231sd	xmm25, xmm3, xmm23      # xmm25 = (xmm3 * xmm23) - xmm25
	vmovsd	xmm3, qword ptr [rsi + 192]
	vmovapd	xmm23, xmm6
	vaddpd	xmm28, xmm28, xmm11
	vmovsd	xmm11, qword ptr [rbp - 224]
	vaddsd	xmm25, xmm1, xmm25
	vmulsd	xmm1, xmm3, xmm29
	vmovsd	qword ptr [rbp - 656], xmm3
	vfmsub231sd	xmm1, xmm4, xmm7        # xmm1 = (xmm4 * xmm7) - xmm1
	vaddsd	xmm1, xmm0, xmm1
	vmulsd	xmm0, xmm3, xmm7
	vmovapd	xmm7, xmmword ptr [rbp - 1040]
	vfmadd231sd	xmm0, xmm4, xmm29       # xmm0 = (xmm4 * xmm29) + xmm0
	vmovsd	xmm4, qword ptr [rbp - 272]
	vmovapd	xmm29, xmm17
	vmovapd	xmmword ptr [rbp - 1840], xmm29
	vaddsd	xmm0, xmm25, xmm0
	vmulpd	xmm25, xmm17, xmm22
	vmovapd	xmm22, xmm31
	vmovapd	xmmword ptr [rbp - 1808], xmm22
	vfmsub213pd	xmm7, xmm31, xmm6       # xmm7 = (xmm31 * xmm7) - xmm6
	vmovapd	xmm6, xmm21
	vfnmadd213pd	xmm6, xmm9, xmm20       # xmm6 = -(xmm9 * xmm6) + xmm20
	vmovapd	xmm21, xmm20
	vmovsd	xmm20, qword ptr [rbp - 960]
	vfmsub231pd	xmm25, xmm27, xmm19     # xmm25 = (xmm27 * xmm19) - xmm25
	vmovapd	xmm19, xmmword ptr [rbp - 816]
	vaddpd	xmm25, xmm16, xmm25
	vmovsd	xmm16, qword ptr [rbp - 248]
	vaddpd	xmm25, xmm25, xmm7
	vmovapd	xmm7, xmmword ptr [rbp - 992]
	vaddpd	xmm25, xmm25, xmm6
	vmovapd	xmm6, xmmword ptr [rbp - 1008]
	vfmsub213pd	xmm6, xmm9, xmm5        # xmm6 = (xmm9 * xmm6) - xmm5
	vmovsd	xmm9, qword ptr [rbp - 312]
	vaddpd	xmm28, xmm28, xmm6
	vmovapd	xmm6, xmmword ptr [rbp - 976]
	vmulpd	xmm27, xmm19, xmm6
	vfmadd231pd	xmm27, xmm10, xmm7      # xmm27 = (xmm10 * xmm7) + xmm27
	vaddpd	xmm2, xmm25, xmm27
	vmulpd	xmm25, xmm19, xmm7
	vmovsd	xmm7, qword ptr [rbp - 280]
	vmovsd	xmm27, qword ptr [rbp - 1624]
	vfnmadd231pd	xmm25, xmm10, xmm6      # xmm25 = -(xmm10 * xmm6) + xmm25
	vmulsd	xmm6, xmm13, xmm4
	vmovapd	xmmword ptr [rbp - 896], xmm2
	vmovsd	xmm10, qword ptr [rbp - 232]
	vfmsub231sd	xmm6, xmm14, xmm18      # xmm6 = (xmm14 * xmm18) - xmm6
	vaddpd	xmm3, xmm28, xmm25
	vmovsd	xmm28, qword ptr [rbp - 56]
	vmovsd	qword ptr [rbp - 704], xmm6
	vmulsd	xmm6, xmm13, xmm18
	vshufpd	xmm18, xmm2, xmm2, 1            # xmm18 = xmm2[1,0]
	vmovsd	xmm2, qword ptr [rbp - 288]
	vshufpd	xmm17, xmm3, xmm3, 1            # xmm17 = xmm3[1,0]
	vmovapd	xmmword ptr [rbp - 880], xmm3
	vmovsd	xmm3, qword ptr [rbp - 296]
	vfmadd231sd	xmm6, xmm14, xmm4       # xmm6 = (xmm14 * xmm4) + xmm6
	vmovsd	xmm4, qword ptr [rbp - 304]
	vmovsd	qword ptr [rbp - 688], xmm6
	vmulsd	xmm6, xmm17, xmm7
	vmulsd	xmm12, xmm15, xmm3
	vfnmadd231sd	xmm6, xmm18, xmm2       # xmm6 = -(xmm18 * xmm2) + xmm6
	vfmsub231sd	xmm12, xmm24, xmm4      # xmm12 = (xmm24 * xmm4) - xmm12
	vmovsd	qword ptr [rbp - 1056], xmm6
	vmulsd	xmm6, xmm17, xmm2
	vmovsd	qword ptr [rbp - 992], xmm12
	vmulsd	xmm12, xmm15, xmm4
	vfmsub231sd	xmm6, xmm18, xmm9       # xmm6 = (xmm18 * xmm9) - xmm6
	vfmadd231sd	xmm12, xmm24, xmm3      # xmm12 = (xmm24 * xmm3) + xmm12
	vmovsd	qword ptr [rbp - 1040], xmm6
	vmulsd	xmm6, xmm13, xmm7
	vmovapd	xmm7, xmmword ptr [rbp - 1072]
	vmovsd	qword ptr [rbp - 976], xmm12
	vmulsd	xmm12, xmm0, xmm3
	vfnmadd231sd	xmm6, xmm14, xmm2       # xmm6 = -(xmm14 * xmm2) + xmm6
	vfmsub231sd	xmm12, xmm1, xmm4       # xmm12 = (xmm1 * xmm4) - xmm12
	vmovsd	qword ptr [rbp - 1024], xmm6
	vmulsd	xmm6, xmm13, xmm2
	vmulsd	xmm2, xmm7, xmm3
	vmovsd	qword ptr [rbp - 312], xmm12
	vfmsub231sd	xmm6, xmm14, xmm9       # xmm6 = (xmm14 * xmm9) - xmm6
	vmulsd	xmm9, xmm0, xmm4
	vfmadd231sd	xmm9, xmm1, xmm3        # xmm9 = (xmm1 * xmm3) + xmm9
	vmovsd	qword ptr [rbp - 1008], xmm6
	vmovapd	xmm6, xmmword ptr [rbp - 784]
	vmovsd	qword ptr [rbp - 304], xmm9
	vmulsd	xmm9, xmm11, xmm0
	vfmadd231sd	xmm9, xmm1, xmm10       # xmm9 = (xmm1 * xmm10) + xmm9
	vfmsub231sd	xmm2, xmm6, xmm4        # xmm2 = (xmm6 * xmm4) - xmm2
	vmovsd	qword ptr [rbp - 1760], xmm9
	vmulsd	xmm9, xmm10, xmm0
	vmovsd	qword ptr [rbp - 1728], xmm2
	vmulsd	xmm2, xmm7, xmm4
	vmovsd	xmm4, qword ptr [rbp - 944]
	vfnmadd231sd	xmm9, xmm1, xmm11       # xmm9 = -(xmm1 * xmm11) + xmm9
	vfmadd231sd	xmm2, xmm6, xmm3        # xmm2 = (xmm6 * xmm3) + xmm2
	vmovsd	qword ptr [rbp - 1752], xmm9
	vmulsd	xmm9, xmm0, xmm16
	vmovsd	qword ptr [rbp - 1736], xmm2
	vmovsd	xmm2, qword ptr [rbp - 208]
	vfmsub231sd	xmm9, xmm1, xmm2        # xmm9 = (xmm1 * xmm2) - xmm9
	vmovsd	qword ptr [rbp - 280], xmm9
	vmulsd	xmm9, xmm0, xmm2
	vfmadd231sd	xmm9, xmm1, xmm16       # xmm9 = (xmm1 * xmm16) + xmm9
	vmovsd	qword ptr [rbp - 272], xmm9
	vmulsd	xmm9, xmm0, xmm4
	vfmsub231sd	xmm9, xmm1, xmm20       # xmm9 = (xmm1 * xmm20) - xmm9
	vmovsd	qword ptr [rbp - 296], xmm9
	vmulsd	xmm9, xmm0, xmm20
	vfmadd231sd	xmm9, xmm1, xmm4        # xmm9 = (xmm1 * xmm4) + xmm9
	vmovsd	qword ptr [rbp - 288], xmm9
	vmovsd	xmm9, qword ptr [rbp - 256]
	vmulsd	xmm12, xmm9, xmm0
	vfmadd231sd	xmm12, xmm1, xmm30      # xmm12 = (xmm1 * xmm30) + xmm12
	vmovsd	qword ptr [rbp - 1768], xmm12
	vmulsd	xmm12, xmm0, xmm30
	vmovsd	xmm0, qword ptr [rbp - 264]
	vfmadd231sd	xmm12, xmm0, xmm1       # xmm12 = (xmm0 * xmm1) + xmm12
	vmulsd	xmm1, xmm9, xmm7
	vfmadd231sd	xmm1, xmm6, xmm30       # xmm1 = (xmm6 * xmm30) + xmm1
	vmovsd	qword ptr [rbp - 1744], xmm12
	vmovsd	qword ptr [rbp - 1600], xmm1
	vmulsd	xmm1, xmm7, xmm30
	vfmadd231sd	xmm1, xmm6, xmm0        # xmm1 = (xmm6 * xmm0) + xmm1
	vmovsd	qword ptr [rbp - 1592], xmm1
	vmulsd	xmm1, xmm15, xmm9
	vfmadd231sd	xmm1, xmm24, xmm30      # xmm1 = (xmm24 * xmm30) + xmm1
	vmovsd	qword ptr [rbp - 1664], xmm1
	vmulsd	xmm1, xmm15, xmm30
	vfmadd231sd	xmm1, xmm24, xmm0       # xmm1 = (xmm24 * xmm0) + xmm1
	vmovsd	qword ptr [rbp - 1656], xmm1
	vmulsd	xmm1, xmm17, xmm0
	vmulsd	xmm0, xmm13, xmm0
	vfmadd231sd	xmm0, xmm14, xmm30      # xmm0 = (xmm14 * xmm30) + xmm0
	vfmadd231sd	xmm1, xmm18, xmm30      # xmm1 = (xmm18 * xmm30) + xmm1
	vmovsd	qword ptr [rbp - 264], xmm0
	vmulsd	xmm0, xmm13, xmm30
	vmovsd	qword ptr [rbp - 1680], xmm1
	vmulsd	xmm1, xmm17, xmm30
	vfmadd231sd	xmm0, xmm14, xmm9       # xmm0 = (xmm14 * xmm9) + xmm0
	vfmadd231sd	xmm1, xmm18, xmm9       # xmm1 = (xmm18 * xmm9) + xmm1
	vmovsd	qword ptr [rbp - 256], xmm0
	vmulsd	xmm0, xmm13, xmm4
	vmovsd	qword ptr [rbp - 1672], xmm1
	vmovapd	xmm1, xmm2
	vfmadd231sd	xmm0, xmm14, xmm20      # xmm0 = (xmm14 * xmm20) + xmm0
	vmovsd	qword ptr [rbp - 1616], xmm0
	vmulsd	xmm0, xmm13, xmm20
	vfnmadd231sd	xmm0, xmm14, xmm4       # xmm0 = -(xmm14 * xmm4) + xmm0
	vmovsd	qword ptr [rbp - 1608], xmm0
	vmovsd	xmm0, qword ptr [rbp - 240]
	vmulsd	xmm3, xmm13, xmm0
	vfmadd231sd	xmm3, xmm14, xmm2       # xmm3 = (xmm14 * xmm2) + xmm3
	vmulsd	xmm2, xmm13, xmm2
	vmovsd	xmm13, qword ptr [r14]
	vfnmadd231sd	xmm2, xmm14, xmm0       # xmm2 = -(xmm14 * xmm0) + xmm2
	vmovsd	qword ptr [rbp - 1720], xmm3
	vmovsd	xmm3, qword ptr [r9]
	movabs	r9, 136143485007520
	vmovsd	qword ptr [rbp - 1712], xmm2
	vmulsd	xmm2, xmm15, xmm1
	vfmadd231sd	xmm2, xmm24, xmm16      # xmm2 = (xmm24 * xmm16) + xmm2
	vmovsd	qword ptr [rbp - 136], xmm3
	vmovsd	qword ptr [rbp - 1648], xmm2
	vmulsd	xmm2, xmm15, xmm16
	vmovsd	xmm16, qword ptr [rbp - 200]
	vfmsub231sd	xmm2, xmm24, xmm1       # xmm2 = (xmm24 * xmm1) - xmm2
	vmovsd	qword ptr [rbp - 1640], xmm2
	vmulsd	xmm2, xmm17, xmm0
	vfmadd231sd	xmm2, xmm18, xmm1       # xmm2 = (xmm18 * xmm1) + xmm2
	vmulsd	xmm1, xmm17, xmm1
	vfnmadd231sd	xmm1, xmm18, xmm0       # xmm1 = -(xmm18 * xmm0) + xmm1
	vmulsd	xmm0, xmm17, xmm4
	vmovsd	qword ptr [rbp - 248], xmm2
	vmulsd	xmm2, xmm11, xmm7
	vfmadd231sd	xmm0, xmm18, xmm20      # xmm0 = (xmm18 * xmm20) + xmm0
	vmovsd	qword ptr [rbp - 240], xmm1
	vmulsd	xmm1, xmm15, xmm10
	vfmadd231sd	xmm2, xmm6, xmm10       # xmm2 = (xmm6 * xmm10) + xmm2
	vfnmadd231sd	xmm1, xmm24, xmm11      # xmm1 = -(xmm24 * xmm11) + xmm1
	vmovsd	qword ptr [rbp - 1696], xmm0
	vmulsd	xmm0, xmm17, xmm20
	vmovsd	qword ptr [rbp - 1552], xmm2
	vmulsd	xmm2, xmm10, xmm7
	vfnmadd231sd	xmm0, xmm18, xmm4       # xmm0 = -(xmm18 * xmm4) + xmm0
	vmovsd	qword ptr [rbp - 1704], xmm1
	vmulsd	xmm1, xmm15, xmm4
	vfnmadd231sd	xmm2, xmm6, xmm11       # xmm2 = -(xmm6 * xmm11) + xmm2
	vfmsub231sd	xmm1, xmm24, xmm20      # xmm1 = (xmm24 * xmm20) - xmm1
	vmovsd	qword ptr [rbp - 1688], xmm0
	vmovapd	xmm0, xmmword ptr [rdi]
	movabs	rdi, 136143485007472
	vmovsd	qword ptr [rbp - 1560], xmm2
	vmulsd	xmm2, xmm15, xmm11
	vmovsd	xmm17, qword ptr [rdi]
	movabs	rdi, 136143485007440
	vmovsd	xmm31, qword ptr [rdi]
	movabs	rdi, 136143485007648
	vmovsd	xmm12, qword ptr [rdi]
	movabs	rdi, 136143485007680
	vmovsd	qword ptr [rbp - 232], xmm1
	vmulsd	xmm1, xmm15, xmm20
	vfmadd231sd	xmm2, xmm24, xmm10      # xmm2 = (xmm24 * xmm10) + xmm2
	vfmadd231sd	xmm1, xmm24, xmm4       # xmm1 = (xmm24 * xmm4) + xmm1
	vmovsd	xmm24, qword ptr [rbp - 72]
	vmovsd	qword ptr [rbp - 208], xmm2
	vmulpd	xmm2, xmm7, xmm0
	vmovsd	qword ptr [rbp - 224], xmm1
	vmulsd	xmm1, xmm7, xmm4
	vmovsd	qword ptr [rbp - 168], xmm12
	vfmsub231sd	xmm1, xmm6, xmm20       # xmm1 = (xmm6 * xmm20) - xmm1
	vmovsd	qword ptr [rbp - 1584], xmm1
	vmulsd	xmm1, xmm7, xmm20
	vmovsd	xmm20, qword ptr [rbp - 216]
	vfmadd231sd	xmm1, xmm6, xmm4        # xmm1 = (xmm6 * xmm4) + xmm1
	vmovsd	qword ptr [rbp - 1576], xmm1
	vmovapd	xmm1, xmmword ptr [r8]
	movabs	r8, 136143485007480
	vmovsd	xmm18, qword ptr [r8]
	movabs	r8, 136143485007448
	vmovsd	xmm30, qword ptr [r8]
	movabs	r8, 136143485007656
	vmovsd	xmm14, qword ptr [r8]
	movabs	r8, 136143485007688
	vfmsub231pd	xmm2, xmm6, xmm1        # xmm2 = (xmm6 * xmm1) - xmm2
	vmulpd	xmm1, xmm7, xmm1
	vfmadd231pd	xmm1, xmm6, xmm0        # xmm1 = (xmm6 * xmm0) + xmm1
	vshufpd	xmm0, xmm7, xmm7, 1             # xmm0 = xmm7[1,0]
	vmovapd	xmm7, xmm3
	vmovsd	qword ptr [rbp - 160], xmm14
	vmovapd	xmmword ptr [rbp - 944], xmm2
	vmulsd	xmm2, xmm0, xmm3
	vmovsd	xmm3, qword ptr [r10]
	movabs	r10, 136143485007536
	vmovsd	xmm11, qword ptr [r10]
	vmovapd	xmmword ptr [rbp - 960], xmm1
	vshufpd	xmm1, xmm6, xmm6, 1             # xmm1 = xmm6[1,0]
	vfnmadd231sd	xmm2, xmm1, xmm3        # xmm2 = -(xmm1 * xmm3) + xmm2
	vmovapd	xmm10, xmm3
	vmovsd	qword ptr [rbp - 144], xmm10
	vmovsd	qword ptr [rbp - 1440], xmm2
	vmulsd	xmm2, xmm0, xmm3
	vmovsd	xmm3, qword ptr [r9]
	movabs	r9, 136143485007528
	vmovsd	xmm5, qword ptr [r9]
	vfmsub231sd	xmm2, xmm1, xmm3        # xmm2 = (xmm1 * xmm3) - xmm2
	vmovapd	xmm9, xmm3
	vmovsd	qword ptr [rbp - 120], xmm3
	vmovsd	xmm3, qword ptr [r15]
	vmovsd	qword ptr [rbp - 112], xmm5
	vmovsd	qword ptr [rbp - 1432], xmm2
	vmulsd	xmm2, xmm0, xmm3
	vmovapd	xmm4, xmm3
	vmovsd	xmm3, qword ptr [rbx]
	vmovsd	qword ptr [rbp - 104], xmm4
	vfmadd231sd	xmm2, xmm1, xmm13       # xmm2 = (xmm1 * xmm13) + xmm2
	vmovsd	qword ptr [rbp - 1504], xmm2
	vmulsd	xmm2, xmm13, xmm0
	vmovapd	xmm6, xmm3
	vmovsd	qword ptr [rbp - 1112], xmm6
	vfmadd231sd	xmm2, xmm1, xmm3        # xmm2 = (xmm1 * xmm3) + xmm2
	vmovsd	xmm3, qword ptr [r11]
	vmovsd	qword ptr [rbp - 1496], xmm2
	vmulsd	xmm2, xmm0, xmm17
	vfmadd231sd	xmm2, xmm1, xmm18       # xmm2 = (xmm1 * xmm18) + xmm2
	vmovsd	qword ptr [rbp - 864], xmm3
	vmovsd	qword ptr [rbp - 1544], xmm2
	vmulsd	xmm2, xmm0, xmm18
	vfnmadd231sd	xmm2, xmm1, xmm17       # xmm2 = -(xmm1 * xmm17) + xmm2
	vmovsd	qword ptr [rbp - 1536], xmm2
	vmulsd	xmm2, xmm0, xmm3
	vmulsd	xmm0, xmm11, xmm0
	vfmadd231sd	xmm0, xmm5, xmm1        # xmm0 = (xmm5 * xmm1) + xmm0
	vfmadd231sd	xmm2, xmm1, xmm11       # xmm2 = (xmm1 * xmm11) + xmm2
	vmovsd	xmm1, qword ptr [rbp - 40]
	vmovsd	qword ptr [rbp - 1072], xmm0
	vmovsd	xmm0, qword ptr [rbp - 48]
	vmovsd	qword ptr [rbp - 784], xmm2
	vmulsd	xmm2, xmm1, xmm31
	vfmadd231sd	xmm2, xmm0, xmm30       # xmm2 = (xmm0 * xmm30) + xmm2
	vmovsd	qword ptr [rbp - 1208], xmm2
	vmulsd	xmm2, xmm1, xmm30
	vfnmadd231sd	xmm2, xmm0, xmm31       # xmm2 = -(xmm0 * xmm31) + xmm2
	vmovsd	qword ptr [rbp - 1200], xmm2
	vmulsd	xmm2, xmm1, xmm5
	vfmadd231sd	xmm2, xmm0, xmm11       # xmm2 = (xmm0 * xmm11) + xmm2
	vmovsd	qword ptr [rbp - 1320], xmm2
	vmulsd	xmm2, xmm11, xmm1
	vfmadd231sd	xmm2, xmm0, xmm3        # xmm2 = (xmm0 * xmm3) + xmm2
	vmovsd	qword ptr [rbp - 1312], xmm2
	vmulsd	xmm2, xmm1, xmm17
	vfmsub231sd	xmm2, xmm0, xmm18       # xmm2 = (xmm0 * xmm18) - xmm2
	vmovsd	qword ptr [rbp - 1456], xmm2
	vmulsd	xmm2, xmm1, xmm18
	vfmadd231sd	xmm2, xmm0, xmm17       # xmm2 = (xmm0 * xmm17) + xmm2
	vmovsd	qword ptr [rbp - 1448], xmm2
	vmulsd	xmm2, xmm1, xmm6
	vfmadd231sd	xmm2, xmm0, xmm13       # xmm2 = (xmm0 * xmm13) + xmm2
	vmovsd	qword ptr [rbp - 1512], xmm2
	vmulsd	xmm2, xmm13, xmm1
	vfmadd231sd	xmm2, xmm0, xmm4        # xmm2 = (xmm0 * xmm4) + xmm2
	vmovsd	qword ptr [rbp - 1520], xmm2
	vmulsd	xmm2, xmm12, xmm1
	vmulsd	xmm1, xmm14, xmm1
	vfmadd231sd	xmm1, xmm12, xmm0       # xmm1 = (xmm12 * xmm0) + xmm1
	vfmsub231sd	xmm2, xmm0, xmm14       # xmm2 = (xmm0 * xmm14) - xmm2
	vmovsd	xmm0, qword ptr [rbp - 544]
	vmovsd	qword ptr [rbp - 48], xmm1
	vmovsd	xmm1, qword ptr [rbp - 536]
	vmovsd	qword ptr [rbp - 1568], xmm2
	vmulsd	xmm2, xmm1, xmm31
	vfmadd231sd	xmm2, xmm0, xmm30       # xmm2 = (xmm0 * xmm30) + xmm2
	vmovsd	qword ptr [rbp - 1152], xmm2
	vmulsd	xmm2, xmm1, xmm30
	vfnmadd231sd	xmm2, xmm0, xmm31       # xmm2 = -(xmm0 * xmm31) + xmm2
	vmovsd	qword ptr [rbp - 1160], xmm2
	vmulsd	xmm2, xmm1, xmm6
	vfmadd231sd	xmm2, xmm0, xmm13       # xmm2 = (xmm0 * xmm13) + xmm2
	vmovsd	qword ptr [rbp - 1232], xmm2
	vmulsd	xmm2, xmm13, xmm1
	vfmadd231sd	xmm2, xmm0, xmm4        # xmm2 = (xmm0 * xmm4) + xmm2
	vmovsd	qword ptr [rbp - 1224], xmm2
	vmulsd	xmm2, xmm1, xmm5
	vfmadd231sd	xmm2, xmm0, xmm11       # xmm2 = (xmm0 * xmm11) + xmm2
	vmovsd	qword ptr [rbp - 1376], xmm2
	vmulsd	xmm2, xmm11, xmm1
	vfmadd231sd	xmm2, xmm0, xmm3        # xmm2 = (xmm0 * xmm3) + xmm2
	vmovsd	qword ptr [rbp - 1368], xmm2
	vmulsd	xmm2, xmm12, xmm1
	vfmsub231sd	xmm2, xmm0, xmm14       # xmm2 = (xmm0 * xmm14) - xmm2
	vmovsd	qword ptr [rbp - 1472], xmm2
	vmulsd	xmm2, xmm14, xmm1
	vmovsd	xmm14, qword ptr [rbp - 64]
	vfmadd231sd	xmm2, xmm0, xmm12       # xmm2 = (xmm0 * xmm12) + xmm2
	vmovsd	xmm12, qword ptr [rbp - 96]
	vmovsd	qword ptr [rbp - 1464], xmm2
	vmulsd	xmm2, xmm1, xmm17
	vmulsd	xmm1, xmm1, xmm18
	vfmadd231sd	xmm1, xmm17, xmm0       # xmm1 = (xmm17 * xmm0) + xmm1
	vfmsub231sd	xmm2, xmm0, xmm18       # xmm2 = (xmm0 * xmm18) - xmm2
	vmovsd	xmm0, qword ptr [rbp - 528]
	vmovsd	qword ptr [rbp - 1528], xmm1
	vmovsd	xmm1, qword ptr [rbp - 520]
	vmovsd	qword ptr [rbp - 40], xmm2
	vmulsd	xmm2, xmm1, xmm17
	vfmadd231sd	xmm2, xmm0, xmm18       # xmm2 = (xmm0 * xmm18) + xmm2
	vmovsd	qword ptr [rbp - 1120], xmm2
	vmulsd	xmm2, xmm1, xmm18
	vfnmadd231sd	xmm2, xmm0, xmm17       # xmm2 = -(xmm0 * xmm17) + xmm2
	vmovsd	qword ptr [rbp - 1128], xmm2
	vmulsd	xmm2, xmm1, xmm7
	vmovsd	xmm7, qword ptr [rbp - 672]
	vfnmadd231sd	xmm2, xmm0, xmm10       # xmm2 = -(xmm0 * xmm10) + xmm2
	vmovsd	qword ptr [rbp - 1176], xmm2
	vmulsd	xmm2, xmm10, xmm1
	vmovsd	xmm10, qword ptr [rbp - 88]
	vfmsub231sd	xmm2, xmm0, xmm9        # xmm2 = (xmm0 * xmm9) - xmm2
	vmovsd	xmm9, qword ptr [rbp - 80]
	vmovsd	qword ptr [rbp - 1168], xmm2
	vmulsd	xmm2, xmm1, xmm3
	vmovsd	xmm3, qword ptr [rdi]
	movabs	rdi, 136143485007728
	vfmadd231sd	xmm2, xmm0, xmm11       # xmm2 = (xmm0 * xmm11) + xmm2
	vmovsd	qword ptr [rbp - 1296], xmm2
	vmulsd	xmm2, xmm11, xmm1
	vmovsd	qword ptr [rbp - 128], xmm3
	vfmadd231sd	xmm2, xmm0, xmm5        # xmm2 = (xmm0 * xmm5) + xmm2
	vmovsd	xmm5, qword ptr [rsi + 112]
	vmovsd	qword ptr [rbp - 1288], xmm2
	vmulsd	xmm2, xmm1, xmm4
	vmovsd	xmm4, qword ptr [r8]
	movabs	r8, 136143485007744
	vfmadd231sd	xmm2, xmm0, xmm13       # xmm2 = (xmm0 * xmm13) + xmm2
	vmovsd	qword ptr [rbp - 1280], xmm5
	vmovsd	qword ptr [rbp - 1416], xmm2
	vmulsd	xmm2, xmm13, xmm1
	vmovsd	qword ptr [rbp - 152], xmm4
	vfmadd231sd	xmm2, xmm0, xmm6        # xmm2 = (xmm0 * xmm6) + xmm2
	vmovsd	qword ptr [rbp - 1424], xmm2
	vmulsd	xmm2, xmm1, xmm3
	vmulsd	xmm1, xmm1, xmm4
	vfmsub231sd	xmm2, xmm0, xmm4        # xmm2 = (xmm0 * xmm4) - xmm2
	vmovapd	xmm4, xmmword ptr [rbp - 848]
	vfmadd231sd	xmm1, xmm3, xmm0        # xmm1 = (xmm3 * xmm0) + xmm1
	vmovapd	xmm0, xmm14
	vfmadd213sd	xmm0, xmm22, xmm23      # xmm0 = (xmm22 * xmm0) + xmm23
	vmovsd	xmm23, qword ptr [rsi + 152]
	vmovsd	qword ptr [rbp - 1480], xmm1
	vmulsd	xmm1, xmm29, xmm10
	vmovsd	qword ptr [rbp - 1488], xmm2
	vmulsd	xmm2, xmm29, xmm12
	vmovapd	xmm29, xmm12
	vfmadd231sd	xmm1, xmm4, xmm12       # xmm1 = (xmm4 * xmm12) + xmm1
	vfnmadd231sd	xmm2, xmm4, xmm10       # xmm2 = -(xmm4 * xmm10) + xmm2
	vaddsd	xmm2, xmm2, qword ptr [rbp - 720]
	vmovsd	xmm12, qword ptr [rbp - 1632]
	vmovapd	xmm4, xmmword ptr [rbp - 832]
	vmovsd	qword ptr [rbp - 1240], xmm23
	vaddsd	xmm1, xmm26, xmm1
	vmulsd	xmm26, xmm23, xmm10
	vaddsd	xmm0, xmm1, xmm0
	vmovapd	xmm1, xmm9
	vfmsub213sd	xmm1, xmm22, qword ptr [rbp - 912] # xmm1 = (xmm22 * xmm1) - mem
	vmulsd	xmm25, xmm5, xmm12
	vaddsd	xmm1, xmm2, xmm1
	vmovapd	xmm2, xmm14
	vfmsub213sd	xmm2, xmm8, xmm21       # xmm2 = (xmm8 * xmm2) - xmm21
	vmovsd	xmm21, qword ptr [rbp - 184]
	vaddsd	xmm0, xmm0, xmm2
	vmovapd	xmm2, xmm24
	vfmsub213sd	xmm2, xmm8, qword ptr [rbp - 928] # xmm2 = (xmm8 * xmm2) - mem
	vfmadd231sd	xmm26, xmm21, xmm29     # xmm26 = (xmm21 * xmm29) + xmm26
	vmovapd	xmm22, xmm21
	vaddsd	xmm1, xmm1, xmm2
	vmulsd	xmm2, xmm19, xmm27
	vfmsub231sd	xmm2, xmm4, xmm12       # xmm2 = (xmm4 * xmm12) - xmm2
	vaddsd	xmm6, xmm0, xmm2
	vmulsd	xmm0, xmm19, xmm12
	vmovsd	xmm19, qword ptr [rbp - 192]
	vmulsd	xmm2, xmm5, xmm27
	vfmadd231sd	xmm0, xmm4, xmm27       # xmm0 = (xmm4 * xmm27) + xmm0
	vaddsd	xmm4, xmm1, xmm0
	vmulsd	xmm1, xmm7, xmm24
	vmovapd	xmm0, xmm14
	vmulsd	xmm7, xmm7, xmm28
	vfmsub231sd	xmm2, xmm19, xmm12      # xmm2 = (xmm19 * xmm12) - xmm2
	vfmadd231sd	xmm25, xmm19, xmm27     # xmm25 = (xmm19 * xmm27) + xmm25
	vfmadd213sd	xmm0, xmm16, xmm1       # xmm0 = (xmm16 * xmm0) + xmm1
	vfmsub231sd	xmm1, xmm16, xmm14      # xmm1 = (xmm16 * xmm14) - xmm1
	vaddsd	xmm0, xmm20, xmm0
	vaddsd	xmm1, xmm20, xmm1
	vaddsd	xmm0, xmm0, xmm2
	vmovapd	xmm2, xmm9
	vmovsd	xmm9, qword ptr [rbp - 32]
	vfmsub213sd	xmm2, xmm16, xmm7       # xmm2 = (xmm16 * xmm2) - xmm7
	vaddsd	xmm2, xmm9, xmm2
	vaddsd	xmm2, xmm2, xmm25
	vmovapd	xmm25, xmm10
	vaddsd	xmm10, xmm0, xmm26
	vmulsd	xmm26, xmm23, xmm29
	vmovsd	xmm0, qword ptr [rbp - 656]
	vfnmadd231sd	xmm26, xmm21, xmm25     # xmm26 = -(xmm21 * xmm25) + xmm26
	vmovsd	xmm21, qword ptr [rbp - 176]
	vaddsd	xmm26, xmm2, xmm26
	vmulsd	xmm8, xmm0, xmm24
	vmulsd	xmm15, xmm21, xmm14
	vmovsd	xmm14, qword ptr [rbp - 864]
	vsubsd	xmm2, xmm15, xmm8
	vaddsd	xmm8, xmm8, xmm15
	vmovsd	xmm15, qword ptr [rbp - 112]
	vaddsd	xmm3, xmm10, xmm2
	vmulsd	xmm10, xmm0, xmm28
	vmovapd	xmm0, xmm24
	vfmsub213sd	xmm0, xmm21, xmm10      # xmm0 = (xmm21 * xmm0) - xmm10
	vaddsd	xmm2, xmm26, xmm0
	vaddsd	xmm26, xmm20, xmm16
	vfmsub213sd	xmm16, xmm24, xmm7      # xmm16 = (xmm24 * xmm16) - xmm7
	vmulsd	xmm7, xmm5, xmm25
	vmovapd	xmm20, xmm27
	vfmadd231sd	xmm7, xmm19, xmm29      # xmm7 = (xmm19 * xmm29) + xmm7
	vaddsd	xmm28, xmm9, xmm16
	vmovsd	xmm9, qword ptr [rbp - 120]
	vmovsd	xmm16, qword ptr [rbp - 160]
	vaddsd	xmm1, xmm1, xmm7
	vaddsd	xmm7, xmm26, xmm19
	vmulsd	xmm26, xmm5, xmm29
	vfnmadd231sd	xmm26, xmm25, xmm19     # xmm26 = -(xmm25 * xmm19) + xmm26
	vaddsd	xmm7, xmm7, xmm22
	vmovsd	xmm19, qword ptr [rsi + 120]
	vaddsd	xmm5, xmm7, xmm21
	vaddsd	xmm26, xmm28, xmm26
	vmulsd	xmm28, xmm23, xmm27
	vmovapd	xmm27, xmm12
	vmovsd	qword ptr [rbp - 1216], xmm5
	vmovsd	xmm5, qword ptr [rbp - 144]
	vfmsub231sd	xmm28, xmm22, xmm12     # xmm28 = (xmm22 * xmm12) - xmm28
	vaddsd	xmm1, xmm1, xmm28
	vmulsd	xmm28, xmm23, xmm12
	vmovsd	xmm12, qword ptr [rbp - 104]
	vmovapd	xmm23, xmm20
	vaddsd	xmm1, xmm8, xmm1
	vmovsd	xmm8, qword ptr [rbp - 136]
	vfmadd231sd	xmm28, xmm20, xmm22     # xmm28 = (xmm20 * xmm22) + xmm28
	vmovsd	xmm22, qword ptr [rbp - 80]
	vaddsd	xmm28, xmm26, xmm28
	vmovsd	xmm26, qword ptr [rbp - 64]
	vmulsd	xmm0, xmm8, xmm2
	vfmsub213sd	xmm21, xmm22, xmm10     # xmm21 = (xmm22 * xmm21) - xmm10
	vmovsd	xmm10, qword ptr [rbp - 168]
	vfnmadd231sd	xmm0, xmm3, xmm5        # xmm0 = -(xmm3 * xmm5) + xmm0
	vaddsd	xmm7, xmm28, xmm21
	vmovsd	xmm28, qword ptr [rbp - 56]
	vmovsd	qword ptr [rbp - 1400], xmm0
	vmulsd	xmm0, xmm2, xmm5
	vfmsub231sd	xmm0, xmm3, xmm9        # xmm0 = (xmm3 * xmm9) - xmm0
	vmovsd	qword ptr [rbp - 1384], xmm0
	vmulsd	xmm0, xmm8, xmm4
	vmovapd	xmm8, xmmword ptr [rbp - 880]
	vfnmadd231sd	xmm0, xmm6, xmm5        # xmm0 = -(xmm6 * xmm5) + xmm0
	vmovsd	qword ptr [rbp - 192], xmm0
	vmulsd	xmm0, xmm4, xmm5
	vmovapd	xmm5, xmmword ptr [rbp - 896]
	vfmsub231sd	xmm0, xmm6, xmm9        # xmm0 = (xmm6 * xmm9) - xmm0
	vmovsd	xmm9, qword ptr [rbp - 1112]
	vmovsd	qword ptr [rbp - 184], xmm0
	vmulsd	xmm0, xmm8, xmm31
	vfmadd231sd	xmm0, xmm5, xmm30       # xmm0 = (xmm5 * xmm30) + xmm0
	vmovsd	qword ptr [rbp - 176], xmm0
	vmulsd	xmm0, xmm8, xmm30
	vfnmadd231sd	xmm0, xmm5, xmm31       # xmm0 = -(xmm5 * xmm31) + xmm0
	vmovsd	qword ptr [rbp - 912], xmm0
	vmulsd	xmm0, xmm7, xmm31
	vfmadd231sd	xmm0, xmm1, xmm30       # xmm0 = (xmm1 * xmm30) + xmm0
	vmovsd	qword ptr [rbp - 216], xmm0
	vmulsd	xmm0, xmm7, xmm30
	vmovapd	xmm30, xmm23
	vfnmadd231sd	xmm0, xmm1, xmm31       # xmm0 = -(xmm1 * xmm31) + xmm0
	vmovapd	xmm31, xmm27
	vmovsd	qword ptr [rbp - 200], xmm0
	vmulsd	xmm0, xmm4, xmm17
	vfmadd231sd	xmm0, xmm6, xmm18       # xmm0 = (xmm6 * xmm18) + xmm0
	vmovsd	qword ptr [rbp - 1272], xmm0
	vmulsd	xmm0, xmm4, xmm18
	vfnmadd231sd	xmm0, xmm6, xmm17       # xmm0 = -(xmm6 * xmm17) + xmm0
	vmovsd	qword ptr [rbp - 1264], xmm0
	vmulsd	xmm0, xmm2, xmm17
	vfmadd231sd	xmm0, xmm3, xmm18       # xmm0 = (xmm3 * xmm18) + xmm0
	vmovsd	qword ptr [rbp - 1408], xmm0
	vmulsd	xmm0, xmm2, xmm18
	vfnmadd231sd	xmm0, xmm3, xmm17       # xmm0 = -(xmm3 * xmm17) + xmm0
	vmovsd	qword ptr [rbp - 1392], xmm0
	vmulsd	xmm0, xmm7, xmm17
	vfmsub231sd	xmm0, xmm1, xmm18       # xmm0 = (xmm1 * xmm18) - xmm0
	vmovsd	qword ptr [rbp - 528], xmm0
	vmulsd	xmm0, xmm7, xmm18
	vmulsd	xmm18, xmm19, xmm28
	vfmadd231sd	xmm0, xmm1, xmm17       # xmm0 = (xmm1 * xmm17) + xmm0
	vmulsd	xmm17, xmm19, xmm24
	vmovsd	qword ptr [rbp - 520], xmm0
	vmulsd	xmm0, xmm15, xmm7
	vfmadd231sd	xmm0, xmm1, xmm11       # xmm0 = (xmm1 * xmm11) + xmm0
	vmovsd	qword ptr [rbp - 1256], xmm0
	vmulsd	xmm0, xmm11, xmm7
	vfmadd231sd	xmm0, xmm1, xmm14       # xmm0 = (xmm1 * xmm14) + xmm0
	vmovsd	qword ptr [rbp - 1248], xmm0
	vmulsd	xmm0, xmm9, xmm7
	vfmadd231sd	xmm0, xmm1, xmm13       # xmm0 = (xmm1 * xmm13) + xmm0
	vmovsd	qword ptr [rbp - 1352], xmm0
	vmulsd	xmm0, xmm13, xmm7
	vfmadd231sd	xmm0, xmm1, xmm12       # xmm0 = (xmm1 * xmm12) + xmm0
	vmovsd	qword ptr [rbp - 1336], xmm0
	vmulsd	xmm0, xmm10, xmm7
	vfmsub231sd	xmm0, xmm1, xmm16       # xmm0 = (xmm1 * xmm16) - xmm0
	vmovsd	qword ptr [rbp - 544], xmm0
	vmulsd	xmm0, xmm7, xmm16
	vmovsd	xmm7, qword ptr [rbp - 152]
	vfmadd231sd	xmm0, xmm1, xmm10       # xmm0 = (xmm1 * xmm10) + xmm0
	vmovsd	xmm1, qword ptr [rbp - 128]
	vmovsd	qword ptr [rbp - 536], xmm0
	vmulsd	xmm0, xmm8, xmm10
	vfmsub231sd	xmm0, xmm5, xmm16       # xmm0 = (xmm5 * xmm16) - xmm0
	vmovsd	qword ptr [rbp - 136], xmm0
	vmulsd	xmm0, xmm8, xmm16
	vfmadd231sd	xmm0, xmm5, xmm10       # xmm0 = (xmm5 * xmm10) + xmm0
	vmovapd	xmm10, xmm22
	vmovsd	qword ptr [rbp - 120], xmm0
	vmulsd	xmm0, xmm2, xmm1
	vfmsub231sd	xmm0, xmm3, xmm7        # xmm0 = (xmm3 * xmm7) - xmm0
	vmovsd	qword ptr [rbp - 1360], xmm0
	vmulsd	xmm0, xmm2, xmm7
	vfmadd231sd	xmm0, xmm3, xmm1        # xmm0 = (xmm3 * xmm1) + xmm0
	vmovsd	qword ptr [rbp - 1344], xmm0
	vmulsd	xmm0, xmm4, xmm1
	vfmsub231sd	xmm0, xmm6, xmm7        # xmm0 = (xmm6 * xmm7) - xmm0
	vmovsd	qword ptr [rbp - 144], xmm0
	vmulsd	xmm0, xmm4, xmm7
	vmovsd	xmm7, qword ptr [rsi + 40]
	vfmadd231sd	xmm0, xmm6, xmm1        # xmm0 = (xmm6 * xmm1) + xmm0
	vmovapd	xmm1, xmmword ptr [r8]
	vmovsd	qword ptr [rbp - 128], xmm0
	vmulsd	xmm0, xmm12, xmm4
	vmulsd	xmm21, xmm7, xmm28
	vfmadd231sd	xmm0, xmm6, xmm13       # xmm0 = (xmm6 * xmm13) + xmm0
	vmovsd	qword ptr [rbp - 1192], xmm0
	vmulsd	xmm0, xmm13, xmm4
	vfmadd231sd	xmm0, xmm6, xmm9        # xmm0 = (xmm6 * xmm9) + xmm0
	vmovsd	qword ptr [rbp - 1184], xmm0
	vmulsd	xmm0, xmm14, xmm4
	vfmadd231sd	xmm0, xmm6, xmm11       # xmm0 = (xmm6 * xmm11) + xmm0
	vmovsd	qword ptr [rbp - 928], xmm0
	vmulsd	xmm0, xmm11, xmm4
	vfmadd231sd	xmm0, xmm15, xmm6       # xmm0 = (xmm15 * xmm6) + xmm0
	vmovapd	xmm6, xmm22
	vmovsd	qword ptr [rbp - 168], xmm0
	vmulsd	xmm0, xmm8, xmm15
	vfmadd231sd	xmm0, xmm5, xmm11       # xmm0 = (xmm5 * xmm11) + xmm0
	vmovsd	qword ptr [rbp - 1144], xmm0
	vmulsd	xmm0, xmm8, xmm11
	vfmadd231sd	xmm0, xmm5, xmm14       # xmm0 = (xmm5 * xmm14) + xmm0
	vmovsd	qword ptr [rbp - 1136], xmm0
	vmulsd	xmm0, xmm14, xmm2
	vmovsd	xmm14, qword ptr [rsi]
	vfmadd231sd	xmm0, xmm3, xmm11       # xmm0 = (xmm3 * xmm11) + xmm0
	vmovsd	qword ptr [rbp - 160], xmm0
	vmulsd	xmm0, xmm11, xmm2
	vfmadd231sd	xmm0, xmm3, xmm15       # xmm0 = (xmm3 * xmm15) + xmm0
	vmovsd	xmm15, qword ptr [rsi + 160]
	vmovsd	qword ptr [rbp - 152], xmm0
	vmulsd	xmm0, xmm12, xmm2
	vfmadd231sd	xmm0, xmm3, xmm13       # xmm0 = (xmm3 * xmm13) + xmm0
	vmovsd	qword ptr [rbp - 1328], xmm0
	vmulsd	xmm0, xmm13, xmm2
	vfmadd231sd	xmm0, xmm3, xmm9        # xmm0 = (xmm3 * xmm9) + xmm0
	vmovsd	xmm3, qword ptr [rsi + 80]
	vmovsd	qword ptr [rbp - 1304], xmm0
	vmulsd	xmm0, xmm8, xmm9
	vmovsd	xmm9, qword ptr [rdx + 120]
	vfmadd231sd	xmm0, xmm5, xmm13       # xmm0 = (xmm5 * xmm13) + xmm0
	vmovsd	qword ptr [rbp - 112], xmm0
	vmulsd	xmm0, xmm8, xmm13
	vmovsd	xmm13, qword ptr [rdx + 160]
	vmulsd	xmm16, xmm9, xmm26
	vfmadd231sd	xmm0, xmm5, xmm12       # xmm0 = (xmm5 * xmm12) + xmm0
	vmulsd	xmm12, xmm3, xmm28
	vmovsd	qword ptr [rbp - 104], xmm0
	vmovapd	xmm0, xmmword ptr [rdi]
	vmulpd	xmm2, xmm8, xmm0
	vfmsub231pd	xmm2, xmm5, xmm1        # xmm2 = (xmm5 * xmm1) - xmm2
	vmulpd	xmm1, xmm8, xmm1
	vmulsd	xmm8, xmm3, xmm24
	vfmadd231pd	xmm1, xmm5, xmm0        # xmm1 = (xmm5 * xmm0) + xmm1
	vmovsd	xmm0, qword ptr [rdx + 40]
	vmovapd	xmm5, xmm26
	vmovapd	xmmword ptr [rbp - 864], xmm2
	vmovsd	xmm2, qword ptr [rdx]
	vmovapd	xmmword ptr [rbp - 896], xmm1
	vmulsd	xmm1, xmm7, xmm25
	vfmadd231sd	xmm1, xmm0, xmm29       # xmm1 = (xmm0 * xmm29) + xmm1
	vaddsd	xmm4, xmm2, xmm1
	vmovsd	xmm1, qword ptr [rdx + 80]
	vfmadd213sd	xmm5, xmm1, xmm8        # xmm5 = (xmm1 * xmm5) + xmm8
	vfmsub213sd	xmm6, xmm1, xmm12       # xmm6 = (xmm1 * xmm6) - xmm12
	vfmsub231sd	xmm8, xmm1, xmm26       # xmm8 = (xmm1 * xmm26) - xmm8
	vaddsd	xmm4, xmm4, xmm5
	vmulsd	xmm5, xmm7, xmm29
	vfnmadd231sd	xmm5, xmm0, xmm25       # xmm5 = -(xmm0 * xmm25) + xmm5
	vaddsd	xmm5, xmm14, xmm5
	vaddsd	xmm5, xmm5, xmm6
	vsubsd	xmm6, xmm16, xmm17
	vaddsd	xmm4, xmm4, xmm6
	vmovapd	xmm6, xmm24
	vfmsub213sd	xmm6, xmm9, xmm18       # xmm6 = (xmm9 * xmm6) - xmm18
	vaddsd	xmm5, xmm5, xmm6
	vmulsd	xmm6, xmm15, xmm20
	vfmsub231sd	xmm6, xmm13, xmm27      # xmm6 = (xmm13 * xmm27) - xmm6
	vaddsd	xmm4, xmm4, xmm6
	vmulsd	xmm6, xmm3, xmm27
	vmovsd	qword ptr [rbp - 880], xmm4
	vmulsd	xmm4, xmm15, xmm27
	vfmadd231sd	xmm6, xmm1, xmm23       # xmm6 = (xmm1 * xmm23) + xmm6
	vfmadd231sd	xmm4, xmm13, xmm20      # xmm4 = (xmm13 * xmm20) + xmm4
	vmulsd	xmm20, xmm7, xmm24
	vaddsd	xmm11, xmm5, xmm4
	vmulsd	xmm5, xmm3, xmm23
	vmovapd	xmm4, xmm26
	vfmadd213sd	xmm4, xmm0, xmm20       # xmm4 = (xmm0 * xmm4) + xmm20
	vmulsd	xmm23, xmm15, xmm24
	vfmsub231sd	xmm20, xmm0, xmm26      # xmm20 = (xmm0 * xmm26) - xmm20
	vfmsub231sd	xmm5, xmm1, xmm27       # xmm5 = (xmm1 * xmm27) - xmm5
	vmulsd	xmm27, xmm15, xmm28
	vaddsd	xmm4, xmm2, xmm4
	vaddsd	xmm4, xmm4, xmm5
	vmovapd	xmm5, xmm22
	vfmsub213sd	xmm5, xmm0, xmm21       # xmm5 = (xmm0 * xmm5) - xmm21
	vmulsd	xmm22, xmm13, xmm26
	vfmsub231sd	xmm21, xmm0, xmm24      # xmm21 = (xmm0 * xmm24) - xmm21
	vaddsd	xmm5, xmm14, xmm5
	vaddsd	xmm21, xmm14, xmm21
	vaddsd	xmm5, xmm5, xmm6
	vmulsd	xmm6, xmm19, xmm25
	vfmadd231sd	xmm6, xmm9, xmm29       # xmm6 = (xmm9 * xmm29) + xmm6
	vaddsd	xmm4, xmm4, xmm6
	vmulsd	xmm6, xmm19, xmm29
	vfnmadd231sd	xmm6, xmm9, xmm25       # xmm6 = -(xmm9 * xmm25) + xmm6
	vaddsd	xmm5, xmm5, xmm6
	vsubsd	xmm6, xmm22, xmm23
	vaddsd	xmm6, xmm4, xmm6
	vmovapd	xmm4, xmm24
	vfmsub213sd	xmm4, xmm13, xmm27      # xmm4 = (xmm13 * xmm4) - xmm27
	vfmsub231sd	xmm27, xmm13, xmm10     # xmm27 = (xmm13 * xmm10) - xmm27
	vaddsd	xmm5, xmm5, xmm4
	vaddsd	xmm4, xmm2, xmm20
	vmulsd	xmm20, xmm3, xmm25
	vfmadd231sd	xmm20, xmm1, xmm29      # xmm20 = (xmm1 * xmm29) + xmm20
	vaddsd	xmm4, xmm4, xmm20
	vaddsd	xmm20, xmm14, xmm7
	vaddsd	xmm20, xmm20, xmm3
	vmulsd	xmm3, xmm3, xmm29
	vfnmadd231sd	xmm3, xmm1, xmm25       # xmm3 = -(xmm1 * xmm25) + xmm3
	vaddsd	xmm20, xmm20, xmm19
	vaddsd	xmm3, xmm21, xmm3
	vmulsd	xmm21, xmm19, xmm30
	vmulsd	xmm19, xmm19, xmm31
	vfmsub231sd	xmm21, xmm9, xmm31      # xmm21 = (xmm9 * xmm31) - xmm21
	vfmadd231sd	xmm19, xmm9, xmm30      # xmm19 = (xmm9 * xmm30) + xmm19
	vaddsd	xmm3, xmm3, xmm19
	vaddsd	xmm4, xmm4, xmm21
	vaddsd	xmm19, xmm23, xmm22
	vmovapd	xmm21, xmm29
	vmovapd	xmm22, xmm25
	vaddsd	xmm4, xmm4, xmm19
	vmulsd	xmm19, xmm7, xmm30
	vmulsd	xmm7, xmm7, xmm31
	vaddsd	xmm3, xmm3, xmm27
	vfmadd231sd	xmm7, xmm30, xmm0       # xmm7 = (xmm30 * xmm0) + xmm7
	vfmsub231sd	xmm19, xmm0, xmm31      # xmm19 = (xmm0 * xmm31) - xmm19
	vaddsd	xmm0, xmm2, xmm0
	vaddsd	xmm0, xmm0, xmm1
	vfmsub213sd	xmm1, xmm24, xmm12      # xmm1 = (xmm24 * xmm1) - xmm12
	vmovsd	xmm12, qword ptr [rbp - 600]
	vaddsd	xmm2, xmm2, xmm19
	vaddsd	xmm7, xmm14, xmm7
	vaddsd	xmm12, xmm12, qword ptr [rbp - 632]
	vmovapd	xmm19, xmm10
	vaddsd	xmm0, xmm9, xmm0
	vfmsub213sd	xmm9, xmm10, xmm18      # xmm9 = (xmm10 * xmm9) - xmm18
	vaddsd	xmm1, xmm7, xmm1
	vaddsd	xmm7, xmm17, xmm16
	vaddsd	xmm2, xmm8, xmm2
	vaddsd	xmm12, xmm12, qword ptr [rbp - 608]
	vmovapd	xmm16, xmm19
	vaddsd	xmm2, xmm2, xmm7
	vmulsd	xmm7, xmm15, xmm25
	vaddsd	xmm12, xmm12, qword ptr [rbp - 592]
	vaddsd	xmm1, xmm9, xmm1
	vaddsd	xmm9, xmm20, xmm15
	vmovsd	xmm20, qword ptr [rbp - 616]
	vfmadd231sd	xmm7, xmm13, xmm29      # xmm7 = (xmm13 * xmm29) + xmm7
	vaddsd	xmm14, xmm12, qword ptr [rbp - 624]
	vaddsd	xmm2, xmm2, xmm7
	vaddsd	xmm7, xmm13, xmm0
	vmulsd	xmm0, xmm15, xmm29
	vmovsd	xmm29, qword ptr [rbp - 880]
	vmovapd	xmm15, xmm26
	vaddsd	xmm12, xmm29, qword ptr [rbp - 576]
	vfnmadd231sd	xmm0, xmm25, xmm13      # xmm0 = -(xmm25 * xmm13) + xmm0
	vaddsd	xmm25, xmm12, qword ptr [rbp - 1208]
	vaddsd	xmm12, xmm11, qword ptr [rbp - 568]
	vmulsd	xmm17, xmm14, xmm30
	vmulsd	xmm18, xmm14, xmm31
	vaddsd	xmm10, xmm12, qword ptr [rbp - 1200]
	vaddsd	xmm12, xmm6, qword ptr [rbp - 1152]
	vaddsd	xmm8, xmm1, xmm0
	vmovapd	xmm0, xmmword ptr [rbp - 736]
	vmovapd	xmm1, xmmword ptr [rbp - 1088]
	vaddsd	xmm1, xmm1, qword ptr [rbp - 768]
	vaddsd	xmm0, xmm0, qword ptr [rbp - 584]
	vfmsub231sd	xmm17, xmm20, xmm31     # xmm17 = (xmm20 * xmm31) - xmm17
	vfmadd231sd	xmm18, xmm20, xmm30     # xmm18 = (xmm20 * xmm30) + xmm18
	vaddsd	xmm27, xmm12, qword ptr [rbp - 360]
	vaddsd	xmm12, xmm5, qword ptr [rbp - 1160]
	vaddsd	xmm1, xmm1, qword ptr [rbp - 800]
	vaddsd	xmm1, xmm1, qword ptr [rbp - 1104]
	vmovsd	qword ptr [rbp - 736], xmm10
	vaddsd	xmm10, xmm12, qword ptr [rbp - 352]
	vaddsd	xmm12, xmm4, qword ptr [rbp - 1120]
	vaddsd	xmm1, xmm1, qword ptr [rbp - 752]
	vmovsd	qword ptr [rbp - 800], xmm10
	vaddsd	xmm10, xmm12, qword ptr [rbp - 328]
	vaddsd	xmm12, xmm3, qword ptr [rbp - 1128]
	vmovsd	qword ptr [rbp - 768], xmm10
	vaddsd	xmm10, xmm12, qword ptr [rbp - 320]
	vaddsd	xmm12, xmm2, qword ptr [rbp - 1552]
	vmovsd	qword ptr [rbp - 1104], xmm10
	vaddsd	xmm10, xmm12, qword ptr [rbp - 1440]
	vaddsd	xmm12, xmm8, qword ptr [rbp - 1560]
	vmovsd	qword ptr [rbp - 752], xmm10
	vaddsd	xmm10, xmm12, qword ptr [rbp - 1432]
	vmulsd	xmm12, xmm1, xmm22
	vfmadd231sd	xmm12, xmm0, xmm21      # xmm12 = (xmm0 * xmm21) + xmm12
	vaddsd	xmm13, xmm12, xmm7
	vmulsd	xmm12, xmm14, xmm24
	vfmadd213sd	xmm15, xmm20, xmm12     # xmm15 = (xmm20 * xmm15) + xmm12
	vmovsd	qword ptr [rbp - 632], xmm10
	vfmsub231sd	xmm12, xmm20, xmm26     # xmm12 = (xmm20 * xmm26) - xmm12
	vaddsd	xmm10, xmm13, xmm15
	vmulsd	xmm13, xmm1, xmm21
	vfnmadd231sd	xmm13, xmm0, xmm22      # xmm13 = -(xmm0 * xmm22) + xmm13
	vmovsd	qword ptr [rbp - 624], xmm10
	vaddsd	xmm15, xmm9, xmm13
	vmulsd	xmm13, xmm14, xmm28
	vfmsub213sd	xmm16, xmm20, xmm13     # xmm16 = (xmm20 * xmm16) - xmm13
	vaddsd	xmm10, xmm15, xmm16
	vaddsd	xmm15, xmm29, qword ptr [rbp - 560]
	vmulsd	xmm16, xmm1, xmm24
	vmovsd	qword ptr [rbp - 584], xmm10
	vaddsd	xmm10, xmm15, qword ptr [rbp - 1320]
	vaddsd	xmm15, xmm11, qword ptr [rbp - 552]
	vmovsd	qword ptr [rbp - 608], xmm10
	vaddsd	xmm10, xmm15, qword ptr [rbp - 1312]
	vaddsd	xmm15, xmm6, qword ptr [rbp - 1232]
	vmovsd	qword ptr [rbp - 576], xmm10
	vaddsd	xmm10, xmm15, qword ptr [rbp - 392]
	vaddsd	xmm15, xmm5, qword ptr [rbp - 1224]
	vmovsd	qword ptr [rbp - 600], xmm10
	vaddsd	xmm10, xmm15, qword ptr [rbp - 384]
	vaddsd	xmm15, xmm4, qword ptr [rbp - 1176]
	vmovsd	qword ptr [rbp - 568], xmm10
	vaddsd	xmm10, xmm15, qword ptr [rbp - 344]
	vaddsd	xmm15, xmm3, qword ptr [rbp - 1168]
	vmovsd	qword ptr [rbp - 592], xmm10
	vaddsd	xmm10, xmm15, qword ptr [rbp - 336]
	vaddsd	xmm15, xmm2, qword ptr [rbp - 1600]
	vmovsd	qword ptr [rbp - 560], xmm10
	vaddsd	xmm10, xmm15, qword ptr [rbp - 1504]
	vaddsd	xmm15, xmm8, qword ptr [rbp - 1592]
	vmovsd	qword ptr [rbp - 1088], xmm10
	vaddsd	xmm10, xmm15, qword ptr [rbp - 1496]
	vmovapd	xmm15, xmm26
	vfmadd213sd	xmm15, xmm0, xmm16      # xmm15 = (xmm0 * xmm15) + xmm16
	vfmsub231sd	xmm16, xmm0, xmm26      # xmm16 = (xmm0 * xmm26) - xmm16
	vaddsd	xmm15, xmm15, xmm7
	vmovsd	qword ptr [rbp - 552], xmm10
	vaddsd	xmm10, xmm15, xmm17
	vmulsd	xmm17, xmm1, xmm28
	vmovapd	xmm15, xmm19
	vfmsub213sd	xmm15, xmm0, xmm17      # xmm15 = (xmm0 * xmm15) - xmm17
	vmovsd	qword ptr [rbp - 392], xmm10
	vfmsub231sd	xmm17, xmm0, xmm24      # xmm17 = (xmm0 * xmm24) - xmm17
	vaddsd	xmm15, xmm9, xmm15
	vaddsd	xmm10, xmm15, xmm18
	vaddsd	xmm15, xmm29, qword ptr [rbp - 512]
	vaddsd	xmm15, xmm15, qword ptr [rbp - 1456]
	vmovsd	qword ptr [rbp - 352], xmm10
	vmovsd	qword ptr [rbp - 512], xmm15
	vaddsd	xmm15, xmm11, qword ptr [rbp - 504]
	vaddsd	xmm15, xmm15, qword ptr [rbp - 1448]
	vmovsd	qword ptr [rbp - 384], xmm15
	vaddsd	xmm15, xmm6, qword ptr [rbp - 1376]
	vaddsd	xmm15, xmm15, qword ptr [rbp - 432]
	vmovsd	qword ptr [rbp - 504], xmm15
	vaddsd	xmm15, xmm5, qword ptr [rbp - 1368]
	vaddsd	xmm15, xmm15, qword ptr [rbp - 424]
	vmovsd	qword ptr [rbp - 360], xmm15
	vaddsd	xmm15, xmm4, qword ptr [rbp - 1296]
	vaddsd	xmm15, xmm15, qword ptr [rbp - 376]
	vmovsd	qword ptr [rbp - 432], xmm15
	vaddsd	xmm15, xmm3, qword ptr [rbp - 1288]
	vaddsd	xmm15, xmm15, qword ptr [rbp - 368]
	vmovsd	qword ptr [rbp - 376], xmm15
	vaddsd	xmm15, xmm2, qword ptr [rbp - 1728]
	vaddsd	xmm15, xmm15, qword ptr [rbp - 1544]
	vmovsd	qword ptr [rbp - 424], xmm15
	vaddsd	xmm15, xmm8, qword ptr [rbp - 1736]
	vaddsd	xmm15, xmm15, qword ptr [rbp - 1536]
	vmovsd	qword ptr [rbp - 368], xmm15
	vaddsd	xmm15, xmm7, xmm16
	vmulsd	xmm16, xmm14, xmm22
	vfmadd231sd	xmm16, xmm20, xmm21     # xmm16 = (xmm20 * xmm21) + xmm16
	vaddsd	xmm15, xmm15, xmm16
	vaddsd	xmm16, xmm9, xmm17
	vmovsd	qword ptr [rbp - 344], xmm15
	vaddsd	xmm15, xmm9, xmm1
	vaddsd	xmm18, xmm15, xmm14
	vmulsd	xmm14, xmm14, xmm21
	vmovapd	xmm15, xmm31
	vfnmadd231sd	xmm14, xmm20, xmm22     # xmm14 = -(xmm20 * xmm22) + xmm14
	vaddsd	xmm14, xmm16, xmm14
	vmovsd	qword ptr [rbp - 320], xmm14
	vaddsd	xmm14, xmm29, qword ptr [rbp - 496]
	vaddsd	xmm14, xmm14, qword ptr [rbp - 1512]
	vmovsd	qword ptr [rbp - 496], xmm14
	vaddsd	xmm14, xmm11, qword ptr [rbp - 488]
	vaddsd	xmm14, xmm14, qword ptr [rbp - 1520]
	vmovsd	qword ptr [rbp - 336], xmm14
	vaddsd	xmm14, xmm6, qword ptr [rbp - 1472]
	vaddsd	xmm14, xmm14, qword ptr [rbp - 448]
	vmovsd	qword ptr [rbp - 488], xmm14
	vaddsd	xmm14, xmm5, qword ptr [rbp - 1464]
	vaddsd	xmm14, xmm14, qword ptr [rbp - 440]
	vmovsd	qword ptr [rbp - 328], xmm14
	vaddsd	xmm14, xmm4, qword ptr [rbp - 1416]
	vaddsd	xmm14, xmm14, qword ptr [rbp - 408]
	vmovsd	qword ptr [rbp - 448], xmm14
	vaddsd	xmm14, xmm3, qword ptr [rbp - 1424]
	vaddsd	xmm14, xmm14, qword ptr [rbp - 400]
	vmovsd	qword ptr [rbp - 408], xmm14
	vaddsd	xmm14, xmm2, qword ptr [rbp - 1584]
	vaddsd	xmm14, xmm14, qword ptr [rbp - 784]
	vmovsd	qword ptr [rbp - 440], xmm14
	vaddsd	xmm14, xmm8, qword ptr [rbp - 1576]
	vaddsd	xmm14, xmm14, qword ptr [rbp - 1072]
	vmovsd	qword ptr [rbp - 400], xmm14
	vmulsd	xmm14, xmm1, xmm30
	vmulsd	xmm1, xmm1, xmm31
	vfmadd231sd	xmm1, xmm30, xmm0       # xmm1 = (xmm30 * xmm0) + xmm1
	vfmsub231sd	xmm14, xmm0, xmm31      # xmm14 = (xmm0 * xmm31) - xmm14
	vaddsd	xmm0, xmm7, xmm0
	vaddsd	xmm1, xmm9, xmm1
	vmovapd	xmm9, xmm20
	vfmsub213sd	xmm9, xmm24, xmm13      # xmm9 = (xmm24 * xmm9) - xmm13
	vaddsd	xmm7, xmm14, xmm7
	vaddsd	xmm31, xmm7, xmm12
	vaddsd	xmm7, xmm0, xmm20
	vmovsd	xmm20, qword ptr [rbp - 1216]
	vaddsd	xmm0, xmm9, xmm1
	vaddsd	xmm1, xmm11, qword ptr [rbp - 472]
	vmovapd	xmm11, xmm24
	vmovsd	qword ptr [rbp - 784], xmm0
	vaddsd	xmm0, xmm29, qword ptr [rbp - 480]
	vmovapd	xmm29, xmm19
	vaddsd	xmm0, xmm0, qword ptr [rbp - 1568]
	vmovsd	qword ptr [rbp - 480], xmm0
	vaddsd	xmm0, xmm1, qword ptr [rbp - 48]
	vaddsd	xmm1, xmm5, qword ptr [rbp - 1528]
	vmulsd	xmm5, xmm20, xmm26
	vmovsd	qword ptr [rbp - 48], xmm0
	vaddsd	xmm0, xmm6, qword ptr [rbp - 40]
	vaddsd	xmm0, xmm0, qword ptr [rbp - 464]
	vmovsd	qword ptr [rbp - 40], xmm0
	vaddsd	xmm0, xmm1, qword ptr [rbp - 456]
	vaddsd	xmm1, xmm3, qword ptr [rbp - 1480]
	vmovsd	qword ptr [rbp - 472], xmm0
	vaddsd	xmm0, xmm4, qword ptr [rbp - 1488]
	vaddsd	xmm0, xmm0, qword ptr [rbp - 416]
	vmovsd	qword ptr [rbp - 464], xmm0
	vaddsd	xmm0, xmm1, qword ptr [rbp - 1776]
	vmovapd	xmm1, xmmword ptr [rbp - 944]
	vmovsd	qword ptr [rbp - 456], xmm0
	vaddsd	xmm0, xmm2, xmm1
	vshufpd	xmm1, xmm1, xmm1, 1             # xmm1 = xmm1[1,0]
	vaddsd	xmm0, xmm0, xmm1
	vmovapd	xmm1, xmmword ptr [rbp - 960]
	vmovsd	qword ptr [rbp - 416], xmm0
	vaddsd	xmm0, xmm8, xmm1
	vshufpd	xmm1, xmm1, xmm1, 1             # xmm1 = xmm1[1,0]
	vaddsd	xmm0, xmm0, xmm1
	vmovsd	qword ptr [rbp - 616], xmm0
	vmovapd	xmm0, xmmword ptr [rbp - 848]
	vaddsd	xmm0, xmm0, qword ptr [rbp - 1824]
	vaddsd	xmm0, xmm0, qword ptr [rbp - 1808]
	vaddsd	xmm0, xmm0, qword ptr [rbp - 1792]
	vaddsd	xmm6, xmm0, qword ptr [rbp - 832]
	vmovapd	xmm0, xmmword ptr [rbp - 1840]
	vaddsd	xmm0, xmm0, qword ptr [rbp - 720]
	vaddsd	xmm0, xmm0, qword ptr [rbp - 1872]
	vaddsd	xmm0, xmm0, qword ptr [rbp - 1856]
	vaddsd	xmm3, xmm0, qword ptr [rbp - 816]
	vmovsd	xmm0, qword ptr [rbp - 32]
	vaddsd	xmm7, xmm7, xmm6
	vaddsd	xmm0, xmm0, qword ptr [rbp - 672]
	vaddsd	xmm7, xmm7, xmm20
	vaddsd	xmm0, xmm0, qword ptr [rbp - 1280]
	vmovsd	qword ptr [rbp - 720], xmm7
	vaddsd	xmm0, xmm0, qword ptr [rbp - 1240]
	vaddsd	xmm8, xmm0, qword ptr [rbp - 656]
	vaddsd	xmm12, xmm18, xmm3
	vmulsd	xmm2, xmm3, xmm30
	vmulsd	xmm4, xmm3, xmm28
	vmulsd	xmm1, xmm15, xmm3
	vmulsd	xmm13, xmm3, xmm22
	vmulsd	xmm14, xmm3, xmm21
	vfmsub231sd	xmm2, xmm6, xmm15       # xmm2 = (xmm6 * xmm15) - xmm2
	vfmsub213sd	xmm11, xmm6, xmm4       # xmm11 = (xmm6 * xmm11) - xmm4
	vfmsub213sd	xmm29, xmm6, xmm4       # xmm29 = (xmm6 * xmm29) - xmm4
	vfmadd231sd	xmm1, xmm6, xmm30       # xmm1 = (xmm6 * xmm30) + xmm1
	vfmadd231sd	xmm13, xmm6, xmm21      # xmm13 = (xmm6 * xmm21) + xmm13
	vfnmadd231sd	xmm14, xmm6, xmm22      # xmm14 = -(xmm6 * xmm22) + xmm14
	vmulsd	xmm6, xmm6, xmm26
	vaddsd	xmm1, xmm1, qword ptr [rbp - 320]
	vaddsd	xmm2, xmm2, qword ptr [rbp - 344]
	vaddsd	xmm7, xmm12, xmm8
	vmulsd	xmm9, xmm8, xmm30
	vmulsd	xmm10, xmm8, xmm15
	vmulsd	xmm4, xmm8, xmm22
	vmulsd	xmm0, xmm8, xmm28
	vmulsd	xmm28, xmm3, xmm24
	vmulsd	xmm3, xmm8, xmm21
	vmovsd	qword ptr [rbp - 56], xmm7
	vaddsd	xmm7, xmm25, qword ptr [rbp - 1272]
	vfmsub231sd	xmm9, xmm20, xmm15      # xmm9 = (xmm20 * xmm15) - xmm9
	vmulsd	xmm15, xmm8, xmm24
	vmovsd	xmm8, qword ptr [rbp - 736]
	vfmadd231sd	xmm10, xmm20, xmm30     # xmm10 = (xmm20 * xmm30) + xmm10
	vfmsub213sd	xmm24, xmm20, xmm0      # xmm24 = (xmm20 * xmm24) - xmm0
	vfmsub231sd	xmm0, xmm20, xmm19      # xmm0 = (xmm20 * xmm19) - xmm0
	vfnmadd231sd	xmm3, xmm20, xmm22      # xmm3 = -(xmm20 * xmm22) + xmm3
	vfmadd231sd	xmm4, xmm20, xmm21      # xmm4 = (xmm20 * xmm21) + xmm4
	vaddsd	xmm8, xmm8, qword ptr [rbp - 1264]
	vaddsd	xmm7, xmm7, qword ptr [rbp - 1760]
	vaddsd	xmm16, xmm1, xmm0
	vmovsd	xmm0, qword ptr [rbp - 496]
	vmovsd	xmm1, qword ptr [rbp - 336]
	vaddsd	xmm0, xmm0, qword ptr [rbp - 144]
	vaddsd	xmm1, xmm1, qword ptr [rbp - 128]
	vmovsd	qword ptr [rbp - 736], xmm7
	vaddsd	xmm7, xmm8, qword ptr [rbp - 1752]
	vmovsd	xmm8, qword ptr [rbp - 800]
	vaddsd	xmm8, xmm8, qword ptr [rbp - 1608]
	vmovsd	qword ptr [rbp - 32], xmm7
	vaddsd	xmm7, xmm27, qword ptr [rbp - 1616]
	vaddsd	xmm7, xmm7, qword ptr [rbp - 1400]
	vmovsd	qword ptr [rbp - 672], xmm7
	vaddsd	xmm7, xmm8, qword ptr [rbp - 1384]
	vmovsd	xmm8, qword ptr [rbp - 1104]
	vaddsd	xmm8, xmm8, qword ptr [rbp - 1656]
	vmovsd	qword ptr [rbp - 656], xmm7
	vmovsd	xmm7, qword ptr [rbp - 768]
	vaddsd	xmm7, xmm7, qword ptr [rbp - 1664]
	vaddsd	xmm7, xmm7, qword ptr [rbp - 1256]
	vmovsd	qword ptr [rbp - 768], xmm7
	vaddsd	xmm7, xmm8, qword ptr [rbp - 1248]
	vmovsd	xmm8, qword ptr [rbp - 632]
	vaddsd	xmm8, xmm8, qword ptr [rbp - 1136]
	vmovsd	qword ptr [rbp - 96], xmm7
	vmovsd	xmm7, qword ptr [rbp - 752]
	vaddsd	xmm7, xmm7, qword ptr [rbp - 1144]
	vaddsd	xmm7, xmm7, qword ptr [rbp - 1680]
	vmovsd	qword ptr [rbp - 752], xmm7
	vaddsd	xmm7, xmm8, qword ptr [rbp - 1672]
	vaddsd	xmm8, xmm11, qword ptr [rbp - 584]
	vmovsd	qword ptr [rbp - 88], xmm7
	vsubsd	xmm7, xmm6, xmm28
	vaddsd	xmm7, xmm7, qword ptr [rbp - 624]
	vaddsd	xmm6, xmm28, xmm6
	vaddsd	xmm6, xmm31, xmm6
	vaddsd	xmm4, xmm6, xmm4
	vaddsd	xmm7, xmm9, xmm7
	vsubsd	xmm9, xmm5, xmm15
	vaddsd	xmm5, xmm15, xmm5
	vaddsd	xmm15, xmm29, qword ptr [rbp - 784]
	vmovapd	xmm29, xmmword ptr [rbp - 864]
	vmovsd	qword ptr [rbp - 80], xmm7
	vaddsd	xmm7, xmm8, xmm10
	vmovsd	xmm8, qword ptr [rbp - 576]
	vaddsd	xmm17, xmm2, xmm5
	vaddsd	xmm8, xmm8, qword ptr [rbp - 1184]
	vmovsd	qword ptr [rbp - 72], xmm7
	vmovsd	xmm7, qword ptr [rbp - 608]
	vaddsd	xmm7, xmm7, qword ptr [rbp - 1192]
	vshufpd	xmm31, xmm29, xmm29, 1          # xmm31 = xmm29[1,0]
	vaddsd	xmm7, xmm7, qword ptr [rbp - 280]
	vaddsd	xmm5, xmm15, xmm3
	vmovsd	qword ptr [rbp - 64], xmm7
	vaddsd	xmm7, xmm8, qword ptr [rbp - 272]
	vmovsd	xmm8, qword ptr [rbp - 568]
	vaddsd	xmm8, xmm8, qword ptr [rbp - 688]
	vmovsd	qword ptr [rbp - 848], xmm7
	vmovsd	xmm7, qword ptr [rbp - 600]
	vaddsd	xmm7, xmm7, qword ptr [rbp - 704]
	vaddsd	xmm7, xmm7, qword ptr [rbp - 1408]
	vmovsd	qword ptr [rbp - 704], xmm7
	vaddsd	xmm7, xmm8, qword ptr [rbp - 1392]
	vmovsd	xmm8, qword ptr [rbp - 560]
	vaddsd	xmm8, xmm8, qword ptr [rbp - 1648]
	vmovsd	qword ptr [rbp - 688], xmm7
	vmovsd	xmm7, qword ptr [rbp - 592]
	vaddsd	xmm7, xmm7, qword ptr [rbp - 1640]
	vaddsd	xmm7, xmm7, qword ptr [rbp - 1352]
	vmovsd	qword ptr [rbp - 832], xmm7
	vaddsd	xmm7, xmm8, qword ptr [rbp - 1336]
	vmovsd	xmm8, qword ptr [rbp - 552]
	vaddsd	xmm8, xmm8, qword ptr [rbp - 912]
	vmovsd	qword ptr [rbp - 816], xmm7
	vmovsd	xmm7, qword ptr [rbp - 1088]
	vaddsd	xmm7, xmm7, qword ptr [rbp - 176]
	vaddsd	xmm30, xmm7, qword ptr [rbp - 1056]
	vaddsd	xmm7, xmm8, qword ptr [rbp - 1040]
	vaddsd	xmm8, xmm14, qword ptr [rbp - 352]
	vaddsd	xmm14, xmm1, qword ptr [rbp - 304]
	vmovsd	xmm1, qword ptr [rbp - 328]
	vaddsd	xmm1, xmm1, qword ptr [rbp - 1712]
	vaddsd	xmm12, xmm1, qword ptr [rbp - 1304]
	vmovsd	xmm1, qword ptr [rbp - 408]
	vaddsd	xmm1, xmm1, qword ptr [rbp - 1704]
	vaddsd	xmm10, xmm1, qword ptr [rbp - 520]
	vmovsd	xmm1, qword ptr [rbp - 400]
	vaddsd	xmm1, xmm1, qword ptr [rbp - 104]
	vmovsd	qword ptr [rbp - 800], xmm7
	vaddsd	xmm7, xmm13, qword ptr [rbp - 392]
	vaddsd	xmm13, xmm0, qword ptr [rbp - 312]
	vmovsd	xmm0, qword ptr [rbp - 488]
	vaddsd	xmm27, xmm8, xmm24
	vmovsd	xmm8, qword ptr [rbp - 384]
	vaddsd	xmm0, xmm0, qword ptr [rbp - 1720]
	vaddsd	xmm8, xmm8, qword ptr [rbp - 184]
	vaddsd	xmm11, xmm0, qword ptr [rbp - 1328]
	vmovsd	xmm0, qword ptr [rbp - 448]
	vaddsd	xmm24, xmm8, qword ptr [rbp - 288]
	vmovsd	xmm8, qword ptr [rbp - 360]
	vaddsd	xmm0, xmm0, qword ptr [rbp - 208]
	vaddsd	xmm8, xmm8, qword ptr [rbp - 1008]
	vaddsd	xmm23, xmm8, qword ptr [rbp - 1344]
	vmovsd	xmm8, qword ptr [rbp - 376]
	vaddsd	xmm8, xmm8, qword ptr [rbp - 976]
	vaddsd	xmm26, xmm7, xmm9
	vmovsd	xmm7, qword ptr [rbp - 512]
	vaddsd	xmm9, xmm0, qword ptr [rbp - 528]
	vmovsd	xmm0, qword ptr [rbp - 440]
	vaddsd	xmm7, xmm7, qword ptr [rbp - 192]
	vaddsd	xmm0, xmm0, qword ptr [rbp - 112]
	vaddsd	xmm21, xmm8, qword ptr [rbp - 200]
	vmovsd	xmm8, qword ptr [rbp - 368]
	vaddsd	xmm8, xmm8, qword ptr [rbp - 120]
	vaddsd	xmm25, xmm7, qword ptr [rbp - 296]
	vmovsd	xmm7, qword ptr [rbp - 504]
	vaddsd	xmm7, xmm7, qword ptr [rbp - 1024]
	vaddsd	xmm19, xmm8, qword ptr [rbp - 1688]
	vaddsd	xmm8, xmm1, qword ptr [rbp - 240]
	vmovsd	xmm1, qword ptr [rbp - 456]
	vaddsd	xmm1, xmm1, qword ptr [rbp - 224]
	vaddsd	xmm22, xmm7, qword ptr [rbp - 1360]
	vmovsd	xmm7, qword ptr [rbp - 432]
	vaddsd	xmm7, xmm7, qword ptr [rbp - 992]
	vaddsd	xmm1, xmm1, qword ptr [rbp - 536]
	vaddsd	xmm20, xmm7, qword ptr [rbp - 216]
	vmovsd	xmm7, qword ptr [rbp - 424]
	vaddsd	xmm7, xmm7, qword ptr [rbp - 136]
	vaddsd	xmm18, xmm7, qword ptr [rbp - 1696]
	vaddsd	xmm7, xmm0, qword ptr [rbp - 248]
	vmovsd	xmm0, qword ptr [rbp - 480]
	vaddsd	xmm6, xmm0, qword ptr [rbp - 928]
	vmovsd	xmm0, qword ptr [rbp - 48]
	vaddsd	xmm15, xmm0, qword ptr [rbp - 168]
	vmovsd	xmm0, qword ptr [rbp - 40]
	vaddsd	xmm28, xmm0, qword ptr [rbp - 264]
	vmovsd	xmm0, qword ptr [rbp - 472]
	vaddsd	xmm0, xmm0, qword ptr [rbp - 256]
	vaddsd	xmm6, xmm6, qword ptr [rbp - 1768]
	vaddsd	xmm15, xmm15, qword ptr [rbp - 1744]
	vaddsd	xmm28, xmm28, qword ptr [rbp - 160]
	vaddsd	xmm3, xmm0, qword ptr [rbp - 152]
	vmovsd	xmm0, qword ptr [rbp - 464]
	vaddsd	xmm0, xmm0, qword ptr [rbp - 232]
	vaddsd	xmm2, xmm0, qword ptr [rbp - 544]
	vaddsd	xmm0, xmm29, qword ptr [rbp - 416]
	vmovapd	xmm29, xmmword ptr [rbp - 896]
	vaddsd	xmm0, xmm0, xmm31
	vaddsd	xmm31, xmm29, qword ptr [rbp - 616]
	vshufpd	xmm29, xmm29, xmm29, 1          # xmm29 = xmm29[1,0]
	vaddsd	xmm29, xmm31, xmm29
	vmovsd	xmm31, qword ptr [rbp - 720]
	vmovsd	qword ptr [rcx], xmm31
	vmovsd	xmm31, qword ptr [rbp - 56]
	mov	rax, qword ptr [rax]
	vmovsd	qword ptr [rax], xmm31
	vmovsd	xmm31, qword ptr [rbp - 736]
	vmovsd	qword ptr [rcx + 8], xmm31
	vmovsd	xmm31, qword ptr [rbp - 32]
	vmovsd	qword ptr [rax + 8], xmm31
	vmovsd	xmm31, qword ptr [rbp - 672]
	vmovsd	qword ptr [rcx + 16], xmm31
	vmovsd	xmm31, qword ptr [rbp - 656]
	vmovsd	qword ptr [rax + 16], xmm31
	vmovsd	xmm31, qword ptr [rbp - 768]
	vmovsd	qword ptr [rcx + 24], xmm31
	vmovsd	xmm31, qword ptr [rbp - 96]
	vmovsd	qword ptr [rax + 24], xmm31
	vmovsd	xmm31, qword ptr [rbp - 752]
	vmovsd	qword ptr [rcx + 32], xmm31
	vmovsd	xmm31, qword ptr [rbp - 88]
	vmovsd	qword ptr [rax + 32], xmm31
	vmovsd	xmm31, qword ptr [rbp - 80]
	vmovsd	qword ptr [rcx + 40], xmm31
	vmovsd	xmm31, qword ptr [rbp - 72]
	vmovsd	qword ptr [rax + 40], xmm31
	vmovsd	xmm31, qword ptr [rbp - 64]
	vmovsd	qword ptr [rcx + 48], xmm31
	vmovsd	xmm31, qword ptr [rbp - 848]
	vmovsd	qword ptr [rax + 48], xmm31
	vmovsd	xmm31, qword ptr [rbp - 704]
	vmovsd	qword ptr [rcx + 56], xmm31
	vmovsd	xmm31, qword ptr [rbp - 688]
	vmovsd	qword ptr [rax + 56], xmm31
	vmovsd	xmm31, qword ptr [rbp - 832]
	vmovsd	qword ptr [rcx + 64], xmm31
	vmovsd	xmm31, qword ptr [rbp - 816]
	vmovsd	qword ptr [rax + 64], xmm31
	vmovsd	xmm31, qword ptr [rbp - 800]
	vmovsd	qword ptr [rcx + 72], xmm30
	vmovsd	qword ptr [rax + 72], xmm31
	vmovsd	qword ptr [rcx + 80], xmm26
	vmovsd	qword ptr [rax + 80], xmm27
	vmovsd	qword ptr [rcx + 88], xmm25
	vmovsd	qword ptr [rax + 88], xmm24
	vmovsd	qword ptr [rcx + 96], xmm22
	vmovsd	qword ptr [rax + 96], xmm23
	vmovsd	qword ptr [rcx + 104], xmm20
	vmovsd	qword ptr [rax + 104], xmm21
	vmovsd	qword ptr [rcx + 112], xmm18
	vmovsd	qword ptr [rax + 112], xmm19
	vmovsd	qword ptr [rcx + 120], xmm17
	vmovsd	qword ptr [rax + 120], xmm16
	vmovsd	qword ptr [rcx + 128], xmm13
	vmovsd	qword ptr [rax + 128], xmm14
	vmovsd	qword ptr [rcx + 136], xmm11
	vmovsd	qword ptr [rax + 136], xmm12
	vmovsd	qword ptr [rcx + 144], xmm9
	vmovsd	qword ptr [rax + 144], xmm10
	vmovsd	qword ptr [rcx + 152], xmm7
	vmovsd	qword ptr [rax + 152], xmm8
	vmovsd	qword ptr [rcx + 160], xmm4
	vmovsd	qword ptr [rax + 160], xmm5
	vmovsd	qword ptr [rcx + 168], xmm6
	vmovsd	qword ptr [rax + 168], xmm15
	vmovsd	qword ptr [rcx + 176], xmm28
	vmovsd	qword ptr [rax + 176], xmm3
	vmovsd	qword ptr [rcx + 184], xmm2
	vmovsd	qword ptr [rax + 184], xmm1
	vmovsd	qword ptr [rcx + 192], xmm0
	movabs	rcx, offset jl_nothing
	vmovsd	qword ptr [rax + 192], xmm29
	mov	rax, qword ptr [rcx]
	add	rsp, 1736
	pop	rbx
	pop	r14
	pop	r15
	pop	rbp
	ret
