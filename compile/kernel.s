	.text
	.file	"LLVMDialectModule"
	.globl	matmul_kernel_0d1d2d34567c89c1011c # -- Begin function matmul_kernel_0d1d2d34567c89c1011c
	.p2align	4, 0x90
	.type	matmul_kernel_0d1d2d34567c89c1011c,@function
matmul_kernel_0d1d2d34567c89c1011c:     # @matmul_kernel_0d1d2d34567c89c1011c
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%rbx
	subq	$232, %rsp
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	movq	%r9, -184(%rbp)                 # 8-byte Spill
	movq	%rcx, -128(%rbp)                # 8-byte Spill
	movq	%rsi, -120(%rbp)                # 8-byte Spill
	movl	88(%rbp), %ebx
	movl	40(%rbp), %r15d
	movl	24(%rbp), %r12d
	movl	16(%rbp), %r14d
	movl	$8256, %edi                     # imm = 0x2040
	callq	malloc@PLT
	movq	%rax, %rcx
	leaq	63(%rax), %r13
	andq	$-64, %r13
	xorl	%eax, %eax
	movq	%r13, %rdx
	jmp	.LBB0_1
	.p2align	4, 0x90
.LBB0_5:                                #   in Loop: Header=BB0_1 Depth=1
	incq	%rax
	addq	$256, %rdx                      # imm = 0x100
.LBB0_1:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_4 Depth 2
	cmpq	$31, %rax
	jg	.LBB0_6
# %bb.2:                                #   in Loop: Header=BB0_1 Depth=1
	xorl	%esi, %esi
	cmpq	$63, %rsi
	jg	.LBB0_5
	.p2align	4, 0x90
.LBB0_4:                                #   Parent Loop BB0_1 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movl	$0, (%rdx,%rsi,4)
	incq	%rsi
	cmpq	$63, %rsi
	jle	.LBB0_4
	jmp	.LBB0_5
.LBB0_6:
	leal	31(%r14), %eax
	leal	62(%r14), %r8d
	testl	%eax, %eax
	cmovnsl	%eax, %r8d
	sarl	$5, %r8d
	leal	63(%r12), %eax
	leal	126(%r12), %esi
	testl	%eax, %eax
	cmovnsl	%eax, %esi
	sarl	$6, %esi
	shll	$3, %esi
	movl	%ebx, %eax
	cltd
	idivl	%esi
	movl	%edx, %esi
	movl	%eax, %edi
	leal	(,%rdi,8), %eax
	subl	%eax, %r8d
	cmpl	$8, %r8d
	movl	$8, %r9d
	cmovll	%r8d, %r9d
	movl	%ebx, %eax
	cltd
	idivl	%r9d
                                        # kill: def $edx killed $edx def $rdx
	leal	(%rdx,%rdi,8), %edi
	movl	%esi, %eax
	cltd
	idivl	%r9d
	movl	%eax, %esi
	shll	$5, %edi
	shll	$6, %esi
	movslq	%r14d, %r8
	movslq	%edi, %r9
	movslq	%r15d, %rdi
	movq	%r9, -168(%rbp)                 # 8-byte Spill
	imulq	%rdi, %r9
	movq	%r9, %rax
	cqto
	idivq	%rdi
	movq	%rdx, %rbx
	movq	%r8, -176(%rbp)                 # 8-byte Spill
	imulq	%rdi, %r8
	movq	%r8, -200(%rbp)                 # 8-byte Spill
	leaq	(%r8,%rdx), %rax
	movq	%r9, -72(%rbp)                  # 8-byte Spill
	subq	%r9, %rax
	cqto
	movq	%rdi, -80(%rbp)                 # 8-byte Spill
	idivq	%rdi
	movq	%rax, -64(%rbp)                 # 8-byte Spill
	movq	-120(%rbp), %rdi                # 8-byte Reload
	movq	(%rdi), %r15
	movslq	%r12d, %r8
	movslq	%esi, %r14
	movq	%r14, %rax
	cqto
	movq	8(%rdi), %rdi
	movq	%rdi, -48(%rbp)                 # 8-byte Spill
	idivq	%r8
	movq	%r14, %r10
	subq	%rdx, %r10
	leaq	64(%rdx), %r9
	cmpq	%r8, %r9
	movq	%r8, -112(%rbp)                 # 8-byte Spill
	cmovgeq	%r8, %r9
	subq	%rdx, %r9
	movl	32(%rbp), %eax
	movq	%rax, %rdx
	addl	$15, %eax
	testl	%eax, %eax
	leal	30(%rdx), %edx
	cmovnsl	%eax, %edx
	movslq	48(%rbp), %rax
	sarl	$4, %edx
	movl	%edx, -92(%rbp)                 # 4-byte Spill
	movq	%rax, -136(%rbp)                # 8-byte Spill
                                        # kill: def $eax killed $eax killed $rax
	shll	$4, %eax
	cltq
	movq	%rax, -192(%rbp)                # 8-byte Spill
	movl	$1, %eax
	movq	%rax, -56(%rbp)                 # 8-byte Spill
	movq	-128(%rbp), %rax                # 8-byte Reload
	movq	(%rax), %rdi
	movq	8(%rax), %r11
	xorl	%edx, %edx
	movl	$16, %r12d
	movq	%r14, %rax
	movq	%r14, -104(%rbp)                # 8-byte Spill
	xorl	%r8d, %r8d
	jmp	.LBB0_7
	.p2align	4, 0x90
.LBB0_43:                               #   in Loop: Header=BB0_7 Depth=1
	movq	-72(%rbp), %r8                  # 8-byte Reload
	addq	$16, %r8
	movq	%r8, %rax
	cqto
	movq	-80(%rbp), %rdi                 # 8-byte Reload
	idivq	%rdi
	movq	%rdx, %rbx
	movq	-200(%rbp), %rax                # 8-byte Reload
	addq	%rdx, %rax
	movq	%r8, -72(%rbp)                  # 8-byte Spill
	subq	%r8, %rax
	cqto
	idivq	%rdi
	movq	%rax, -64(%rbp)                 # 8-byte Spill
	movq	-120(%rbp), %rax                # 8-byte Reload
	movq	(%rax), %r15
	movq	8(%rax), %rax
	movq	%rax, -48(%rbp)                 # 8-byte Spill
	movl	$1, %eax
	movq	%rax, -56(%rbp)                 # 8-byte Spill
	movl	$16, %r12d
	movq	-256(%rbp), %r8                 # 8-byte Reload
	addq	-192(%rbp), %r8                 # 8-byte Folded Reload
	movq	-104(%rbp), %rax                # 8-byte Reload
	leaq	(%r8,%rax), %r14
	movq	%r14, %rax
	cqto
	movq	-112(%rbp), %rdi                # 8-byte Reload
	idivq	%rdi
	movq	%r14, %r10
	subq	%rdx, %r10
	leaq	64(%rdx), %r9
	cmpq	%rdi, %r9
	cmovgeq	%rdi, %r9
	subq	%rdx, %r9
	movq	-128(%rbp), %rax                # 8-byte Reload
	movq	(%rax), %rdi
	movq	8(%rax), %r11
	movq	-264(%rbp), %rdx                # 8-byte Reload
	incl	%edx
	movq	%rsi, %r13
.LBB0_7:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_10 Depth 2
                                        #       Child Loop BB0_13 Depth 3
                                        #     Child Loop BB0_17 Depth 2
                                        #       Child Loop BB0_20 Depth 3
                                        #     Child Loop BB0_23 Depth 2
                                        #       Child Loop BB0_26 Depth 3
                                        #     Child Loop BB0_29 Depth 2
                                        #       Child Loop BB0_31 Depth 3
                                        #         Child Loop BB0_34 Depth 4
                                        #     Child Loop BB0_38 Depth 2
                                        #       Child Loop BB0_41 Depth 3
	cmpl	-92(%rbp), %edx                 # 4-byte Folded Reload
	jge	.LBB0_44
# %bb.8:                                #   in Loop: Header=BB0_7 Depth=1
	movq	%r15, -144(%rbp)                # 8-byte Spill
	movq	%r14, -224(%rbp)                # 8-byte Spill
	movq	%rbx, -232(%rbp)                # 8-byte Spill
	movq	%r11, -152(%rbp)                # 8-byte Spill
	movq	%rdi, -160(%rbp)                # 8-byte Spill
	movq	%r10, -240(%rbp)                # 8-byte Spill
	movq	%r9, -248(%rbp)                 # 8-byte Spill
	movq	%r8, -256(%rbp)                 # 8-byte Spill
	movq	%rdx, -264(%rbp)                # 8-byte Spill
	movl	%edx, %eax
	shll	$4, %eax
	movl	32(%rbp), %ecx
	movl	%ecx, %r14d
	subl	%eax, %r14d
	movl	$2048, %edi                     # imm = 0x800
	callq	malloc@PLT
	movq	%rax, %rbx
	movslq	%r14d, %rax
	cmpq	$16, %rax
	movl	$16, %r15d
	movq	%rax, -88(%rbp)                 # 8-byte Spill
	cmovlq	%rax, %r15
	cmpq	$15, %r15
	jg	.LBB0_15
# %bb.9:                                #   in Loop: Header=BB0_7 Depth=1
	movq	%rbx, %rax
	xorl	%ecx, %ecx
	jmp	.LBB0_10
	.p2align	4, 0x90
.LBB0_14:                               #   in Loop: Header=BB0_10 Depth=2
	incq	%rcx
	addq	$64, %rax
.LBB0_10:                               #   Parent Loop BB0_7 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB0_13 Depth 3
	cmpq	$31, %rcx
	jg	.LBB0_15
# %bb.11:                               #   in Loop: Header=BB0_10 Depth=2
	xorl	%edx, %edx
	cmpq	$15, %rdx
	jg	.LBB0_14
	.p2align	4, 0x90
.LBB0_13:                               #   Parent Loop BB0_7 Depth=1
                                        #     Parent Loop BB0_10 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	movl	$0, (%rax,%rdx,4)
	incq	%rdx
	cmpq	$15, %rdx
	jle	.LBB0_13
	jmp	.LBB0_14
	.p2align	4, 0x90
.LBB0_15:                               #   in Loop: Header=BB0_7 Depth=1
	movq	%rsp, %rax
	leaq	-16(%rax), %rsp
	movq	%rsp, -208(%rbp)                # 8-byte Spill
	movq	%r12, -216(%rbp)                # 8-byte Spill
	movq	%r12, -8(%rax)
	movq	-64(%rbp), %rsi                 # 8-byte Reload
	movq	%rsi, -16(%rax)
	cmpq	$32, %rsi
	movl	$32, %eax
	cmovgeq	%rax, %rsi
	movl	$32, %r12d
	subq	%rsi, %r12
	movq	%rsi, %r14
	shlq	$4, %r14
	movq	%rsp, %rax
	leaq	-64(%rax), %rcx
	movq	%rcx, %rsp
	movq	-56(%rbp), %rdx                 # 8-byte Reload
	movq	%rdx, -16(%rax)
	movq	-80(%rbp), %rdx                 # 8-byte Reload
	movq	%rdx, -24(%rax)
	movq	%r15, -32(%rax)
	movq	%rsi, -40(%rax)
	movq	-72(%rbp), %rdx                 # 8-byte Reload
	movq	%rdx, -48(%rax)
	movq	-48(%rbp), %rdx                 # 8-byte Reload
	movq	%rdx, -56(%rax)
	movq	-144(%rbp), %rdx                # 8-byte Reload
	movq	%rdx, -64(%rax)
	movq	%rsp, %rax
	leaq	-64(%rax), %rdi
	movq	%rdi, %rsp
	movq	%r15, -32(%rax)
	movq	%rsi, -40(%rax)
	movq	%rbx, -56(%rax)
	movq	%rbx, -64(%rax)
	movq	$1, -16(%rax)
	movq	$16, -24(%rax)
	movq	$0, -48(%rax)
	movq	%rsp, %rax
	leaq	-16(%rax), %rsi
	movq	%rsi, %rsp
	movq	%rcx, -8(%rax)
	movq	$2, -16(%rax)
	movq	%rsp, %rax
	leaq	-16(%rax), %rdx
	movq	%rdx, %rsp
	movq	%rdi, -8(%rax)
	movq	$2, -16(%rax)
	movl	$4, %edi
	callq	memrefCopy@PLT
	movq	-208(%rbp), %rsp                # 8-byte Reload
	movq	%rsp, -64(%rbp)                 # 8-byte Spill
	movq	%rsp, %rax
	leaq	-64(%rax), %rcx
	movq	%rcx, %rsp
	movq	-56(%rbp), %rdx                 # 8-byte Reload
	movq	%rdx, -16(%rax)
	movq	-80(%rbp), %rdx                 # 8-byte Reload
	movq	%rdx, -24(%rax)
	movq	%r15, -32(%rax)
	movq	%r12, -40(%rax)
	movq	-232(%rbp), %rdx                # 8-byte Reload
	movq	%rdx, -48(%rax)
	movq	-48(%rbp), %rdx                 # 8-byte Reload
	movq	%rdx, -56(%rax)
	movq	-144(%rbp), %rdx                # 8-byte Reload
	movq	%rdx, -64(%rax)
	movq	%rsp, %rax
	leaq	-64(%rax), %rdi
	movq	%rdi, %rsp
	movq	%r15, -32(%rax)
	movq	%r12, -40(%rax)
	movq	%r14, -48(%rax)
	movq	%rbx, -56(%rax)
	movq	%rbx, -64(%rax)
	movq	$1, -16(%rax)
	movq	$16, -24(%rax)
	movq	%rsp, %rax
	leaq	-16(%rax), %rsi
	movq	%rsi, %rsp
	movq	%rcx, -8(%rax)
	movq	$2, -16(%rax)
	movq	%rsp, %rax
	leaq	-16(%rax), %rdx
	movq	%rdx, %rsp
	movq	%rdi, -8(%rax)
	movq	$2, -16(%rax)
	movl	$4, %edi
	callq	memrefCopy@PLT
	movq	-64(%rbp), %rsp                 # 8-byte Reload
	movl	$4096, %edi                     # imm = 0x1000
	callq	malloc@PLT
	movq	%rax, %r15
	movq	-88(%rbp), %rsi                 # 8-byte Reload
	cmpq	$16, %rsi
	movl	$16, %eax
	cmovgeq	%rax, %rsi
	cmpq	$15, %rsi
	jg	.LBB0_22
# %bb.16:                               #   in Loop: Header=BB0_7 Depth=1
	movq	%r15, %rax
	xorl	%ecx, %ecx
	jmp	.LBB0_17
	.p2align	4, 0x90
.LBB0_21:                               #   in Loop: Header=BB0_17 Depth=2
	incq	%rcx
	addq	$256, %rax                      # imm = 0x100
.LBB0_17:                               #   Parent Loop BB0_7 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB0_20 Depth 3
	cmpq	$15, %rcx
	jg	.LBB0_22
# %bb.18:                               #   in Loop: Header=BB0_17 Depth=2
	xorl	%edx, %edx
	cmpq	$63, %rdx
	jg	.LBB0_21
	.p2align	4, 0x90
.LBB0_20:                               #   Parent Loop BB0_7 Depth=1
                                        #     Parent Loop BB0_17 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	movl	$0, (%rax,%rdx,4)
	incq	%rdx
	cmpq	$63, %rdx
	jle	.LBB0_20
	jmp	.LBB0_21
	.p2align	4, 0x90
.LBB0_22:                               #   in Loop: Header=BB0_7 Depth=1
	movq	%rsp, %rax
	leaq	-16(%rax), %rsp
	movq	%rsp, -48(%rbp)                 # 8-byte Spill
	movq	-216(%rbp), %rcx                # 8-byte Reload
	movq	%rcx, -16(%rax)
	movq	-248(%rbp), %r12                # 8-byte Reload
	movq	%r12, -8(%rax)
	cmpq	$64, %r12
	movl	$64, %eax
	cmovgeq	%rax, %r12
	movl	$64, %r14d
	subq	%r12, %r14
	movq	%rsp, %rax
	leaq	-64(%rax), %rcx
	movq	%rcx, %rsp
	movq	-56(%rbp), %rdx                 # 8-byte Reload
	movq	%rdx, -16(%rax)
	movq	-136(%rbp), %rdx                # 8-byte Reload
	movq	%rdx, -24(%rax)
	movq	%r12, -32(%rax)
	movq	%rsi, -40(%rax)
	movq	-224(%rbp), %rdx                # 8-byte Reload
	movq	%rdx, -48(%rax)
	movq	-152(%rbp), %rdx                # 8-byte Reload
	movq	%rdx, -56(%rax)
	movq	-160(%rbp), %rdx                # 8-byte Reload
	movq	%rdx, -64(%rax)
	movq	%rsp, %rax
	leaq	-64(%rax), %rdi
	movq	%rdi, %rsp
	movq	%r12, -32(%rax)
	movq	%rsi, -40(%rax)
	movq	%r15, -56(%rax)
	movq	%r15, -64(%rax)
	movq	$1, -16(%rax)
	movq	$64, -24(%rax)
	movq	$0, -48(%rax)
	movq	%rsp, %rax
	movq	%rsi, -88(%rbp)                 # 8-byte Spill
	leaq	-16(%rax), %rsi
	movq	%rsi, %rsp
	movq	%rcx, -8(%rax)
	movq	$2, -16(%rax)
	movq	%rsp, %rax
	leaq	-16(%rax), %rdx
	movq	%rdx, %rsp
	movq	%rdi, -8(%rax)
	movq	$2, -16(%rax)
	movl	$4, %edi
	callq	memrefCopy@PLT
	movq	-48(%rbp), %rsp                 # 8-byte Reload
	movq	%rsp, %rax
	leaq	-64(%rax), %rcx
	movq	%rcx, %rsp
	movq	-56(%rbp), %rdx                 # 8-byte Reload
	movq	%rdx, -16(%rax)
	movq	-136(%rbp), %rdx                # 8-byte Reload
	movq	%rdx, -24(%rax)
	movq	%r14, -32(%rax)
	movq	-88(%rbp), %rsi                 # 8-byte Reload
	movq	%rsi, -40(%rax)
	movq	-240(%rbp), %rdx                # 8-byte Reload
	movq	%rdx, -48(%rax)
	movq	-152(%rbp), %rdx                # 8-byte Reload
	movq	%rdx, -56(%rax)
	movq	-160(%rbp), %rdx                # 8-byte Reload
	movq	%rdx, -64(%rax)
	movq	%rsp, %rax
	leaq	-64(%rax), %rdi
	movq	%rdi, %rsp
	movq	%r14, -32(%rax)
	movq	%rsi, -40(%rax)
	movq	%r12, -48(%rax)
	movq	%r15, -56(%rax)
	movq	%r15, -64(%rax)
	movq	$1, -16(%rax)
	movq	$64, -24(%rax)
	movq	%rsp, %rax
	leaq	-16(%rax), %rsi
	movq	%rsi, %rsp
	movq	%rcx, -8(%rax)
	movq	$2, -16(%rax)
	movq	%rsp, %rax
	leaq	-16(%rax), %rdx
	movq	%rdx, %rsp
	movq	%rdi, -8(%rax)
	movq	$2, -16(%rax)
	movl	$4, %edi
	callq	memrefCopy@PLT
	movq	-48(%rbp), %rsp                 # 8-byte Reload
	movl	$8256, %edi                     # imm = 0x2040
	callq	malloc@PLT
	movq	%rax, %rcx
	leaq	63(%rax), %rsi
	andq	$-64, %rsi
	movq	%rsi, %rax
	xorl	%edx, %edx
	jmp	.LBB0_23
	.p2align	4, 0x90
.LBB0_27:                               #   in Loop: Header=BB0_23 Depth=2
	incq	%rdx
	addq	$256, %rax                      # imm = 0x100
.LBB0_23:                               #   Parent Loop BB0_7 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB0_26 Depth 3
	cmpq	$31, %rdx
	jg	.LBB0_28
# %bb.24:                               #   in Loop: Header=BB0_23 Depth=2
	xorl	%edi, %edi
	cmpq	$63, %rdi
	jg	.LBB0_27
	.p2align	4, 0x90
.LBB0_26:                               #   Parent Loop BB0_7 Depth=1
                                        #     Parent Loop BB0_23 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	movl	$0, (%rax,%rdi,4)
	incq	%rdi
	cmpq	$63, %rdi
	jle	.LBB0_26
	jmp	.LBB0_27
	.p2align	4, 0x90
.LBB0_28:                               #   in Loop: Header=BB0_7 Depth=1
	xorl	%eax, %eax
	jmp	.LBB0_29
	.p2align	4, 0x90
.LBB0_36:                               #   in Loop: Header=BB0_29 Depth=2
	incq	%rax
	addq	$64, %rbx
.LBB0_29:                               #   Parent Loop BB0_7 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB0_31 Depth 3
                                        #         Child Loop BB0_34 Depth 4
	cmpq	$31, %rax
	jg	.LBB0_37
# %bb.30:                               #   in Loop: Header=BB0_29 Depth=2
	movq	%r15, %rdx
	xorl	%edi, %edi
	jmp	.LBB0_31
	.p2align	4, 0x90
.LBB0_35:                               #   in Loop: Header=BB0_31 Depth=3
	incq	%rdi
	addq	$4, %rdx
.LBB0_31:                               #   Parent Loop BB0_7 Depth=1
                                        #     Parent Loop BB0_29 Depth=2
                                        # =>    This Loop Header: Depth=3
                                        #         Child Loop BB0_34 Depth 4
	cmpq	$63, %rdi
	jg	.LBB0_36
# %bb.32:                               #   in Loop: Header=BB0_31 Depth=3
	movq	%rdx, %r8
	xorl	%r9d, %r9d
	cmpq	$15, %r9
	jg	.LBB0_35
	.p2align	4, 0x90
.LBB0_34:                               #   Parent Loop BB0_7 Depth=1
                                        #     Parent Loop BB0_29 Depth=2
                                        #       Parent Loop BB0_31 Depth=3
                                        # =>      This Inner Loop Header: Depth=4
	movss	(%rbx,%r9,4), %xmm0             # xmm0 = mem[0],zero,zero,zero
	movq	%rax, %r10
	shlq	$6, %r10
	addq	%rdi, %r10
	mulss	(%r8), %xmm0
	addss	(%rsi,%r10,4), %xmm0
	movss	%xmm0, (%rsi,%r10,4)
	incq	%r9
	addq	$256, %r8                       # imm = 0x100
	cmpq	$15, %r9
	jle	.LBB0_34
	jmp	.LBB0_35
	.p2align	4, 0x90
.LBB0_37:                               #   in Loop: Header=BB0_7 Depth=1
	xorl	%eax, %eax
	movq	%rsi, %rdx
	xorl	%edi, %edi
	jmp	.LBB0_38
	.p2align	4, 0x90
.LBB0_42:                               #   in Loop: Header=BB0_38 Depth=2
	incq	%rdi
	addq	$256, %rdx                      # imm = 0x100
	addq	$256, %rax                      # imm = 0x100
.LBB0_38:                               #   Parent Loop BB0_7 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB0_41 Depth 3
	cmpq	$31, %rdi
	jg	.LBB0_43
# %bb.39:                               #   in Loop: Header=BB0_38 Depth=2
	xorl	%r8d, %r8d
	cmpq	$63, %r8
	jg	.LBB0_42
	.p2align	4, 0x90
.LBB0_41:                               #   Parent Loop BB0_7 Depth=1
                                        #     Parent Loop BB0_38 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	movss	(%rdx,%r8,4), %xmm0             # xmm0 = mem[0],zero,zero,zero
	leaq	(%r13,%rax), %r9
	addss	(%r9,%r8,4), %xmm0
	movss	%xmm0, (%rdx,%r8,4)
	incq	%r8
	cmpq	$63, %r8
	jle	.LBB0_41
	jmp	.LBB0_42
.LBB0_44:
	movq	%rsp, %rbx
	movslq	56(%rbp), %rdx
	movq	-168(%rbp), %r8                 # 8-byte Reload
	movq	%r8, %rax
	imulq	%rdx, %rax
	movq	-104(%rbp), %r11                # 8-byte Reload
	addq	%r11, %rax
	leaq	32(%r8), %rsi
	movq	-176(%rbp), %r9                 # 8-byte Reload
	cmpq	%r9, %rsi
	cmovlq	%rsi, %r9
	movq	-184(%rbp), %rdi                # 8-byte Reload
	movq	(%rdi), %rsi
	movq	8(%rdi), %rdi
	subq	%r8, %r9
	leaq	64(%r11), %r8
	movq	-112(%rbp), %r10                # 8-byte Reload
	cmpq	%r10, %r8
	cmovlq	%r8, %r10
	subq	%r11, %r10
	cmpq	$32, %r9
	movl	$32, %r8d
	cmovlq	%r9, %r8
	cmpq	$64, %r10
	movl	$64, %r9d
	cmovlq	%r10, %r9
	movq	%rsp, %r10
	leaq	-64(%r10), %r11
	movq	%r11, %rsp
	movq	%r9, -32(%r10)
	movq	%r8, -40(%r10)
	movq	%r13, -56(%r10)
	movq	%rcx, -64(%r10)
	movq	$1, -16(%r10)
	movq	$64, -24(%r10)
	movq	$0, -48(%r10)
	movq	%rsp, %rcx
	leaq	-64(%rcx), %r10
	movq	%r10, %rsp
	movq	%rdx, -24(%rcx)
	movq	%r9, -32(%rcx)
	movq	%r8, -40(%rcx)
	movq	%rax, -48(%rcx)
	movq	%rdi, -56(%rcx)
	movq	%rsi, -64(%rcx)
	movq	$1, -16(%rcx)
	movq	%rsp, %rax
	leaq	-16(%rax), %rsi
	movq	%rsi, %rsp
	movq	%r11, -8(%rax)
	movq	$2, -16(%rax)
	movq	%rsp, %rax
	leaq	-16(%rax), %rdx
	movq	%rdx, %rsp
	movq	%r10, -8(%rax)
	movq	$2, -16(%rax)
	movl	$4, %edi
	callq	memrefCopy@PLT
	movq	%rbx, %rsp
	leaq	-40(%rbp), %rsp
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end0:
	.size	matmul_kernel_0d1d2d34567c89c1011c, .Lfunc_end0-matmul_kernel_0d1d2d34567c89c1011c
	.cfi_endproc
                                        # -- End function
	.section	".note.GNU-stack","",@progbits
