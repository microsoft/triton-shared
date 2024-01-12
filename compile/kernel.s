	.text
	.file	"LLVMDialectModule"
	.globl	triton__0d1d2de                 # -- Begin function triton__0d1d2de
	.p2align	4, 0x90
	.type	triton__0d1d2de,@function
triton__0d1d2de:                        # @triton__0d1d2de
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
	pushq	%rax
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	movq	%rcx, -48(%rbp)                 # 8-byte Spill
	movl	32(%rbp), %eax
	shll	$2, %eax
	movslq	%eax, %r15
	shlq	$7, %r15
	movq	(%rsi), %r13
	movq	8(%rsi), %r12
	movl	$1024, %edi                     # imm = 0x400
	callq	malloc@PLT
	movq	%rax, %r14
	xorl	%eax, %eax
	movq	%r14, %rcx
	jmp	.LBB0_1
	.p2align	4, 0x90
.LBB0_5:                                #   in Loop: Header=BB0_1 Depth=1
	incq	%rax
	addq	$256, %rcx                      # imm = 0x100
.LBB0_1:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_4 Depth 2
	cmpq	$3, %rax
	jg	.LBB0_6
# %bb.2:                                #   in Loop: Header=BB0_1 Depth=1
	xorl	%edx, %edx
	cmpq	$63, %rdx
	jg	.LBB0_5
	.p2align	4, 0x90
.LBB0_4:                                #   Parent Loop BB0_1 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movl	$77, (%rcx,%rdx,4)
	incq	%rdx
	cmpq	$63, %rdx
	jle	.LBB0_4
	jmp	.LBB0_5
.LBB0_6:
	movq	%rsp, %rbx
	movq	%rsp, %rax
	leaq	-64(%rax), %rcx
	movq	%rcx, %rsp
	movq	%r15, -48(%rax)
	movq	%r12, -56(%rax)
	movq	%r13, -64(%rax)
	movq	$1, -16(%rax)
	movq	$128, -24(%rax)
	movq	$32, -32(%rax)
	movq	$4, -40(%rax)
	movq	%rsp, %rax
	leaq	-64(%rax), %rdi
	movq	%rdi, %rsp
	movq	%r14, -56(%rax)
	movq	%r14, -64(%rax)
	movq	$1, -16(%rax)
	movq	$64, -24(%rax)
	movq	$32, -32(%rax)
	movq	$4, -40(%rax)
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
	movq	%rbx, %rsp
	movq	%rsp, %r15
	movq	-48(%rbp), %rcx                 # 8-byte Reload
	movq	(%rcx), %rax
	movq	8(%rcx), %rcx
	movq	%rsp, %rdx
	leaq	-64(%rdx), %rdi
	movq	%rdi, %rsp
	movl	$1, %esi
	movq	%rsi, -16(%rdx)
	movl	$64, %esi
	movq	%rsi, -24(%rdx)
	movq	%rsi, -32(%rdx)
	movl	$4, %esi
	movq	%rsi, -40(%rdx)
	movq	%r14, -56(%rdx)
	movq	%r14, -64(%rdx)
	movq	$0, -48(%rdx)
	movq	%rsp, %rdx
	leaq	-64(%rdx), %r8
	movq	%r8, %rsp
	movq	%rcx, -56(%rdx)
	movq	%rax, -64(%rdx)
	movq	$4, -16(%rdx)
	movq	$1, -24(%rdx)
	movq	$64, -32(%rdx)
	movq	$4, -40(%rdx)
	movq	$0, -48(%rdx)
	movq	%rsp, %rax
	leaq	-16(%rax), %rsi
	movq	%rsi, %rsp
	movq	%rdi, -8(%rax)
	movq	$2, -16(%rax)
	movq	%rsp, %rax
	leaq	-16(%rax), %rdx
	movq	%rdx, %rsp
	movq	%r8, -8(%rax)
	movq	$2, -16(%rax)
	movl	$4, %edi
	callq	memrefCopy@PLT
	movq	%r15, %rsp
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
	.size	triton__0d1d2de, .Lfunc_end0-triton__0d1d2de
	.cfi_endproc
                                        # -- End function
	.section	".note.GNU-stack","",@progbits
