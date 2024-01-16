	.text
	.file	"LLVMDialectModule"
	.globl	softmax_kernel_0d1d234          # -- Begin function softmax_kernel_0d1d234
	.p2align	4, 0x90
	.type	softmax_kernel_0d1d234,@function
softmax_kernel_0d1d234:                 # @softmax_kernel_0d1d234
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
	subq	$24, %rsp
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	movl	%r9d, -48(%rbp)                 # 4-byte Spill
	movq	%rsi, -64(%rbp)                 # 8-byte Spill
	movslq	16(%rbp), %rbx
	imull	48(%rbp), %r8d
	movslq	%r8d, %r12
	movq	(%rcx), %r15
	movq	8(%rcx), %r13
	movl	$4096, %edi                     # imm = 0x1000
	callq	malloc@PLT
	movq	%rax, %r14
	cmpq	$1024, %rbx                     # imm = 0x400
	movl	$1024, %eax                     # imm = 0x400
	movq	%rbx, %rcx
	movq	%rbx, -56(%rbp)                 # 8-byte Spill
	cmovlq	%rbx, %rax
	cmpq	$1023, %rax                     # imm = 0x3FF
	jg	.LBB0_4
# %bb.1:
	xorl	%ecx, %ecx
	cmpq	$1023, %rcx                     # imm = 0x3FF
	jg	.LBB0_4
	.p2align	4, 0x90
.LBB0_3:                                # =>This Inner Loop Header: Depth=1
	movl	$-8388608, (%r14,%rcx,4)        # imm = 0xFF800000
	incq	%rcx
	cmpq	$1023, %rcx                     # imm = 0x3FF
	jle	.LBB0_3
.LBB0_4:
	movq	%rsp, %rbx
	movq	%rsp, %rcx
	leaq	-48(%rcx), %rdx
	movq	%rdx, %rsp
	movq	%rax, -24(%rcx)
	movq	%r12, -32(%rcx)
	movq	%r13, -40(%rcx)
	movq	%r15, -48(%rcx)
	movq	$1, -16(%rcx)
	movq	%rsp, %rcx
	leaq	-48(%rcx), %rdi
	movq	%rdi, %rsp
	movq	%rax, -24(%rcx)
	movq	%r14, -40(%rcx)
	movq	%r14, -48(%rcx)
	movq	$1, -16(%rcx)
	movq	$0, -32(%rcx)
	movq	%rsp, %rax
	leaq	-16(%rax), %rsi
	movq	%rsi, %rsp
	movq	%rdx, -8(%rax)
	movq	$1, -16(%rax)
	movq	%rsp, %rax
	leaq	-16(%rax), %rdx
	movq	%rdx, %rsp
	movq	%rdi, -8(%rax)
	movq	$1, -16(%rax)
	movl	$4, %edi
	callq	memrefCopy@PLT
	movq	%rbx, %rsp
	movl	$68, %edi
	callq	malloc@PLT
	addq	$63, %rax
	andq	$-64, %rax
	movl	$-8388608, (%rax)               # imm = 0xFF800000
	xorl	%ecx, %ecx
	cmpq	$1023, %rcx                     # imm = 0x3FF
	jg	.LBB0_7
	.p2align	4, 0x90
.LBB0_6:                                # =>This Inner Loop Header: Depth=1
	movss	(%r14,%rcx,4), %xmm0            # xmm0 = mem[0],zero,zero,zero
	movss	(%rax), %xmm1                   # xmm1 = mem[0],zero,zero,zero
	movaps	%xmm1, %xmm2
	maxss	%xmm0, %xmm2
	cmpunordss	%xmm0, %xmm0
	movaps	%xmm0, %xmm3
	andnps	%xmm2, %xmm3
	andps	%xmm1, %xmm0
	orps	%xmm3, %xmm0
	movss	%xmm0, (%rax)
	incq	%rcx
	cmpq	$1023, %rcx                     # imm = 0x3FF
	jle	.LBB0_6
.LBB0_7:
	movss	(%rax), %xmm0                   # xmm0 = mem[0],zero,zero,zero
	movss	%xmm0, -44(%rbp)                # 4-byte Spill
	movl	$4160, %edi                     # imm = 0x1040
	callq	malloc@PLT
	movss	-44(%rbp), %xmm0                # 4-byte Reload
                                        # xmm0 = mem[0],zero,zero,zero
	addq	$63, %rax
	andq	$-64, %rax
	xorl	%ecx, %ecx
	cmpq	$1023, %rcx                     # imm = 0x3FF
	jg	.LBB0_10
	.p2align	4, 0x90
.LBB0_9:                                # =>This Inner Loop Header: Depth=1
	movss	%xmm0, (%rax,%rcx,4)
	incq	%rcx
	cmpq	$1023, %rcx                     # imm = 0x3FF
	jle	.LBB0_9
.LBB0_10:
	xorl	%ecx, %ecx
	cmpq	$1023, %rcx                     # imm = 0x3FF
	jg	.LBB0_13
	.p2align	4, 0x90
.LBB0_12:                               # =>This Inner Loop Header: Depth=1
	movss	(%r14,%rcx,4), %xmm0            # xmm0 = mem[0],zero,zero,zero
	subss	(%rax,%rcx,4), %xmm0
	movss	%xmm0, (%r14,%rcx,4)
	incq	%rcx
	cmpq	$1023, %rcx                     # imm = 0x3FF
	jle	.LBB0_12
.LBB0_13:
	xorl	%ebx, %ebx
	cmpq	$1023, %rbx                     # imm = 0x3FF
	jg	.LBB0_16
	.p2align	4, 0x90
.LBB0_15:                               # =>This Inner Loop Header: Depth=1
	movss	(%r14,%rbx,4), %xmm0            # xmm0 = mem[0],zero,zero,zero
	callq	expf@PLT
	movss	%xmm0, (%r14,%rbx,4)
	incq	%rbx
	cmpq	$1023, %rbx                     # imm = 0x3FF
	jle	.LBB0_15
.LBB0_16:
	movl	$68, %edi
	callq	malloc@PLT
	addq	$63, %rax
	andq	$-64, %rax
	movl	$0, (%rax)
	xorl	%ecx, %ecx
	cmpq	$1023, %rcx                     # imm = 0x3FF
	jg	.LBB0_19
	.p2align	4, 0x90
.LBB0_18:                               # =>This Inner Loop Header: Depth=1
	movss	(%r14,%rcx,4), %xmm0            # xmm0 = mem[0],zero,zero,zero
	addss	(%rax), %xmm0
	movss	%xmm0, (%rax)
	incq	%rcx
	cmpq	$1023, %rcx                     # imm = 0x3FF
	jle	.LBB0_18
.LBB0_19:
	movss	(%rax), %xmm0                   # xmm0 = mem[0],zero,zero,zero
	movss	%xmm0, -44(%rbp)                # 4-byte Spill
	movl	$4160, %edi                     # imm = 0x1040
	callq	malloc@PLT
	movss	-44(%rbp), %xmm0                # 4-byte Reload
                                        # xmm0 = mem[0],zero,zero,zero
	addq	$63, %rax
	andq	$-64, %rax
	xorl	%ecx, %ecx
	cmpq	$1023, %rcx                     # imm = 0x3FF
	jg	.LBB0_22
	.p2align	4, 0x90
.LBB0_21:                               # =>This Inner Loop Header: Depth=1
	movss	%xmm0, (%rax,%rcx,4)
	incq	%rcx
	cmpq	$1023, %rcx                     # imm = 0x3FF
	jle	.LBB0_21
.LBB0_22:
	xorl	%ecx, %ecx
	cmpq	$1023, %rcx                     # imm = 0x3FF
	jg	.LBB0_25
	.p2align	4, 0x90
.LBB0_24:                               # =>This Inner Loop Header: Depth=1
	movss	(%r14,%rcx,4), %xmm0            # xmm0 = mem[0],zero,zero,zero
	divss	(%rax,%rcx,4), %xmm0
	movss	%xmm0, (%r14,%rcx,4)
	incq	%rcx
	cmpq	$1023, %rcx                     # imm = 0x3FF
	jle	.LBB0_24
.LBB0_25:
	movq	%rsp, %r15
	movl	48(%rbp), %eax
	imull	-48(%rbp), %eax                 # 4-byte Folded Reload
	cltq
	movslq	-56(%rbp), %rcx                 # 4-byte Folded Reload
	cmpq	$1024, %rcx                     # imm = 0x400
	movl	$1024, %edx                     # imm = 0x400
	cmovlq	%rcx, %rdx
	movq	-64(%rbp), %rsi                 # 8-byte Reload
	movq	(%rsi), %rcx
	movq	8(%rsi), %rsi
	movq	%rsp, %rdi
	leaq	-48(%rdi), %r8
	movq	%r8, %rsp
	movq	%rdx, -24(%rdi)
	movq	%r14, -40(%rdi)
	movq	%r14, -48(%rdi)
	movq	$1, -16(%rdi)
	movq	$0, -32(%rdi)
	movq	%rsp, %rdi
	leaq	-48(%rdi), %r9
	movq	%r9, %rsp
	movq	%rdx, -24(%rdi)
	movq	%rax, -32(%rdi)
	movq	%rsi, -40(%rdi)
	movq	%rcx, -48(%rdi)
	movq	$1, -16(%rdi)
	movq	%rsp, %rax
	leaq	-16(%rax), %rsi
	movq	%rsi, %rsp
	movq	%r8, -8(%rax)
	movq	$1, -16(%rax)
	movq	%rsp, %rax
	leaq	-16(%rax), %rdx
	movq	%rdx, %rsp
	movq	%r9, -8(%rax)
	movq	$1, -16(%rax)
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
	.size	softmax_kernel_0d1d234, .Lfunc_end0-softmax_kernel_0d1d234
	.cfi_endproc
                                        # -- End function
	.section	".note.GNU-stack","",@progbits
