	.text
	.file	"LLVMDialectModule"
	.globl	reduce_kernel_2d_0d             # -- Begin function reduce_kernel_2d_0d
	.p2align	4, 0x90
	.type	reduce_kernel_2d_0d,@function
reduce_kernel_2d_0d:                    # @reduce_kernel_2d_0d
	.cfi_startproc
# %bb.0:
	movq	8(%rsi), %rax
	movslq	%r9d, %rcx
	xorl	%edx, %edx
	jmp	.LBB0_1
	.p2align	4, 0x90
.LBB0_5:                                #   in Loop: Header=BB0_1 Depth=1
	incl	%edx
.LBB0_1:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_4 Depth 2
	cmpl	$7, %edx
	jg	.LBB0_6
# %bb.2:                                #   in Loop: Header=BB0_1 Depth=1
	xorps	%xmm0, %xmm0
	cvtsi2ss	%edx, %xmm0
	xorl	%esi, %esi
	cmpl	$1, %esi
	jg	.LBB0_5
	.p2align	4, 0x90
.LBB0_4:                                #   Parent Loop BB0_1 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movss	%xmm0, (%rax,%rcx,4)
	incq	%rcx
	incl	%esi
	cmpl	$1, %esi
	jle	.LBB0_4
	jmp	.LBB0_5
.LBB0_6:
	retq
.Lfunc_end0:
	.size	reduce_kernel_2d_0d, .Lfunc_end0-reduce_kernel_2d_0d
	.cfi_endproc
                                        # -- End function
	.section	".note.GNU-stack","",@progbits
