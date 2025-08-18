#include <cstdint>

// #include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "triton-shared/Conversion/TPtrToLLVM/TPtrToLLVM.h"
#include "triton-shared/Dialect/TPtr/IR/TPtrDialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
namespace mlir {
namespace tptr {

#define DEBUG_TYPE "tptr-to-llvm"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")
#define LDBG_ENDL() LLVM_DEBUG(llvm::dbgs() << "\n")
Type convertMemRefType(MemRefType type) {
  auto ctx = type.getContext();
  auto rank = type.getShape().size();
  auto i64Ty = IntegerType::get(ctx, 64);
  SmallVector<Type, 5> types;

  // struct { ptr base_ptr, ptr aligned_ptr, i64 offset, array<rank x i64>
  // sizes, array<rank x i64> strides }
  types.push_back(LLVM::LLVMPointerType::get(ctx));  // base pointer
  types.push_back(LLVM::LLVMPointerType::get(ctx));  // aligned pointer
  types.push_back(i64Ty);                            // offset
  types.push_back(LLVM::LLVMArrayType::get(ctx, i64Ty, rank));  // sizes
  types.push_back(LLVM::LLVMArrayType::get(ctx, i64Ty, rank));  // strides

  Type ret = LLVM::LLVMStructType::getLiteral(ctx, types);
  LDBG(" From MemrefConverter convertMemRefType: " << type << " -> " << ret);
  return ret;
}
// 1. PtrAddOp -> llvm.gep
class PtrAddConverter : public OpConversionPattern<tptr::PtrAddOp> {
  using OpConversionPattern<tptr::PtrAddOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      tptr::PtrAddOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    LDBG("matchAndRewrite: ptradd " << op);
    assert(isa<triton::PointerType>(op.getType()));
    auto ptrTy = cast<triton::PointerType>(op.getType());
    Type elemTy = getTypeConverter()->convertType(ptrTy.getPointeeType());
    Type resTy = getTypeConverter()->convertType(ptrTy);
    Value ptr = rewriter.getRemappedValue(op.getBase());
    Value offset = rewriter.getRemappedValue(op.getOffset());
    auto gep =
        rewriter.create<LLVM::GEPOp>(op.getLoc(), resTy, elemTy, ptr, offset);
    rewriter.replaceOp(op, gep);
    LDBG("matchAndRewrite: ptradd done " << gep << "\n");
    return success();
  }
};

// 2. ToMemrefOp -> 直接构建LLVM struct
class ToMemrefConverter : public OpConversionPattern<tptr::ToMemrefOp> {
  using OpConversionPattern<tptr::ToMemrefOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      tptr::ToMemrefOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    LDBG("matchAndRewrite: to_memref " << op);
    auto input = adaptor.getArg();
    if (Operation *defOp = input.getDefiningOp()) {
      if (auto unrealizedCast =
              dyn_cast<mlir::UnrealizedConversionCastOp>(defOp)) {
        if (unrealizedCast.getInputs().size() == 1) {
          Value castInput = unrealizedCast.getInputs()[0];
          if (isa<LLVM::LLVMPointerType>(castInput.getType())) {
            input = castInput;
          }
        }
      }
    }
    if (!isa<LLVM::LLVMPointerType>(input.getType())) {
      return failure();
    }
    if (!isa<MemRefType>(op.getType())) {
      return failure();
    }
    Type targetType = convertMemRefType(op.getType());
    if (!targetType) {
      return failure();
    }

    auto loc = op.getLoc();
    auto i64Ty = rewriter.getIntegerType(64);

    Value result = rewriter.create<LLVM::UndefOp>(loc, targetType);

    result = rewriter.create<LLVM::InsertValueOp>(loc, result, input, 0);

    result = rewriter.create<LLVM::InsertValueOp>(loc, result, input, 1);

    Value zero_offset = rewriter.create<LLVM::ConstantOp>(
        loc, i64Ty, rewriter.getIntegerAttr(i64Ty, 0));
    result = rewriter.create<LLVM::InsertValueOp>(loc, result, zero_offset, 2);

    auto memrefType = cast<MemRefType>(op.getType());
    for (auto [i, size] : llvm::enumerate(memrefType.getShape())) {
      Value sizeVal = rewriter.create<LLVM::ConstantOp>(
          loc, i64Ty, rewriter.getIntegerAttr(i64Ty, size));
      result = rewriter.create<LLVM::InsertValueOp>(
          loc, result, sizeVal, ArrayRef<int64_t>{3, static_cast<int64_t>(i)});

      Value strideVal = rewriter.create<LLVM::ConstantOp>(
          loc, i64Ty, rewriter.getIntegerAttr(i64Ty, 1));
      result = rewriter.create<LLVM::InsertValueOp>(
          loc, result, strideVal,
          ArrayRef<int64_t>{4, static_cast<int64_t>(i)});
    }

    LDBG("matchAndRewrite: to_memref done " << result);
    rewriter.replaceOp(op, result);
    return success();
  }
};

// 3. FromMemrefOp -> llvm.extractvalue
class FromMemrefConverter : public OpConversionPattern<tptr::FromMemrefOp> {
  using OpConversionPattern<tptr::FromMemrefOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      tptr::FromMemrefOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // TODO: 实现 from_memref 到 llvm.extractvalue 的转换
    LDBG("matchAndRewrite: from_memref " << op);
    Value input = adaptor.getInput();
    if (isa<MemRefType>(input.getType())) {
      input = rewriter
                  .create<UnrealizedConversionCastOp>(
                      op.getLoc(),
                      convertMemRefType(dyn_cast<MemRefType>(input.getType())),
                      input)
                  .getResult(0);
    }
    auto positionAttr =
        rewriter.getDenseI64ArrayAttr({0});  // 提取第0个元素（通常是指针）

    Type resultType = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto newOp = rewriter.create<LLVM::ExtractValueOp>(op.getLoc(), resultType,
                                                       input, positionAttr);

    rewriter.replaceOp(op, newOp);
    LDBG("matchAndRewrite: from_memref done " << newOp << "\n");
    return success();
  }
};

// 4. remove useles unrealized_cast
class UnrealizedCastConverter
    : public OpConversionPattern<UnrealizedConversionCastOp> {
  using OpConversionPattern<UnrealizedConversionCastOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      UnrealizedConversionCastOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (!isOneToOneCast(op)) {
      return failure();
    }
    auto input = adaptor.getInputs().front();
    auto input_type = input.getType();
    auto output_type = op.getOutputs().front().getType();

    if (input_type == output_type) {
      LDBG("matchAndRewrite: unrealized_cast identical types, removing cast");
      rewriter.replaceOp(op, input);
      return success();
    }

    if (isa<triton::PointerType>(output_type)) {
      LDBG("matchAndRewrite: unrealized_cast to triton type, rejecting!");
      return rewriter.notifyMatchFailure(
          op, "conversion to triton pointer type not allowed");
    }

    if (isa<LLVM::LLVMStructType>(input_type) && isa<MemRefType>(output_type)) {
      rewriter.replaceOp(op, input);
      return success();
    }

    if (isa<MemRefType>(input_type) && isa<LLVM::LLVMStructType>(output_type)) {
      rewriter.replaceOp(op, input);
      return success();
    }

    if (isa<LLVM::LLVMPointerType>(input_type) &&
        isa<MemRefType>(output_type)) {
      return rewriter.notifyMatchFailure(
          op, "direct llvm.ptr to memref conversion not allowed");
    }
    return failure();
  }
};

// 5. Legalize target block arguments.
static LogicalResult legalizeBlockArguments(Block &block, Operation *op,
                                            PatternRewriter &rewriter,
                                            const TypeConverter &converter) {
  auto builder = OpBuilder::atBlockBegin(&block);
  for (unsigned i = 0; i < block.getNumArguments(); ++i) {
    BlockArgument arg = block.getArgument(i);
    if (converter.isLegal(arg.getType())) continue;
    Type ty = arg.getType();
    Type newTy = converter.convertType(ty);
    if (!newTy) {
      return rewriter.notifyMatchFailure(op, "failed to convert argument type");
    }
    unsigned argNum = arg.getArgNumber();
    Location loc = arg.getLoc();
    Value newArg = block.insertArgument(argNum, newTy, loc);
    arg.replaceAllUsesWith(newArg);
    block.eraseArgument(argNum + 1);
  }
  return success();
}

// 5. control flow operations conversion
struct ConvertControlFlowOp : public OpConversionPattern<cf::CondBranchOp> {
  using OpConversionPattern<cf::CondBranchOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      cf::CondBranchOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    LDBG("matchAndRewrite: cond_branch " << op);

    // 先处理目标基本块的参数类型转换
    if (failed(legalizeBlockArguments(*op.getTrueDest(), op, rewriter,
                                      *getTypeConverter())))
      return failure();

    if (failed(legalizeBlockArguments(*op.getFalseDest(), op, rewriter,
                                      *getTypeConverter())))
      return failure();

    auto convertedTrueOps = adaptor.getTrueDestOperands();
    auto convertedFalseOps = adaptor.getFalseDestOperands();
    auto newOp = rewriter.create<cf::CondBranchOp>(
        op.getLoc(), adaptor.getCondition(), op.getTrueDest(), convertedTrueOps,
        op.getFalseDest(), convertedFalseOps);
    rewriter.replaceOp(op, newOp);
    LDBG("matchAndRewrite: cond_branch done " << newOp);
    return success();
  }
};

// 6. Control Flow operations conversion for cf::BranchOp
class ConvertBranchOp : public OpConversionPattern<cf::BranchOp> {
  using OpConversionPattern<cf::BranchOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      cf::BranchOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    LDBG("matchAndRewrite: cf.br " << op);

    // Legalize destination block arguments first
    if (failed(legalizeBlockArguments(*op.getDest(), op, rewriter,
                                      *getTypeConverter())))
      return failure();

    // Create new branch with converted operands
    rewriter.replaceOpWithNewOp<cf::BranchOp>(op, op.getDest(),
                                              adaptor.getDestOperands());
    LDBG("matchAndRewrite: cf.br done");
    return success();
  }
};

void populateTPtrToLLVMConversionPatterns(RewritePatternSet &patterns,
                                          TypeConverter &typeConverter) {

  patterns.add<PtrAddConverter, ToMemrefConverter, FromMemrefConverter,
               ConvertBranchOp, ConvertControlFlowOp>(typeConverter,
                                                      patterns.getContext());
}
}  // namespace tptr
}  // namespace mlir