#include <cstdint>

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Ptr/IR/PtrTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Transforms/DialectConversion.h"
#include "triton-shared/Conversion/TPtrToLLVM/TPtrToLLVM.h"
#include "triton-shared/Dialect/TPtr/IR/TPtrDialect.h"

namespace mlir {
namespace tptr {

#define DEBUG_TYPE "tptr-to-llvm"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

static bool isOneToOneCast(UnrealizedConversionCastOp op) {
  return (op.getInputs().size() == 1 && op->getNumResults() == 1);
}


// PtrAddOp -> llvm.getelementptr conversion
struct PtrAddConverter : OpConversionPattern<tptr::PtrAddOp> {
  using OpConversionPattern<tptr::PtrAddOp>::OpConversionPattern;

  Type convertPtrPointerType(ptr::PtrType type) const {
    auto ctx = type.getContext();
    return LLVM::LLVMPointerType::get(ctx);
  }

  LogicalResult
  matchAndRewrite(tptr::PtrAddOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    LDBG("matchAndRewrite: ptradd " << op);

    if (!isa<ptr::PtrType>(op.getType())) {
      return rewriter.notifyMatchFailure(op, "expected ptr type");
    }
    auto ptrTy = cast<ptr::PtrType>(op.getType());

    // Infer element type
    Type elemTy = nullptr;
    auto origOffset = op.getOffset();
    if (auto mulOp = origOffset.getDefiningOp<LLVM::MulOp>()) {
      if (auto typeOffsetOp =
              mulOp.getRhs().getDefiningOp<tptr::TypeOffsetOp>()) {
        elemTy = typeOffsetOp.getBaseType();
      }
    } else if (auto mulOp = origOffset.getDefiningOp<arith::MulIOp>()) {
      if (auto typeOffsetOp =
              mulOp.getRhs().getDefiningOp<tptr::TypeOffsetOp>()) {
        elemTy = typeOffsetOp.getBaseType();
      }
    }

    if (!elemTy) {
      elemTy = rewriter.getIntegerType(8); // default to i8
    }

    Type resTy = convertPtrPointerType(ptrTy);

    Value elementIndex;
    if (auto mulOp = adaptor.getOffset().getDefiningOp<LLVM::MulOp>()) {
      elementIndex = mulOp.getLhs();
    } else if (auto mulOp =
                   adaptor.getOffset().getDefiningOp<arith::MulIOp>()) {
      elementIndex = mulOp.getLhs();
    } else {
      LDBG("Warning: ptradd offset is not MulOp pattern, using raw offset");
      elementIndex = adaptor.getOffset();
    }

    auto gep = rewriter.create<LLVM::GEPOp>(op.getLoc(), resTy, elemTy,
                                            adaptor.getBase(), elementIndex);
    rewriter.replaceOp(op, gep);
    LDBG("matchAndRewrite: ptradd done " << gep);
    return success();
  }
};

// ToMemrefOp -> build LLVM memref struct
struct ToMemrefConverter : OpConversionPattern<tptr::ToMemrefOp> {
  using OpConversionPattern<tptr::ToMemrefOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tptr::ToMemrefOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    LDBG("matchAndRewrite: to_memref (before) " << op);

    auto input = adaptor.getArg();

    // Try to extract real pointer from UnrealizedConversionCast
    if (Operation *defOp = input.getDefiningOp()) {
      if (auto unrealizedCast = dyn_cast<UnrealizedConversionCastOp>(defOp)) {
        if (unrealizedCast.getInputs().size() == 1) {
          Value castInput = unrealizedCast.getInputs()[0];
          if (isa<LLVM::LLVMPointerType>(castInput.getType())) {
            input = castInput;
          }
        }
      }
    }

    Type targetType = getTypeConverter()->convertType(cast<MemRefType>(op.getType()));
    LDBG("matchAndRewrite: to_memref (typeconverted) " <<  cast<MemRefType>(op.getType()) << " -> " << targetType);
    if (!targetType) {
      return rewriter.notifyMatchFailure(op, "failed to convert memref type");
    }

    auto loc = op.getLoc();
    auto i64Ty = rewriter.getIntegerType(64);
    auto shape = cast<MemRefType>(op.getType()).getShape();
    auto rank = shape.size();

    Value result = rewriter.create<LLVM::UndefOp>(loc, targetType);
    result =
        rewriter.create<LLVM::InsertValueOp>(loc, result, input, 0); // base_ptr
    result = rewriter.create<LLVM::InsertValueOp>(loc, result, input,
                                                  1); // aligned_ptr

    Value zeroOffset = rewriter.create<LLVM::ConstantOp>(
        loc, i64Ty, rewriter.getIntegerAttr(i64Ty, 0));
    result = rewriter.create<LLVM::InsertValueOp>(loc, result, zeroOffset, 2);

    SmallVector<int64_t> strides(rank, 1);
    for (int i = rank - 2; i >= 0; --i) {
      if (shape[i + 1] != ShapedType::kDynamic) {
        strides[i] = strides[i + 1] * shape[i + 1];
      }
    }

    for (auto [i, size] : llvm::enumerate(shape)) {
      Value sizeVal = rewriter.create<LLVM::ConstantOp>(
          loc, i64Ty, rewriter.getIntegerAttr(i64Ty, size));
      result = rewriter.create<LLVM::InsertValueOp>(
          loc, result, sizeVal, ArrayRef<int64_t>{3, static_cast<int64_t>(i)});

      Value strideVal = rewriter.create<LLVM::ConstantOp>(
          loc, i64Ty, rewriter.getIntegerAttr(i64Ty, strides[i]));
      result = rewriter.create<LLVM::InsertValueOp>(
          loc, result, strideVal,
          ArrayRef<int64_t>{4, static_cast<int64_t>(i)});
    }

    rewriter.replaceOp(op, result);
    LDBG("matchAndRewrite: to_memref (after) -> " << result);
    return success();
  }
};

// FromMemrefOp -> llvm.extractvalue
struct FromMemrefConverter : OpConversionPattern<tptr::FromMemrefOp> {
  using OpConversionPattern<tptr::FromMemrefOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tptr::FromMemrefOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    LDBG("matchAndRewrite: from_memref (before) " << op);

    Value input = adaptor.getInput();
    if (isa<MemRefType>(input.getType())) {
      input = rewriter
                  .create<UnrealizedConversionCastOp>(
                      op.getLoc(),
                      getTypeConverter()->convertType(
                          cast<MemRefType>(input.getType())),
                      input)
                  .getResult(0);
    }

    // Extract base_ptr (index 0)
    Type resultType = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto extractOp = rewriter.create<LLVM::ExtractValueOp>(
        op.getLoc(), resultType, input, rewriter.getDenseI64ArrayAttr({0}));

    rewriter.replaceOp(op, extractOp);
    LDBG("matchAndRewrite: from_memref (after) -> " << extractOp);
    return success();
  }
};

// Clean up unused UnrealizedConversionCast
struct UnrealizedCastConverter
    : OpConversionPattern<UnrealizedConversionCastOp> {
  using OpConversionPattern<UnrealizedConversionCastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(UnrealizedConversionCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isOneToOneCast(op)) {
      return failure();
    }

    auto input = adaptor.getInputs().front();
    auto inputType = input.getType();
    auto outputType = op.getOutputs().front().getType();

    // Same type, directly remove cast
    if (inputType == outputType) {
      rewriter.replaceOp(op, input);
      return success();
    }

    if (isa<ptr::PtrType>(outputType) ||
        (isa<LLVM::LLVMPointerType>(inputType) && isa<MemRefType>(outputType))) {
      LDBG("UnrealizedCast (reject): unsafe pointer conversion " << op);
      return rewriter.notifyMatchFailure(op, "unsafe pointer conversion");
    }

    if ((isa<LLVM::LLVMStructType>(inputType) && isa<MemRefType>(outputType)) ||
        (isa<MemRefType>(inputType) && isa<LLVM::LLVMStructType>(outputType))) {
      LDBG("matchAndRewrite: UnrealizedCast (after) " << op << " -> " << input);
      rewriter.replaceOp(op, input);
      return success();
    }

    return failure();
  }
};

// Basic block argument type legalization
static LogicalResult legalizeBlockArguments(Block &block, Operation *op,
                                            PatternRewriter &rewriter,
                                            const TypeConverter &converter) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(&block);

  for (unsigned i = 0; i < block.getNumArguments(); ++i) {
    BlockArgument arg = block.getArgument(i);
    if (converter.isLegal(arg.getType())) {
      continue;
    }

    Type newTy = converter.convertType(arg.getType());
    if (!newTy) {
      return rewriter.notifyMatchFailure(op, "failed to convert argument type");
    }

    unsigned argNum = arg.getArgNumber();
    Value newArg = block.insertArgument(argNum, newTy, arg.getLoc());
    arg.replaceAllUsesWith(newArg);
    block.eraseArgument(argNum + 1);
  }
  return success();
}

// Conditional branch conversion
struct ConvertControlFlowOp : OpConversionPattern<cf::CondBranchOp> {
  using OpConversionPattern<cf::CondBranchOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cf::CondBranchOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    LDBG("matchAndRewrite: cond_branch (before) " << op);

    if (failed(legalizeBlockArguments(*op.getTrueDest(), op, rewriter,
                                      *getTypeConverter())) ||
        failed(legalizeBlockArguments(*op.getFalseDest(), op, rewriter,
                                      *getTypeConverter()))) {
      return failure();
    }

    auto newOp = rewriter.replaceOpWithNewOp<cf::CondBranchOp>(
        op, adaptor.getCondition(), op.getTrueDest(),
        adaptor.getTrueDestOperands(), op.getFalseDest(),
        adaptor.getFalseDestOperands());

    LDBG("matchAndRewrite: cond_branch (after) -> " << newOp);
    return success();
  }
};

// Unconditional branch conversion
struct ConvertBranchOp : OpConversionPattern<cf::BranchOp> {
  using OpConversionPattern<cf::BranchOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cf::BranchOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    LDBG("matchAndRewrite: cf.br (before) " << op);

    if (failed(legalizeBlockArguments(*op.getDest(), op, rewriter,
                                      *getTypeConverter()))) {
      return failure();
    }

    auto newOp =
        rewriter.replaceOpWithNewOp<cf::BranchOp>(op, op.getDest(),
                                                  adaptor.getDestOperands());
    LDBG("matchAndRewrite: cf.br (after) -> " << newOp);
    return success();
  }
};

// MemRef allocation with pointer element types -> LLVM malloc + struct
// construction
struct MemRefAllocConverter : OpConversionPattern<memref::AllocOp> {
  using OpConversionPattern<memref::AllocOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::AllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    LDBG("matchAndRewrite: memref.alloc (before) " << op);

    auto oldMemRefType = op.getType();
    auto elementType = oldMemRefType.getElementType();

    // Only handle memref with pointer element types
    if (!isa<ptr::PtrType>(elementType)) {
      return failure();
    }

    // Convert to LLVM struct type
    Type llvmStructType = getTypeConverter()->convertType(oldMemRefType);
    if (!llvmStructType) {
      return rewriter.notifyMatchFailure(op, "failed to convert memref type");
    }

    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();
    auto i64Ty = rewriter.getIntegerType(64);
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);
    auto shape = oldMemRefType.getShape();
    auto rank = shape.size();

    // Calculate total size (number of elements * pointer size)
    int64_t totalElements = 1;
    for (auto dim : shape) {
      if (dim == ShapedType::kDynamic) {
        return rewriter.notifyMatchFailure(op,
                                           "dynamic shapes not supported yet");
      }
      totalElements *= dim;
    }

    // For now, use alloca instead of malloc to avoid complex call setup
    Value totalSize = rewriter.create<LLVM::ConstantOp>(
        loc, i64Ty, rewriter.getIntegerAttr(i64Ty, totalElements));

    Value allocatedPtr = rewriter.create<LLVM::AllocaOp>(
        loc, ptrTy, ptrTy, totalSize, /*alignment=*/0);

    // Build memref descriptor struct
    Value result = rewriter.create<LLVM::UndefOp>(loc, llvmStructType);
    result = rewriter.create<LLVM::InsertValueOp>(loc, result, allocatedPtr,
                                                  0); // base_ptr
    result = rewriter.create<LLVM::InsertValueOp>(loc, result, allocatedPtr,
                                                  1); // aligned_ptr

    Value zeroOffset = rewriter.create<LLVM::ConstantOp>(
        loc, i64Ty, rewriter.getIntegerAttr(i64Ty, 0));
    result = rewriter.create<LLVM::InsertValueOp>(loc, result, zeroOffset, 2);

    // Set sizes and strides
    SmallVector<int64_t> strides(rank, 1);
    for (int i = rank - 2; i >= 0; --i) {
      strides[i] = strides[i + 1] * shape[i + 1];
    }

    for (auto [i, size] : llvm::enumerate(shape)) {
      Value sizeVal = rewriter.create<LLVM::ConstantOp>(
          loc, i64Ty, rewriter.getIntegerAttr(i64Ty, size));
      result = rewriter.create<LLVM::InsertValueOp>(
          loc, result, sizeVal, ArrayRef<int64_t>{3, static_cast<int64_t>(i)});

      Value strideVal = rewriter.create<LLVM::ConstantOp>(
          loc, i64Ty, rewriter.getIntegerAttr(i64Ty, strides[i]));
      result = rewriter.create<LLVM::InsertValueOp>(
          loc, result, strideVal,
          ArrayRef<int64_t>{4, static_cast<int64_t>(i)});
    }

    rewriter.replaceOp(op, result);
    LDBG("matchAndRewrite: memref.alloc (after) -> " << result);
    return success();
  }
};

// MemRef store with pointer element types -> LLVM GEP + store
struct MemRefStoreConverter : OpConversionPattern<memref::StoreOp> {
  using OpConversionPattern<memref::StoreOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    LDBG("matchAndRewrite: memref.store (before) " << op);

    auto memrefType = op.getMemRef().getType();
    if (auto memrefTy = dyn_cast<MemRefType>(memrefType)) {
      auto elementType = memrefTy.getElementType();

      // Only handle memref with pointer element types
      if (!isa<ptr::PtrType>(elementType)) {
        return failure();
      }

      auto loc = op.getLoc();
      auto ctx = rewriter.getContext();
      auto ptrTy = LLVM::LLVMPointerType::get(ctx);
      auto i64Ty = rewriter.getIntegerType(64);

      // Extract base pointer from memref descriptor (index 0)
      Value memrefDescriptor = adaptor.getMemref();
      Value basePtr = rewriter.create<LLVM::ExtractValueOp>(
          loc, ptrTy, memrefDescriptor, rewriter.getDenseI64ArrayAttr({0}));

      // Calculate linear index from multi-dimensional indices
      Value linearIndex = nullptr;
      if (adaptor.getIndices().size() == 1) {
        // Single dimension case
        Value index = adaptor.getIndices()[0];
        // Convert index to i64 if needed
        if (index.getType() != i64Ty) {
          if (isa<IndexType>(index.getType())) {
            index =
                rewriter.create<UnrealizedConversionCastOp>(loc, i64Ty, index)
                    .getResult(0);
          }
        }
        linearIndex = index;
      } else {
        // Multi-dimensional: linearIndex = i0*stride0 + i1*stride1 + ...
        linearIndex = rewriter.create<LLVM::ConstantOp>(
            loc, i64Ty, rewriter.getIntegerAttr(i64Ty, 0));

        for (auto [i, index] : llvm::enumerate(adaptor.getIndices())) {
          // Convert index to i64 if needed
          Value convertedIndex = index;
          if (index.getType() != i64Ty) {
            if (isa<IndexType>(index.getType())) {
              convertedIndex =
                  rewriter.create<UnrealizedConversionCastOp>(loc, i64Ty, index)
                      .getResult(0);
            }
          }

          Value stride = rewriter.create<LLVM::ExtractValueOp>(
              loc, i64Ty, memrefDescriptor,
              rewriter.getDenseI64ArrayAttr({4, static_cast<int64_t>(i)}));
          Value contribution =
              rewriter.create<LLVM::MulOp>(loc, convertedIndex, stride);
          linearIndex =
              rewriter.create<LLVM::AddOp>(loc, linearIndex, contribution);
        }
      }

      // GEP to get the address of the element
      Value elementPtr =
          rewriter.create<LLVM::GEPOp>(loc, ptrTy, ptrTy, basePtr, linearIndex);

      // Store the value
      auto storeOp =
          rewriter.create<LLVM::StoreOp>(loc, adaptor.getValue(), elementPtr);
      rewriter.eraseOp(op);
      LDBG("matchAndRewrite: memref.store (after) -> " << storeOp);
      return success();
    }

    return failure();
  }
};

// MemRef load with pointer element types -> LLVM GEP + load
struct MemRefLoadConverter : OpConversionPattern<memref::LoadOp> {
  using OpConversionPattern<memref::LoadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    LDBG("matchAndRewrite: memref.load (before) " << op);

    auto memrefType = op.getMemRef().getType();
    if (auto memrefTy = dyn_cast<MemRefType>(memrefType)) {
      auto elementType = memrefTy.getElementType();

      // Only handle memref with pointer element types
      if (!isa<ptr::PtrType>(elementType)) {
        return failure();
      }

      // Convert result type through type converter
      Type newResultType = getTypeConverter()->convertType(op.getType());
      if (!newResultType) {
        return rewriter.notifyMatchFailure(op, "failed to convert result type");
      }

      auto loc = op.getLoc();
      auto ctx = rewriter.getContext();
      auto ptrTy = LLVM::LLVMPointerType::get(ctx);
      auto i64Ty = rewriter.getIntegerType(64);

      // Extract base pointer from memref descriptor (index 0)
      Value memrefDescriptor = adaptor.getMemref();
      Value basePtr = rewriter.create<LLVM::ExtractValueOp>(
          loc, ptrTy, memrefDescriptor, rewriter.getDenseI64ArrayAttr({0}));

      // Calculate linear index from multi-dimensional indices
      Value linearIndex = nullptr;
      if (adaptor.getIndices().size() == 1) {
        // Single dimension case
        Value index = adaptor.getIndices()[0];
        // Convert index to i64 if needed
        if (index.getType() != i64Ty) {
          if (isa<IndexType>(index.getType())) {
            index =
                rewriter.create<UnrealizedConversionCastOp>(loc, i64Ty, index)
                    .getResult(0);
          }
        }
        linearIndex = index;
      } else {
        // Multi-dimensional: linearIndex = i0*stride0 + i1*stride1 + ...
        linearIndex = rewriter.create<LLVM::ConstantOp>(
            loc, i64Ty, rewriter.getIntegerAttr(i64Ty, 0));

        for (auto [i, index] : llvm::enumerate(adaptor.getIndices())) {
          // Convert index to i64 if needed
          Value convertedIndex = index;
          if (index.getType() != i64Ty) {
            if (isa<IndexType>(index.getType())) {
              convertedIndex =
                  rewriter.create<UnrealizedConversionCastOp>(loc, i64Ty, index)
                      .getResult(0);
            }
          }

          Value stride = rewriter.create<LLVM::ExtractValueOp>(
              loc, i64Ty, memrefDescriptor,
              rewriter.getDenseI64ArrayAttr({4, static_cast<int64_t>(i)}));
          Value contribution =
              rewriter.create<LLVM::MulOp>(loc, convertedIndex, stride);
          linearIndex =
              rewriter.create<LLVM::AddOp>(loc, linearIndex, contribution);
        }
      }

      // GEP to get the address of the element
      Value elementPtr =
          rewriter.create<LLVM::GEPOp>(loc, ptrTy, ptrTy, basePtr, linearIndex);

      // Load the value
      Value loadedValue =
          rewriter.create<LLVM::LoadOp>(loc, newResultType, elementPtr);
      rewriter.replaceOp(op, loadedValue);

      LDBG("matchAndRewrite: memref.load (after) -> " << loadedValue);
      return success();
    }

    return failure();
  }
};

// TypeOffsetOp -> constant conversion
struct TypeOffsetConverter : OpConversionPattern<tptr::TypeOffsetOp> {
  using OpConversionPattern<tptr::TypeOffsetOp>::OpConversionPattern;

  llvm::TypeSize
  getTypeSize(tptr::TypeOffsetOp op,
              std::optional<DataLayout> layout = std::nullopt) const {
    if (layout)
      return layout->getTypeSize(op.getBaseType());
    DataLayout dl = DataLayout::closest(op);
    return dl.getTypeSize(op.getBaseType());
  }

  LogicalResult
  matchAndRewrite(tptr::TypeOffsetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    LDBG("matchAndRewrite: type_offset (before) " << op);

    auto size = getTypeSize(op);
    if (size.isScalable()) {
      return rewriter.notifyMatchFailure(op, "scalable type size unsupported");
    }
    auto fixedSize = static_cast<int64_t>(size.getFixedValue());
    auto constOp = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), op.getType(),
        rewriter.getIntegerAttr(op.getType(), fixedSize));

    rewriter.replaceOp(op, constOp);
    LDBG("matchAndRewrite: type_offset (after) -> " << constOp);
    return success();
  }
};

void populateTPtrToLLVMConversionPatterns(RewritePatternSet &patterns,
                                          TypeConverter &typeConverter) {
  patterns.add<TypeOffsetConverter, PtrAddConverter, FromMemrefConverter,
               ConvertBranchOp, ConvertControlFlowOp, ToMemrefConverter,
               UnrealizedCastConverter, MemRefAllocConverter,
               MemRefStoreConverter, MemRefLoadConverter>(
      typeConverter, patterns.getContext());
}

} // namespace tptr
} // namespace mlir
