#include <memory>

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "triton-shared/Conversion/TPtrToLLVM/TPtrToLLVM.h"
#include "triton-shared/Dialect/TPtr/IR/TPtrDialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

#define DEBUG_TYPE "tptr-to-llvm"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace tptr;

namespace mlir {
namespace tptr {
#define GEN_PASS_DEF_TPTRTOLLVM
#include "triton-shared/Conversion/TPtrToLLVM/Passes.h.inc"
} // namespace tptr
} // namespace mlir

namespace {

// Simplify chained UnrealizedConversionCast operations
struct SimplifyUnrealizedCast
    : public OpRewritePattern<UnrealizedConversionCastOp> {
  SimplifyUnrealizedCast(MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern<UnrealizedConversionCastOp>(context, benefit) {}

  LogicalResult matchAndRewrite(UnrealizedConversionCastOp op,
                                PatternRewriter &rewriter) const override {
    if (!isOneToOneCast(op)) {
      return failure();
    }

    auto input = op.getInputs().front();
    if (auto unrealizedCast =
            input.getDefiningOp<UnrealizedConversionCastOp>()) {
      if (!isOneToOneCast(unrealizedCast)) {
        return failure();
      }

      // Collapse chained casts: cast(cast(x)) -> cast(x)
      auto prevInput = unrealizedCast.getInputs().front();
      auto newCast = rewriter.create<UnrealizedConversionCastOp>(
          op->getLoc(), op->getResultTypes(), ValueRange{prevInput});

      rewriter.replaceOp(op, newCast);
      return success();
    }
    return failure();
  }
};

// Type converter for TPtr to LLVM conversion
class TptrToLLVMTypeConverter : public TypeConverter {
public:
  Type convertPtrPointerType(ptr::PtrType type) {
    auto ctx = type.getContext();
    auto pointeeType = type.getElementType();

    if (pointeeType && isa<RankedTensorType>(pointeeType)) {
      // Convert tensor pointer to struct with metadata
      auto tensorTy = cast<RankedTensorType>(pointeeType);
      auto rank = tensorTy.getShape().size();
      auto i64Ty = IntegerType::get(ctx, 64);

      SmallVector<Type, 4> types{
          LLVM::LLVMPointerType::get(ctx),             // base_ptr
          LLVM::LLVMArrayType::get(ctx, i64Ty, rank),  // offsets
          LLVM::LLVMArrayType::get(ctx, i64Ty, rank),  // shape
          LLVM::LLVMArrayType::get(ctx, i64Ty, rank)}; // strides

      return LLVM::LLVMStructType::getLiteral(ctx, types);
    }

    return LLVM::LLVMPointerType::get(ctx);
  }

  TptrToLLVMTypeConverter(MLIRContext *ctx) {
    // Identity conversion for non-pointer types
    addConversion([](Type type) -> Type { return type; });

    // Convert ptr::PtrType to LLVM pointer types
    addConversion([&](ptr::PtrType type) -> std::optional<Type> {
      return convertPtrPointerType(type);
    });

    // Convert memref containing pointer types to LLVM struct
    addConversion([&](MemRefType type) -> std::optional<Type> {
      auto elementType = type.getElementType();

      // Only convert memref if it contains pointer types
      if (!isa<ptr::PtrType>(elementType)) {
        return std::nullopt; // Use default conversion
      }

      // Convert memref<NxPtr> to LLVM struct (similar to standard memref
      // lowering)
      auto ctx = type.getContext();
      auto rank = type.getShape().size();
      auto i64Ty = IntegerType::get(ctx, 64);

      SmallVector<Type, 5> types{
          LLVM::LLVMPointerType::get(ctx),             // base_ptr
          LLVM::LLVMPointerType::get(ctx),             // aligned_ptr
          i64Ty,                                       // offset
          LLVM::LLVMArrayType::get(ctx, i64Ty, rank),  // sizes
          LLVM::LLVMArrayType::get(ctx, i64Ty, rank)}; // strides

      return LLVM::LLVMStructType::getLiteral(ctx, types);
    });

    // Materialization functions for unrealized casts
    auto createUnrealizedCast = [](OpBuilder &builder, Type resultType,
                                   ValueRange inputs, Location loc) -> Value {
      return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
          .getResult(0);
    };
    addSourceMaterialization(createUnrealizedCast);
    addTargetMaterialization(createUnrealizedCast);
  }
};

// Main conversion pass implementation
class TPtrToLLVMPass : public tptr::impl::TPtrToLLVMBase<TPtrToLLVMPass> {
  using TPtrToLLVMBase<TPtrToLLVMPass>::TPtrToLLVMBase;

public:
  void runOnOperation() override {
    auto moduleOp = getOperation();

    // Phase 1: Simplify unrealized casts
    RewritePatternSet simplifyPatterns(&getContext());
    simplifyPatterns.add<SimplifyUnrealizedCast>(&getContext());
    if (failed(applyPatternsGreedily(moduleOp, std::move(simplifyPatterns)))) {
      signalPassFailure();
      return;
    }
    LDBG("runOnOperation: simplify unrealized cast done");

    // Phase 2: Main TPtr to LLVM conversion
    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());
    TptrToLLVMTypeConverter typeConverter(&getContext());

    // Legal dialects after conversion
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addLegalDialect<cf::ControlFlowDialect>();
    target.addLegalDialect<BuiltinDialect>();

    // Illegal dialects to be converted
    target.addIllegalDialect<tptr::TPtrDialect>();

    // Dynamic legality for control flow ops with pointer operands
    target.addDynamicallyLegalOp<cf::CondBranchOp>([&](cf::CondBranchOp op) {
      // Check operands
      for (auto operand : op.getOperands()) {
        if (isa<ptr::PtrType>(operand.getType())) {
          LDBG("CondBranchOp marked illegal due to operand type: "
               << operand.getType());
          return false;
        }
      }

      // Check block arguments
      for (auto dest : {op.getTrueDest(), op.getFalseDest()}) {
        for (auto arg : dest->getArguments()) {
          if (isa<triton::PointerType, ptr::PtrType>(arg.getType())) {
            LDBG("CondBranchOp marked illegal due to block arg type: "
                 << arg.getType());
            return false;
          }
        }
      }
      return true;
    });

    target.addDynamicallyLegalOp<cf::BranchOp>([&](cf::BranchOp op) {
      // Check operands
      for (auto operand : op.getOperands()) {
        if (isa<triton::PointerType, ptr::PtrType>(operand.getType())) {
          LDBG("BranchOp marked illegal due to operand type: "
               << operand.getType());
          return false;
        }
      }

      // Check destination block arguments
      for (auto arg : op.getDest()->getArguments()) {
        if (isa<triton::PointerType, ptr::PtrType>(arg.getType())) {
          LDBG("BranchOp marked illegal due to block arg type: "
               << arg.getType());
          return false;
        }
      }
      return true;
    });

    // Unrealized casts involving pointer types are illegal
    target.addDynamicallyLegalOp<UnrealizedConversionCastOp>(
        [&](UnrealizedConversionCastOp op) {
          for (auto type : op.getResultTypes()) {
            if (isa<triton::PointerType, ptr::PtrType>(type)) {
              return false;
            }
          }
          for (auto operand : op.getOperands()) {
            if (isa<triton::PointerType, ptr::PtrType>(operand.getType())) {
              return false;
            }
          }
          return true;
        });

    // MemRef operations with pointer element types need conversion
    target.addDynamicallyLegalOp<memref::AllocOp>([&](memref::AllocOp op) {
      auto memrefType = op.getType();
      auto elementType = memrefType.getElementType();
      if (isa<ptr::PtrType>(elementType)) {
        LDBG("AllocOp marked illegal due to pointer element type: "
             << elementType);
        return false;
      }
      return true;
    });

    target.addDynamicallyLegalOp<memref::StoreOp>([&](memref::StoreOp op) {
      auto memrefType = op.getMemRef().getType();
      if (auto memrefTy = dyn_cast<MemRefType>(memrefType)) {
        auto elementType = memrefTy.getElementType();
        if (isa<ptr::PtrType>(elementType)) {
          LDBG("StoreOp marked illegal due to pointer element type: "
               << elementType);
          return false;
        }
      }
      return true;
    });

    target.addDynamicallyLegalOp<memref::LoadOp>([&](memref::LoadOp op) {
      auto memrefType = op.getMemRef().getType();
      if (auto memrefTy = dyn_cast<MemRefType>(memrefType)) {
        auto elementType = memrefTy.getElementType();
        if (isa<ptr::PtrType>(elementType)) {
          LDBG("LoadOp marked illegal due to pointer element type: "
               << elementType);
          return false;
        }
      }
      return true;
    });

    target.addLegalOp<ModuleOp>();

    // Apply conversion patterns
    populateTPtrToLLVMConversionPatterns(patterns, typeConverter);
    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
    LDBG("runOnOperation: after conversion");

    // Phase 3: Cleanup and optimization
    {
      mlir::PassManager pm(&getContext(), getOperation().getOperationName());
      pm.addPass(mlir::createCanonicalizerPass());
      pm.addPass(mlir::createCSEPass());
      if (failed(runPipeline(pm, getOperation()))) {
        signalPassFailure();
        return;
      }
    }

    LDBG("runOnOperation: done");
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> tptr::createTPtrToLLVMPass() {
  return std::make_unique<TPtrToLLVMPass>();
}
