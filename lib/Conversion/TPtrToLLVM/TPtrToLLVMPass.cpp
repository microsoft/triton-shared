#include <memory>

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Ptr/IR/PtrAttrs.h"
#include "mlir/Dialect/Ptr/IR/PtrDialect.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "triton-shared/Conversion/TPtrToLLVM/TPtrToLLVM.h"
#include "triton-shared/Dialect/TPtr/IR/TPtrDialect.h"

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

struct TptrToLLVMTypeConverter : TypeConverter {
  TptrToLLVMTypeConverter(MLIRContext *ctx) {
    addConversion([](Type type) -> Type { return type; });

    addConversion([&](MemRefType type) -> std::optional<Type> {
      auto elementType = type.getElementType();
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
    addTypeAttributeConversion(
        [&](PtrLikeTypeInterface type, ptr::GenericSpaceAttr memorySpace)
            -> TypeConverter::AttributeConversionResult {
          if (type.getMemorySpace() != memorySpace)
            return TypeConverter::AttributeConversionResult::na();
          return IntegerAttr::get(IntegerType::get(type.getContext(), 32), 0);
        });
    addTypeAttributeConversion(
        [&](PtrLikeTypeInterface type, tptr::DefaultMemorySpaceAttr memorySpace)
            -> TypeConverter::AttributeConversionResult {
          if (type.getMemorySpace() != memorySpace)
            return TypeConverter::AttributeConversionResult::na();
          // Default memory space maps to LLVM addrspace(0).
          return IntegerAttr::get(IntegerType::get(type.getContext(), 32), 0);
        });

    // Add type conversions.
    addConversion([&](ptr::PtrType type) -> Type {
      LDBG("MemorySpace " << type.getMemorySpace());
      std::optional<Attribute> maybeAttr =
          convertTypeAttribute(type, type.getMemorySpace());
      auto memSpace =
          maybeAttr ? dyn_cast_or_null<IntegerAttr>(*maybeAttr) : IntegerAttr();
      if (!memSpace) {
        return {};
      }
      return LLVM::LLVMPointerType::get(type.getContext(),
                                        memSpace.getValue().getSExtValue());
    });

    auto createUnrealizedCast = [](OpBuilder &builder, Type resultType,
                                   ValueRange inputs, Location loc) -> Value {
      return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
          .getResult(0);
    };
    addSourceMaterialization(createUnrealizedCast);
    addTargetMaterialization(createUnrealizedCast);
  }
};

class TPtrToLLVMPass : public tptr::impl::TPtrToLLVMBase<TPtrToLLVMPass> {
  using TPtrToLLVMBase<TPtrToLLVMPass>::TPtrToLLVMBase;

public:
  void runOnOperation() override {
    auto moduleOp = getOperation();

    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());
    TptrToLLVMTypeConverter typeConverter(&getContext());

    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addLegalDialect<cf::ControlFlowDialect>();
    target.addLegalDialect<BuiltinDialect>();
    target.addIllegalDialect<tptr::TPtrDialect>();
    target.addDynamicallyLegalOp<cf::CondBranchOp>([&](cf::CondBranchOp op) {
      for (auto operand : op.getOperands()) {
        if (isa<ptr::PtrType>(operand.getType())) {
          LDBG("CondBranchOp marked illegal due to operand type: "
               << operand.getType());
          return false;
        }
      }
      for (auto dest : {op.getTrueDest(), op.getFalseDest()}) {
        for (auto arg : dest->getArguments()) {
          if (isa<ptr::PtrType>(arg.getType())) {
            LDBG("CondBranchOp marked illegal due to block arg type: "
                 << arg.getType());
            return false;
          }
        }
      }
      return true;
    });

    target.addDynamicallyLegalOp<cf::BranchOp>([&](cf::BranchOp op) {
      for (auto operand : op.getOperands()) {
        if (isa<ptr::PtrType>(operand.getType())) {
          LDBG("BranchOp marked illegal due to operand type: "
               << operand.getType());
          return false;
        }
      }

      for (auto arg : op.getDest()->getArguments()) {
        if (isa<ptr::PtrType>(arg.getType())) {
          LDBG("BranchOp marked illegal due to block arg type: "
               << arg.getType());
          return false;
        }
      }
      return true;
    });

    target.addDynamicallyLegalOp<UnrealizedConversionCastOp>(
        [&](UnrealizedConversionCastOp op) {
          for (auto type : op.getResultTypes()) {
            if (isa<ptr::PtrType>(type)) {
              return false;
            }
          }
          for (auto operand : op.getOperands()) {
            if (isa<ptr::PtrType>(operand.getType())) {
              return false;
            }
          }
          return true;
        });

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

    populateTPtrToLLVMConversionPatterns(patterns, typeConverter);
    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
    LDBG("runOnOperation: after conversion\n" << moduleOp);

    {
      mlir::PassManager pm(&getContext(), getOperation().getOperationName());
      pm.addPass(mlir::createCanonicalizerPass());
      pm.addPass(mlir::createCSEPass());
      if (failed(runPipeline(pm, getOperation()))) {
        signalPassFailure();
        return;
      }
    }

    LDBG("runOnOperation: done\n" << moduleOp);
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> tptr::createTPtrToLLVMPass() {
  return std::make_unique<TPtrToLLVMPass>();
}
