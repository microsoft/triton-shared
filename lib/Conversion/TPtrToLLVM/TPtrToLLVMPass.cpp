// mlir
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

// matcher
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

// rewriter
#include "mlir/Transforms/DialectConversion.h"

// pass
#include "triton-shared/Conversion/TPtrToLLVM/TPtrToLLVM.h"

// dialect
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
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
}  // namespace tptr
}  // namespace mlir

namespace {

struct SimplifyUnrealizedCast
    : public OpRewritePattern<UnrealizedConversionCastOp> {
  SimplifyUnrealizedCast(MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern<UnrealizedConversionCastOp>(context, benefit) {}

  LogicalResult matchAndRewrite(UnrealizedConversionCastOp op,
                                PatternRewriter &rewriter) const override {
    if (!isOneToOneCast(op)) {
      return failure();
    }
    auto in = op.getInputs().front();

    if (auto unrealizedCast = in.getDefiningOp<UnrealizedConversionCastOp>()) {
      if (!isOneToOneCast(unrealizedCast)) {
        return failure();
      }

      auto prevInput = unrealizedCast.getInputs().front();
      auto newCast = rewriter.create<UnrealizedConversionCastOp>(
          op->getLoc(), op->getResultTypes(), ValueRange{prevInput});

      rewriter.replaceOp(op, newCast);
      return success();
    }
    return failure();
  }
};

class TptrToLLVMTypeConverter : public TypeConverter {
 public:
  Type convertTritonPointerType(triton::PointerType type) {
    auto ctx = type.getContext();
    auto pointeeType = type.getPointeeType();
    Type ret;
    if (isa<RankedTensorType>(pointeeType)) {
      // struct {
      //   ptr base_ptr;
      //   array<rank x i64> offsets;
      //   array<rank x i64> shape;
      //   array<rank x i64> strides;
      // }
      auto tensorTy = cast<RankedTensorType>(pointeeType);
      auto rank = tensorTy.getShape().size();
      auto i64Ty = IntegerType::get(ctx, 64);
      SmallVector<Type, 4> types;
      types.push_back(LLVM::LLVMPointerType::get(ctx));
      types.push_back(LLVM::LLVMArrayType::get(ctx, i64Ty, rank));
      types.push_back(LLVM::LLVMArrayType::get(ctx, i64Ty, rank));
      types.push_back(LLVM::LLVMArrayType::get(ctx, i64Ty, rank));
      ret = LLVM::LLVMStructType::getLiteral(ctx, types);
    } else {
      ret = LLVM::LLVMPointerType::get(ctx);
    }
    LDBG("convertTritonPointerType: " << type << " -> " << ret << "\n");
    return ret;
  }


  TptrToLLVMTypeConverter(MLIRContext *ctx) {
    addConversion([ctx](Type type) -> Type { return type; });
    addConversion([&](triton::PointerType type) -> std::optional<Type> {
      return convertTritonPointerType(type);
    });
    auto createUnrealizedCast = [&](OpBuilder &builder, Type resultType,
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

    RewritePatternSet r_patterns(&getContext());
    r_patterns.add<SimplifyUnrealizedCast>(&getContext());
    if (failed(applyPatternsGreedily(moduleOp, std::move(r_patterns)))) {
      signalPassFailure();
    }
    LDBG("runOnOperation: simplify unrealized cast done\nmoduleOp: \n"
         << moduleOp);

    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());
    TptrToLLVMTypeConverter typeConverter(&getContext());

    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addLegalDialect<cf::ControlFlowDialect>();
    target.addLegalDialect<BuiltinDialect>();

    target.addIllegalDialect<tptr::TPtrDialect>();

    target.addDynamicallyLegalOp<cf::CondBranchOp>([&](cf::CondBranchOp op) {
      for (auto operand : op.getOperands()) {
        if (isa<triton::PointerType>(operand.getType())) {
          LDBG(
              "CondBranchOp marked illegal due to "
              "operand type: "
              << operand.getType() << "\n");
          return false;
        }
      }
      for (auto dest : {op.getTrueDest(), op.getFalseDest()}) {
        for (auto arg : dest->getArguments()) {
          if (isa<triton::PointerType>(arg.getType())) {
            LDBG(
                "CondBranchOp marked illegal due to "
                "block arg type: "
                << arg.getType() << "\n");
            return false;
          }
        }
      }
      return true;
    });

    target.addDynamicallyLegalOp<cf::BranchOp>([&](cf::BranchOp op) {
      for (auto operand : op.getOperands()) {
        if (isa<triton::PointerType>(operand.getType())) {
          LDBG("BranchOp marked illegal due to operand type: "
               << operand.getType() << "\n");
          return false;
        }
      }
      for (auto arg : op.getDest()->getArguments()) {
        if (isa<triton::PointerType>(arg.getType())) {
          LDBG("BranchOp marked illegal due to block arg type: "
               << arg.getType() << "\n");
          return false;
        }
      }
      return true;
    });

    target.addDynamicallyLegalOp<UnrealizedConversionCastOp>(
        [&](UnrealizedConversionCastOp op) {
          for (auto type : op.getResultTypes()) {
            if (isa<triton::PointerType>(type)) return false;
          }
          for (auto operand : op.getOperands()) {
            if (isa<triton::PointerType>(operand.getType())) return false;
          }


          return true;
        });

    target.addLegalOp<ModuleOp>();

    populateTPtrToLLVMConversionPatterns(patterns, typeConverter);
    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) {
      signalPassFailure();
    }
    LDBG("runOnOperation: after conversion\nmoduleOp: \n" << moduleOp);
    {
      mlir::PassManager pm(&getContext(), getOperation().getOperationName());
      pm.addPass(mlir::createCanonicalizerPass());
      pm.addPass(mlir::createCSEPass());
      if (failed(runPipeline(pm, getOperation()))) {
        signalPassFailure();
      }
    }

    LDBG("runOnOperation: done\nmoduleOp: \n" << moduleOp);
  }
};
}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> tptr::createTPtrToLLVMPass() {
  return std::make_unique<TPtrToLLVMPass>();
}
