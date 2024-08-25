//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "triton-shared/Conversion/LinalgToLinearAlgebraSubprograms/LinalgToLinearAlgebraSubprograms.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "triton-shared/Dialect/TritonTilingExt/IR/TritonTilingExtDialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "linalg-to-las"

using namespace mlir;
using namespace triton;

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_LINALGTOLINEARALGEBRASUBPROGRAMS
#include "triton-shared/Conversion/LinalgToLinearAlgebraSubprograms/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

struct MatmulConverter : public OpConversionPattern<linalg::MatmulOp> {
  using OpConversionPattern<linalg::MatmulOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(linalg::MatmulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    if (op.getInputs().size() != 2) {
      LLVM_DEBUG(llvm::dbgs() << "Cannot replace, must be exactly two input matrices\n");
      return failure();
    }

    Operation *resultOp;

    Value A = op.getInputs()[0];
    Value B = op.getInputs()[1];
    Value C;

    Value matmulResult = op.getResults()[0];
    bool otherUsers = false;
    bool found = false;

    for (Operation *user : matmulResult.getUsers()) {
        if (auto addFOp = dyn_cast<arith::AddFOp>(user)) {
          if (!found) {
            found = true;
            C = addFOp.getLhs() == matmulResult ? addFOp.getRhs() : addFOp.getLhs();
            resultOp = addFOp;
            continue;
          }
        }
        otherUsers = true;
    }

    bool replacingFOp = true;
    if (otherUsers || !found) {
      C = op.getOutputs()[0];
      resultOp = op;
      replacingFOp = false;
    }

    auto tensorA = cast<RankedTensorType>(A.getType());
    auto tensorB = cast<RankedTensorType>(B.getType());
    auto tensorC = cast<RankedTensorType>(C.getType());

    if (tensorA.getElementType() != tensorB.getElementType() ||
        tensorC.getElementType() != tensorB.getElementType()) {
      LLVM_DEBUG(llvm::dbgs() << "Cannot replace, different element types\n");
      return failure();
    }

    if (!tensorA.getElementType().isF32() && !tensorA.getElementType().isF64()) {
      LLVM_DEBUG(llvm::dbgs() << "Cannot replace, unsupported type\n");
      return failure();
    }

    auto floatType = tensorA.getElementType();

    // since tensors are immutable, we need to allocate a buffer for the result
    Value memrefConst = rewriter.create<bufferization::ToMemrefOp>(loc, MemRefType::get(tensorC.getShape(), tensorC.getElementType()), C);
    auto memrefType = MemRefType::get(tensorC.getShape(), floatType);
    Value memrefC = rewriter.create<memref::AllocOp>(loc, memrefType);
    auto copyOp = rewriter.create<linalg::CopyOp>(loc, ValueRange{memrefConst}, ValueRange{memrefC});

    ModuleOp module = op->getParentOfType<ModuleOp>();

    auto intType = rewriter.getI32Type();
    auto int64Type = rewriter.getI64Type();
    auto ptrType = LLVM::LLVMPointerType::get(op.getContext(), 0); // default address space

    auto funcType = FunctionType::get(op.getContext(),
        {intType, intType, intType, intType, intType, intType, floatType,
         ptrType, intType, ptrType, intType, floatType,
         ptrType, intType}, {});

    bool usingF64 = floatType.isF64();
    const char *funcName = usingF64 ? "cblas_dgemm" : "cblas_sgemm";
    auto func = module.lookupSymbol<func::FuncOp>(funcName);
    if (!func) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      func = rewriter.create<func::FuncOp>(loc, funcName, funcType);
      func.setVisibility(SymbolTable::Visibility::Private);
    }

    auto memrefToPointer = [&rewriter, &loc, &int64Type, &ptrType](Value &memref) {
      auto indexPtr = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, memref);
      auto castOp = rewriter.create<arith::IndexCastOp>(loc, int64Type, indexPtr);
      return rewriter.create<LLVM::IntToPtrOp>(loc, ptrType, castOp);
    };

    auto tensorToPointer = [&rewriter, &loc, &memrefToPointer](Value &V, RankedTensorType &T) {
      if (auto tensorOp = V.getDefiningOp<bufferization::ToTensorOp>()) {
        Value ref = tensorOp.getMemref();
        return memrefToPointer(ref);
      }

      Value memref = rewriter.create<bufferization::ToMemrefOp>(loc, MemRefType::get(T.getShape(), T.getElementType()), V);
      return memrefToPointer(memref);
    };

    Value ptrA = tensorToPointer(A, tensorA);
    Value ptrB = tensorToPointer(B, tensorB);
    Value ptrC = memrefToPointer(memrefC);

    int32_t M = tensorA.getShape()[0];
    int32_t K = tensorA.getShape()[1];
    int32_t N = tensorB.getShape()[1];

    Value alpha = rewriter.create<arith::ConstantOp>(loc, floatType, usingF64 ? rewriter.getF64FloatAttr(1.0) : rewriter.getF32FloatAttr(1.0));
    Value beta = alpha;

    auto constOp = [&rewriter, &loc, &intType](int32_t V) {
      return rewriter.create<arith::ConstantOp>(loc, intType, rewriter.getI32IntegerAttr(V));
      };

    // constants below are from OpenBLAS library, check variable names for interpretation
    // for more information check: https://github.com/OpenMathLib/OpenBLAS/blob/develop/cblas.h
    Value CblasRowMajor = constOp(101), CblasNoTrans = constOp(111);
    Value MVal = constOp(M), NVal = constOp(N), KVal = constOp(K);
    Value LDA = KVal, LDB = NVal, LDC = NVal;

    auto funcOp = rewriter.create<func::CallOp>(loc, func, ValueRange{
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        MVal, NVal, KVal,
        alpha, ptrA, LDA,
        ptrB, LDB, beta,
        ptrC, LDC
    });

    auto toTensorOp = rewriter.create<bufferization::ToTensorOp>(loc,
        tensorC, memrefC, true /* restrict */, true /* writable */);

    if (!replacingFOp) {
      rewriter.replaceOp(op, toTensorOp);
    } else {
      rewriter.eraseOp(op);
      rewriter.replaceOp(resultOp, toTensorOp);
    }

    return success();
  }
};

class LinalgToLinearAlgebraSubprogramsPass
    : public triton::impl::LinalgToLinearAlgebraSubprogramsBase<LinalgToLinearAlgebraSubprogramsPass> {
  using LinalgToLinearAlgebraSubprogramsBase<
      LinalgToLinearAlgebraSubprogramsPass>::LinalgToLinearAlgebraSubprogramsBase;

public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<linalg::LinalgDialect, func::FuncDialect, arith::ArithDialect, math::MathDialect, bufferization::BufferizationDialect,
            affine::AffineDialect, scf::SCFDialect, tensor::TensorDialect, LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());

    patterns.add<MatmulConverter>(patterns.getContext());

    target.addLegalDialect<
        func::FuncDialect, arith::ArithDialect, math::MathDialect,
        affine::AffineDialect, scf::SCFDialect, linalg::LinalgDialect,
        cf::ControlFlowDialect, tensor::TensorDialect,
        bufferization::BufferizationDialect, memref::MemRefDialect, LLVM::LLVMDialect>();


    target.addDynamicallyLegalOp<linalg::MatmulOp>([](linalg::MatmulOp op) {
        Value A = op.getInputs()[0];
        Value B = op.getInputs()[1];

        auto tensorA = cast<RankedTensorType>(A.getType());
        auto tensorB = cast<RankedTensorType>(B.getType());

        if (tensorA.getElementType() != tensorB.getElementType()) {
          // no need to replace if types are different
          return true;
        }

        if (!tensorA.getElementType().isF32() && !tensorA.getElementType().isF64()) {
          // unsupported types
          return true;
        }

        return false;  // MatmulOp is illegal, and transformation is needed
    });

    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
triton::createLinalgToLinearAlgebraSubprogramsPass() {
  return std::make_unique<LinalgToLinearAlgebraSubprogramsPass>();
}
