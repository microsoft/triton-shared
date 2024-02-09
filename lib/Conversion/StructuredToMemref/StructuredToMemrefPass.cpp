//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation, Meta Platforms.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "triton-shared/Conversion/StructuredToMemref/StructuredToMemref.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "triton-shared/Dialect/TritonTilingExt/IR/TritonTilingExtDialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "llvm/Support/Debug.h"
#include <cassert>
#include <optional>

#define DEBUG_TYPE "structured-to-memref"

using namespace mlir;
using namespace triton;

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_STRUCTUREDTOMEMREF
#include "triton-shared/Conversion/StructuredToMemref/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

class TritonTypeConverter : public TypeConverter {
public:
  TritonTypeConverter() {
    // The order of type conversion is important: later ones are tried earlier.
    addConversion([](Type type) { return type; });
    addConversion([](triton::PointerType ptrType) {
      return UnrankedMemRefType::get(ptrType.getPointeeType(), 0);
    });
  }
};

class LoopTypeConverter : public TypeConverter {
public:
  LoopTypeConverter(MLIRContext *context) {
    // The order of type conversion is important: later ones are tried earlier.
    addConversion([](Type type) { return type; });
    addConversion([&](triton::PointerType ptrType) {
      SmallVector<int64_t> strides{1};

      auto layout =
          StridedLayoutAttr::get(context, ShapedType::kDynamic, strides);

      auto elemType = ptrType.getPointeeType();
      auto memrefType = MemRefType::get({1}, elemType, layout);
      return memrefType;
    });

    addArgumentMaterialization([&](OpBuilder &builder, Type resultType,
                                   ValueRange inputs,
                                   Location loc) -> std::optional<Value> {
      // return builder.create<UnrealizedConversionCastOp>(loc, resultType,
      // inputs)
      //     .getResult(0);
      if (auto memrefType = dyn_cast<MemRefType>(resultType)) {
        if (isa<UnrankedMemRefType>(inputs[0].getType())) {
          auto shape = memrefType.getShape();
          if (shape.size() == 1 && shape[0] == 1) {
            auto t = builder.create<memref::ReinterpretCastOp>(
                loc, memrefType, inputs[0], 0, ArrayRef<int64_t>{1},
                ArrayRef<int64_t>{1});
            t->dump();
            return t;
          }
        }
      }
      return std::nullopt;
    });
  }
};

class StructuredToMemrefPass
    : public triton::impl::StructuredToMemrefBase<StructuredToMemrefPass> {
  using StructuredToMemrefBase<StructuredToMemrefPass>::StructuredToMemrefBase;

public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect, arith::ArithDialect, math::MathDialect,
                    linalg::LinalgDialect, affine::AffineDialect,
                    scf::SCFDialect, tensor::TensorDialect,
                    bufferization::BufferizationDialect, triton::TritonDialect,
                    ttx::TritonTilingExtDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();

    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());
    TritonTypeConverter typeConverter;

    target.addLegalDialect<
        func::FuncDialect, arith::ArithDialect, math::MathDialect,
        linalg::LinalgDialect, affine::AffineDialect, scf::SCFDialect,
        cf::ControlFlowDialect, tensor::TensorDialect,
        bufferization::BufferizationDialect, ttx::TritonTilingExtDialect,
        memref::MemRefDialect>();

    target.addIllegalDialect<tts::TritonStructuredDialect>();

    target.addLegalOp<UnrealizedConversionCastOp>();

    // Update function signature to use memrefs
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return typeConverter.isSignatureLegal(op.getFunctionType());
    });

    target.addDynamicallyLegalOp<scf::ForOp>([](Operation *op) {
      return llvm::all_of(op->getResultTypes(), [](Type t) {
        if (isa<triton::PointerType>(t)) {
          return false;
        }
        if (auto shapedType = dyn_cast<ShapedType>(t)) {
          return shapedType.getElementType().isIntOrFloat();
        }
        assert(t.isIntOrIndexOrFloat());
        return true;
      });
    });

    LoopTypeConverter loop(patterns.getContext());
    triton::populateStructuredToMemrefConversionPatterns(patterns, loop);

    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);

    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) {
      signalPassFailure();
    }

    // Erase dead code and fold constants created during lowering
    PassManager pm(&getContext(), moduleOp.getOperationName());
    pm.addPass(createCanonicalizerPass());
    if (failed(runPipeline(pm, getOperation()))) {
      signalPassFailure();
    }
  }
};
} // namespace
