//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation, Meta Platforms.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LogicalResult.h"
#include "triton-shared/Analysis/OpFoldResultUtils.h"
#include "triton-shared/AnalysisStructured/PtrAnalysis.h"
#include "triton-shared/Conversion/TritonToStructured/TritonToStructured.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/OneToNTypeConversion.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include <cassert>
#include <optional>

#define DEBUG_TYPE "triton-to-structured"

using namespace mlir;
using namespace triton;

#define GEN_PASS_CLASSES
#include "triton-shared/Conversion/TritonToStructured/Passes.h.inc"

namespace {

class TritonToStructuredPass
    : public TritonToStructuredBase<TritonToStructuredPass> {

public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<arith::ArithDialect, math::MathDialect, affine::AffineDialect,
                scf::SCFDialect, tensor::TensorDialect, triton::TritonDialect,
                tts::TritonStructuredDialect>();
  }

  LogicalResult test() {
    auto moduleOp = getOperation();

    RewritePatternSet patterns(&getContext());

    auto context = &getContext();
    OneToNTypeConverter converter;
    converter.addConversion([](Type type) { return type; });

    // We are doing a 1->1 type conversion here, where a triton pointer type
    // maps to a pair of {memref, index} type for the the buffer and offset.
    converter.addConversion(
        [context](RankedTensorType tensorType, SmallVectorImpl<Type> &types)
            -> std::optional<LogicalResult> {
          // tensorType.dump();
          if (auto ptrType =
                  dyn_cast<triton::PointerType>(tensorType.getElementType())) {
            auto rank = tensorType.getRank();
            auto offsetAndStrideTuple = TupleType::get(
                context, SmallVector<Type>(rank * 2, IndexType::get(context)));
            auto tupleType = TupleType::get(
                context, SmallVector<Type>{tensorType, offsetAndStrideTuple});
            types = SmallVector<Type>{tupleType};
            // assert(0);
            return success();
          }
          return std::nullopt;
        });

    converter.addConversion(
        [context](triton::PointerType ptrType, SmallVectorImpl<Type> &types)
            -> std::optional<LogicalResult> {
          if (auto tensorType =
                  llvm::dyn_cast<RankedTensorType>(ptrType.getPointeeType())) {
            // Block ptr
            auto rank = tensorType.getRank();
            auto offsetAndStrideTuple = TupleType::get(
                context, SmallVector<Type>(rank * 2, IndexType::get(context)));
            auto tupleType = TupleType::get(
                context, SmallVector<Type>{ptrType, offsetAndStrideTuple});
            types = SmallVector<Type>{tupleType};

          } else {
            auto tupleType = TupleType::get(
                context, SmallVector<Type>{ptrType, IndexType::get(context)});
            types = SmallVector<Type>{tupleType};
          }
          return success();
        });

    // Hooks to compute the correct materialization, "argument" and "source"
    // materialization are used when we need to convert a pair of {memref,
    // index} type back to the original triton pointer type.
    // These are used when there are ops that still need to use the original
    // pointer type. For instance, we convert the result of tt.addptr from
    // tt.ptr type to a pair of {memref, index}, but the original ptr result is
    // still being used by another tt.load or tt.store.
    auto materialize = [](OpBuilder &builder, Type resultType,
                          ValueRange inputs, Location loc) {
      auto op =
          builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs);
      op->setAttr("state_placeholder", UnitAttr::get(builder.getContext()));
      return op.getResult(0);
    };

    converter.addArgumentMaterialization(materialize);
    converter.addSourceMaterialization(materialize);

    // Compute the target materialization, given a value with the pointer type,
    // convert that value to a pair of {memref, index} type.

    converter.addTargetMaterialization(
        [](OpBuilder &builder, TypeRange resultTypes, Value input,
           Location loc) -> std::optional<SmallVector<Value>> {
          return builder
              .create<UnrealizedConversionCastOp>(loc, resultTypes, input)
              ->getResults();
        });

    scf::populateSCFStructuralOneToNTypeConversions(converter, patterns);

    if (failed(applyPartialOneToNConversion(getOperation(), converter,
                                            std::move(patterns)))) {
      return failure();
    }

    PassManager pm(&getContext(), moduleOp.getOperationName());
    pm.addPass(createCanonicalizerPass());
    if (failed(runPipeline(pm, getOperation()))) {
      return failure();
    }

    return success();
  }

  LogicalResult test2() {
    auto moduleOp = getOperation();

    auto context = &getContext();
    OneToNTypeConverter converter;
    converter.addConversion([](Type type) { return type; });

    // We are doing a 1->1 type conversion here, where a triton pointer type
    // maps to a pair of {memref, index} type for the the buffer and offset.
    converter.addConversion(
        [context](TupleType tupleType, SmallVectorImpl<Type> &types)
            -> std::optional<LogicalResult> {
          tupleType.getFlattenedTypes(types);
          return success();
        });

    // Hooks to compute the correct materialization, "argument" and "source"
    // materialization are used when we need to convert a pair of {memref,
    // index} type back to the original triton pointer type.
    // These are used when there are ops that still need to use the original
    // pointer type. For instance, we convert the result of tt.addptr from
    // tt.ptr type to a pair of {memref, index}, but the original ptr result is
    // still being used by another tt.load or tt.store.
    auto materialize = [](OpBuilder &builder, Type resultType,
                          ValueRange inputs,
                          Location loc) { return inputs[0]; };
    converter.addArgumentMaterialization(materialize);
    converter.addSourceMaterialization(materialize);

    // Compute the target materialization, given a value with the pointer
    // type, convert that value to a pair of {memref, index} type.
    converter.addTargetMaterialization([](OpBuilder &builder,
                                          TypeRange resultTypes, Value input,
                                          Location loc) {
      auto placeholder = builder.create<tts::GetStructuredStateOp>(
          loc, resultTypes, input.getDefiningOp()->getOperand(0));
      return SmallVector<Value>{placeholder->getResults()};
    });

    RewritePatternSet patterns(&getContext());
    scf::populateSCFStructuralOneToNTypeConversions(converter, patterns);
    if (failed(applyPartialOneToNConversion(getOperation(), converter,
                                            std::move(patterns)))) {
      return failure();
    }

    PassManager pm(&getContext(), moduleOp.getOperationName());
    pm.addPass(mlir::createReconcileUnrealizedCastsPass());
    if (failed(runPipeline(pm, getOperation()))) {
      signalPassFailure();
    }

    return success();
  }

  void runOnOperation() override {
    (void)test();
    // assert(0);
    // return;
    // assert(0);
    (void)test2();
    auto moduleOp = getOperation();

    // return;

    mlir::tts::PtrAnalysis ptrAnalysis;
    if (ptrAnalysis.rewriteOp(moduleOp).failed()) {
      moduleOp->emitWarning("PtrAnalysis failed");
    }

    moduleOp.walk([&ptrAnalysis](tts::GetStructuredStateOp op) {
      if (failed(ptrAnalysis.rewriteGetStructuredStateOp(op))) {
        op.emitWarning("Rewriting GetStructuredStateOp failed.");
      }
    });
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
triton::createTritonToStructuredPass() {
  return std::make_unique<TritonToStructuredPass>();
}
