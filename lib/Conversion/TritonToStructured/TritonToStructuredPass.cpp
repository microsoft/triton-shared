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

  LogicalResult convertToPointerTupleWithOffsetsAndStrides() {
    auto moduleOp = getOperation();

    RewritePatternSet patterns(&getContext());

    auto context = &getContext();
    OneToNTypeConverter converter;
    converter.addConversion([](Type type) { return type; });

    // We are doing a 1->1 type conversion here, where a triton pointer type
    // maps to a tuple of {pointer, offset_0, offset_1,..., stride_0,
    // stride_1,...} type.

    // Case 1: Unstructured pointers (tensor<!tt.ptr<type>>)
    converter.addConversion(
        [context](RankedTensorType tensorType, SmallVectorImpl<Type> &types)
            -> std::optional<LogicalResult> {
          if (auto ptrType =
                  dyn_cast<triton::PointerType>(tensorType.getElementType())) {
            auto rank = tensorType.getRank();
            auto offsetAndStrideTuple = TupleType::get(
                context, SmallVector<Type>(rank * 2, IndexType::get(context)));
            auto tupleType = TupleType::get(
                context, SmallVector<Type>{tensorType, offsetAndStrideTuple});
            types = SmallVector<Type>{tupleType};
            return success();
          }
          return std::nullopt;
        });

    // Case 2: Block pointers (!tt.ptr<tensor<>> or !tt.ptr<type>)
    converter.addConversion(
        [context](triton::PointerType ptrType, SmallVectorImpl<Type> &types)
            -> std::optional<LogicalResult> {
          if (auto tensorType =
                  llvm::dyn_cast<RankedTensorType>(ptrType.getPointeeType())) {
            // Block pointers
            auto rank = tensorType.getRank();
            auto offsetAndStrideTuple = TupleType::get(
                context, SmallVector<Type>(rank * 2, IndexType::get(context)));
            auto tupleType = TupleType::get(
                context, SmallVector<Type>{ptrType, offsetAndStrideTuple});
            types = SmallVector<Type>{tupleType};
          } else {
            // Scalar pointers
            // The only relevant state that can be updated in loops for scalar
            // pointers are offset. No need to include stride here.
            auto tupleType = TupleType::get(
                context, SmallVector<Type>{ptrType, IndexType::get(context)});
            types = SmallVector<Type>{tupleType};
          }
          return success();
        });

    // Hooks to compute the correct materialization, "argument" and "source"
    // materialization are used when we need to convert the tuple type back to
    // the original triton pointer type. These are used when there are ops that
    // still need to use the original pointer type. For instance, we convert the
    // result of tt.addptr from tt.ptr type to a tuple, but the original ptr
    // result is still being used by another tt.load or tt.store.
    auto materialize = [](OpBuilder &builder, Type resultType,
                          ValueRange inputs, Location loc) {
      return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
          .getResult(0);
    };

    converter.addArgumentMaterialization(materialize);
    converter.addSourceMaterialization(materialize);

    // Compute the target materialization, given a value with the pointer type,
    // convert that value to a tuple type.
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

  LogicalResult decomposePointerTuple() {
    auto moduleOp = getOperation();

    auto context = &getContext();
    OneToNTypeConverter converter;
    converter.addConversion([](Type type) { return type; });

    // We are doing a 1->N type conversion here, where a pointer tuple type
    // maps to a sequence of {pointer, offset_0, offset_1,..., stride_0,
    // stride_1,...}
    converter.addConversion(
        [context](TupleType tupleType, SmallVectorImpl<Type> &types)
            -> std::optional<LogicalResult> {
          tupleType.getFlattenedTypes(types);
          return success();
        });

    // Hooks to compute the correct materialization, "argument" and "source"
    // materialization are used when we need to convert a series of {pointer,
    // offset_0, offset_1,..., stride_0, stride_1,...} type back to the "pointer
    // tuple type".
    //
    // Because we actually want to get rid of the tuple type, return `inputs[0]`
    // which corresponds to a "triton pointer type". This approach will work as
    // intended because the ops that currently take "pointer tuple type" are
    // `unrealized_conversion_cast` ops which will get removed below during
    // reconcile-unrealized-conversion-casts.
    auto materialize = [](OpBuilder &builder, Type resultType,
                          ValueRange inputs,
                          Location loc) { return inputs[0]; };
    converter.addArgumentMaterialization(materialize);
    converter.addSourceMaterialization(materialize);

    // For each value of "pointer tuple type" that gets decomposed into a
    // sequence of {pointer, offset_0, offset_1,..., stride_0, stride_1,...},
    // create a `tts.get_structured_state` op that serves as a placeholder.
    // The return values for this op will be used as the init-args for scf.for.
    // At the end of pointer analysis, we will use the PtrState to create the
    // correct offsets, strides, and remove these ops.
    converter.addTargetMaterialization([](OpBuilder &builder,
                                          TypeRange resultTypes, Value input,
                                          Location loc) {
      auto placeholder = builder.create<tts::GetStructuredStateOp>(
          loc, resultTypes, input.getDefiningOp()->getOperand(0));
      return placeholder->getResults();
    });

    RewritePatternSet patterns(&getContext());
    scf::populateSCFStructuralOneToNTypeConversions(converter, patterns);
    if (failed(applyPartialOneToNConversion(getOperation(), converter,
                                            std::move(patterns)))) {
      return failure();
    }

    // Note:
    // Be careful not to run canonicalization here, because the
    // tts.get_structured_state ops created above are just placeholders and
    // don't have any effects. Canonicalization will remove them altogether.
    PassManager pm(&getContext(), moduleOp.getOperationName());
    pm.addPass(mlir::createReconcileUnrealizedCastsPass());
    if (failed(runPipeline(pm, getOperation()))) {
      signalPassFailure();
    }

    return success();
  }

  // Prepass that inserts `tts.get_structured_state` ops. These ops are used as
  // placeholders to make passing structured pointer state into scf.for loop's
  // init args easier, especially with multiple levels of loops.
  //
  // Background:
  // If a triton pointer is updated and returned in a scf.for op, it means
  // that we have to carry its offsets and strides in the scf.for's iterargs.
  // Previously, we have to manually rewrite the loops to include the
  // relevant information from a PtrState which was rather involved and
  // error-prone; this was also hard to scale up to multiple level of loops
  // because there are several book-keeping data structures that we have to
  // maintain.
  //
  // With the introduction of the prepass that inserts
  // `tts.get_structured_state`, the return values of these ops, which include a
  // triton pointer and its corresponding offsets and strides, will be used as
  // "placeholders" into the scf.for's init-args. We leverage standard MLIR
  // infrastructure 1->N conversion to perform this rewrite, which helps
  // simplify the logic significantly.
  //
  // After PtrAnalysis finishes, the return values of these
  // `tts.get_structured_state` ops will be remapped to the correct
  // initialization of the pointer's offsets and strides through the pointer's
  // computed PtrState.
  //
  // Implementation details:
  // In essence, what we really want to do in the prepass is, for every value
  // of triton-pointer-like type (tt.ptr or tensor<tt.ptr<>>), we want to
  // create an op `tts.get_structured_state` that takes in the original triton
  // pointer value and returns a series of values:
  //
  // {triton_ptr, offset_0, offset_1, ..., stride_0, stride_1,...}
  //
  // Applying the above conversion will also mean that any structural ops such
  // as scf.for and scf.yield that originally takes the triton pointer will
  // then take {triton_ptr, offset_0, offset_1, ..., stride_0, stride_1,...}.
  //
  // The 1->N type conversion is a perfect fit for this transformation.
  // Unfortunately, we cannot do this is one pass, because the current 1->N
  // type conversion implementation for scf.for ops doesn't provide us with a
  // way to detect that a type conversion is recursive. So a triton_ptr type
  // that gets converted to a {triton_ptr, offset_0, offset_1, ..., stride_0,
  // stride_1,...} will recursively trigger other conversions.
  //
  // To fix this issue, we have to first convert triton_ptr to
  // tuple<triton_ptr, offset_0, offset_1, ..., stride_0, stride_1,...>.
  // Finally, we decompose these tuples into the desired sequence.
  LogicalResult runTritonToStructuredPrepass() {
    if (failed(convertToPointerTupleWithOffsetsAndStrides())) {
      return failure();
    }

    return decomposePointerTuple();
  }

  void runOnOperation() override {
    if (failed(runTritonToStructuredPrepass())) {
      signalPassFailure();
      return;
    }

    if (runPrepassOnly) {
      return;
    }

    auto moduleOp = getOperation();
    mlir::tts::PtrAnalysis ptrAnalysis;
    if (failed(ptrAnalysis.rewriteOp(moduleOp))) {
      moduleOp->emitWarning("PtrAnalysis failed");
    }

    // Now that all the PtrStates have been populated, we can wire up the states
    // with the tts.get_structured_state ops inserted in the prepass.
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
