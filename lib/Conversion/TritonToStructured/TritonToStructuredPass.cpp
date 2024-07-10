//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation, Meta Platforms.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
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

Type getType(MLIRContext *context, triton::PointerType ptrType) {
  if (auto tensorType = cast<RankedTensorType>(ptrType.getPointeeType())) {
    auto rank = tensorType.getRank();
    auto offsetTuple = TupleType::get(
        context, SmallVector<Type>(rank, IndexType::get(context)));
    auto strideTuple = TupleType::get(
        context, SmallVector<Type>(rank, IndexType::get(context)));
    auto tupleType = TupleType::get(
        context, SmallVector<Type>{ptrType, offsetTuple, strideTuple});
    tupleType.dump();
    return tupleType;
  } else {
    return TupleType::get(context,
                          SmallVector<Type>{ptrType, IndexType::get(context),
                                            IndexType::get(context)});
  }
}

class TupleConverter : public TypeConverter {
public:
  TupleConverter(MLIRContext *context) {
    // The order of type conversion is important: later ones are tried earlier.
    addConversion([](Type type) { return type; });
    addConversion([context](triton::PointerType ptrType) -> TupleType {
      return TupleType::get(context,
                            SmallVector<Type>{ptrType, IndexType::get(context),
                                              IndexType::get(context)});
    });

    addConversion([context](
                      RankedTensorType tensorType) -> std::optional<TupleType> {
      if (auto ptrType =
              dyn_cast<triton::PointerType>(tensorType.getElementType())) {
        auto rank = tensorType.getRank();
        auto offsetTuple = TupleType::get(
            context, SmallVector<Type>(rank, IndexType::get(context)));
        auto strideTuple = TupleType::get(
            context, SmallVector<Type>(rank, IndexType::get(context)));
        auto tupleType = TupleType::get(
            context, SmallVector<Type>{tensorType, offsetTuple, strideTuple});
        tupleType.dump();
        return tupleType;
      }
      return std::nullopt;
    });
    // Used for converting tuple<tt.ptr, tuple<offset0, offset1,...>,
    // tuple<stride0, stride1,...>> back to tt.ptr type
    addSourceMaterialization([&](OpBuilder &builder,
                                 triton::PointerType ptrType, ValueRange inputs,
                                 Location loc) -> std::optional<Value> {
      auto cast =
          builder.create<UnrealizedConversionCastOp>(loc, ptrType, inputs);
      cast->setAttr("insert", UnitAttr::get(context));
      return cast->getResult(0);
    });
  }
};

static std::optional<SmallVector<Value>>
buildCastAndOffsetOps(OpBuilder &builder, TypeRange resultTypes, Value input,
                      Location loc) {

  auto cast =
      builder.create<UnrealizedConversionCastOp>(loc, resultTypes, input);
  cast->setAttr("make_state", UnitAttr::get(builder.getContext()));
  return SmallVector<Value>{cast->getResult(0)};
}

static std::optional<SmallVector<Value>>
buildCastAndOffsetOps2(OpBuilder &builder, TypeRange resultTypes, Value input,
                       Location loc) {
  assert(0);
  auto cast =
      builder.create<UnrealizedConversionCastOp>(loc, resultTypes, input);
  cast->setAttr("zz", UnitAttr::get(builder.getContext()));
  return SmallVector<Value>{cast->getResult(0)};
}

static std::optional<Value> buildCastOp(OpBuilder &builder, Type resultType,
                                        ValueRange inputs, Location loc) {
  // llvm::dbgs() << "build cast op\n";
  // llvm::dbgs() << "result type\n";
  // resultType.dump();
  // llvm::dbgs() << "inputs:\n";
  for (auto v : inputs) {
    // v.dump();
  }
  auto op = builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs);
  op->setAttr("state_placeholder", UnitAttr::get(builder.getContext()));
  return op.getResult(0);
}

struct UnrealizedCastConverter : public OpConversionPattern<triton::AddPtrOp> {
  using OpConversionPattern<triton::AddPtrOp>::OpConversionPattern;

  UnrealizedCastConverter(TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<triton::AddPtrOp>(typeConverter, context) {}

  LogicalResult
  matchAndRewrite(triton::AddPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto origType = op.getResult().getType();
    origType.dump();
    auto newType = getTypeConverter()->convertType(origType);
    auto cast = rewriter.create<UnrealizedConversionCastOp>(
        op->getLoc(), newType, op.getResult());
    cast->setAttr("make_state", UnitAttr::get(rewriter.getContext()));
    auto clone = rewriter.clone(*op.getOperation());
    rewriter.replaceOp(op, clone);
    rewriter.replaceAllUsesWith(clone->getResult(0), cast->getResult(0));
    return success();
  }
};

struct UnrealizedConverter
    : public OneToNOpConversionPattern<UnrealizedConversionCastOp> {
  using OneToNOpConversionPattern::OneToNOpConversionPattern;

  UnrealizedConverter(TypeConverter &typeConverter, MLIRContext *context)
      : OneToNOpConversionPattern<UnrealizedConversionCastOp>(typeConverter,
                                                              context) {}

  LogicalResult
  matchAndRewrite(UnrealizedConversionCastOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {

    auto loc = op->getLoc();

    if (op->hasAttr("make_state")) {

      auto tupleType = cast<TupleType>(op.getResult(0).getType());
      SmallVector<Type> resTypes;
      tupleType.getFlattenedTypes(resTypes);

      auto cast = rewriter.create<UnrealizedConversionCastOp>(
          op->getLoc(), resTypes, op.getInputs());
      cast->setAttr("make_state_new", UnitAttr::get(rewriter.getContext()));

      rewriter.replaceOp(op, cast->getResults(), adaptor.getResultMapping());
      return success();

    } else if (op->hasAttr("state_placeholder")) {
      auto input = op.getInputs()[0];
      // auto argCast = input.getDefiningOp();
      // argCast->dump();
      // op->dump();
      // input.dump();
      // argCast->getOperands()[0].dump();
      rewriter.replaceOp(op, input);
      return success();
    }
    return failure();
  }
};

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

          return failure();
        });

    // Hooks to compute the correct materialization, "argument" and "source"
    // materialization are used when we need to convert a pair of {memref,
    // index} type back to the original triton pointer type.
    // These are used when there are ops that still need to use the original
    // pointer type. For instance, we convert the result of tt.addptr from
    // tt.ptr type to a pair of {memref, index}, but the original ptr result is
    // still being used by another tt.load or tt.store.
    converter.addArgumentMaterialization(buildCastOp);
    converter.addSourceMaterialization(buildCastOp);

    // Compute the target materialization, given a value with the pointer type,
    // convert that value to a pair of {memref, index} type.
    converter.addTargetMaterialization(buildCastAndOffsetOps);

    // patterns.add<ScalarAddptrConverter>(converter, context);

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

  static std::optional<Value> buildCastOp2(OpBuilder &builder, Type resultType,
                                           ValueRange inputs, Location loc) {
    // assert(0);
    // llvm::dbgs() << "build cast op2\n";
    // llvm::dbgs() << "result type\n";
    // resultType.dump();
    // llvm::dbgs() << "inputs:\n";
    // for (auto v : inputs) {
    //   v.dump();
    // }
    return inputs[0];
  }

  LogicalResult test2() {
    auto moduleOp = getOperation();

    RewritePatternSet patterns(&getContext());

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
    converter.addArgumentMaterialization(buildCastOp2);
    // converter.addSourceMaterialization(buildCastOp2);

    // Compute the target materialization, given a value with the pointer type,
    // convert that value to a pair of {memref, index} type.
    converter.addTargetMaterialization(buildCastAndOffsetOps2);

    patterns.add<UnrealizedConverter>(converter, context);

    scf::populateSCFStructuralOneToNTypeConversions(converter, patterns);

    if (failed(applyPartialOneToNConversion(getOperation(), converter,
                                            std::move(patterns)))) {
      return failure();
    }

    // PassManager pm(&getContext(), moduleOp.getOperationName());
    // pm.addPass(createCanonicalizerPass());
    // if (failed(runPipeline(pm, getOperation()))) {
    //   return failure();
    // }

    return success();
  }

  LogicalResult decomposeMakeStateOp() {
    auto moduleOp = getOperation();

    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());

    target.addIllegalDialect<tts::TritonStructuredDialect>();

    target.addDynamicallyLegalOp<UnrealizedConversionCastOp>(
        [](Operation *op) { return !op->hasAttr("make_state"); });

    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) {
      signalPassFailure();
    }
    return success();
  }

  void runOnOperation() override {
    (void)test();
    // assert(0);
    (void)test2();
    // return;

    auto moduleOp = getOperation();

    mlir::tts::PtrAnalysis ptrAnalysis;
    if (ptrAnalysis.rewriteOp(moduleOp).failed()) {
      moduleOp->emitWarning("PtrAnalysis failed");
    }

    moduleOp.walk([&ptrAnalysis](UnrealizedConversionCastOp op) {
      OpBuilder builder(op);
      SmallVector<Value> replacements;
      if (op->hasAttr("make_state_new")) {
        auto origPtr = op.getInputs()[0];
        assert(ptrAnalysis.knownPtrs.contains(origPtr));
        tts::PtrState state = ptrAnalysis.knownPtrs[origPtr];

        replacements.push_back(origPtr);

        for (auto [j, s] : llvm::enumerate(state.offsets)) {
          auto sIntAttr = getIntAttr(s);
          if (sIntAttr) {
            auto constOp = builder.create<arith::ConstantOp>(
                op.getLoc(), builder.getIndexAttr(sIntAttr.value()));
            replacements.push_back(constOp.getResult());
          } else {
            replacements.push_back(s.get<Value>());
          }
        }

        for (auto [j, s] : llvm::enumerate(state.strides)) {
          auto sIntAttr = getIntAttr(s);
          if (sIntAttr) {
            auto constOp = builder.create<arith::ConstantOp>(
                op.getLoc(), builder.getIndexAttr(sIntAttr.value()));
            replacements.push_back(constOp.getResult());
          } else {
            replacements.push_back(s.get<Value>());
          }
        }

        op->replaceAllUsesWith(replacements);
        op->erase();
      }
    });
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
triton::createTritonToStructuredPass() {
  return std::make_unique<TritonToStructuredPass>();
}
