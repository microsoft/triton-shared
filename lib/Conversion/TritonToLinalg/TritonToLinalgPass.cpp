//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "triton-shared/Analysis/UseAnalysis.h"
#include "triton-shared/Conversion/TritonToLinalg/TritonToLinalg.h"
#include "triton-shared/Dialect/TritonTilingExt/IR/TritonTilingExtDialect.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"
#include <cassert>
#include <cstdint>

#define DEBUG_TYPE "triton-to-linalg"

using namespace mlir;
using namespace triton;

#define GEN_PASS_CLASSES
#include "triton-shared/Conversion/TritonToLinalg/Passes.h.inc"

namespace {

class TritonTypeConverter : public TypeConverter {
public:
  TritonTypeConverter() {
    // The order of type conversion is important: later ones are tried earlier.
    addConversion([](Type type) { return type; });
    addConversion([](triton::PointerType ptrType) {
      return UnrankedMemRefType::get(ptrType.getPointeeType(), 0);
    });
    addConversion([](TensorType tensorType) -> Type {
      auto elemType = tensorType.getElementType();
      if (auto ptrType = dyn_cast<triton::PointerType>(elemType)) {
        elemType = ptrType.getPointeeType();
      }
      return MemRefType::get(tensorType.getShape(), elemType);
    });
  }
};

struct LoadOpConverter : public OpConversionPattern<triton::LoadOp> {
private:
  using OpConversionPattern<triton::LoadOp>::OpConversionPattern;

  static MemRefType getMemrefTypeForScalarPtr(triton::PointerType ptrType,
                                              MLIRContext *context) {
    SmallVector<int64_t> strides{1};
    auto layout =
        StridedLayoutAttr::get(context, ShapedType::kDynamic, strides);

    auto elemType = ptrType.getPointeeType();
    auto memrefType = MemRefType::get({1}, elemType, layout);
    return memrefType;
  }

public:
  LoadOpConverter(TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<triton::LoadOp>(typeConverter, context) {}

  LogicalResult
  matchAndRewrite(triton::LoadOp loadOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto op = loadOp.getPtr().getDefiningOp<UnrealizedConversionCastOp>();

    auto results = op->getResultTypes();

    if (op.getInputs().size() != 2) {
      return failure();
    }

    auto loc = op->getLoc();
    auto ptr = op.getInputs()[0];
    auto offset = op.getInputs()[1];
    auto offsetType = dyn_cast<ShapedType>(offset.getType());

    if (!offsetType) {
      return failure();
    }

    // auto ptrType =
    // dyn_cast<triton::PointerType>(offsetType.getElementType());
    // SmallVector<int64_t> strides(offsetType.getRank(), ShapedType::kDynamic);
    // auto layout = StridedLayoutAttr::get(rewriter.getContext(), 0, strides);
    // auto elemType = ptrType.getPointeeType();
    // auto memrefType = MemRefType::get(offsetType.getShape(), elemType,
    // layout);

    auto resultType = dyn_cast<RankedTensorType>(loadOp.getResult().getType());

    auto memref = rewriter.create<memref::CastOp>(
        loc,
        MemRefType::get({ShapedType::kDynamic}, resultType.getElementType()),
        ptr);

    // Treat this as a 1-d tensor
    auto tensor = rewriter.create<bufferization::ToTensorOp>(
        loc,
        RankedTensorType::get(SmallVector<int64_t>(1, ShapedType::kDynamic),
                              resultType.getElementType()),
        memref, true /* restrict */, false /* writable */);

    auto emptyTensor = rewriter
                           .create<tensor::EmptyOp>(loc, resultType.getShape(),
                                                    resultType.getElementType())
                           .getResult();

    SmallVector<AffineMap, 2> affineMaps(
        loadOp.getMask() ? 3 : 2,
        rewriter.getMultiDimIdentityMap(resultType.getRank()));

    // auto genericOp = rewriter.create<linalg::GenericOp>(loc);

    SmallVector<Value> inputs{offset};
    if (loadOp.getMask()) {
      inputs.push_back(loadOp.getMask());
    }

    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, ArrayRef<Type>({resultType}), inputs, ValueRange{emptyTensor},
        affineMaps,
        SmallVector<utils::IteratorType>(resultType.getRank(),
                                         utils::IteratorType::parallel),
        [&](OpBuilder &b, Location loc, ValueRange args) {
          if (!loadOp.getMask()) {

            auto indexValue = args[0];
            // auto index0 = rewriter.create<linalg::IndexOp>(loc, 0);
            Value index0 = rewriter.create<arith::IndexCastOp>(
                loc, rewriter.getIndexType(), indexValue);

            Value extract = rewriter.create<tensor::ExtractOp>(
                loc, tensor, ValueRange{index0});
            rewriter.create<linalg::YieldOp>(loc, extract);
          } else {
            auto mask = args[1];

            auto ifOp = rewriter.create<scf::IfOp>(
                loc, mask,
                [&](OpBuilder &b, Location loc) {
                  auto indexValue = args[0];
                  // auto index0 = rewriter.create<linalg::IndexOp>(loc, 0);
                  Value index0 = rewriter.create<arith::IndexCastOp>(
                      loc, rewriter.getIndexType(), indexValue);

                  Value extract = rewriter.create<tensor::ExtractOp>(
                      loc, tensor, ValueRange{index0});
                  b.create<scf::YieldOp>(loc, extract);
                },
                [&](OpBuilder &b, Location loc) {
                  Value extract = rewriter.create<arith::ConstantOp>(
                      loc, b.getF32FloatAttr(0));
                  // b.getFloatAttr()
                  b.create<scf::YieldOp>(loc, extract);
                });

            rewriter.create<linalg::YieldOp>(loc, ifOp->getResult(0));
          }
        });

    rewriter.replaceOp(loadOp, genericOp);

    return success();
  }
};

class TritonToLinalgPass : public TritonToLinalgBase<TritonToLinalgPass> {

  static auto constexpr LAUNCH_GRID_RANK = getMaxEnumValForProgramIDDim() + 1;
  static unsigned int constexpr TRITON_PROGRAM_INFO_ARG_COUNT =
      LAUNCH_GRID_RANK * 2;

  // Add additional I32 arguments to represent:
  // - num_programs, 3 in total, one for each axis of the launch grid
  // - program_id, 3 in total, one for each axis of the launch grid
  static void addProgramInfo(triton::FuncOp func) {
    OpBuilder b(func);

    auto origFuncType = func.getFunctionType();
    auto origInputTypes = origFuncType.getInputs();
    SmallVector<Type> newInputTypes(origInputTypes);
    newInputTypes.append(TRITON_PROGRAM_INFO_ARG_COUNT, b.getI32Type());

    auto newFuncType =
        b.getFunctionType(newInputTypes, origFuncType.getResults());

    func.setFunctionType(newFuncType);

    // Add empty attributes for each new argument if needed
    if (func.getAllArgAttrs()) {
      SmallVector<DictionaryAttr> newArgAttrs;
      func.getAllArgAttrs(newArgAttrs);
      newArgAttrs.append(TRITON_PROGRAM_INFO_ARG_COUNT, DictionaryAttr());
      func.setAllArgAttrs(newArgAttrs);
    }

    // Add the corresponding arguments to function body
    for (unsigned int i = 0; i < TRITON_PROGRAM_INFO_ARG_COUNT; i++) {
      func.getBody().front().addArgument(b.getI32Type(), func.getLoc());
    }
  }

public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<func::FuncDialect, arith::ArithDialect, math::MathDialect,
                linalg::LinalgDialect, affine::AffineDialect, scf::SCFDialect,
                tensor::TensorDialect, bufferization::BufferizationDialect,
                memref::MemRefDialect, ttx::TritonTilingExtDialect>();
  }

  void convert() {
    auto moduleOp = getOperation();
    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());

    target.addLegalDialect<
        func::FuncDialect, arith::ArithDialect, math::MathDialect,
        linalg::LinalgDialect, affine::AffineDialect, scf::SCFDialect,
        cf::ControlFlowDialect, tensor::TensorDialect,
        bufferization::BufferizationDialect, memref::MemRefDialect,
        ttx::TritonTilingExtDialect>();

    target.addDynamicallyLegalOp<triton::LoadOp>([](triton::LoadOp op) {
      return !isa<ShapedType>(op.getResult().getType());
    });

    TritonTypeConverter converter;
    patterns.add<LoadOpConverter>(converter, patterns.getContext());

    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns))))
      signalPassFailure();
  }

  void runOnOperation() override {
    convert();
    return;

    auto moduleOp = getOperation();

    {
      RewritePatternSet patterns(&getContext());
      populateTritonToLinalgCanonicalizationPatterns(patterns);
      if (failed(applyPatternsAndFoldGreedily(moduleOp, std::move(patterns)))) {
        signalPassFailure();
      }
    }

    moduleOp.walk([this](triton::FuncOp op) {
      if (failed(runUseAnalysis(op))) {
        signalPassFailure();
      }
    });

    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());
    TritonTypeConverter tritonTypeConverter;

    target.addLegalDialect<
        func::FuncDialect, arith::ArithDialect, math::MathDialect,
        linalg::LinalgDialect, affine::AffineDialect, scf::SCFDialect,
        cf::ControlFlowDialect, tensor::TensorDialect,
        bufferization::BufferizationDialect, memref::MemRefDialect,
        ttx::TritonTilingExtDialect>();

    target.addLegalOp<ModuleOp>();

    // Update function signature to use memrefs
    target.addDynamicallyLegalOp<triton::FuncOp>([&](triton::FuncOp op) {
      return tritonTypeConverter.isSignatureLegal(op.getFunctionType());
    });

    // Lower dense constant to linalg.fill
    target.addDynamicallyLegalOp<arith::ConstantOp>([](arith::ConstantOp op) {
      if (!isa<RankedTensorType>(op.getResult().getType())) {
        return true;
      }

      if (auto denseAttr = dyn_cast<DenseElementsAttr>(op.getValue())) {
        if (denseAttr.isSplat() &&
            isa<FloatType, IntegerType>(denseAttr.getElementType())) {
          return false;
        }
      }
      return true;
    });

    target.addDynamicallyLegalOp<scf::ForOp, scf::YieldOp>([](Operation *op) {
      return llvm::all_of(op->getOperandTypes(), [](Type t) {
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

    target.addDynamicallyLegalDialect<arith::ArithDialect, math::MathDialect>(
        [](Operation *op) {
          if (op->hasAttr("MetaUse")) {
            return false;
          }

          if (isa<arith::ConstantOp>(op)) {
            return true;
          }

          bool operateOnTensors =
              llvm::all_of(op->getOperandTypes(), [](Type type) {
                return isa<RankedTensorType>(type);
              });

          return !operateOnTensors;
        });

    triton::populateTritonToLinalgConversionPatterns(
        tritonTypeConverter, patterns, LAUNCH_GRID_RANK);

    for (auto func : getOperation().getOps<triton::FuncOp>())
      addProgramInfo(func);

    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns))))
      signalPassFailure();

    // Convert tt.func and tt.return into func's counterparts
    moduleOp.walk([&](triton::FuncOp func) {
      OpBuilder builder(func);

      auto name = func.getName();
      auto type = func.getFunctionType();

      SmallVector<DictionaryAttr> argAttrs, resAttrs;
      func.getAllArgAttrs(argAttrs);
      func.getAllResultAttrs(resAttrs);

      auto funcFunc = builder.create<func::FuncOp>(func.getLoc(), name, type);
      funcFunc.setAllArgAttrs(argAttrs);
      funcFunc.setAllResultAttrs(resAttrs);

      auto &funcFuncBody = funcFunc.getBody();
      auto &funcBody = func.getBody();

      IRMapping map;
      funcBody.cloneInto(&funcFuncBody, map);

      for (Block &block : funcFuncBody.getBlocks()) {
        auto term = block.getTerminator();
        builder.setInsertionPoint(term);
        builder.create<func::ReturnOp>(func.getLoc(), term->getOperands());
        term->erase();
      }
      func.erase();
    });

    // Erase dead code and fold constants created during lowering
    PassManager pm(&getContext(), moduleOp.getOperationName());
    pm.addPass(createCanonicalizerPass());
    if (failed(runPipeline(pm, getOperation()))) {
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>> triton::createTritonToLinalgPass() {
  return std::make_unique<TritonToLinalgPass>();
}
