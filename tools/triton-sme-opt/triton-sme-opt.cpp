#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/TransformOps/VectorTransformOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"

#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Interfaces/VectorInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include <fstream>
#include <iostream>
#include <llvm/Support/raw_ostream.h>
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"

#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace matmul_conversion {

struct MatmulTileConversion : public OpRewritePattern<linalg::MatmulOp> {
  explicit MatmulTileConversion(MLIRContext *context, bool enableSME)
      : OpRewritePattern<linalg::MatmulOp>(context), enableSME(enableSME) {}

  LogicalResult matchAndRewrite(linalg::MatmulOp op,
                                PatternRewriter &rewriter) const override {
    linalg::LinalgTilingOptions tilingOptions;

    tilingOptions.setTileSizeComputationFunction(
        [&](OpBuilder &b, Operation *) {
          SmallVector<Value, 4> sizes;
          sizes.reserve(3);

          Location loc = op.getLoc();
          Value vscale = b.create<vector::VectorScaleOp>(loc, b.getIndexType());

          if (enableSME) {
            Value tileM = b.create<arith::ConstantIndexOp>(loc, 4);
            Value tileMScaled = b.create<arith::MulIOp>(loc, tileM, vscale);
            sizes.push_back(tileMScaled);
          } else {
            Value tileM = b.create<arith::ConstantIndexOp>(loc, 2);
            sizes.push_back(tileM);
          }
          Value tileN = b.create<arith::ConstantIndexOp>(loc, 4);
          Value tileNScaled = b.create<arith::MulIOp>(loc, tileN, vscale);
          sizes.push_back(tileNScaled);
          Value tileK = b.create<arith::ConstantIndexOp>(loc, 1);
          sizes.push_back(tileK);

          return sizes;
        });


    auto tiledOpResult = tileLinalgOp(rewriter, op, tilingOptions);
    if (failed(tiledOpResult)) {
      std::cout << "TILING FAILED" << std::endl;
      return failure();
    }

    SmallVector<int64_t, 4> inputVectorSizes = {enableSME ? 4 : 2, 4, 1};
    SmallVector<bool, 4> inputScalableVecDims = {enableSME, true, false};

    if (failed(linalg::vectorize(
            rewriter, cast<linalg::LinalgOp>(tiledOpResult->op.getOperation()),
            inputVectorSizes, inputScalableVecDims))) {
      std::cout << "Vectorization FAILED" << std::endl;
      return failure();
    }

    MLIRContext *context = getContext();
    rewriter.replaceOp(op, tiledOpResult->tensorResults);

    return success();
  }


private:
  bool enableSME;
};

struct MatmulTileConversionPass
    : public PassWrapper<MatmulTileConversionPass,
                          OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MatmulTileConversionPass)

  explicit MatmulTileConversionPass(bool enableSME) : enableSME(enableSME) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, func::FuncDialect, scf::SCFDialect,
                    vector::VectorDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);
    patterns.add<MatmulTileConversion>(context, enableSME);

    ConversionTarget target(*context);
    target.addLegalDialect<linalg::LinalgDialect, func::FuncDialect,
                           vector::VectorDialect, affine::AffineDialect,
                           scf::SCFDialect>();

    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      signalPassFailure();
    }
  }

private:
  bool enableSME;
};

struct OuterProductVectorizationPass
    : public PassWrapper<OuterProductVectorizationPass,
                          OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OuterProductVectorizationPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vector::VectorDialect, func::FuncDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *context = funcOp.getContext();
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);

    vector::populateVectorMaskLoweringPatternsForSideEffectingOps(patterns);
    vector::populateVectorReductionToContractPatterns(patterns);
    vector::populateVectorMaskOpLoweringPatterns(patterns);
    vector::populateVectorTransferDropUnitDimsPatterns(patterns);
    vector::VectorTransformsOptions vectorTransformOptions;

    vectorTransformOptions.setVectorTransformsOptions(vector::VectorContractLowering::OuterProduct);
    vector::populateVectorContractLoweringPatterns(patterns, vectorTransformOptions);

    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

std::unique_ptr<Pass> createOuterProductVectorizationPass() {
  return std::make_unique<OuterProductVectorizationPass>();
}

std::unique_ptr<Pass> createMatmulTileConversionPass(bool enableSME) {
  return std::make_unique<MatmulTileConversionPass>(enableSME);
}

} // namespace matmul_conversion

int main(int argc, char **argv) {
 DialectRegistry registry;

  registry.insert<func::FuncDialect, arith::ArithDialect, math::MathDialect,
                  linalg::LinalgDialect, affine::AffineDialect, scf::SCFDialect,
                  tensor::TensorDialect, bufferization::BufferizationDialect,
                  vector::VectorDialect, memref::MemRefDialect,
                  func::FuncDialect, arm_sme::ArmSMEDialect>();

  registerAllDialects(registry);
  registerAllPasses();

  MLIRContext context;
  context.appendDialectRegistry(registry);
  context.loadAllAvailableDialects();


  PassPipelineRegistration<> smeConversionPipeline(
      "sme-conversion",
      "Converts linalg.matmul to a more optimized form using SME",
      [](OpPassManager &pm) {
        pm.addPass(matmul_conversion::createMatmulTileConversionPass(true));
        pm.addPass(mlir::tensor::createTensorBufferizePass());
        pm.addPass(createLinalgBufferizePass());
        //adddiontal bufferization lowering may be necessary
        pm.addPass(matmul_conversion::createOuterProductVectorizationPass());
      });

  return asMainReturnCode(
      MlirOptMain(argc, argv, "Optimizer Driver\n", registry));
}