#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Interfaces/VectorInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/DialectConversion.h"
#include <iostream>
#include "mlir/IR/Operation.h"
#include <llvm/Support/raw_ostream.h>


using namespace mlir;


namespace {
struct OuterProductVectorizationPass : public PassWrapper<OuterProductVectorizationPass, OperationPass<func::FuncOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vector::VectorDialect, func::FuncDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *context = funcOp.getContext();
    RewritePatternSet patterns(context);

    // Step 4: Lower vector.multi_reduction to vector.contract (+ some helpful patterns)
    vector::VectorTransformsOptions vectorTransformsOptions;
    vectorTransformsOptions.setVectorTransformsOptions(vector::VectorContractLowering::OuterProduct);
    vector::populateVectorTransferDropUnitDimsPatterns(patterns);
    vector::populateVectorReductionToContractPatterns(patterns);

    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      return signalPassFailure();
    }

    // Step 5: Lower vector.contract to vector.outerproduct. Also drop unit dims.
    patterns.clear();
    vectorTransformsOptions.setVectorTransformsOptions(vector::VectorContractLowering::OuterProduct);
    vector::populateVectorTransferDropUnitDimsPatterns(patterns);

    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

    struct MatmulTileConversion : public OpRewritePattern<linalg::MatmulOp> {
      using OpRewritePattern<linalg::MatmulOp>::OpRewritePattern;

      LogicalResult matchAndRewrite(linalg::MatmulOp op,
                                    PatternRewriter &rewriter) const override {
        linalg::LinalgTilingOptions tilingOptions;

        tilingOptions.setTileSizeComputationFunction([&](OpBuilder &b,
                                                        Operation *) {
          SmallVector<mlir::Value, 4> sizes;
          sizes.reserve(3);

          Location loc = op.getLoc();
          Value vscale = b.create<vector::VectorScaleOp>(loc, b.getIndexType());
          Value tileM = b.create<arith::ConstantIndexOp>(loc, 4);
          Value tileMScaled = b.create<arith::MulIOp>(loc, tileM, vscale);
          sizes.push_back(tileMScaled);
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

        // Specify vector sizes and scalable dimensions for each dimension
        SmallVector<int64_t, 4> inputVectorSizes = {4, 4, 1};
        SmallVector<bool, 4> inputScalableVecDims = {true, true, false};

        if (failed(linalg::vectorize(rewriter,
                                    cast<linalg::LinalgOp>(
                                        tiledOpResult->op.getOperation()),
                                    inputVectorSizes, inputScalableVecDims))) {
          std::cout << "Vectorization FAILED" << std::endl;
          return failure();
        }

        MLIRContext *context = getContext();
        rewriter.replaceOp(op, tiledOpResult->tensorResults);

        return success();
      }
    };
  class MatmulTileConversionPass
    : public PassWrapper <MatmulTileConversionPass, OperationPass <func::FuncOp>> {
      public: void getDependentDialects(DialectRegistry & registry) const override {
        registry.insert <linalg::LinalgDialect, func::FuncDialect, scf::SCFDialect, vector::VectorDialect> ();
      }

      void runOnOperation() override {
        func::FuncOp funcOp = getOperation();
        MLIRContext *context = &getContext();
      

        RewritePatternSet patterns(context);
        patterns.add<MatmulTileConversion>(context);

        ConversionTarget target( * context);
        target.addLegalDialect <linalg::LinalgDialect, func::FuncDialect, vector::VectorDialect, affine::AffineDialect, scf::SCFDialect> ();

        if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
          signalPassFailure();
        }
      }
    };

  std::unique_ptr <Pass> createOuterProductVectorizationPass() {
    return std::make_unique <OuterProductVectorizationPass>();
  }
    std::unique_ptr <Pass> createMatmulTileConversionPass() {
    return std::make_unique <MatmulTileConversionPass>();
  }
}

int main(int argc, char ** argv) {
  mlir::DialectRegistry registry;

  registry.insert<func::FuncDialect, arith::ArithDialect, math::MathDialect,
    linalg::LinalgDialect, affine::AffineDialect, scf::SCFDialect,
    tensor::TensorDialect, bufferization::BufferizationDialect,
    vector::VectorDialect, memref::MemRefDialect, func::FuncDialect>();


  registerAllDialects(registry);
  registerAllPasses();

  MLIRContext context;
  context.appendDialectRegistry(registry);
  context.loadAllAvailableDialects();

  PassPipelineRegistration<> pipeline(
    "sme-converison",
    "Converts linalg.matmul to a more optimized form",
    [](OpPassManager & pm) {
      pm.addPass(createMatmulTileConversionPass());
      pm.addPass(createOuterProductVectorizationPass());
    }
  );


  return asMainReturnCode(
    MlirOptMain(argc, argv, "Optimizer Driver\n", registry)
  );
}