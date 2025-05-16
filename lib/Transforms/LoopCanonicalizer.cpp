//===- LoopCanonicalizer.cpp - Canonicalize MLIR operations
//-------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transformation pass converts operations into their canonical forms by
// folding constants, applying operation identity transformations etc.
//
//===----------------------------------------------------------------------===//

#include "triton-shared/Transforms/Passes.h"

#include "mlir/Analysis/DataFlow/LivenessAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_LOOPCANONICALIZER
#include "triton-shared/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::dataflow;

namespace {

// Fold away ForOp iter arguments when:
// 1) The op yields the iter arguments.
// 2) The argument's corresponding outer region iterators (inputs) are yielded.
// 3) The iter arguments have no use and the corresponding (operation) results
// have no use.
//
// These arguments must be defined outside of the ForOp region and can just be
// forwarded after simplifying the op inits, yields and returns.
//
// The implementation uses `inlineBlockBefore` to steal the content of the
// original ForOp and avoid cloning.
struct ForOpIterArgsFolder : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const final {
    bool canonicalize = false;

    // An internal flat vector of block transfer
    // arguments `newBlockTransferArgs` keeps the 1-1 mapping of original to
    // transformed block argument mappings. This plays the role of a
    // IRMapping for the particular use case of calling into
    // `inlineBlockBefore`.
    int64_t numResults = forOp.getNumResults();
    SmallVector<bool, 4> keepMask;
    keepMask.reserve(numResults);
    SmallVector<Value, 4> newBlockTransferArgs, newIterArgs, newYieldValues,
        newResultValues;
    newBlockTransferArgs.reserve(1 + numResults);
    newBlockTransferArgs.push_back(Value()); // iv placeholder with null value
    newIterArgs.reserve(forOp.getInitArgs().size());
    newYieldValues.reserve(numResults);
    newResultValues.reserve(numResults);
    DenseMap<std::pair<Value, Value>, std::pair<Value, Value>> initYieldToArg;
    for (auto [init, arg, result, yielded] :
         llvm::zip(forOp.getInitArgs(),       // iter from outside
                   forOp.getRegionIterArgs(), // iter inside region
                   forOp.getResults(),        // op results
                   forOp.getYieldedValues()   // iter yield
                   )) {
      // Forwarded is `true` when:
      // 1) The region `iter` argument is yielded.
      // 2) The region `iter` argument the corresponding input is yielded.
      // 3) The region `iter` argument has no use, and the corresponding op
      // result has no use.
      bool forwarded = (arg == yielded) || (init == yielded) ||
                       (arg.use_empty() && result.use_empty());
      if (forwarded) {
        canonicalize = true;
        keepMask.push_back(false);
        newBlockTransferArgs.push_back(init);
        newResultValues.push_back(init);
        continue;
      }

      // This value is kept.
      initYieldToArg.insert({{init, yielded}, {arg, result}});
      keepMask.push_back(true);
      newIterArgs.push_back(init);
      newYieldValues.push_back(yielded);
      newBlockTransferArgs.push_back(Value()); // placeholder with null value
      newResultValues.push_back(Value());      // placeholder with null value
    }

    if (!canonicalize)
      return failure();

    scf::ForOp newForOp = rewriter.create<scf::ForOp>(
        forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
        forOp.getStep(), newIterArgs);
    newForOp->setAttrs(forOp->getAttrs());
    Block &newBlock = newForOp.getRegion().front();

    // Replace the null placeholders with newly constructed values.
    newBlockTransferArgs[0] = newBlock.getArgument(0); // iv
    for (unsigned idx = 0, collapsedIdx = 0, e = newResultValues.size();
         idx != e; ++idx) {
      Value &blockTransferArg = newBlockTransferArgs[1 + idx];
      Value &newResultVal = newResultValues[idx];
      assert((blockTransferArg && newResultVal) ||
             (!blockTransferArg && !newResultVal));
      if (!blockTransferArg) {
        blockTransferArg = newForOp.getRegionIterArgs()[collapsedIdx];
        newResultVal = newForOp.getResult(collapsedIdx++);
      }
    }

    Block &oldBlock = forOp.getRegion().front();
    assert(oldBlock.getNumArguments() == newBlockTransferArgs.size() &&
           "unexpected argument size mismatch");

    // No results case: the scf::ForOp builder already created a zero
    // result terminator. Merge before this terminator and just get rid of the
    // original terminator that has been merged in.
    if (newIterArgs.empty()) {
      auto newYieldOp = cast<scf::YieldOp>(newBlock.getTerminator());
      rewriter.inlineBlockBefore(&oldBlock, newYieldOp, newBlockTransferArgs);
      rewriter.eraseOp(newBlock.getTerminator()->getPrevNode());
      rewriter.replaceOp(forOp, newResultValues);
      return success();
    }

    // No terminator case: merge and rewrite the merged terminator.
    auto cloneFilteredTerminator = [&](scf::YieldOp mergedTerminator) {
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPoint(mergedTerminator);
      SmallVector<Value, 4> filteredOperands;
      filteredOperands.reserve(newResultValues.size());
      for (unsigned idx = 0, e = keepMask.size(); idx < e; ++idx)
        if (keepMask[idx])
          filteredOperands.push_back(mergedTerminator.getOperand(idx));
      rewriter.create<scf::YieldOp>(mergedTerminator.getLoc(),
                                    filteredOperands);
    };

    rewriter.mergeBlocks(&oldBlock, &newBlock, newBlockTransferArgs);
    auto mergedYieldOp = cast<scf::YieldOp>(newBlock.getTerminator());
    cloneFilteredTerminator(mergedYieldOp);
    rewriter.eraseOp(mergedYieldOp);
    rewriter.replaceOp(forOp, newResultValues);
    return success();
  }
};

/// Canonicalize operations in nested regions.
struct LoopCanonicalizer
    : public impl::LoopCanonicalizerBase<LoopCanonicalizer> {
  LoopCanonicalizer() = default;

  void runOnOperation() override {
    auto &la = getAnalysis<RunLivenessAnalysis>();
    getOperation()->walk([&](scf::ForOp forOp) {
      llvm::dbgs() << "processing\n";
      forOp->dump();

      for (auto [init, arg, result, yielded] :
           llvm::zip(forOp.getInitArgs(),       // iter from outside
                     forOp.getRegionIterArgs(), // iter inside region
                     forOp.getResults(),        // op results
                     forOp.getYieldedValues()   // iter yield
                     )) {
        llvm::dbgs() << "number: " << arg.getArgNumber() << "\n";
        llvm::dbgs() << "init:";
        la.getLiveness(init)->dump();
        llvm::dbgs() << "\n";
        llvm::dbgs() << "arg:";
        la.getLiveness(arg)->dump();
        llvm::dbgs() << "\n";
        llvm::dbgs() << "result:";
        la.getLiveness(result)->dump();
        llvm::dbgs() << "\n";
        llvm::dbgs() << "yielded:";
        la.getLiveness(yielded)->dump();
        llvm::dbgs() << "\n";
        llvm::dbgs() << "+++\n";
      }
      llvm::dbgs() << "~~~\n";
    });
    // Canonicalization is best-effort. Non-convergence is not a pass failure.
  }
  GreedyRewriteConfig config;
  std::shared_ptr<const FrozenRewritePatternSet> patterns;
};
} // namespace

/// Create a LoopCanonicalizer pass.
std::unique_ptr<Pass> mlir::createLoopCanonicalizerPass() {
  return std::make_unique<LoopCanonicalizer>();
}
