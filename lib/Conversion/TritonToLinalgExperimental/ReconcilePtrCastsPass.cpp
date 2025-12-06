//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//
// Throughout the conversion process, we convert !tt.ptr -> {!ptr.ptr or
// memref<*>}. This process leaves around unrealized_conversion_cast ops between
// these types. We want to remove these unrealized casts and use the proper
// conversion ops in the PtrDialect: to_memref or from_memref. To do this, we
// use a pattern that simplifies the chain of conversions by removing
// intermediate conversion cast ops. At the end, we are left with just pointer
// to memref or vice versa. We then convert the unrealized cast to to_memref or
// from_memref accordingly.
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Ptr/IR/PtrTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton-shared/Conversion/TritonToLinalgExperimental/ReconcilePtrCasts.h"

#include "triton/Dialect/Triton/IR/Types.h"

#include "triton-shared/Conversion/TritonToLinalgExperimental/ReconcilePtrCasts.h"
#include "triton-shared/Dialect/TPtr/IR/TPtrDialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

using namespace mlir;
using namespace triton;

#define GEN_PASS_CLASSES
#include "triton-shared/Conversion/TritonToLinalgExperimental/Passes.h.inc"

namespace {

static bool isOneToOneCast(UnrealizedConversionCastOp op) {
  return (op.getInputs().size() == 1 && op->getNumResults() == 1);
}

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

struct FromMemrefConverter
    : public OpRewritePattern<UnrealizedConversionCastOp> {
  FromMemrefConverter(MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern<UnrealizedConversionCastOp>(context, benefit) {}

  LogicalResult matchAndRewrite(UnrealizedConversionCastOp op,
                                PatternRewriter &rewriter) const override {
    if (!isOneToOneCast(op)) {
      return failure();
    }

    Location loc = op.getLoc();
    auto input = op.getInputs().front();
    auto unrankedInput = dyn_cast<UnrankedMemRefType>(input.getType());
    auto output = op.getResult(0);
    auto outType = output.getType();

    if (unrankedInput && isa<triton::PointerType, ptr::PtrType>(outType)) {
      // from_memref only takes ranked memref, cast the unranked memref to
      // ranked memref first.
      Value rankedMemref = rewriter.create<memref::CastOp>(
          op.getLoc(), MemRefType::get({1}, unrankedInput.getElementType()),
          input);

      // The output of the cast may have a different element type than the
      // input memref. In that case we should use an unrealized conversion cast
      // to match the element type.
      Type elementTy;
      if (auto ttPtr = dyn_cast<triton::PointerType>(outType))
        elementTy = ttPtr.getPointeeType();
      else if (auto genericPtr = dyn_cast<ptr::PtrType>(outType))
        elementTy = genericPtr.getElementType();
      else
        return failure();

      if (elementTy && elementTy != unrankedInput.getElementType()) {
        // Insert an unrealized conversion cast to match element type
        auto castOp = rewriter.create<UnrealizedConversionCastOp>(
            loc, MemRefType::get({1}, elementTy), rankedMemref);

        // Use the result of the cast op
        rankedMemref = castOp.getResult(0);
      }

      auto memrefToPtr = rewriter.create<tptr::FromMemrefOp>(
          op->getLoc(),
          ptr::PtrType::get(
              rewriter.getContext(),
              tptr::DefaultMemorySpaceAttr::get(rewriter.getContext())),
          rankedMemref);

      rewriter.replaceAllUsesWith(output, memrefToPtr);
      rewriter.eraseOp(op);

      return success();
    }

    return failure();
  }
};

static std::optional<arith::AtomicRMWKind> mapTritonToMLIR(uint32_t triKind) {
  switch (triKind) {
  case 1:
    return arith::AtomicRMWKind::andi; // Triton AND
  case 2:
    return arith::AtomicRMWKind::ori; // Triton OR
  case 3:
    return std::nullopt; // Triton XOR not supported
  case 4:
    return arith::AtomicRMWKind::addi; // Triton ADD
  case 5:
    return arith::AtomicRMWKind::addf; // Triton FADD
  case 6:
    return arith::AtomicRMWKind::maxs; // Triton signed max
  case 7:
    return arith::AtomicRMWKind::mins; // Triton signed min
  case 8:
    return arith::AtomicRMWKind::maxu; // Triton unsigned max
  case 9:
    return arith::AtomicRMWKind::minu; // Triton unsigned min
  case 10:
    return arith::AtomicRMWKind::assign; // Triton XCHG → assign
  default:
    return std::nullopt;
  }
}

struct AtomicrmwConverter : public OpRewritePattern<triton::AtomicRMWOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::AtomicRMWOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto val = op.getVal();
    auto tritonPtr = op.getPtr();
    auto mask = op.getMask();

    // Recover memref from Triton pointer
    Value memref;
    if (auto fromMemref = tritonPtr.getDefiningOp<tptr::FromMemrefOp>())
      memref = fromMemref.getOperand();

    // Get Triton's atomic kind integer
    auto kindIntAttr = op->getAttrOfType<IntegerAttr>("atomic_rmw_op");
    if (!kindIntAttr)
      return rewriter.notifyMatchFailure(op, "missing Triton atomic kind");

    uint32_t triKind = kindIntAttr.getInt();
    auto mlirKindOpt = mapTritonToMLIR(triKind);
    if (!mlirKindOpt)
      return rewriter.notifyMatchFailure(op, "unsupported Triton atomic kind");

    auto kindAttr =
        arith::AtomicRMWKindAttr::get(rewriter.getContext(), *mlirKindOpt);

    // Use index 0 for rank-1 memrefs for now
    Value idx = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    SmallVector<Value, 1> indices{idx};

    Value atomic;
    
    if (mask) {
      auto ifOp = rewriter.create<scf::IfOp>(
          loc, mask,
          /*thenBuilder=*/
          [&](OpBuilder &b, Location l) {
            atomic = b.create<memref::AtomicRMWOp>(l, kindAttr, val, memref,
                                                   indices);
            b.create<scf::YieldOp>(l);
          },
          /*elseBuilder=*/
          [&](OpBuilder &b, Location l) { b.create<scf::YieldOp>(l); });

      rewriter.replaceOp(op, atomic);
    } else {
      // Replace with memref.atomic_rmw
      rewriter.replaceOpWithNewOp<memref::AtomicRMWOp>(op, kindAttr, val,
                                                       memref, indices);
    }

    return success();
  }
};

struct ToMemrefConverter : public OpRewritePattern<UnrealizedConversionCastOp> {
  ToMemrefConverter(MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern<UnrealizedConversionCastOp>(context, benefit) {}

  LogicalResult matchAndRewrite(UnrealizedConversionCastOp op,
                                PatternRewriter &rewriter) const override {
    if (!isOneToOneCast(op)) {
      return failure();
    }
    auto input = op.getInputs().front();
    auto inType = input.getType();
    auto output = op.getResult(0);
    auto outUnrankedMemrefType = dyn_cast<UnrankedMemRefType>(output.getType());
    if (isa<triton::PointerType, ptr::PtrType>(inType) &&
        outUnrankedMemrefType) {
      // to_memref can only cast to ranked static shape memref, we have to cast
      // the resulting memref back to unranked
      auto elemType = outUnrankedMemrefType.getElementType();
      auto ptrToMemref = rewriter.create<tptr::ToMemrefOp>(
          op->getLoc(), MemRefType::get({1}, elemType), input);

      SmallVector<OpFoldResult> sizes = {rewriter.getIndexAttr(1)};
      SmallVector<OpFoldResult> newStrides = {rewriter.getIndexAttr(1)};
      auto newUnrankedMemref = rewriter.create<memref::ReinterpretCastOp>(
          op->getLoc(), MemRefType::get({ShapedType::kDynamic}, elemType),
          ptrToMemref, rewriter.getIndexAttr(0), sizes, newStrides);

      rewriter.replaceAllUsesWith(output, newUnrankedMemref);
      rewriter.eraseOp(op);
      return success();
    }

    return failure();
  }
};

class ReconcilePtrCastsPass
    : public ReconcilePtrCastsBase<ReconcilePtrCastsPass> {

public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tptr::TPtrDialect, memref::MemRefDialect, BuiltinDialect,
                    arith::ArithDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();

    // === Phase 1: Greedy rewrites ===
    {
      RewritePatternSet greedyPatterns(&getContext());
      greedyPatterns
          .add<SimplifyUnrealizedCast, FromMemrefConverter, ToMemrefConverter>(
              &getContext());

      if (failed(applyPatternsGreedily(moduleOp, std::move(greedyPatterns)))) {
        signalPassFailure();
        return;
      }
    }

    // === Phase 2: Conversion patterns ===
    {
      RewritePatternSet conversionPatterns(&getContext());
      conversionPatterns.add<AtomicrmwConverter>(&getContext());

      ConversionTarget target(getContext());
      target.addIllegalOp<triton::AtomicRMWOp>();
      target.addLegalDialect<arith::ArithDialect>();
      target.addLegalDialect<memref::MemRefDialect>();
      target.addLegalDialect<scf::SCFDialect>();

      if (failed(applyPartialConversion(moduleOp, target,
                                        std::move(conversionPatterns)))) {
        signalPassFailure();
        return;
      }
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>> triton::createReconcilePtrCastsPass() {
  return std::make_unique<ReconcilePtrCastsPass>();
}
