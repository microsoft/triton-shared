//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//
// This pass lowers all triton ops on pointer to their equivalent form in the
// proposed Pointer Dialect:
// https://discourse.llvm.org/t/rfc-ptr-dialect-modularizing-ptr-ops-in-the-llvm-dialect/75142
//
// This pass is intended to be used after all running
// triton-arith-to-linalg="tensor-ptr-to-linalg=true".
// All triton ops on tensors of pointers are expected to have been lowered to
// linalg ops, and that only triton ops on single pointers remain.
//
// Implementation notes:
// Because triton pointers are typed whereas the !ptr.ptr type isn't. The
// lowering for addptr will have to manually scale the offsets by pointee type.
// As a result, bitcasts are no-op after this pass.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Ptr/IR/PtrDialect.h"
#include "mlir/Dialect/Ptr/IR/PtrTypes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

#include "triton-shared/Analysis/OpFoldResultUtils.h"
#include "triton-shared/AnalysisStructured/PtrAnalysis.h"
#include "triton-shared/Conversion/TritonToLinalgExperimental/TritonToPtr.h"
#include "triton-shared/Dialect/TPtr/IR/TPtrDialect.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "triton-shared/Utils/Utils.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "llvm/ADT/STLExtras.h"

#define DEBUG_TYPE "triton-to-ptr"

using namespace mlir;

namespace {

#define GEN_PASS_DEF_TRITONTOPTR
#include "triton-shared/Conversion/TritonToLinalgExperimental/Passes.h.inc"

// Convert tensor.insert_slice to use ptr.ptr type. This insert_slice op must
// have been lowered from tl.cat
struct InsertSliceConverter
    : public OpConversionPattern<tensor::InsertSliceOp> {
  using OpConversionPattern<tensor::InsertSliceOp>::OpConversionPattern;

  InsertSliceConverter(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<tensor::InsertSliceOp>(typeConverter, context) {}

  LogicalResult
  matchAndRewrite(tensor::InsertSliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<tensor::InsertSliceOp>(
        op, adaptor.getSource(), adaptor.getDest(), op.getMixedOffsets(),
        op.getMixedSizes(), op.getMixedStrides());
    return success();
  }
};

// Convert tensor.empty with !tt.ptr to tensor.empty with !ptr.ptr
struct EmptyTensorConverter : public OpConversionPattern<tensor::EmptyOp> {
  using OpConversionPattern<tensor::EmptyOp>::OpConversionPattern;

  EmptyTensorConverter(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<tensor::EmptyOp>(typeConverter, context) {}

  LogicalResult
  matchAndRewrite(tensor::EmptyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<tensor::EmptyOp>(
        op, op.getType().getShape(),
        ptr::PtrType::get(
            rewriter.getContext(),
            tptr::DefaultMemorySpaceAttr::get(rewriter.getContext())));
    return success();
  }
};

// This expand shape op must have been lowered from tt.expand_dims which could
// operate on tensor of pointers.
// Convert expand shape op to operate on !ptr.ptr instead of !tt.ptr.
struct ExpandShapeConverter
    : public OpConversionPattern<tensor::ExpandShapeOp> {
  using OpConversionPattern<tensor::ExpandShapeOp>::OpConversionPattern;

  ExpandShapeConverter(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<tensor::ExpandShapeOp>(typeConverter, context) {}

  LogicalResult
  matchAndRewrite(tensor::ExpandShapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<tensor::ExpandShapeOp>(
        op, getTypeConverter()->convertType(op.getType()), adaptor.getSrc(),
        op.getReassociationExprs());
    return success();
  }
};

// arith.select could operate on triton pointers. Convert to use !ptr.ptr
struct SelectOpConverter : public OpConversionPattern<arith::SelectOp> {
  using OpConversionPattern<arith::SelectOp>::OpConversionPattern;

  SelectOpConverter(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<arith::SelectOp>(typeConverter, context) {}

  LogicalResult
  matchAndRewrite(arith::SelectOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<arith::SelectOp>(
        op, getTypeConverter()->convertType(op.getType()),
        adaptor.getCondition(), adaptor.getTrueValue(),
        adaptor.getFalseValue());
    return success();
  }
};

// Convert bitcast which is a no-op because !ptr.ptr is opaque with no pointee
// type.
struct BitCastConverter : public OpConversionPattern<triton::BitcastOp> {
  using OpConversionPattern<triton::BitcastOp>::OpConversionPattern;

  BitCastConverter(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<triton::BitcastOp>(typeConverter, context) {}

  LogicalResult
  matchAndRewrite(triton::BitcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (isa<ShapedType>(op.getType())) {
      return failure();
    }
    // Bitcast is a no-op, simply forward the src
    rewriter.replaceOp(op, adaptor.getSrc());
    return success();
  }
};

// Convert tt.addptr to ptr.ptradd. Since the !ptr.ptr type is opaque, we scale
// the offset explicitly using type_offset op. This approach means that bitcast
// is a no-op.
struct AddPtrConverter : public OpConversionPattern<triton::AddPtrOp> {
  using OpConversionPattern<triton::AddPtrOp>::OpConversionPattern;

  AddPtrConverter(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<triton::AddPtrOp>(typeConverter, context) {}

  LogicalResult
  matchAndRewrite(triton::AddPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (isa<ShapedType>(op.getType())) {
      return failure();
    }
    auto loc = op->getLoc();
    auto pointeeType = cast<triton::PointerType>(op.getType()).getPointeeType();
    auto offsetType = op.getOffset().getType();
    auto pointeeSizeInBytes =
        rewriter.create<tptr::TypeOffsetOp>(loc, offsetType, pointeeType);
    auto scaledOffset =
        rewriter.create<arith::MulIOp>(loc, op.getOffset(), pointeeSizeInBytes);
    rewriter.replaceOpWithNewOp<tptr::PtrAddOp>(
        op,
        ptr::PtrType::get(
            rewriter.getContext(),
            tptr::DefaultMemorySpaceAttr::get(rewriter.getContext())),
        adaptor.getPtr(), scaledOffset);
    return success();
  }
};

// Convert tt.load which loads from a single pointer into a pair of
// to_memref and memref.load op.
// In the case of mask, the load is guarded by an scf.if
struct LoadConverter : public OpConversionPattern<triton::LoadOp> {
  using OpConversionPattern<triton::LoadOp>::OpConversionPattern;

  LoadConverter(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<triton::LoadOp>(typeConverter, context) {}

  LogicalResult
  matchAndRewrite(triton::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (isa<ShapedType>(op.getType())) {
      return failure();
    }
    auto ptr = op.getPtr();
    auto pointeeType =
        cast<triton::PointerType>(ptr.getType()).getPointeeType();

    auto memref = rewriter.create<tptr::ToMemrefOp>(
        op->getLoc(), MemRefType::get({1}, pointeeType), adaptor.getPtr());

    auto zero = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);

    if (op.getMask()) {
      auto ifOp = rewriter.create<scf::IfOp>(
          op->getLoc(), op.getMask(),
          [&](OpBuilder &b, Location loc) {
            // Truthy case, load from the index.
            Value memrefLoad = rewriter.create<memref::LoadOp>(
                op->getLoc(), memref, ValueRange{zero});
            b.create<scf::YieldOp>(loc, memrefLoad);
          },
          [&](OpBuilder &b, Location loc) {
            // Falsy case, yield `other` or 0 as the default value.
            if (op.getOther()) {
              b.create<scf::YieldOp>(loc, op.getOther());
            } else {
              auto elemType = op.getType();
              auto zeroAttr = b.getZeroAttr(elemType);
              assert(zeroAttr && "unexpected element type");
              Value val = b.create<arith::ConstantOp>(loc, zeroAttr);
              b.create<scf::YieldOp>(loc, val);
            }
          });
      rewriter.replaceOp(op, ifOp);
    } else {
      auto memrefLoad = rewriter.create<memref::LoadOp>(op->getLoc(), memref,
                                                        ValueRange{zero});

      rewriter.replaceOp(op, memrefLoad);
    }
    return success();
  }
};

// Convert tt.store which stores to a single pointer into a pair of
// to_memref and memref.store op.
// In the case of mask, the store is guarded by an scf.if
struct StoreConverter : public OpConversionPattern<triton::StoreOp> {
  using OpConversionPattern<triton::StoreOp>::OpConversionPattern;

  StoreConverter(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<triton::StoreOp>(typeConverter, context) {}

  LogicalResult
  matchAndRewrite(triton::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (isa<ShapedType>(op.getValue().getType())) {
      return failure();
    }
    auto ptr = op.getPtr();
    auto pointeeType =
        cast<triton::PointerType>(ptr.getType()).getPointeeType();

    IRRewriter::InsertionGuard g(rewriter);
    if (op.getMask()) {
      auto ifOp = rewriter.create<scf::IfOp>(op->getLoc(), op.getMask(),
                                             /*withElseRegion*/ false);
      rewriter.setInsertionPointToStart(
          &ifOp.getThenRegion().getBlocks().front());
    }

    auto memref = rewriter.create<tptr::ToMemrefOp>(
        op->getLoc(), MemRefType::get({1}, pointeeType), adaptor.getPtr());
    auto zero = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);

    rewriter.create<memref::StoreOp>(op->getLoc(), op.getValue(), memref,
                                     ValueRange{zero});

    rewriter.eraseOp(op);

    return success();
  }
};

// Convert tt.ptr_to_int to ptr.ptrtoint
struct PtrToIntConverter : public OpConversionPattern<triton::PtrToIntOp> {
  using OpConversionPattern<triton::PtrToIntOp>::OpConversionPattern;

  PtrToIntConverter(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<triton::PtrToIntOp>(typeConverter, context) {}

  LogicalResult
  matchAndRewrite(triton::PtrToIntOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (isa<ShapedType>(op.getType())) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<tptr::PtrToIntOp>(op, op.getType(),
                                                  adaptor.getSrc());
    return success();
  }
};

// Convert tt.int_to_ptr to ptr.ptrtoint
struct IntToPtrConverter : public OpConversionPattern<triton::IntToPtrOp> {
  using OpConversionPattern<triton::IntToPtrOp>::OpConversionPattern;

  IntToPtrConverter(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<triton::IntToPtrOp>(typeConverter, context) {}

  LogicalResult
  matchAndRewrite(triton::IntToPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (isa<ShapedType>(op.getType())) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<tptr::IntToPtrOp>(
        op,
        ptr::PtrType::get(
            rewriter.getContext(),
            tptr::DefaultMemorySpaceAttr::get(rewriter.getContext())),
        adaptor.getSrc());
    return success();
  }
};

// Convert a linalg op on triton pointer to use !ptr.ptr
// The conversion infrastructrure will recursively handle the inner op
// which could be either tt.load, tt.store, tt.bitcast, tt.int_to_ptr, and
// tt.ptr_to_int and use their corresponding converters.
struct LinalgPtrConverter : public OpConversionPattern<linalg::GenericOp> {
  using OpConversionPattern<linalg::GenericOp>::OpConversionPattern;

  LinalgPtrConverter(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<linalg::GenericOp>(typeConverter, context) {}

  LogicalResult
  matchAndRewrite(linalg::GenericOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    SmallVector<Type> convertedTypes;
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(),
                                                      convertedTypes))) {
      return failure();
    }

    auto replacement = rewriter.create<linalg::GenericOp>(
        op.getLoc(), convertedTypes, adaptor.getInputs(), adaptor.getOutputs(),
        op.getIndexingMapsArray(), op.getIteratorTypesArray());

    Region &region = op.getRegion();
    Block &block = region.front();

    TypeConverter::SignatureConversion mapping(block.getArgumentTypes().size());
    if (failed(typeConverter->convertSignatureArgs(block.getArgumentTypes(),
                                                   mapping)))
      return failure();

    // Perform signature conversion on the body block.
    rewriter.applySignatureConversion(&block, mapping);

    // Splice the old body region into the new for-op.
    Region &dstRegion = replacement.getBodyRegion();
    rewriter.inlineRegionBefore(op.getRegion(), dstRegion, dstRegion.end());

    rewriter.replaceOp(op, replacement);

    return success();
  }
};

// The linalg.yield op is still yielding the original !tt.ptr results, convert
// them to use the new !ptr.ptr results
struct LinalgYieldConverter : public OpConversionPattern<linalg::YieldOp> {
  using OpConversionPattern<linalg::YieldOp>::OpConversionPattern;

  LinalgYieldConverter(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<linalg::YieldOp>(typeConverter, context) {}

  LogicalResult
  matchAndRewrite(linalg::YieldOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<linalg::YieldOp>(op, adaptor.getOperands());
    return success();
  }
};

// Convert linalg.fill to use !ptr.ptr. linalg.fill on triton pointer is lowered
// from tt.splat on a triton pointer.
struct LinalgFillPtrConverter : public OpConversionPattern<linalg::FillOp> {
  using OpConversionPattern<linalg::FillOp>::OpConversionPattern;

  LinalgFillPtrConverter(const TypeConverter &typeConverter,
                         MLIRContext *context)
      : OpConversionPattern<linalg::FillOp>(typeConverter, context) {}

  LogicalResult
  matchAndRewrite(linalg::FillOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<linalg::FillOp>(op, adaptor.getInputs(),
                                                adaptor.getOutputs());
    return success();
  }
};

class TritonPtrTypeConverter : public TypeConverter {
public:
  TritonPtrTypeConverter(MLIRContext *context) {
    addConversion([](Type type) { return type; });
    addConversion([context](triton::PointerType ptrType) {
      return ptr::PtrType::get(context,
                               tptr::DefaultMemorySpaceAttr::get(context));
    });
    addConversion([context](RankedTensorType tensorType) {
      if (isa<triton::PointerType>(tensorType.getElementType())) {
        return RankedTensorType::get(
            tensorType.getShape(),
            ptr::PtrType::get(context,
                              tptr::DefaultMemorySpaceAttr::get(context)));
      }
      return tensorType;
    });
    auto createCast = [&](OpBuilder &builder, Type resultType,
                          ValueRange inputs, Location loc) -> Value {
      return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
          .getResult(0);
    };
    addTargetMaterialization(createCast);
    addSourceMaterialization(createCast);
  }
};

class TritonToPtrPass : public impl::TritonToPtrBase<TritonToPtrPass> {

public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<
        arith::ArithDialect, math::MathDialect, affine::AffineDialect,
        scf::SCFDialect, tensor::TensorDialect, triton::TritonDialect,
        tts::TritonStructuredDialect, ptr::PtrDialect, tptr::TPtrDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();

    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());
    TritonPtrTypeConverter typeConverter(&getContext());

    target.addIllegalOp<triton::AddPtrOp, triton::BitcastOp, triton::IntToPtrOp,
                        triton::PtrToIntOp>();

    // We do not want to lower triton load and store on block pointers
    target.addDynamicallyLegalOp<triton::LoadOp, triton::StoreOp>([](auto op) {
      auto ptrType = op->getOperand(0).getType();
      if (triton::isTensorPointerType(ptrType)) {
        return true;
      }
      return !triton::isPtrTypeLike(ptrType);
    });

    target.addDynamicallyLegalOp<
        linalg::FillOp, linalg::GenericOp, linalg::YieldOp, tensor::EmptyOp,
        tensor::ExpandShapeOp, tensor::InsertSliceOp, arith::SelectOp>(
        [](auto op) {
          return llvm::all_of(
              llvm::concat<Value>(op->getOperands(), op->getResults()),
              [&](Value v) { return !triton::isPtrTypeLike(v.getType()); });
        });

    target.addLegalDialect<arith::ArithDialect, linalg::LinalgDialect,
                           tensor::TensorDialect, affine::AffineDialect,
                           tptr::TPtrDialect, memref::MemRefDialect>();

    patterns
        .add<AddPtrConverter, BitCastConverter, StoreConverter, LoadConverter,
             PtrToIntConverter, IntToPtrConverter, ExpandShapeConverter,
             SelectOpConverter, InsertSliceConverter, EmptyTensorConverter,
             LinalgFillPtrConverter, LinalgPtrConverter, LinalgYieldConverter>(
            typeConverter, patterns.getContext());

    mlir::scf::populateSCFStructuralTypeConversionsAndLegality(
        typeConverter, patterns, target);
    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>> triton::createTritonToPtrPass() {
  return std::make_unique<TritonToPtrPass>();
}
