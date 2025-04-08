//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
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
    auto add = rewriter.create<tptr::PtrAddOp>(
        loc, ptr::PtrType::get(rewriter.getContext()), adaptor.getPtr(),
        scaledOffset);
    rewriter.replaceOp(op, add);
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
    auto newYield =
        rewriter.create<linalg::YieldOp>(op->getLoc(), adaptor.getOperands());
    rewriter.replaceOp(op, newYield);
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
    auto newEmptyOp = rewriter.create<tensor::EmptyOp>(
        op->getLoc(), op.getType().getShape(),
        ptr::PtrType::get(rewriter.getContext()));
    rewriter.replaceOp(op, newEmptyOp);
    return success();
  }
};

struct ExpandShapeConverter
    : public OpConversionPattern<tensor::ExpandShapeOp> {
  using OpConversionPattern<tensor::ExpandShapeOp>::OpConversionPattern;

  ExpandShapeConverter(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<tensor::ExpandShapeOp>(typeConverter, context) {}

  LogicalResult
  matchAndRewrite(tensor::ExpandShapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newExpandShapeOp = rewriter.create<tensor::ExpandShapeOp>(
        op->getLoc(), getTypeConverter()->convertType(op.getType()),
        adaptor.getSrc(), op.getReassociationExprs());
    rewriter.replaceOp(op, newExpandShapeOp);
    return success();
  }
};

struct SelectOpConverter : public OpConversionPattern<arith::SelectOp> {
  using OpConversionPattern<arith::SelectOp>::OpConversionPattern;

  SelectOpConverter(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<arith::SelectOp>(typeConverter, context) {}

  LogicalResult
  matchAndRewrite(arith::SelectOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newSelectOp = rewriter.create<arith::SelectOp>(
        op->getLoc(), getTypeConverter()->convertType(op.getType()),
        adaptor.getCondition(), adaptor.getTrueValue(),
        adaptor.getFalseValue());
    rewriter.replaceOp(op, newSelectOp);
    return success();
  }
};

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
    auto replacement = rewriter.create<tptr::PtrToIntOp>(
        op->getLoc(), op.getType(), adaptor.getSrc());
    rewriter.replaceOp(op, replacement);
    return success();
  }
};

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

struct LinalgFillPtrConverter : public OpConversionPattern<linalg::FillOp> {
  using OpConversionPattern<linalg::FillOp>::OpConversionPattern;

  LinalgFillPtrConverter(const TypeConverter &typeConverter,
                         MLIRContext *context)
      : OpConversionPattern<linalg::FillOp>(typeConverter, context) {}

  LogicalResult
  matchAndRewrite(linalg::FillOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto loc = op.getLoc();
    auto replacement = rewriter
                           .create<linalg::FillOp>(loc, adaptor.getInputs(),
                                                   adaptor.getOutputs())
                           .result();

    rewriter.replaceOp(op, replacement);
    return success();
  }
};

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
    auto storeOp = rewriter.create<tptr::IntToPtrOp>(
        op->getLoc(), ptr::PtrType::get(rewriter.getContext()),
        adaptor.getSrc());
    rewriter.replaceOp(op, storeOp);
    return success();
  }
};

struct TensorExtractConverter : public OpConversionPattern<tensor::ExtractOp> {
  using OpConversionPattern<tensor::ExtractOp>::OpConversionPattern;

  TensorExtractConverter(const TypeConverter &typeConverter,
                         MLIRContext *context)
      : OpConversionPattern<tensor::ExtractOp>(typeConverter, context) {}

  LogicalResult
  matchAndRewrite(tensor::ExtractOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto in = op.getTensor();
    if (!isa<triton::PointerType>(in.getType().getElementType())) {
      return failure();
    }
    IRMapping map;
    map.map(op.getTensor(), adaptor.getTensor());
    auto newExtract = rewriter.clone(*op, map);
    newExtract->getResult(0).setType(ptr::PtrType::get(rewriter.getContext()));
    rewriter.replaceOp(op, newExtract);
    return success();
  }
};

class TritonPtrTypeConverter : public TypeConverter {
public:
  TritonPtrTypeConverter(MLIRContext *context) {
    addConversion([](Type type) { return type; });
    addConversion([context](triton::PointerType ptrType) {
      return ptr::PtrType::get(context);
    });
    addConversion([context](RankedTensorType tensorType) {
      if (isa<triton::PointerType>(tensorType.getElementType())) {
        return RankedTensorType::get(tensorType.getShape(),
                                     ptr::PtrType::get(context));
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
    addArgumentMaterialization(createCast);
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

    target.addIllegalOp<triton::AddPtrOp, triton::BitcastOp, triton::LoadOp,
                        triton::StoreOp, triton::IntToPtrOp,
                        triton::PtrToIntOp>();

    target.addDynamicallyLegalOp<
        linalg::FillOp, linalg::GenericOp, linalg::YieldOp, tensor::ExtractOp,
        tensor::EmptyOp, tensor::ExpandShapeOp, arith::SelectOp>([](auto op) {
      return llvm::all_of(
          llvm::concat<Value>(op->getOperands(), op->getResults()),
          [&](Value v) { return !triton::isPtrTypeLike(v.getType()); });
    });

    target.addLegalDialect<arith::ArithDialect, linalg::LinalgDialect,
                           tensor::TensorDialect, affine::AffineDialect,
                           tptr::TPtrDialect, memref::MemRefDialect>();

    patterns
        .add<AddPtrConverter, BitCastConverter, StoreConverter, LoadConverter,
             PtrToIntConverter, IntToPtrConverter, TensorExtractConverter,
             ExpandShapeConverter, SelectOpConverter, EmptyTensorConverter,
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
