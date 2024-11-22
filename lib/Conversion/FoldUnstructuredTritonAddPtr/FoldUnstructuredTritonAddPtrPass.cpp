//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation, Meta Platforms.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "triton-shared/Analysis/OpFoldResultUtils.h"
#include "triton-shared/AnalysisStructured/PtrAnalysis.h"
#include "triton-shared/Conversion/FoldUnstructuredTritonAddPtr/FoldUnstructuredTritonAddPtr.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/OneToNTypeConversion.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"

#include <algorithm>
#include <cassert>
#include <optional>
#include <queue>

#define DEBUG_TYPE "triton-ptr-to-index"

using namespace mlir;
using namespace triton;

#define GEN_PASS_CLASSES
#include "triton-shared/Conversion/FoldUnstructuredTritonAddPtr/Passes.h.inc"

namespace {

static bool isPtrTypeLike(Type t) {
  if (auto tensorType = dyn_cast<RankedTensorType>(t)) {
    return isa<triton::PointerType>(tensorType.getElementType());
  }
  return isa<triton::PointerType>(t);
}

static Type getPtrOffsetType(Type type, unsigned int bitWidth) {
  if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
    if (auto ptrType =
            dyn_cast<triton::PointerType>(tensorType.getElementType())) {
      return RankedTensorType::get(
          tensorType.getShape(), IntegerType::get(type.getContext(), bitWidth));
    }
  } else if (auto ptrType =
                 dyn_cast<triton::PointerType>(tensorType.getElementType())) {
    return IntegerType::get(type.getContext(), bitWidth);
  }
  return nullptr;
}

static unsigned int getBitWidth(Type type) {
  if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
    if (auto integerType = dyn_cast<IntegerType>(tensorType.getElementType())) {
      return integerType.getWidth();
    }
  } else if (auto integerType = dyn_cast<IntegerType>(type)) {
    return integerType.getWidth();
  }
  type.dump();
  assert(0);
  return 0;
}

class TritonTypeConverter : public TypeConverter {
public:
  TritonTypeConverter(MLIRContext *context) {
    addConversion([](Type type) { return type; });

    addConversion([context](RankedTensorType tensorType)
                      -> std::optional<RankedTensorType> {
      if (auto ptrType =
              dyn_cast<triton::PointerType>(tensorType.getElementType())) {
        return RankedTensorType::get(tensorType.getShape(),
                                     IntegerType::get(context, 64));
      }
      return std::nullopt;
    });

    addConversion([context](triton::PointerType ptrType) -> Type {
      return IntegerType::get(context, 64);
    });

    addSourceMaterialization([&](OpBuilder &builder, Type type,
                                 ValueRange inputs,
                                 Location loc) -> std::optional<Value> {
      if (inputs.size() == 1 && isPtrTypeLike(type)) {
        auto op = builder.create<UnrealizedConversionCastOp>(loc, type, inputs);
        op->setAttr("reconvert-offset-to-ptr",
                    UnitAttr::get(builder.getContext()));
        return op->getResult(0);
      }
      return std::nullopt;
    });

    addTargetMaterialization([&](OpBuilder &builder, IntegerType type,
                                 ValueRange inputs,
                                 Location loc) -> std::optional<Value> {
      if (inputs.size() == 1 && isa<triton::PointerType>(inputs[0].getType())) {
        auto op = builder.create<UnrealizedConversionCastOp>(loc, type, inputs);
        op->setAttr("convert-arg-ptr-to-offset",
                    UnitAttr::get(builder.getContext()));
        return op->getResult(0);
      }
      return std::nullopt;
    });

    addArgumentMaterialization([&](OpBuilder &builder, Type type,
                                   ValueRange inputs,
                                   Location loc) -> std::optional<Value> {
      if (inputs.size() == 1 && isPtrTypeLike(type)) {
        auto op = builder.create<UnrealizedConversionCastOp>(loc, type, inputs);
        op->setAttr("reconvert-offset-to-ptr",
                    UnitAttr::get(builder.getContext()));
        return op->getResult(0);
      }
      return std::nullopt;
    });
  }
};

class DummyTypeConverter : public TypeConverter {
public:
  DummyTypeConverter(MLIRContext *context) {
    addConversion([](Type type) { return type; });

    addSourceMaterialization([&](OpBuilder &builder, Type type,
                                 ValueRange inputs,
                                 Location loc) -> std::optional<Value> {
      // assert(0);
      auto op = builder.create<UnrealizedConversionCastOp>(loc, type, inputs);
      op->setAttr("src-mat", UnitAttr::get(builder.getContext()));
      return op->getResult(0);

      if (inputs.size() == 1 && isPtrTypeLike(type)) {
        auto op = builder.create<UnrealizedConversionCastOp>(loc, type, inputs);
        op->setAttr("reconvert-offset-to-ptr",
                    UnitAttr::get(builder.getContext()));
        return op->getResult(0);
      }
      return std::nullopt;
    });

    addTargetMaterialization([&](OpBuilder &builder, IntegerType type,
                                 ValueRange inputs,
                                 Location loc) -> std::optional<Value> {
      // assert(0);
      auto op = builder.create<UnrealizedConversionCastOp>(loc, type, inputs);
      op->setAttr("target-mat", UnitAttr::get(builder.getContext()));
      return op->getResult(0);

      if (inputs.size() == 1 && isa<triton::PointerType>(inputs[0].getType())) {
        auto op = builder.create<UnrealizedConversionCastOp>(loc, type, inputs);
        op->setAttr("convert-arg-ptr-to-offset",
                    UnitAttr::get(builder.getContext()));
        return op->getResult(0);
      }
      return std::nullopt;
    });

    addArgumentMaterialization([&](OpBuilder &builder, Type type,
                                   ValueRange inputs,
                                   Location loc) -> std::optional<Value> {
      // assert(0);
      // if (inputs.size() == 1 && isPtrTypeLike(type)) {
      auto op = builder.create<UnrealizedConversionCastOp>(loc, type, inputs);
      op->setAttr("arg-mat", UnitAttr::get(builder.getContext()));
      return op->getResult(0);
      // }
      return std::nullopt;
    });
  }
};

struct SplatConverter : public OpConversionPattern<triton::SplatOp> {
  using OpConversionPattern<triton::SplatOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::SplatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto replacement = rewriter.create<triton::SplatOp>(
        loc, getTypeConverter()->convertType(op.getResult().getType()),
        adaptor.getSrc());
    rewriter.replaceOp(op, replacement);
    return success();
  }
};

struct BroadcastConverter : public OpConversionPattern<triton::BroadcastOp> {
  using OpConversionPattern<triton::BroadcastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::BroadcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto replacement = rewriter.create<triton::BroadcastOp>(
        loc, getTypeConverter()->convertType(op.getResult().getType()),
        adaptor.getSrc());
    rewriter.replaceOp(op, replacement);
    return success();
  }
};

struct LoadConverter : public OpConversionPattern<triton::LoadOp> {
  using OpConversionPattern<triton::LoadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto ptrType = op.getPtr().getType();

    auto cast =
        rewriter
            .create<UnrealizedConversionCastOp>(loc, ptrType, adaptor.getPtr())
            ->getResult(0);

    auto replacement =
        dyn_cast<triton::LoadOp>(rewriter.clone(*op.getOperation()));
    // replacement
    replacement.getPtrMutable().set(cast);
    rewriter.replaceOp(op, replacement);
    return success();
  }
};

struct StoreConverter : public OpConversionPattern<triton::StoreOp> {
  using OpConversionPattern<triton::StoreOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto ptrType = op.getPtr().getType();

    auto cast =
        rewriter
            .create<UnrealizedConversionCastOp>(loc, ptrType, adaptor.getPtr())
            ->getResult(0);

    auto replacement =
        dyn_cast<triton::StoreOp>(rewriter.clone(*op.getOperation()));
    // replacement
    replacement.getPtrMutable().set(cast);
    rewriter.replaceOp(op, replacement);
    return success();
  }
};

struct CastConverter : public OpConversionPattern<UnrealizedConversionCastOp> {
  using OpConversionPattern<UnrealizedConversionCastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(UnrealizedConversionCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getInputs()[0]);
    return success();
  }
};

struct AddPtrConverter : public OpConversionPattern<triton::AddPtrOp> {
  using OpConversionPattern<triton::AddPtrOp>::OpConversionPattern;

  Type getType(Type t) const {
    if (auto shapedType = dyn_cast<ShapedType>(t)) {
      return RankedTensorType::get(shapedType.getShape(),
                                   IntegerType::get(getContext(), 64));
    }
    return IntegerType::get(getContext(), 64);
  }

  LogicalResult
  matchAndRewrite(triton::AddPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto ptr = op.getPtr();
    auto func = op->getParentOfType<triton::FuncOp>();
    auto loc = op->getLoc();

    bool isArg = false;
    for (auto arg : func.getArguments()) {
      if (arg == ptr) {
        isArg = true;
        break;
      }
    }

    auto targetType = getType(op.getType());
    Value off = op.getOffset();
    if (targetType != op.getOffset().getType()) {
      off = rewriter.create<arith::ExtSIOp>(loc, targetType, op.getOffset());
    }

    auto prevOff = adaptor.getPtr();
    auto accumulatedOff = rewriter.create<arith::AddIOp>(loc, prevOff, off);
    rewriter.replaceOp(op, accumulatedOff);

    return success();
  }
};

class FoldUnstructuredTritonAddPtrPass
    : public FoldUnstructuredTritonAddPtrBase<
          FoldUnstructuredTritonAddPtrPass> {

  void bfs(Operation *op) {
    std::queue<std::pair<Value, Value>> q;
    DenseSet<Value> visited;

    op->walk([&q, &visited](UnrealizedConversionCastOp op) {
      if (op->hasAttr("convert-arg-ptr-to-offset")) {
        auto value = op->getResult(0);
        q.push({value, op.getInputs()[0]});
        visited.insert(value);
      }
    });

    // // Consider ptrs used directly without addptr
    // op->walk([&q, &visited](triton::FuncOp op) {
    //   for (auto arg : op.getArguments()) {
    //     if (isa<triton::PointerType>(arg.getType())) {
    //       q.push({arg, arg});
    //     }
    //   }
    // });

    while (!q.empty()) {
      auto [v, arg] = q.front();
      // llvm::dbgs() << "visiting: \n";
      // v.dump();
      q.pop();
      for (auto user : v.getUsers()) {
        // scf.for is a special case. We have 2 set of values to consider:
        // - iter-args
        // - loop results
        // for every init arg that originates from a
        // `tts.get_structured_state` op, its corresponding iter-arg and loop
        // result will also be considered "maybeStructured".
        if (auto castOp = dyn_cast<UnrealizedConversionCastOp>(user)) {
          castOp->dump();
          assert(castOp->hasAttr("reconvert-offset-to-ptr"));
          // assert(castOp.getInputs().size() == 1);
          // assert(castOp->getResults().size() == 1);
          for (auto user : castOp->getUsers()) {
            auto isLoadStore = llvm::isa<triton::LoadOp, triton::StoreOp>(user);
            if (!isLoadStore) {
              user->dump();
              assert(false);
            }
          }

          OpBuilder b(castOp);
          auto createPtrOp = b.create<tts::CreatePtrOp>(
              castOp->getLoc(), castOp->getResult(0).getType(), arg,
              castOp->getOperands()[0]);

          castOp.replaceAllUsesWith(createPtrOp);
        } else if (auto forOp = dyn_cast<scf::ForOp>(user)) {
          auto it = llvm::find(forOp.getInitArgs(), v);
          // assert(0);

          if (it == forOp.getInitArgs().end()) {
            continue;
          }

          // assert(0);
          auto argIndex = std::distance(forOp.getInitArgs().begin(), it);
          auto iterArg = forOp.getRegionIterArg(argIndex);
          auto tiedLoopRes = forOp.getTiedLoopResult(iterArg);

          SmallVector<Value> neighbors{iterArg, tiedLoopRes};
          for (auto neighbor : neighbors) {
            if (!visited.contains(neighbor)) {
              visited.insert(neighbor);
              q.push({neighbor, arg});
            }
          }

        } else {
          for (auto res : user->getResults()) {
            if (res.getType() != v.getType()) {
              // continue;
            }
            if (!visited.contains(res)) {
              visited.insert(res);
              q.push({res, arg});
            }
          }
        }
      }
    }

    op->walk([&q, &visited](UnrealizedConversionCastOp cast) {
      if (cast->hasAttr("convert-arg-ptr-to-offset")) {
        IRRewriter b(cast);
        b.replaceOp(cast, b.create<arith::ConstantOp>(cast->getLoc(),
                                                      b.getI64IntegerAttr(0)));
      }
    });
  }

public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<arith::ArithDialect, math::MathDialect, affine::AffineDialect,
                scf::SCFDialect, tensor::TensorDialect, triton::TritonDialect,
                tts::TritonStructuredDialect>();
  }

  void convertLoop() {
    auto moduleOp = getOperation();
    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());

    TritonTypeConverter converter(&getContext());
    scf::populateSCFStructuralTypeConversionsAndLegality(converter, patterns,
                                                         target);

    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }

  struct PtrOffset {
    unsigned int bitWidth;
    Value srcPtr;
    Type ptrType;
    Type offsetType;
  };

  auto computePtrType(unsigned int defaultBitWidth = 32) {
    auto moduleOp = getOperation();

    llvm::DenseMap<Value, PtrOffset> offsetMap;
    std::queue<Value> workList;
    llvm::DenseMap<Value, Value> ptrToOffset;

    moduleOp.walk([&](triton::FuncOp func) {
      for (auto arg : func.getArguments()) {
        if (!isPtrTypeLike(arg.getType())) {
          continue;
        }

        OpBuilder b(func.getRegion());

        auto zero = b.create<arith::ConstantOp>(
            arg.getLoc(),
            b.getIntegerAttr(IntegerType::get(&getContext(), defaultBitWidth),
                             0));

        // auto initialOffsetMarker = b.create<UnrealizedConversionCastOp>(
        //     arg.getLoc(), getPtrOffsetType(arg.getType(), defaultBitWidth),
        //     arg, zero);

        // initialOffsetMarker->setAttr("src-ptr-marker",
        //                              UnitAttr::get(&getContext()));

        // auto initialOffset = initialOffsetMarker->getResult(0);

        auto initialOffset = zero;
        ptrToOffset[initialOffset] = initialOffset;

        arg.replaceUsesWithIf(initialOffset, [](OpOperand &operand) {
          auto owner = operand.getOwner();
          return isa<triton::TritonDialect>(owner->getDialect());
        });

        offsetMap.insert(
            {initialOffset,
             {defaultBitWidth, arg, arg.getType(), zero.getType()}});
        workList.push(initialOffset);
      }
    });

    moduleOp->dump();

    SmallVector<std::pair<triton::AddPtrOp, Value>> toDelete;
    // llvm::DenseMap<Value, Value> ptrToOffset;

    while (!workList.empty()) {
      auto val = workList.front();
      // llvm::dbgs() << "processing val\n";
      val.dump();
      workList.pop();

      for (auto &use : val.getUses()) {
        auto user = use.getOwner();
        // llvm::dbgs() << "processing user\n";
        // user->dump();

        llvm::TypeSwitch<Operation *>(user)
            .Case<triton::AddPtrOp>([&](triton::AddPtrOp addptr) {
              // assert(0);
              OpBuilder rewriter{addptr};
              auto off = addptr.getOffset();
              auto prevOff = addptr.getPtr();
              auto offsetInfo = offsetMap.at(prevOff);
              auto loc = addptr->getLoc();

              // if (isPtrTypeLike(prevOff.getType())) {
              //   prevOff = rewriter
              //                 .create<UnrealizedConversionCastOp>(
              //                     loc, off.getType(), prevOff)
              //                 ->getResult(0);
              // }

              auto lhsWidth = offsetInfo.bitWidth;
              auto rhsWidth = getBitWidth(off.getType());
              auto resWidth = std::max(lhsWidth, rhsWidth);

              if (lhsWidth > rhsWidth) {
                off = rewriter.create<arith::ExtSIOp>(
                    loc, IntegerType::get(&getContext(), resWidth), off);
              }

              auto accumulatedOff = rewriter.create<arith::AddIOp>(
                  loc, off.getType(), prevOff, off);
              llvm::dbgs() << "~~~\n";
              llvm::dbgs() << "accumulate\n";
              accumulatedOff->dump();
              addptr->dump();
              llvm::dbgs() << "~~~\n";

              auto type = addptr.getType();
              toDelete.push_back({addptr, accumulatedOff.getResult()});
              // addptr->replaceAllUsesWith(accumulatedOff);
              // addptr.erase();
              // rewriter.replaceOp(addptr, accumulatedOff);

              PtrOffset newOffsetInfo{resWidth, offsetInfo.srcPtr, type,
                                      accumulatedOff.getType()};

              offsetMap.insert({addptr, newOffsetInfo});
              // offsetMap.insert({accumulatedOff, newOffsetInfo});
              workList.push(addptr);
              ptrToOffset[addptr] = accumulatedOff;
            })
            .Case<triton::SplatOp, triton::BroadcastOp>([&](Operation *op) {
              // assert(0);

              auto ptr = op->getOperand(0);
              auto res = op->getResult(0);
              auto ptrType = res.getType();
              auto offsetInfo = offsetMap.at(ptr);
              auto offsetType =
                  getPtrOffsetType(res.getType(), offsetInfo.bitWidth);

              offsetInfo.ptrType = ptrType;
              offsetInfo.offsetType = offsetType;
              res.setType(offsetType);

              offsetMap.insert({
                  res,
                  offsetInfo,
              });

              workList.push(res);
              op->dump();
            })
            .Case<triton::LoadOp, triton::StoreOp>([&](Operation *op) {
              OpBuilder rewriter{op};

              auto offset = op->getOperand(0);
              auto offsetInfo = offsetMap.at(offset);

              // if (isPtrTypeLike(offset.getType())) {
              //   offset = rewriter
              //                .create<UnrealizedConversionCastOp>(
              //                    op->getLoc(),
              //                    getPtrOffsetType(offset.getType(),
              //                                     offsetInfo.bitWidth),
              //                    offset)
              //                ->getResult(0);
              // }

              auto srcPtr = offsetInfo.srcPtr;

              offsetInfo.ptrType.dump();

              auto cast = rewriter.create<tts::CreatePtrOp>(
                  op->getLoc(), offsetInfo.ptrType, srcPtr, offset);

              op->setOperand(0, cast.getResult());
            })
            .Case<scf::ForOp>([&](scf::ForOp forOp) {
              // map init arg to iter-arg
              // map init arg to result
              // forOp.getBody();
              llvm::dbgs() << "arg number: " << use.getOperandNumber() << "\n";
              // use.get().dump();
              llvm::dbgs() << "init arg size\n";
              llvm::dbgs() << forOp.getInitArgsMutable().size() << "\n";
              llvm::dbgs() << "num region iter-args\n";
              llvm::dbgs() << forOp.getNumRegionIterArgs() << "\n";
              llvm::dbgs() << "dump from that index\n";
              auto init =
                  // forOp->getBlock().get
                  forOp.getInitArgs()[use.getOperandNumber() - 3];
              // init.dump();
              // forOp.getInitArgs()[use.getOperandNumber()].dump();

              llvm::dbgs() << "iter arg\n";
              auto offsetInfo = offsetMap.at(init);
              auto iterArg = forOp.getRegionIterArg(use.getOperandNumber() - 3);
              iterArg.setType(offsetInfo.offsetType);
              // iterArg.dump();

              auto res = forOp.getResult(use.getOperandNumber() - 3);
              res.setType(offsetInfo.offsetType);

              llvm::dbgs() << "init arg\n";

              workList.push(iterArg);
              workList.push(res);
              offsetMap.insert({
                  iterArg,
                  offsetInfo,
              });
              offsetMap.insert({
                  res,
                  offsetInfo,
              });

              for (auto arg : forOp.getInitArgs()) {
                arg.dump();
                // if (isPtrTypeLike(arg))
              }

              // llvm::dbgs() << "inits\n";
              // for (auto arg : forOp.getInits()) {
              //   arg.dump();
              // }

              // IRRewriter rewriter{forOp};

              // scf::ForOp newOp = rewriter.create<scf::ForOp>(
              //     forOp.getLoc(), forOp.getLowerBound(),
              //     forOp.getUpperBound(), forOp.getStep(), flatArgs);

              // return WalkResult::advance();
            })
            .Case<scf::YieldOp>([&](scf::YieldOp yieldOp) {
              llvm::dbgs() << "++++++++++++++\n";
              llvm::dbgs() << "yield op\n";
              yieldOp->dump();
              llvm::dbgs() << "val index: " << use.getOperandNumber() << "\n";
              use.get().dump();
              llvm::dbgs() << "++++++++++++++\n";
              // yieldOp->getResult(use.getOperandNumber())
              //     .setType(offsetMap.at(val).offsetType);
            })

            // .Case<scf::IfOp>([&](scf::IfOp ifOp) { IRRewriter rewriter{ifOp};
            // })

            // .Case<scf::WhileOp>(
            //     [&](scf::WhileOp whileOp) { IRRewriter rewriter{whileOp}; })

            // .Case<scf::ConditionOp>([&](scf::ConditionOp conditionOp) {
            //   IRRewriter rewriter{conditionOp};
            // })

            .Default([&](auto) {});
      }
    }

    moduleOp->dump();

    for (auto [op, replacement] : toDelete) {
      op.replaceAllUsesWith(replacement);
      op->erase();
    }

    return offsetMap;
  }

  struct ForConverter : public OpConversionPattern<scf::ForOp> {
    using OpConversionPattern<scf::ForOp>::OpConversionPattern;

    const llvm::DenseMap<Value, PtrOffset> &offsetMap;

    ForConverter(const llvm::DenseMap<Value, PtrOffset> &offsetMap,
                 const TypeConverter &typeConverter, MLIRContext *context)
        : OpConversionPattern<scf::ForOp>(typeConverter, context),
          offsetMap(offsetMap) {}

    void initialize() {
      // This pattern recursively unpacks one dimension at a time. The recursion
      // bounded as the rank is strictly decreasing.
      setHasBoundedRewriteRecursion();
    }

    LogicalResult
    matchAndRewrite(scf::ForOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {
      rewriter.startOpModification(op);

      llvm::dbgs() << "~~~~~~~~~~~~~~~~~~~\n";
      // op->dump();
      // for (auto i : op.getInitArgs()) {
      //   offsetMap.at(i).offsetType.dump();
      // }
      llvm::dbgs() << "~~~~~~~~~~~~~~~~~~~\n";

      auto loc = op->getLoc();
      Region &region = op.getRegion();
      llvm::dbgs() << "block count: " << region.getBlocks().size() << "\n";

      for (Block &block : llvm::make_early_inc_range(region)) {
        TypeConverter::SignatureConversion conversion(
            /*numOrigInputs=*/block.getNumArguments());

        llvm::dbgs() << "block arg count: " << block.getNumArguments() << "\n";
        llvm::dbgs() << "init arg count: " << op.getInitArgs().size() << "\n";
        int diff = block.getNumArguments() - op.getInitArgs().size();
        for (BlockArgument blockArgument : block.getArguments()) {
          int idx = blockArgument.getArgNumber();
          // blockArgument.s

          if (isPtrTypeLike(blockArgument.getType())) {
            llvm::dbgs() << "mapping " << idx << " to\n";
            // -1 for the induction var
            auto replacement =
                offsetMap.at(op.getInitArgs()[idx - 1]).offsetType;
            replacement.dump();
            op.getInitArgs()[idx - 1].dump();
            conversion.addInputs(idx, {replacement});
          } else {
            llvm::dbgs() << "reusing type for " << idx << " to\n";
            block.getArgumentTypes()[idx].dump();
            conversion.addInputs(idx, {block.getArgumentTypes()[idx]});
          }
        }

        auto term = block.getTerminator();
        for (auto r : term->getResults()) {
          if (isPtrTypeLike(r.getType())) {
            r.setType(
                offsetMap.at(op.getInitArgs()[r.getResultNumber()]).offsetType);
          }
        }

        // DummyTypeConverter converter(rewriter.getContext());
        auto converter = getTypeConverter();
        // converter = nullptr;
        rewriter.applySignatureConversion(&block, conversion, converter);
      }

      for (auto r : op->getResults()) {
        if (isPtrTypeLike(r.getType())) {
          r.setType(
              offsetMap.at(op.getInitArgs()[r.getResultNumber()]).offsetType);
        }
      }

      rewriter.finalizeOpModification(op);

      return success();
    }
  };

  void fixUp(const llvm::DenseMap<Value, PtrOffset> &offsetMap) {
    auto moduleOp = getOperation();

    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());

    target.addLegalDialect<
        func::FuncDialect, arith::ArithDialect, math::MathDialect,
        affine::AffineDialect, scf::SCFDialect, cf::ControlFlowDialect,
        tensor::TensorDialect, bufferization::BufferizationDialect>();

    target.addLegalOp<ModuleOp>();

    target.addDynamicallyLegalOp<scf::ForOp>([](scf::ForOp op) {
      for (auto arg : op.getRegionIterArgs()) {
        if (isPtrTypeLike(arg.getType())) {
          return false;
        }
      }
      return true;
    });

    // target.addIllegalOp<UnrealizedConversionCastOp>();

    DummyTypeConverter typeConverter(patterns.getContext());
    patterns.add<ForConverter>(offsetMap, typeConverter, patterns.getContext());
    // patterns.add<CastConverter>(patterns.getContext());

    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) {
      signalPassFailure();
    }
    moduleOp.walk([](UnrealizedConversionCastOp op) {
      // op.replaceAllUsesWith(ValuesT && values)
      Value v = op.getInputs()[0];
      op.replaceAllUsesWith(ValueRange{v});
    });
  }

  void runOnOperation() override {
    auto z = computePtrType(32);
    getOperation().dump();
    // fixUp(z);
    return;

    auto moduleOp = getOperation();

    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());

    target.addLegalDialect<
        func::FuncDialect, arith::ArithDialect, math::MathDialect,
        affine::AffineDialect, scf::SCFDialect, cf::ControlFlowDialect,
        tensor::TensorDialect, bufferization::BufferizationDialect>();

    target.addLegalOp<ModuleOp>();

    target.addIllegalOp<triton::AddPtrOp>();

    target.addDynamicallyLegalOp<UnrealizedConversionCastOp>([](Operation *op) {
      auto resType = op->getResultTypes()[0];
      return !isa<triton::PointerType>(resType);
    });

    target.addDynamicallyLegalOp<triton::SplatOp>([](triton::SplatOp op) {
      auto resType = op.getResult().getType();
      if (auto shapedType = dyn_cast<ShapedType>(resType)) {
        return !isa<triton::PointerType>(shapedType.getElementType());
      }
      return !isa<triton::PointerType>(resType);
    });

    target.addDynamicallyLegalOp<triton::BroadcastOp>(
        [](triton::BroadcastOp op) {
          auto resType = op.getResult().getType();
          if (auto shapedType = dyn_cast<ShapedType>(resType)) {
            return !isa<triton::PointerType>(shapedType.getElementType());
          }
          return !isa<triton::PointerType>(resType);
        });

    TritonTypeConverter converter(&getContext());

    patterns.add<AddPtrConverter, SplatConverter, BroadcastConverter>(
        converter, &getContext());
    scf::populateSCFStructuralTypeConversionsAndLegality(converter, patterns,
                                                         target);

    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) {
      signalPassFailure();
    }

    // not sure why we cannot run this together
    // convertLoop();

    PassManager pm(&getContext(), moduleOp.getOperationName());
    pm.addPass(createReconcileUnrealizedCastsPass());
    if (failed(runPipeline(pm, getOperation()))) {
      signalPassFailure();
    }

    bfs(moduleOp.getOperation());
    // return;

    moduleOp->dump();

    moduleOp.walk([&](triton::FuncOp func) {
      for (auto arg : func.getArguments()) {
        bool shouldProcess = false;
        if (isa<triton::PointerType>(arg.getType())) {
          shouldProcess = true;
        } else if (auto tensorType =
                       dyn_cast<RankedTensorType>(arg.getType())) {
          shouldProcess = isa<triton::PointerType>(tensorType.getElementType());
        }

        if (!shouldProcess) {
          continue;
        }

        bool skip = false;
        for (auto user : arg.getUsers()) {
          if (isa<tts::CreatePtrOp>(user)) {
            skip = true;
            break;
          }
        }

        if (skip) {
          continue;
        }

        triton::AddPtrOp op;
        // op.getResult().setType(Type newType)

        // ok i remember this now,
        // this is for args that aren't in any of the addptr chain but used
        // in load/store directly
        OpBuilder b(func.getRegion());
        auto loc = func->getLoc();
        auto zero = b.create<arith::ConstantOp>(loc, b.getI64IntegerAttr(0));
        auto cast = b.create<tts::CreatePtrOp>(loc, arg.getType(), arg, zero);
        arg.replaceUsesWithIf(cast->getResult(0), [](OpOperand &operand) {
          auto owner = operand.getOwner();
          return isa<triton::TritonDialect>(owner->getDialect());
        });
      }
    });
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
triton::createFoldUnstructuredTritonAddPtrPass() {
  return std::make_unique<FoldUnstructuredTritonAddPtrPass>();
}
