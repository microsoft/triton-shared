//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation, Meta Platforms.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
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
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#include <algorithm>
#include <cassert>
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
  type.dump();
  if (type.isInteger()) {
    // assert(type.getIntOrFloatBitWidth() == bitWidth);
    auto t = IntegerType::get(type.getContext(), bitWidth);
    t.dump();
    return t;
  } else if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
    if (auto ptrType =
            dyn_cast<triton::PointerType>(tensorType.getElementType())) {
      return RankedTensorType::get(
          tensorType.getShape(), IntegerType::get(type.getContext(), bitWidth));
    } else if (tensorType.getElementType().isInteger()) {
      return RankedTensorType::get(
          tensorType.getShape(), IntegerType::get(type.getContext(), bitWidth));
    }
  } else if (auto ptrType = dyn_cast<triton::PointerType>(type)) {
    return IntegerType::get(type.getContext(), bitWidth);
  }
  assert(0);
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

class FoldUnstructuredTritonAddPtrPass
    : public FoldUnstructuredTritonAddPtrBase<
          FoldUnstructuredTritonAddPtrPass> {

public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<arith::ArithDialect, math::MathDialect, affine::AffineDialect,
                scf::SCFDialect, tensor::TensorDialect, triton::TritonDialect,
                tts::TritonStructuredDialect>();
  }

  struct PtrOffset {
    Value ptr;
    Type ptrType;
    unsigned int bitWidth;
    Value offset;
    // Type offsetType;
  };

  auto computePtrType(unsigned int defaultBitWidth = 32) {
    auto moduleOp = getOperation();

    llvm::DenseMap<Value, PtrOffset> offsetMap;
    std::queue<Value> workList;
    llvm::DenseMap<Value, Value> ptrToOffset;

    /*
    algorithm:
    addptr -> addi
    src: pointer args

    - map each operand of tt.addptr to offset (ptr-to-offset)
      - special case: the ptr arg has offset == 0 (zero)


    problems:
    - do not want to set operands during traversal (use-def chains are changed)

    */

    moduleOp.walk([&](FunctionOpInterface func) {
      for (auto arg : func.getArguments()) {
        if (!isPtrTypeLike(arg.getType())) {
          continue;
        }

        // arg.dump();

        OpBuilder b(func->getRegion(0));

        Value zero = b.create<arith::ConstantOp>(
            arg.getLoc(),
            b.getIntegerAttr(IntegerType::get(&getContext(), defaultBitWidth),
                             0));

        // arg.replaceUsesWithIf(initialOffset, [](OpOperand &operand) {
        //   auto owner = operand.getOwner();
        //   // return isa<triton::TritonDialect>(owner->getDialect());
        //   return !isa<tts::MakeTensorPtrOp>(owner);
        // });

        offsetMap.insert({arg, {arg, arg.getType(), defaultBitWidth, zero}});
        workList.push(arg);
      }
    });

    moduleOp->dump();

    llvm::SmallVector<Operation *> toDelete;
    llvm::SmallVector<std::pair<Operation *, Value>> loadStores;

    /*

     */
    while (!workList.empty()) {
      auto val = workList.front();
      llvm::dbgs() << "processing val\n";
      val.dump();
      workList.pop();
      llvm::dbgs() << "these are the uses:\n";
      for (auto &use : val.getUses()) {
        use.getOwner()->dump();
      }

      llvm::dbgs() << "actual processing\n";

      for (auto &use : val.getUses()) {
        auto user = use.getOwner();
        llvm::dbgs() << "processing user\n";
        user->dump();

        llvm::TypeSwitch<Operation *>(user)
            .Case<triton::AddPtrOp>([&](triton::AddPtrOp addptr) {
              OpBuilder rewriter{addptr};
              auto offsetInfo = offsetMap.at(addptr.getPtr());

              auto prevOff = offsetInfo.offset;
              auto off = addptr.getOffset();

              auto loc = addptr->getLoc();

              auto lhsWidth = offsetInfo.bitWidth;
              auto rhsWidth = getBitWidth(off.getType());
              auto resWidth = std::max(lhsWidth, rhsWidth);

              if (lhsWidth != resWidth) {
                prevOff = rewriter.create<arith::ExtSIOp>(
                    loc, getPtrOffsetType(offsetInfo.ptrType, resWidth),
                    prevOff);
              }

              if (rhsWidth != resWidth) {
                off = rewriter.create<arith::ExtSIOp>(
                    loc, getPtrOffsetType(offsetInfo.ptrType, resWidth), off);
              }

              auto accumulatedOff = rewriter.create<arith::AddIOp>(
                  loc, getPtrOffsetType(addptr.getType(), resWidth), prevOff,
                  off);

              PtrOffset newOffsetInfo{offsetInfo.ptr, addptr.getType(),
                                      resWidth, accumulatedOff};

              offsetMap.insert({addptr, newOffsetInfo});
              workList.push(addptr);
              toDelete.push_back(addptr);
            })
            .Case<triton::SplatOp, triton::BroadcastOp>([&](Operation *op) {
              auto res = op->getResult(0);
              auto resType = res.getType();

              if (!isPtrTypeLike(resType)) {
                return;
              }
              auto loc = op->getLoc();

              auto ptr = op->getOperand(0);
              auto offsetInfo = offsetMap.at(ptr);

              offsetInfo.ptrType = resType;

              OpBuilder rewriter{op};
              auto clone = rewriter.create(
                  loc, op->getName().getIdentifier(),
                  ValueRange{offsetInfo.offset},
                  TypeRange{getPtrOffsetType(resType, offsetInfo.bitWidth)});

              PtrOffset newOffsetInfo{offsetInfo.ptr, resType,
                                      offsetInfo.bitWidth, clone->getResult(0)};

              offsetMap.insert({
                  res,
                  newOffsetInfo,
              });

              workList.push(res);
              toDelete.push_back(op);
            })
            .Case<triton::LoadOp, triton::StoreOp
                  // TODO: where do we put this?
                  // , tts::MakeTensorPtrOp
                  >([&](Operation *op) {
              OpBuilder rewriter{op};

              auto ptr = op->getOperand(0);
              // assert(toDelete.count(ptr.get));
              auto offsetInfo = offsetMap.at(ptr);

              auto srcPtr = offsetInfo.ptr;

              offsetInfo.ptrType.dump();

              auto cast = rewriter.create<tts::CreatePtrOp>(
                  op->getLoc(), offsetInfo.ptrType, srcPtr, offsetInfo.offset);

              loadStores.push_back({op, cast.getResult()});

              // op->setOperand(0, cast.getResult());
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
              auto offsetType =
                  getPtrOffsetType(offsetInfo.ptrType, offsetInfo.bitWidth);

              auto iterArg = forOp.getRegionIterArg(use.getOperandNumber() - 3);
              iterArg.setType(offsetType);
              // iterArg.dump();

              auto res = forOp.getResult(use.getOperandNumber() - 3);
              res.setType(offsetType);

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
            // .Case<scf::YieldOp>([&](scf::YieldOp yieldOp) {
            //   llvm::dbgs() << "++++++++++++++\n";
            //   llvm::dbgs() << "yield op\n";
            //   yieldOp->dump();
            //   llvm::dbgs() << "val index: " << use.getOperandNumber() <<
            //   "\n"; use.get().dump(); llvm::dbgs() << "++++++++++++++\n";
            //   // yieldOp->getResult(use.getOperandNumber())
            //   //     .setType(offsetMap.at(val).offsetType);
            // })

            // .Case<scf::IfOp>([&](scf::IfOp ifOp) { IRRewriter rewriter{ifOp};
            // })

            // .Case<scf::WhileOp>(
            //     [&](scf::WhileOp whileOp) { IRRewriter rewriter{whileOp}; })

            // .Case<scf::ConditionOp>([&](scf::ConditionOp conditionOp) {
            //   IRRewriter rewriter{conditionOp};
            // })

            .Default([&](auto) {});
      }
      llvm::dbgs() << "~~~~\n";
    }

    moduleOp->dump();

    llvm::dbgs() << "total addptr count: " << toDelete.size() << "\n";
    for (auto op : toDelete) {
      llvm::dbgs() << "deleting\n";
      op->dump();
      // op.replaceAllUsesWith(replacement);
      // op->erase();
    }

    for (auto [op, val] : loadStores) {
      op->setOperand(0, val);
    }

    for (auto op : toDelete) {
      // llvm::dbgs() << "deleting\n";
      // op->dump();
      auto ptrInfo = offsetMap.at(op->getResult(0));
      op->replaceAllUsesWith(ValueRange{ptrInfo.offset});
      op->erase();
    }

    return offsetMap;
  }

  void runOnOperation() override { auto z = computePtrType(64); }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
triton::createFoldUnstructuredTritonAddPtrPass() {
  return std::make_unique<FoldUnstructuredTritonAddPtrPass>();
}
