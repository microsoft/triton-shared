//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//
//
////////////////////////////////////////////////////////////////////////////////
// Overview
////////////////////////////////////////////////////////////////////////////////
//
// This pass attempts to simplify the uses of triton pointers in the IR which
// will help lowering triton to other mlir dialects easier.
//
// A triton pointer has two pieces of info:
//   - a base pointer which comes from the kernel arguments
//   - an offset which could be either a tensor of offset or a single integer
//   offset
//
// Triton pointers are created and manipulated through a sequence of tt.addptr,
// tt.splat, or tt.broadcast ops. If a triton pointer is created through
// `tt.addptr %ptr %offset`, the new pointer will contain the same base pointer
// as the original pointer; its offset will also be accumulated. Triton pointers
// created through tt.splat and tt.broadcast retain their base pointers and
// offsets. Tensors of pointers cannot have different bases by design. In other
// words, the base pointer is fixed throughout a chain of pointer manipulation
// ops.
//
// Leveraging these insights, we can simplify chains of tt.addptr,
// tt.splat, and tt.broadcast which produce tt.addptr to just a sequence of
// offset manipulation ops and a base pointer.
//
// In essence, this pass transforms all sequences of tt.addptr into sequences of
// offset accumulation ops which are then fed into a single op
// tts.make_unstructured_tptr that takes:
//
//   - a base pointer from the kernel arguments
//   - a tensor of offsets (or single offset) that indicates the offsets from
//   the base pointer
//
// This simplification makes it easier for subsequent passes to lower these load
// and store ops.
//
// All intermediate tt.addptr ops are converted to arith.addi ops that compute
// the offsets. All pointer shape manipulation ops such as tt.splat and
// tt.broadcast will instead operate on the offsets.
//
// Ops that use tensor pointers (or pointers) produced by tt.addptr, tt.splat,
// or tt.broadcast to perform load and store (tt.load, tt.store,
// tt.make_tensor_ptr, tts.make_tptr) will operate on the result of
// tts.make_unstructured_tptr instead.
//
// Note that if the load/store ops operate on a pointer directly from the kernel
// arguments, tts.make_unstructured_tptr op will not be created. See
// test/Conversion/FoldUnstructuredPtr/make_tensor_ptr.mlir
//
// In essence, after the pass, the only two cases where a pointer value is
// introduced are:
// 1) from the kernel arguments
// 2) returned from a tts.make_unstructured_tptr
//
// As a simple example:
//
// ```
// %cst = arith.constant dense<10> : tensor<16xi32>
// %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
// %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
// %2 = tt.addptr %1, %0 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
// %3 = tt.addptr %2, %cst : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
// %4 = tt.load %3 : tensor<16x!tt.ptr<f32>>
// ```
//
// becomes
//
// ```
// %cst = arith.constant dense<10> : tensor<16xi32>
// %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
// %1 = arith.addi %0, %cst : tensor<16xi32>
// %2 = "tts.make_unstructured_tptr"(%arg0, %1) : (!tt.ptr<f32>, tensor<16xi32>)
//   -> tensor<16x!tt.ptr<f32>>
// %3 = tt.load %2 : tensor<16x!tt.ptr<f32>>
// ```
//
// By default, the pass uses i32 for the initial offsets of all pointers
// (configurable via offset-bit-width=width). If any intermediate tt.addptr
// introduces a larger bitwidth offset, the offsets will be sign-extended to the
// larger bitwidth type.
//
////////////////////////////////////////////////////////////////////////////////
// Algorithm
////////////////////////////////////////////////////////////////////////////////
//
// This pass uses a standard worklist-based algorithm to walk the use-def chains
// of all pointer arguments and create replacement ops that operate on offsets
// instead of tt.ptr types.
//
// In cases such as tt.addptr, tt.splat, and tt.broadcast, we create
// corresponding replacement ops which will then be used to map the results
// at the end of the algorithm. We do not want to modify these ops in-place
// because the use-def chains may be changed. In special cases like scf.for, we
// also set the type of the iter-arg and result directly which is usually frown
// upon (but justified).
//
// This approach is used in favor of the traditional ConversionPatternRewriter
// which converts all pointer type into an offset integer type because
// TypeConverter does not support dynamic type based on value. This limitation
// means we have to decide the same bitwidth for all tt.addptr sequences which
// is not ideal.
//
// For instance, assuming we have two sequences of tt.addptr: one operates on
// 32-bit offsets while the other operates on 64-bit offsets. If we set the
// default bitwidth to 64, the 32-bit sequence will require unncessary
// sign-extending when computing the offsets. Contrast this with the manual
// approach, we will only sign-extend where necessary.

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/Passes.h"
#include "triton-shared/Analysis/OpFoldResultUtils.h"
#include "triton-shared/AnalysisStructured/PtrAnalysis.h"
#include "triton-shared/Conversion/FoldUnstructuredPtr/FoldUnstructuredPtr.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Pass/PassManager.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"

#include <algorithm>
#include <queue>

#define DEBUG_TYPE "fold-unstructured-triton-ptr"

using namespace mlir;
using namespace triton;

#define GEN_PASS_CLASSES
#include "triton-shared/Conversion/FoldUnstructuredPtr/Passes.h.inc"

namespace {

static bool isPtrTypeLike(Type t) {
  if (auto tensorType = dyn_cast<RankedTensorType>(t)) {
    return isa<triton::PointerType>(tensorType.getElementType());
  }
  return isa<triton::PointerType>(t);
}

// Given a type, return the offset type corresponding to that type with the
// specified width.
// If the type is a tensor, return a tensor of offsets of the same shape. If the
// type is a pointer, return a single offset type.
static Type getPtrOffsetType(Type type, unsigned int bitWidth) {
  if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
    if (auto ptrType =
            dyn_cast<triton::PointerType>(tensorType.getElementType())) {
      return RankedTensorType::get(
          tensorType.getShape(), IntegerType::get(type.getContext(), bitWidth));
    }
  }

  if (auto ptrType = dyn_cast<triton::PointerType>(type)) {
    return IntegerType::get(type.getContext(), bitWidth);
  }

  llvm_unreachable("unexpected type");
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

  llvm_unreachable("unexpected type");
  return 0;
}

class FoldUnstructuredPtrPass
    : public FoldUnstructuredPtrBase<FoldUnstructuredPtrPass> {

public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<arith::ArithDialect, math::MathDialect, affine::AffineDialect,
                scf::SCFDialect, tensor::TensorDialect, triton::TritonDialect,
                tts::TritonStructuredDialect>();
  }

  struct PtrOffset {
    // the source pointer which comes from the kernel argument
    Value ptr;
    // the pointer type that corresponds to this offset; used when
    // creating tts.make_unstructured_tptr
    Type ptrType;
    // bitwidth that is used for this offset, used to track if sign-extension is
    // necessary
    unsigned int bitWidth;
    // the offset value
    Value offset;
  };

  void foldUnstructuredAddptr(unsigned int defaultBitWidth = 32) {
    llvm::SmallDenseSet<Value> ptrArgs;
    llvm::DenseMap<Value, PtrOffset> offsetMap;
    std::queue<Value> workList;

    getOperation().walk([&](FunctionOpInterface func) {
      for (auto arg : func.getArguments()) {
        if (!isPtrTypeLike(arg.getType())) {
          continue;
        }

        OpBuilder b(func->getRegion(0));
        Value zero = b.create<arith::ConstantOp>(
            arg.getLoc(),
            b.getIntegerAttr(IntegerType::get(&getContext(), defaultBitWidth),
                             0));

        ptrArgs.insert(arg);
        offsetMap.insert({arg, {arg, arg.getType(), defaultBitWidth, zero}});
        workList.push(arg);
      }
    });

    llvm::SmallVector<Operation *> toDelete;
    llvm::SmallVector<std::pair<Operation *, Value>> ptrUsers;

    while (!workList.empty()) {
      auto val = workList.front();
      workList.pop();

      for (auto &use : val.getUses()) {
        auto user = use.getOwner();

        llvm::TypeSwitch<Operation *>(user)
            .Case<triton::AddPtrOp>([&](triton::AddPtrOp addptr) {
              OpBuilder rewriter{addptr};
              auto loc = addptr->getLoc();

              auto offsetInfo = offsetMap.at(addptr.getPtr());

              auto prevOff = offsetInfo.offset;
              auto off = addptr.getOffset();

              auto lhsWidth = offsetInfo.bitWidth;
              auto rhsWidth = getBitWidth(off.getType());
              auto resWidth = std::max(lhsWidth, rhsWidth);

              if (lhsWidth < resWidth) {
                prevOff = rewriter.create<arith::ExtSIOp>(
                    loc, getPtrOffsetType(offsetInfo.ptrType, resWidth),
                    prevOff);
              }

              if (rhsWidth < resWidth) {
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

              auto ptr = op->getOperand(0);
              auto offsetInfo = offsetMap.at(ptr);

              OpBuilder rewriter{op};
              auto clone = rewriter.create(
                  op->getLoc(), op->getName().getIdentifier(),
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
            .Case<triton::LoadOp, triton::StoreOp, triton::MakeTensorPtrOp,
                  tts::MakeTensorPtrOp>([&](Operation *op) {
              // Special case:
              // We do not want to create "unstructured tensor pointer" into
              // tts.make_tptr if the base pointer is directly from the kernel
              // arguments.
              if (auto makeTensorPtr = dyn_cast<tts::MakeTensorPtrOp>(op)) {
                if (ptrArgs.contains(makeTensorPtr.getBase())) {
                  return;
                }
              }

              auto ptr = op->getOperand(0);
              auto offsetInfo = offsetMap.at(ptr);

              OpBuilder rewriter{op};
              auto makePtrOp =
                  rewriter.create<tts::MakeUnstructuredTensorPtrOp>(
                      op->getLoc(), offsetInfo.ptrType, offsetInfo.ptr,
                      offsetInfo.offset);

              ptrUsers.push_back({op, makePtrOp.getResult()});
            })
            .Case<scf::ForOp>([&](scf::ForOp forOp) {
              // Index of the init-arg corresponding to this use, note that we
              // have to subtract by 3 from the operand number because scf.for
              // ops always have 3 leading operands for start, end, and step.
              auto argIndex = use.getOperandNumber() - 3;
              auto init = forOp.getInitArgs()[argIndex];

              auto offsetInfo = offsetMap.at(init);
              auto offsetType =
                  getPtrOffsetType(offsetInfo.ptrType, offsetInfo.bitWidth);

              // We're setting both the types of the iter-arg and the
              // corresponding result directly to the offset type.
              // At this point, the IR is in an invalid state because the
              // init-args still have tt.ptr. But at the end, we will replace
              // all uses of the tt.ptr to offset values.
              auto iterArg = forOp.getRegionIterArg(argIndex);
              iterArg.setType(offsetType);

              auto res = forOp.getResult(argIndex);
              res.setType(offsetType);

              // For other ops, we only need to push the result into the
              // worklist. But for scf.for, the iter-arg corresponding to the
              // init-arg is used in the op's body instead, we have to process
              // uses of the iter-arg.
              offsetMap.insert({
                  iterArg,
                  offsetInfo,
              });
              offsetMap.insert({
                  res,
                  offsetInfo,
              });
              workList.push(iterArg);
              workList.push(res);
            })
            .Default([&](auto) {});
      }
    }

    for (auto [op, val] : ptrUsers) {
      op->setOperand(0, val);
    }

    for (auto op : toDelete) {
      auto ptrInfo = offsetMap.at(op->getResult(0));
      op->replaceAllUsesWith(ValueRange{ptrInfo.offset});
      op->erase();
    }
  }

  void runOnOperation() override {
    foldUnstructuredAddptr(offsetBitWidth);

    PassManager pm(&getContext(), getOperation().getOperationName());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    if (failed(runPipeline(pm, getOperation()))) {
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
triton::createFoldUnstructuredPtrPass() {
  return std::make_unique<FoldUnstructuredPtrPass>();
}
