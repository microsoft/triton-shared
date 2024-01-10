//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation, Meta Platforms.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_ANALYSISSTRUCTURED_PTRANALYSIS_H
#define TRITON_ANALYSISSTRUCTURED_PTRANALYSIS_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include <set>

namespace mlir {

class OpBuilder;

namespace triton {

const extern std::string ptrAnalysisAttr;

// Data structure used to decode pointer arithmetics. offsets, sizes, and
// strides are in unit of elements in a linearly laid-out memory, which is the
// same as pointer arithmetic operations in Triton language. scalar is a
// shortcut used when the entire state describes a single scalar value. source
// is the base pointer.
class PtrSState {

public:
  SmallVector<OpFoldResult> offsets;
  SmallVector<OpFoldResult> sizes;
  SmallVector<OpFoldResult> strides;
  SmallVector<OpFoldResult> modulos;

  Value source;
  Value scalar;

  int32_t getRank() const;

  bool isEmpty() const;

  bool hasModulo() const;

  bool dimHasModulo(uint32_t dim) const;

  // Process addition of two PtrSStates.
  LogicalResult addState(const PtrSState &lhsState, const PtrSState &rhsState,
                         Operation *op, OpBuilder &builder);

  // Process multiplication of two PtrSStates
  LogicalResult mulState(const PtrSState &lhsState, const PtrSState &rhsState,
                         Operation *op, OpBuilder &builder);

  tts::MakeTensorPtrOp createTTSMakeTensorPtrOp(OpBuilder &builder,
                                                Location loc);

  static void swap(PtrSState &&a, PtrSState &&b);
};

struct PtrAnalysis {
  using IndexMapSet = std::map<int, std::set<int>>;

  IndexMapSet levelToBlockArgIndex;
  int level = 0;

  llvm::SmallDenseMap<Value, PtrSState> knownPtrs;

  IRMapping map;

  // Recursively parse a Value; call the corresponding
  // function based on the defining operation and argument type.
  LogicalResult visitOperand(Value operand, PtrSState &state, const Location loc,
                             OpBuilder &builder);

  // Operand is the result of arith.addi. Process both arguments and insert any
  // arith.addi instruction as needed.
  // Main assumptions:
  //  Only one of lhsState and rhsState has source field set
  //  Current PtrSState should be empty
  // Expected result:
  //  source = lhsState.source ? lhsState.source : rhsState.source
  //  sizes[i] = lhsState.sizes[i] (which should match rhsState.sizes[i])
  //  offsets[i] = lhsState.offsets[i] + rhsState.offsets[i]
  //  strides[i] = lhsState.strides[i] + rhsState.strides[i]
  LogicalResult visitOperandAdd(arith::AddIOp addOp, PtrSState &state,
                                const Location loc, OpBuilder &builder);

  // Operand is the result of arith.muli. Process both arguments and insert any
  // arith.muli instruction as needed.
  // Main assumptions:
  //  Neither lhsState nor rhsState has source field set
  //  Current PtrSState should be empty
  //  Currently only support one of the operand is a scalar index
  // Expected result (scalar and tensorState represent the two operands):
  //  source = null
  //  sizes[i] = tensorState.sizes[i]
  //  offsets[i] = tensorState.offsets[i] * scalar
  //  strides[i] = tensorState.strides[i] * scalar
  LogicalResult visitOperandMul(arith::MulIOp mulOp, PtrSState &state,
                                const Location loc, OpBuilder &builder);

  LogicalResult visitOperandRem(arith::RemSIOp mulOp, PtrSState &state,
                                const Location loc, OpBuilder &builder);

  // Operand is the result of make_range.
  // Main assumptions:
  //  start, end, and shape are all statically known
  //  The output of make_range is 1-dimensional
  //  Does not check validity of inputs (e.g., stride > 0)
  // Expected result:
  //  source = null
  //  sizes[0] = shape[0]
  //  offset[0] = start
  //  strides[0] = ceiling( (end - start) / shape[0] )
  LogicalResult visitOperandMakeRange(triton::MakeRangeOp rangeOp,
                                      PtrSState &state, Location loc,
                                      OpBuilder &builder);

  // Operand is the result of expand_dims
  // Main assumptions:
  //  Only 1 dimension changes for each invocation of reshape
  //  The changed dimension must have size of 1
  // Expected result:
  //  Insert a dimension of size 1, stride 0, and offset 0
  LogicalResult visitOperandExpandDims(triton::ExpandDimsOp expandDimsOp,
                                       PtrSState &state, const Location loc,
                                       OpBuilder &builder);

  // Operand is the result of broadcast
  // Main assumptions:
  //  Rank of soure and result is the same
  // Expected result:
  //  Update sizes[i] only, no changes to other fields
  LogicalResult visitOperandBroadcast(triton::BroadcastOp broadcastOp,
                                      PtrSState &state, const Location loc,
                                      OpBuilder &builder);

  // Operand is the result of splat
  // Main assumptions:
  //  Source is a scalar value (i.e., an integer or a pointer, not a tensor)
  // Expected result:
  //  sizes[i] reflect the shape of the result, strides[i] = 0,  offsets[i] = 0
  //  if source is an integer, offset[0] = scalar = source
  LogicalResult visitOperandSplat(triton::SplatOp splatOp, PtrSState &state,
                                  const Location loc, OpBuilder &builder);

  // Operand is the result of arith.constant that is a splat
  // Main assumptions:
  //  Source is a constant op that produces a constant dense tensor where all
  //  elements are the same (i.e.: a constant that is splatted)
  // Expected result:
  //  sizes[i] reflect the shape of the result, strides[i] = 0,  offsets[i] =
  //  splat value if i == 0, otherwise 0
  LogicalResult visitOperandConstSplat(arith::ConstantOp op, PtrSState &state,
                                       const Location loc, OpBuilder &builder);

  // Operand is the result of addptr.
  // Main assumptions:
  //  The ptr field should populate the source field
  //  ptr and offset fields should result in same rank
  // Expected result:
  //  The resulting state for ptr and offset wil be added
  LogicalResult visitOperandAddptr(triton::AddPtrOp addptrOp, PtrSState &state,
                                   const Location loc, OpBuilder &builder);

  // Operand is the result of make_tptr.
  // Main assumptions:
  //  This function is only called when rewriting a loop
  // Expected result:
  //  Directly grab all corresponding fields from make_tptr.
  LogicalResult visitOperandMakeTPtr(tts::MakeTensorPtrOp makeTPtrOp,
                                     PtrSState &state, const Location loc,
                                     OpBuilder &builder);

  // Parse the state of AddPtrOp, insert any instruction needed to
  // calculate strides and offsets, build PtrSState for this operand, and record
  // PtrSState for knownPtrs.
  LogicalResult rewriteAddptrOp(triton::AddPtrOp op);

  // Parse the state of YieldOp, insert any instruction needed to calculate
  // strides and offsets, build PtrSState for this operand, and record PtrSState
  // in knownPtrs.
  LogicalResult
  rewriteYieldOp(scf::YieldOp op,
                 llvm::SmallDenseMap<int, PtrSState> &knownPtrsFor);

  // Rewrite eligible tt.addptr in loop init args so loop can update the such
  // pointers over iterations. Insert any instruction needed to calculate
  // strides, offsets, and modulos.
  LogicalResult rewriteForOp(scf::ForOp op);

  LogicalResult rewriteLoadOp(triton::LoadOp op);

  LogicalResult rewriteStoreOp(triton::StoreOp op);

  LogicalResult rewriteOp(Operation *op);
};

} // namespace triton

} // namespace mlir

#endif
