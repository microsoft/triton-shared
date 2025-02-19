#pragma once
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "triton-shared/Conversion/StructuredToMemref/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "triton/Dialect/Triton/Transforms/Passes.h"

#include "triton-shared/Conversion/StructuredToMemref/Passes.h"
#include "triton-shared/Conversion/TritonArithToLinalg/Passes.h"
#include "triton-shared/Conversion/TritonPtrToMemref/Passes.h"
#include "triton-shared/Conversion/TritonToLinalg/Passes.h"
#include "triton-shared/Conversion/TritonToLinalgExperimental/Passes.h"
#include "triton-shared/Conversion/TritonToStructured/Passes.h"
#include "triton-shared/Conversion/TritonToUnstructured/Passes.h"
#include "triton-shared/Conversion/UnstructuredToMemref/Passes.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "triton-shared/Dialect/TritonTilingExt/IR/TritonTilingExtDialect.h"

#include "mlir/InitAllPasses.h"

namespace mlir {
namespace test {
void registerTestAliasPass();
void registerTestAlignmentPass();
void registerTestAllocationPass();
void registerTestMembarPass();
} // namespace test
} // namespace mlir

inline void registerTritonSharedDialects(mlir::DialectRegistry &registry) {
  mlir::registerAllPasses();
  mlir::registerTritonPasses();
  mlir::registerLinalgPasses();
  mlir::test::registerTestAliasPass();
  mlir::test::registerTestAlignmentPass();
  mlir::test::registerTestAllocationPass();
  mlir::test::registerTestMembarPass();
  mlir::triton::registerTritonToLinalgPass();
  mlir::triton::registerTritonToLinalgExperimentalPass();
  mlir::triton::registerTritonToStructuredPass();
  mlir::triton::registerTritonPtrToMemref();
  mlir::triton::registerUnstructuredToMemref();
  mlir::triton::registerTritonToUnstructuredPasses();
  mlir::triton::registerTritonArithToLinalgPasses();
  mlir::triton::registerStructuredToMemrefPasses();

  // TODO: register Triton & TritonGPU passes
  registry.insert<
      mlir::ttx::TritonTilingExtDialect, mlir::tts::TritonStructuredDialect,
      mlir::triton::TritonDialect, mlir::cf::ControlFlowDialect,
      mlir::math::MathDialect, mlir::arith::ArithDialect, mlir::scf::SCFDialect,
      mlir::gpu::GPUDialect, mlir::linalg::LinalgDialect,
      mlir::func::FuncDialect, mlir::tensor::TensorDialect,
      mlir::memref::MemRefDialect, mlir::bufferization::BufferizationDialect>();
}
