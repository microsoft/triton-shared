#pragma once
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"

#include "triton/Conversion/TritonToTritonGPU/Passes.h"

#include "triton-shared/Conversion/TritonArithToLinalg/Passes.h"
#include "triton-shared/Conversion/TritonToLinalg/Passes.h"
#include "triton-shared/Conversion/TritonToStructured/Passes.h"
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
  mlir::registerTritonGPUPasses();
  mlir::test::registerTestAliasPass();
  mlir::test::registerTestAlignmentPass();
  mlir::test::registerTestAllocationPass();
  mlir::test::registerTestMembarPass();
  mlir::triton::registerTritonToLinalgPass();
  mlir::triton::registerTritonToStructuredPass();
  mlir::triton::registerTritonArithToLinalgPasses();
  mlir::triton::registerConvertTritonToTritonGPUPass();

  // TODO: register Triton & TritonGPU passes
  registry
      .insert<mlir::ttx::TritonTilingExtDialect,
              mlir::tts::TritonStructuredDialect, mlir::triton::TritonDialect,
              mlir::cf::ControlFlowDialect, mlir::triton::gpu::TritonGPUDialect,
              mlir::math::MathDialect, mlir::arith::ArithDialect,
              mlir::scf::SCFDialect, mlir::gpu::GPUDialect>();
}
