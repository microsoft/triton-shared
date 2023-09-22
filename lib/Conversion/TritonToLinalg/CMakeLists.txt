#===------------------------------------------------------------------------===#
#
# Copyright (c) Triton Project Contributors.
#
#===------------------------------------------------------------------------===#

add_mlir_conversion_library(TritonToLinalg
  TritonToLinalg.cpp
  TritonToLinalgPass.cpp

  DEPENDS
  TritonToLinalgConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRDialectUtils
  MLIRIR
  MLIRMathDialect
  MLIRPass
  MLIRTensorDialect
  MLIRTransforms
  MLIRSupport
  TritonAnalysis
  TritonIR
  TritonTransforms
  TritonSharedAnalysis
)
