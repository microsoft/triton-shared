#===------------------------------------------------------------------------===#
#
# Copyright (c) Triton Project Contributors.
#
#===------------------------------------------------------------------------===#

add_mlir_conversion_library(TritonToLinalgLegacy
  TritonToLinalgLegacy.cpp
  TritonToLinalgLegacyPass.cpp

  DEPENDS
  TritonToLinalgLegacyConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  TritonTilingExtIR
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
