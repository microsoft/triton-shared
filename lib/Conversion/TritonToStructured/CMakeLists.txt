add_mlir_conversion_library(TritonToStructured
  TritonToStructuredPass.cpp

  DEPENDS
  TritonStructuredTableGen
  TritonToStructuredConversionPassIncGen

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
  TritonIR
  TritonTransforms
  TritonSharedAnalysisStructured
  TritonStructuredIR
)
