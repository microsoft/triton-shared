add_triton_library(TritonToStructured
  TritonToStructuredPass.cpp

  DEPENDS
  TritonStructuredTableGen
  TritonToStructuredConversionPassIncGen

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
