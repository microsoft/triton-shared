add_triton_library(TritonToUnstructured
  TritonToUnstructuredPass.cpp

  DEPENDS
  TritonStructuredTableGen
  TritonToUnstructuredConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRDialectUtils
  MLIRIR
  MLIRMathDialect
  MLIRPass
  MLIRShapeDialect
  MLIRTensorDialect
  MLIRTransforms
  MLIRSupport
  MLIRReconcileUnrealizedCasts
  TritonIR
  TritonTransforms
  TritonSharedAnalysisStructured
  TritonStructuredIR
  TritonSharedUtils
)
