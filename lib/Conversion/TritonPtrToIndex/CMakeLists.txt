add_triton_library(TritonPtrToIndex
  TritonPtrToIndexPass.cpp

  DEPENDS
  TritonStructuredTableGen
  TritonPtrToIndexConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRDialectUtils
  MLIRIR
  MLIRMathDialect
  MLIRPass
  MLIRTensorDialect
  MLIRTransforms
  MLIRSupport
  MLIRReconcileUnrealizedCasts
  TritonIR
  TritonTransforms
  TritonSharedAnalysisStructured
  TritonStructuredIR
)
