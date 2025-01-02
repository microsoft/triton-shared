add_triton_library(FoldUnstructuredTritonAddPtr
  FoldUnstructuredTritonAddPtrPass.cpp

  DEPENDS
  TritonStructuredTableGen
  FoldUnstructuredTritonAddPtrConversionPassIncGen

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
