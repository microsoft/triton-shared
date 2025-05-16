add_triton_library(TritonSharedTransforms
  LoopCanonicalizer.cpp

  DEPENDS
  TritonSharedTransformsPassIncGen

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRDialectUtils
  MLIRIR
  MLIRMathDialect
  MLIRPass
  MLIRTensorDialect
  MLIRTransforms
  MLIRSupport
)
