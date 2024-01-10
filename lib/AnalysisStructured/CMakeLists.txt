add_mlir_library(TritonSharedAnalysisStructured
  MaskAnalysis.cpp
  PtrAnalysis.cpp

  DEPENDS
  TritonAnalysis
  TritonTableGen
  TritonStructuredTableGen
  TritonGPUAttrDefsIncGen

  LINK_LIBS PUBLIC
  TritonStructuredIR
  MLIRAnalysis
)
