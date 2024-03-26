add_triton_library(TritonSharedAnalysisStructured
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
