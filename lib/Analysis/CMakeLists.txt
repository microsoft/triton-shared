add_mlir_library(TritonSharedAnalysis
  MaskAnalysis.cpp
  OpFoldResultUtils.cpp
  PtrAnalysis.cpp
  UseAnalysis.cpp

  DEPENDS
  TritonAnalysis
  TritonTableGen
  TritonGPUAttrDefsIncGen

  LINK_LIBS PUBLIC
  MLIRAnalysis
)
