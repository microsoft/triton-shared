add_triton_library(StructuredToMemref
  StructuredToMemref.cpp
  StructuredToMemrefPass.cpp

  DEPENDS
  StructuredToMemrefConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRSCFTransforms
  MLIRArithDialect
  MLIRDialectUtils
  MLIRIR
  MLIRMathDialect
  MLIRPass
  MLIRTensorDialect
  MLIRTransforms
  MLIRSupport
  TritonIR
  TPtrIR
  TritonTransforms
  TritonTilingExtIR
  TritonStructuredIR
)
