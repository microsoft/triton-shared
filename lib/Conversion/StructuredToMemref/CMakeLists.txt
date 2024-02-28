add_mlir_conversion_library(StructuredToMemref
  StructuredToMemref.cpp
  StructuredToMemrefPass.cpp

  DEPENDS
  StructuredToMemrefConversionPassIncGen

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
  TritonTilingExtIR
  TritonStructuredIR
)
