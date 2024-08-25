add_triton_library(LinalgToLinearAlgebraSubprograms
  LinalgToLinearAlgebraSubprogramsPass.cpp

  DEPENDS
  LinalgToLinearAlgebraSubprogramsConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRLinalgTransforms
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
