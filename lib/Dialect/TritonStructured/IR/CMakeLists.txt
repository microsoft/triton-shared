add_mlir_dialect_library(TritonStructuredIR
  TritonStructuredOps.cpp
  TritonStructuredDialect.cpp

  DEPENDS
  TritonStructuredTableGen

  LINK_LIBS PUBLIC
  TritonIR
  MLIRIR
  )
