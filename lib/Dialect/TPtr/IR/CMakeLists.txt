add_triton_library(TPtrIR
  TPtrOps.cpp
  TPtrDialect.cpp

  DEPENDS
  TPtrTableGen

  LINK_LIBS PUBLIC
  TritonIR
  MLIRIR
  )
