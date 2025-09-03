add_triton_library(TPtrToLLVM
  TPtrToLLVMPass.cpp
  TPtrToLLVM.cpp

  DEPENDS
  TPtrToLLVMConversionPassIncGen
  TPtrTableGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRSupport
  MLIRDialectUtils
  TPtrIR
)
