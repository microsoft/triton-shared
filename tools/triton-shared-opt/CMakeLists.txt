get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_llvm_executable(triton-shared-opt triton-shared-opt.cpp PARTIAL_SOURCES_INTENDED)

# TODO: what's this?
llvm_update_compile_flags(triton-shared-opt)
target_link_libraries(triton-shared-opt PRIVATE
  TritonAnalysis
  TritonTransforms
  TritonGPUTransforms
  TritonTestDialectTritonGPU
  TritonSharedAnalysis
  ${dialect_libs}
  ${conversion_libs}
  # tests
  TritonTestAnalysis
  # MLIR core
  MLIROptLib
  MLIRPass
  MLIRTransforms
)

mlir_check_all_link_libraries(triton-shared-opt)
