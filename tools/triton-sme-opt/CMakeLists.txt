get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_llvm_executable(triton-sme-opt triton-sme-opt.cpp PARTIAL_SOURCES_INTENDED)

llvm_update_compile_flags(triton-sme-opt)
target_link_libraries(triton-sme-opt PRIVATE

  ${dialect_libs}
  ${conversion_libs}
  # tests
  TritonTestAnalysis
  # MLIR core
  MLIROptLib
  MLIRPass
  MLIRTransforms
)

mlir_check_all_link_libraries(triton-sme-opt)
