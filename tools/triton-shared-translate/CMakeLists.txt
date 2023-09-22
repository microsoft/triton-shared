
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_llvm_executable(triton-shared-translate triton-shared-translate.cpp PARTIAL_SOURCES_INTENDED)
llvm_update_compile_flags(triton-shared-translate)
 target_link_libraries(triton-shared-translate PRIVATE
         TritonAnalysis
         TritonTransforms
         TritonGPUTransforms
         TritonNvidiaGPUTransforms
         TritonLLVMIR
         TritonPTX
         ${dialect_libs}
         ${conversion_libs}
         # tests
         TritonTestAnalysis

         TritonSharedAnalysis

         LLVMCore
         LLVMSupport
         LLVMOption
         LLVMCodeGen
         LLVMAsmParser

         # MLIR core
         MLIROptLib
         MLIRIR
         MLIRLLVMDialect
         MLIRPass
         MLIRSupport
         MLIRTransforms
         MLIRExecutionEngine
         MLIRMathToLLVM
         MLIRTransformUtils
         MLIRLLVMToLLVMIRTranslation
         MLIRNVVMToLLVMIRTranslation
         MLIRROCDLToLLVMIRTranslation
         )
mlir_check_all_link_libraries(triton-shared-translate)
