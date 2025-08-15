add_subdirectory(Analysis)
add_subdirectory(AnalysisStructured)
add_subdirectory(Conversion)
add_subdirectory(Dialect)

if(TRITON_SHARED_BUILD_TRITON_SAN)
  add_subdirectory(Sanitizer)
endif(TRITON_SHARED_BUILD_TRITON_SAN)

add_subdirectory(Transform)
add_subdirectory(Utils)
