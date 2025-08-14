add_subdirectory(Analysis)
add_subdirectory(AnalysisStructured)
add_subdirectory(Conversion)
add_subdirectory(Dialect)

if(TRITON_SAN_ENABLED)
  add_subdirectory(Sanitizer)
endif(TRITON_SAN_ENABLED)

add_subdirectory(Transform)
add_subdirectory(Utils)
