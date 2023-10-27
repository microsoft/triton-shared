
# Python module
if(TRITON_BUILD_PYTHON_MODULE)
    message(STATUS "Adding Triton-Shared Reference CPU Backend")
    file(INSTALL
         ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
         ${CMAKE_CURRENT_SOURCE_DIR}/ExecutionEngine/Msan.h
         ${CMAKE_CURRENT_SOURCE_DIR}/ExecutionEngine/CRunnerUtils.h
         ${CMAKE_CURRENT_SOURCE_DIR}/ExecutionEngine/CRunnerUtils.cpp
         DESTINATION ${PYTHON_THIRD_PARTY_PATH}/cpu/)
    # TODO: perhaps we want to install binary files used in __init__.py
endif()
