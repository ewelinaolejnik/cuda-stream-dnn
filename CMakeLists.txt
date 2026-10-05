cmake_minimum_required(VERSION 3.24)
project(cuda_stream_dnn LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")
foreach(config Debug Release RelWithDebInfo MinSizeRel)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${config} "${CMAKE_SOURCE_DIR}/bin")
endforeach()

find_package(CUDAToolkit REQUIRED)

set(CUDNN_ROOT "" CACHE PATH "Root folder of the cuDNN installation")
set(CUDNN_CUDA_VARIANT "" CACHE STRING
  "cuDNN CUDA variant subdirectory, for example 12.9; empty selects the first installed variant")
find_path(CUDNN_INCLUDE_DIR cudnn.h
  HINTS ${CUDNN_ROOT} ENV CUDNN_ROOT
  PATH_SUFFIXES include)
find_library(CUDNN_LIBRARY cudnn
  HINTS ${CUDNN_ROOT} ENV CUDNN_ROOT
  PATH_SUFFIXES lib lib/x64 lib64)

# cuDNN 9 on Windows can use a versioned layout such as
# include/12.9/cudnn.h and lib/12.9/x64/cudnn.lib. Discover it when the
# traditional flat SDK layout above is not present.
if((NOT CUDNN_INCLUDE_DIR OR NOT CUDNN_LIBRARY) AND CUDNN_ROOT)
  if(CUDNN_CUDA_VARIANT)
    set(CUDNN_VARIANTS ${CUDNN_CUDA_VARIANT})
  else()
    file(GLOB CUDNN_VARIANT_DIRECTORIES LIST_DIRECTORIES true
      "${CUDNN_ROOT}/include/*")
    foreach(CUDNN_VARIANT_DIRECTORY IN LISTS CUDNN_VARIANT_DIRECTORIES)
      get_filename_component(CUDNN_VARIANT "${CUDNN_VARIANT_DIRECTORY}" NAME)
      list(APPEND CUDNN_VARIANTS ${CUDNN_VARIANT})
    endforeach()
  endif()
  foreach(CUDNN_VARIANT IN LISTS CUDNN_VARIANTS)
    if(EXISTS "${CUDNN_ROOT}/include/${CUDNN_VARIANT}/cudnn.h" AND
       EXISTS "${CUDNN_ROOT}/lib/${CUDNN_VARIANT}/x64/cudnn.lib")
      set(CUDNN_INCLUDE_DIR "${CUDNN_ROOT}/include/${CUDNN_VARIANT}")
      set(CUDNN_LIBRARY "${CUDNN_ROOT}/lib/${CUDNN_VARIANT}/x64/cudnn.lib")
      set(CUDNN_BIN_DIR "${CUDNN_ROOT}/bin/${CUDNN_VARIANT}/x64")
      break()
    endif()
  endforeach()
endif()

if(NOT CUDNN_INCLUDE_DIR OR NOT CUDNN_LIBRARY)
  message(FATAL_ERROR
    "cuDNN was not found. Install the cuDNN SDK, then configure with "
    "-DCUDNN_ROOT=path/to/cudnn (the folder that contains include/cudnn.h and lib/x64/cudnn.lib).")
endif()

add_executable(cuda_stream_dnn
  src/main.cpp
  src/options.cpp
  src/image_data.cpp
  src/cudnn_image_processor.cpp)
target_include_directories(cuda_stream_dnn PRIVATE include ${CUDNN_INCLUDE_DIR})
target_link_libraries(cuda_stream_dnn PRIVATE CUDA::cudart ${CUDNN_LIBRARY})

# Make the matching cuDNN runtime DLLs available beside the Windows executable.
if(WIN32 AND CUDNN_BIN_DIR)
  add_custom_command(TARGET cuda_stream_dnn POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
      "${CUDNN_BIN_DIR}" "$<TARGET_FILE_DIR:cuda_stream_dnn>")
endif()
