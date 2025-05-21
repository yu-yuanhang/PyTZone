cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(PyTZone LANGUAGES C CXX)
# cmake -DTORCHZONE=1 -DCMAKE_PREFIX_PATH="$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')" ..
# cmake -DTORCHZONE=0 -DLIBTORCH_EXT=0 -DCMAKE_PREFIX_PATH="$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')" ..
# cmake -DTORCHZONE=0 -DLIBTORCH_EXT=0 -DLOGCPP_EXT=0 -DCMAKE_PREFIX_PATH="$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')" ..
# cmake -DTORCHZONE=0 -DLIBTORCH_EXT=0 -DLOGCPP_EXT=0 -DRELEASE=1 -DOPENMP=0 -DCMAKE_PREFIX_PATH="$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')" ..

if(NOT DEFINED TORCHZONE)
    set(TORCHZONE 0 CACHE STRING "value of TORCHZONE")
endif()
if(NOT DEFINED LIBTORCH_EXT)
    set(LIBTORCH_EXT 0 CACHE STRING "value of LIBTORCH_EXT")
endif()
if(NOT DEFINED LOGCPP_EXT)
    set(LOGCPP_EXT 0 CACHE STRING "value of LOGCPP_EXT")
endif()
if(NOT DEFINED CUDA)
    set(CUDA 0 CACHE STRING "value of CPU")
endif()
if(NOT DEFINED OPENMP)
    set(OPENMP 0 CACHE STRING "value of OPENMP")
endif()
if(NOT DEFINED RELEASE)
    set(RELEASE 0 CACHE STRING "value of RELEASE")
endif()
# ${CMAKE_INSTALL_PREFIX}
set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_BINARY_DIR})
message(STATUS "Default install prefix: ${CMAKE_INSTALL_PREFIX}")
###########################################################################
# file(GLOB_RECURSE MAIN_SOURCES
file(GLOB MAIN_SOURCES_CPP
    "${CMAKE_CURRENT_SOURCE_DIR}/RunMgr/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/Logger/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/core/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/core/utils/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/core/ops/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/core/thread/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/core/ops/cpu/*.cc")

file(GLOB MAIN_SOURCES_C
    "${CMAKE_CURRENT_SOURCE_DIR}/core/ops/cpu/*.c")

message(STATUS "MAIN_SOURCES_CPP = ${MAIN_SOURCES_CPP}")
message(STATUS "MAIN_SOURCES_C = ${MAIN_SOURCES_C}")

add_library(${PROJECT_NAME} SHARED 
    ${MAIN_SOURCES_CPP}
    ${MAIN_SOURCES_C})


# CMAKE_CURRENT_SOURCE_DIR
# 当前处理的 CMakeLists.txt 所在的源目录

# CMAKE_SOURCE_DIR
# 项目的顶层源目录(即顶层的 CMakeLists.txt 所在的目录)

# message(STATUS "CMake source directory is: ${CMAKE_SOURCE_DIR}")
message(STATUS "CMake source directory is: ${CMAKE_CURRENT_SOURCE_DIR}")
# message(STATUS "CMAKE_INSTALL_BINDIR is set to ${CMAKE_INSTALL_BINDIR}")
# message(STATUS "CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX}")
# message(STATUS "CMAKE_INSTALL_BINDIR: ${CMAKE_INSTALL_BINDIR}")
# message(STATUS "CMAKE_INSTALL_LIBDIR: ${CMAKE_INSTALL_LIBDIR}")

# Logger : 路径名
# add_subdirectory(Logger)

# add_dependencies(${PROJECT_NAME} logger)

target_include_directories(${PROJECT_NAME}
                PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
                PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/RunMgr
                PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/generic
                PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../log4cpp/2.build/include
                PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../log4cpp/2.build/include/threading
                PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/Logger
                PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/core
                PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/core/utils
                PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/core/ops
                PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/core/thread
                PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/core/ops/cpu)
                
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)                

# 设置编译选项（跨平台兼容）
target_compile_options(${PROJECT_NAME} PRIVATE
    # 对所有源文件添加 -fPIC（位置无关代码）
    $<$<COMPILE_LANGUAGE:CXX>:-fPIC>
    $<$<COMPILE_LANGUAGE:C>:-fPIC>
)

# Release 模式配置
if(RELEASE EQUAL 1)
    # 明确指定优化级别（覆盖默认的 Debug 配置）
    # target_compile_options(${PROJECT_NAME} PRIVATE
    #     # $<$<NOT:$<CONFIG:Debug>>:-O3>  # Release 用 -O3
    #     $<$<NOT:$<CONFIG:Debug>>:-O0> 
    # )
    message(STATUS "Release 用 -O0")
    target_compile_options(${PROJECT_NAME} PRIVATE -O0)
    # 如果确实需要禁用优化（调试时）
    # target_compile_options(${PROJECT_NAME} PRIVATE $<$<CONFIG:Debug>:-O0>)
else()
    # Debug 模式默认 -O0（CMake 的 Debug 配置已自动处理）
    # target_compile_options(${PROJECT_NAME} PRIVATE
    #     $<$<CONFIG:Debug,RelWithDebInfo>:-g>
    #     $<$<CONFIG:Debug>:-O0>
    # )
    message(STATUS "Debug 配置用 -g -O0")
    target_compile_options(${PROJECT_NAME} PRIVATE -g -O0)
endif()

target_link_libraries(${PROJECT_NAME} PRIVATE 
    ${CMAKE_DL_LIBS}  # 如果需要动态加载
    pthread          
    # Threads::Threads  # 如果使用了线程
)

if(OPENMP EQUAL 1)
    message(STATUS "OPENMP : Enabled")
    target_compile_options(${PROJECT_NAME} PRIVATE -fopenmp -lgomp)
    find_package(OpenMP REQUIRED)  # 自动检测 OpenMP 支持
    target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX) 
endif()

###########################################################################
if(LIBTORCH_EXT EQUAL 1)
    message(STATUS "LIBTORCH_EXT : Enabled")
    find_package(Torch REQUIRED)
    find_package(PythonLibs REQUIRED)
endif()
# find_package(pybind11 REQUIRED)

if(LIBTORCH_EXT EQUAL 1)
    # TORCH_INCLUDE_DIRS 是通过 find_package(Torch REQUIRED) 查找 PyTorch 时设置的变量 指向 PyTorch 的头文件目录 
    target_include_directories(${PROJECT_NAME} PRIVATE ${TORCH_INCLUDE_DIRS})
    # PYTHON_INCLUDE_DIRS 是通过 find_package(PythonLibs REQUIRED) 
    # 查找 Python 开发包时自动设置的变量 它指向 Python 的头文件目录(如 Python.h 所在的目录)
    include_directories(${PYTHON_INCLUDE_DIRS})  # 添加 Python 头文件路径
endif()
message(STATUS "Python Include Dir: ${PYTHON_INCLUDE_DIRS}")

if(LIBTORCH_EXT EQUAL 1)
    target_link_libraries(${PROJECT_NAME} PRIVATE "${TORCH_LIBRARIES}")
endif()
###########################################################################
if(LOGCPP_EXT EQUAL 1)
    find_library(LOG4CPP_LIB log4cpp PATHS 
        ${CMAKE_CURRENT_SOURCE_DIR}/../../log4cpp/2.build/lib 
        # /another/path/to/library 
        REQUIRED) 

    # REQUIRED 参数会在库找不到时自动停止并报错 如果你需要自定义错误 可以手动判断
    if(NOT LOG4CPP_LIB)
        message(FATAL_ERROR "log4cpp library not found")
    endif()
    message(STATUS "LOG4CPP_LIB : ${LOG4CPP_LIB}")
    # 链接到某个外部库 (假设外部库名为 some_external_lib)
    target_link_libraries(${PROJECT_NAME} PRIVATE ${LOG4CPP_LIB})
endif()
###########################################################################
# Link against LibTorch
# pybind11
# target_link_libraries(PyTZone PRIVATE ${TORCH_LIBRARIES} pybind11::module)

# target_link_libraries(${PROJECT_NAME} PRIVATE logger)
###########################################################################

if(TORCHZONE)
    find_library(TORCHZONE_LIB TorchZone PATHS 
        ${CMAKE_CURRENT_SOURCE_DIR}/../../TorchZone/host
        # /another/path/to/library 
        REQUIRED) 
    target_link_libraries(${PROJECT_NAME} PRIVATE ${TORCHZONE_LIB})
    
    target_include_directories(${PROJECT_NAME}
                PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
                PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/C
                PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/kyee
                PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/ta)
endif()
###########################################################################
# add_definitions() 
target_compile_definitions(${PROJECT_NAME} PRIVATE TORCHZONE=${TORCHZONE})
target_compile_definitions(${PROJECT_NAME} PRIVATE LIBTORCH_EXT=${LIBTORCH_EXT})
target_compile_definitions(${PROJECT_NAME} PRIVATE LOGCPP_EXT=${LOGCPP_EXT})
# target_compile_definitions(${PROJECT_NAME} PRIVATE CUDA=${CUDA})
target_compile_definitions(${PROJECT_NAME} PRIVATE OPENMP=${OPENMP})
# target_compile_definitions(${PROJECT_NAME} PRIVATE RELEASE=${RELEASE})
###########################################################################
# 关于头文件的发布 ......
# install(DIRECTORY include/
#         DESTINATION include  
#         FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")

file(GLOB_RECURSE HEADER_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/../TExt/*.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/../TExt/RunMgr/*.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/../TExt/generic/*.h"
        # "${CMAKE_CURRENT_SOURCE_DIR}/../log4cpp/2.build/include/*.h"
        # "${CMAKE_CURRENT_SOURCE_DIR}/../log4cpp/2.build/include/threading/*.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/../TExt/Logger/*.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/../TExt/core/*.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/../TExt/core/utils/*.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/../TExt/core/ops/*.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/../TExt/core/thread/*.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/../TExt/core/ops/cpu/*.h"
)

# 安装所有头文件，保持相对路径
foreach(HEADER_FILE ${HEADER_FILES})
    get_filename_component(HEADER_PATH ${HEADER_FILE} DIRECTORY)
    string(REPLACE "${CMAKE_CURRENT_SOURCE_DIR}/../" "" RELATIVE_PATH ${HEADER_PATH})
    # 打印安装信息
    message(STATUS "Install header file: ${HEADER_FILE} to ${CMAKE_INSTALL_PREFIX}/include/${RELATIVE_PATH}")
    install(FILES ${HEADER_FILE}
            DESTINATION include/${RELATIVE_PATH})  # 在指定的位置安装
endforeach()

###########################################################################
# install (TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_INSTALL_BINDIR})
# install(TARGETS exetorchca  
#     # LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
#     LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}                                                                                            
#     # ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} # 如果还有静态库版本  
#     # RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} # 如果还有可执行文件版本  
# )

###############################################################################

# cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
# project(custom_class)

# find_package(Torch REQUIRED)

# # Define our library target
# add_library(custom_class SHARED class.cpp)
# set(CMAKE_CXX_STANDARD 14)
# # Link against LibTorch
# target_link_libraries(custom_class "${TORCH_LIBRARIES}")
