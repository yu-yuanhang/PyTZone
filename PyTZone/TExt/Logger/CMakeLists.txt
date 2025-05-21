# 你可以通过 make VERBOSE=1 命令查看 CMake 在编译时调用的实际 g++ 命令 

# 设置项目名称以及使用的C++标准
# cmake_minimum_required(VERSION 3.10)
project(logger CXX)

# 设置C++标准为C++11
# set(CMAKE_CXX_STANDARD 11)
# set(CMAKE_CXX_STANDARD_REQUIRED True)

# 自动递归收集 src/ 和 src/utils/ 目录下的所有 .cpp 文件
# message(STATUS "${CMAKE_CURRENT_SOURCE_DIR}/*.cc")
file(GLOB_RECURSE LOG_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cc")

add_library(${PROJECT_NAME} SHARED ${LOG_SOURCES})
# 为该库启用 -fPIC 共享库需要生成位置无关代码
set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)

#############################################################

# /usr/bin/c++  
# -I/home/yyh/2.Programs/3.workplace/3.pytorch/2.demo/cppExtension_ca/CppExt/include  
# -MD -MT 
# CMakeFiles/CppExt.dir/Logger/logger.o 
# -MF 
# CMakeFiles/CppExt.dir/Logger/logger.o.d 
# -o CMakeFiles/CppExt.dir/Logger/logger.o 
# -c /home/yyh/2.Programs/3.workplace/3.pytorch/2.demo/cppExtension_ca/CppExt/Logger/logger.cc

# Include directories
target_include_directories(${PROJECT_NAME}
                PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} 
                # PRIVATE ${CMAKE_SOURCE_DIR}/../log4cpp/2.build/include)
                # 这里直接在顶层 CMKAE 文件中添加路径代替 PUBLIC
                # PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../log4cpp/2.build/include
                PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../log4cpp/2.build/include
                PRIVATE ${CMAKE_SOURCE_DIR}/generic)

#############################################################
# cd /home/yyh/2.Programs/3.workplace/3.pytorch/2.demo/cppExtension_ca/CppExt/build/Logger && /home/yyh/anaconda3/envs/pytorch-cu/bin/cmake -E cmake_link_script CMakeFiles/logger.dir/link.txt --verbose=1
# /usr/bin/c++ -fPIC -shared -Wl,-soname,liblogger.so -o liblogger.so CMakeFiles/logger.dir/logger.o  -llog4cpp 

# 查找库文件
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
#############################################################


# 指定编译选项 CXX11
# set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 11)
