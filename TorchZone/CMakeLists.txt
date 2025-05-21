project (TorchZone CXX)

# set(CMAKE_CXX_COMPILER g++)
# set(CMAKE_CXX_STANDARD 11)
# /home/yyh/2.Programs/2.workplace/pytorch/2.demo/TorchZone/CMakeLists.txt
# set(CMAKE_TOOLCHAIN_FILE "./toolchain.cmake")
# set(CMAKE_TOOLCHAIN_FILE "/home/yyh/2.Programs/2.workplace/pytorch/2.demo/TorchZone/toolchain.cmake")
####################################################################
# set (SRC host/main.c)
if(NOT DEFINED KYEE)
    set(KYEE 0 CACHE STRING "value of KYEE")
endif()
####################################################################
# 原来是希望将 pytzone 中的 TExt 作为 子项目与 TorchZone 一起编译
# 但是 optee 的 qemu 项目似乎并不支持 c++ 的编译
# 单纯的配置交叉编译工具似乎不起作用
# 这里索性就将 pytzone 的编译过程放到系统启动之后了
# 但是默认的小文件系统中也不支持 c++ 但是可以为其配置编译工具
if(NOT DEFINED PYTZONE)
    set(PYTZONE 0 CACHE STRING "value of PYTZONE")
endif()
####################################################################
if(PYTZONE EQUAL 1)
	if(NOT DEFINED TORCHZONE)
		set(TORCHZONE 1 CACHE STRING "value of TORCHZONE")
	endif()
	if(NOT DEFINED LIBTORCH_EXT)
		set(LIBTORCH_EXT 0 CACHE STRING "value of LIBTORCH_EXT")
	endif()
	if(NOT DEFINED LOGCPP_EXT)
		set(LOGCPP_EXT 0 CACHE STRING "value of LOGCPP_EXT")
	endif()
endif()
####################################################################
####################################################################
set(ALL_SOURCES)
file(GLOB SRC 
    "host/*.c"
    "host/C/*.c"
	"host/kyee/*.c")
foreach(source IN LISTS SRC)
	list(APPEND ALL_SOURCES ${source})
endforeach()

if(PYTZONE EQUAL 1)
	file(GLOB PYTZONE_SOURCES 
		"host/TExt/RunMgr/*.cc"
		"host/TExt/Logger/*.cc"
		"host/TExt/core/*.cc"
		"host/TExt/core/utils/*.cc"
		"host/TExt/core/ops/*.cc")
	foreach(source IN LISTS PYTZONE_SOURCES)
		list(APPEND ALL_SOURCES ${source})
	endforeach()
endif()

# add_executable (${PROJECT_NAME} ${SRC})
add_library(${PROJECT_NAME} SHARED ${SRC})

if(PYTZONE EQUAL 1)
	target_include_directories(${PROJECT_NAME}
					PRIVATE host/TExt
					PRIVATE host/TExt/RunMgr
					PRIVATE host/TExt/generic
					PRIVATE host/TExt/../log4cpp/2.build/include
					PRIVATE host/TExt/Logger
					PRIVATE host/TExt/core
					PRIVATE host/TExt/core/utils
					PRIVATE host/TExt/core/ops)
endif()

target_include_directories(${PROJECT_NAME}
			   PRIVATE ta/include
			   PRIVATE include
			   PRIVATE host/C
			   PRIVATE host/kyee/include
			   PRIVATE host)

target_link_libraries (${PROJECT_NAME} PRIVATE teec)
#######################################################################
target_compile_definitions(${PROJECT_NAME} PRIVATE KYEE=${KYEE})
if(PYTZONE EQUAL 1)
	target_compile_definitions(${PROJECT_NAME} PRIVATE TORCHZONE=${TORCHZONE})
	target_compile_definitions(${PROJECT_NAME} PRIVATE LIBTORCH_EXT=${LIBTORCH_EXT})
	target_compile_definitions(${PROJECT_NAME} PRIVATE LOGCPP_EXT=${LOGCPP_EXT})
endif()

#################################### Ofast 编译优化
add_compile_options(-Ofast)
#######################################################################
if(PYTZONE EQUAL 1)
	file(GLOB_RECURSE HEADER_FILES
			"${CMAKE_CURRENT_SOURCE_DIR}/host/TExt/*.h"
			"${CMAKE_CURRENT_SOURCE_DIR}/host/TExt/RunMgr/*.h"
			"${CMAKE_CURRENT_SOURCE_DIR}/host/TExt/generic/*.h"
			# "${CMAKE_CURRENT_SOURCE_DIR}/../log4cpp/2.build/include/*.h"
			# "${CMAKE_CURRENT_SOURCE_DIR}/../log4cpp/2.build/include/threading/*.h"
			"${CMAKE_CURRENT_SOURCE_DIR}/host/TExt/Logger/*.h"
			"${CMAKE_CURRENT_SOURCE_DIR}/host/TExt/core/*.h"
			"${CMAKE_CURRENT_SOURCE_DIR}/host/TExt/core/utils/*.h"
			"${CMAKE_CURRENT_SOURCE_DIR}/host/TExt/core/ops/*.h"
	)

	# 安装所有头文件，保持相对路径
	foreach(HEADER_FILE ${HEADER_FILES})
		get_filename_component(HEADER_PATH ${HEADER_FILE} DIRECTORY)
		string(REPLACE "${CMAKE_CURRENT_SOURCE_DIR}/host/" "" RELATIVE_PATH ${HEADER_PATH})
		# 打印安装信息
		message(STATUS "Install header file: ${HEADER_FILE} to ${CMAKE_INSTALL_PREFIX}/include/${RELATIVE_PATH}")
		install(FILES ${HEADER_FILE}
				DESTINATION pytz/include/${RELATIVE_PATH})  # 在指定的位置安装
	endforeach()
endif()
#######################################################################

# install (TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_INSTALL_BINDIR})
install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_INSTALL_LIBDIR})
# CMAKE_INSTALL_LIBDIR : lib
# message(STATUS "!!!!!!!!CMAKE_INSTALL_LIBDIR!!!!!!!!: ${CMAKE_INSTALL_LIBDIR}")
# message(STATUS "!!!!!!!!CMAKE_CURRENT_SOURCE_DIR!!!!!!!!: ${CMAKE_CURRENT_SOURCE_DIR}")