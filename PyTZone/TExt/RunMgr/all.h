#ifndef __ALL_H_CA__
#define __ALL_H_CA__

#include <iostream>
#include <memory>
#include <Logger/logger.h>
#include <cstddef>
#include <cstdlib>  // 包含 exit 函数的头文件
#include <random> 
#include <ctime>   // 包含时间函数的头文件
#include <chrono>
#include <unistd.h>

#include <generic/CustomArray.h>
#include <generic/templateList.h>

using namespace PyTZone;
using namespace PyTZone::SingleLog4;

// pytorch 中定义有 float32 和 float64 
// 目前只考虑默认的 float 为 32 位
// 项目只支持编译前预处理指定 float 类型
// #define FLOATCA float

#if TORCHZONE
#ifdef __cplusplus
extern "C" {
#endif
#include <torchzone.h>
#ifdef __cplusplus
}
#endif
#else 
enum ACTIVATION{
    NONE_ACTIV = 0,
    LOGISTIC_ACTIV = 1, 
    RELU_ACTIV, 
    RELU6_ACTIV,
    RELIE_ACTIV, 
    LINEAR_ACTIV, 
    RAMP_ACTIV, 
    TANH_ACTIV, 
    PLSE_ACTIV, 
    LEAKY_ACTIV,  // leaky_activate_TA
    ELU_ACTIV, 
    LOGGY_ACTIV, 
    STAIR_ACTIV, 
    HARDTAN_ACTIV, 
    LHTAN_ACTIV, 
    SELU_ACTIV
};

enum STATION {
    NONE_STATION = 0,
    ADD_STATION,
};

// 代指输入的维度(包括通道数)
enum DIMENSIONALITY {
    DN = 1, // 无效值
    D1 = 2,
    D2,
};
// #define MAX_CONV_DIMENSIONS 8

enum PADDING_MODE {
    ZEROS_PADDING = 200,
};

// #define INVALID_VALUE_U 0

enum LAYER_TYPE {
    NULLTYPE = 100,
    CONV_TYPE,
    FCONNECTED_TYPE,
    BATCHNORM_TYPE,
    MAXPOOL_TYPE,
    AVGPOOL_TYPE,
    ACTIV_TYPE,
    // =================== 以下是非常规的功能算子 
    // 表示无输入输出数据以及维度信息
    // 上面类似与激活函数等理论上也可以无视输入输出唯独信息
    // 但是上面的算子主要通过其他手段优化
    // 这里要将功能算子进行区分
    TSTATION_TYPE
};

typedef float FLOATCA;
#endif


#define FLOAT_SIZE (4)

// #include <torch/extension.h>	
#ifndef LIBTORCH_EXT
#define LIBTORCH_EXT 1
#endif
#ifndef TORCHZONE
#define TORCHZONE 0
#endif

#if LIBTORCH_EXT
#include <torch/script.h>
#include <torch/custom_class.h>
#endif // LIBTORCH_EXT


// 1. 标准宏
// 这些宏通常是标准库或编译器提供的 供开发者使用来获取环境、版本、特性等信息 

// __cplusplus: 表示支持的 C++ 标准的版本号 例如: 
// 199711L (C++98)
// 201103L (C++11)
// 201402L (C++14)
// 201703L (C++17)
// 202002L (C++20)
// __FILE__: 表示当前文件的名称 (字符串形式) 
// __LINE__: 表示当前的行号 (整数) 
// __DATE__: 编译时的日期 (字符串形式) 
// __TIME__: 编译时的时间 (字符串形式) 
// __STDC__: 如果定义 表示编译器支持标准 C 
// 2. 与平台或编译器相关的宏
// 这些宏可以帮助开发者判断代码正在运行的环境或使用的编译器 以便编写跨平台代码 

// __GNUC__: 表示 GCC 的主版本号 
// __clang__: 如果定义 表示正在使用 Clang 编译器 
// _MSC_VER: 表示 Microsoft 编译器的版本号 
// __unix__、__APPLE__、_WIN32: 这些宏表示目标平台类型 分别表示 Unix 系统、苹果系统和 Windows 系统 
// 3. 特性检测宏
// 这些宏用于检查编译器是否支持特定的特性: 

// __has_include(<header>): 用于检查编译器是否支持特定的头文件 
// __has_cpp_attribute(attribute): 用于检查是否支持某个 C++ 属性 
// __has_builtin(builtin): 用于检查是否支持某个内建函数 
// 4. 预定义的处理器架构宏
// __x86_64__: 如果定义 表示目标平台为 64 位 x86 架构 
// __i386__: 如果定义 表示目标平台为 32 位 x86 架构 
// __arm__: 如果定义 表示目标平台为 ARM 架构 
// 5. 标准库相关的宏
// 这些宏与 C++ 标准库和标准 C 库有关 

// BUFSIZ: 在 C 标准库中定义的缓冲区大小 通常用于文件 I/O 
// NULL: 空指针的宏定义 通常在 C/C++ 中定义为 0 或 nullptr 
// EOF: 表示文件结束的宏 通常用于文件流操作 
// 6. 字节和类型相关的宏
// CHAR_BIT: 表示 char 类型的位数 通常为 8 位 
// SCHAR_MIN、SCHAR_MAX: 表示 signed char 的最小值和最大值 
// INT_MIN、INT_MAX: 表示 int 类型的最小值和最大值 
// SIZE_MAX: 表示 size_t 类型的最大值 
// 7. 线程相关的宏
// 如果你的编译环境支持多线程 以下宏可能会定义: 

// _REENTRANT: 表示可重入代码 多用于 POSIX 线程 (pthreads) 
// _OPENMP: 表示支持 OpenMP 并行编程 

#if CHECKS
// #if 1
#define ARGC_CHECK(argc, num) {\
    if(argc != num) {\
        LogError("args error!");\
        return -1;\
    }\
}

#define ERROR_CHECK(ret, num, msg) {\
    if((ret) == (num)) {\
        LogError(msg);\
        return -1;\
    }\
}

#define RET_ERROR_CHECK(ret, num, msg, errorNum) {\
    if((ret) == (num)) {\
        LogError(msg);\
        return errorNum;\
    }\
}

#define EXIT_ERROR_CHECK(ret, num, msg) {\
    if((ret) == (num)) {\
        LogError(msg);\
        exit(EXIT_FAILURE);\
    }\
}
#else
#define ARGC_CHECK(argc, num)
#define ERROR_CHECK(ret, num, msg)
#define RET_ERROR_CHECK(ret, num, msg, errorNum)
#define EXIT_ERROR_CHECK(ret, num, msg)
#endif

namespace PyTZone {

#if TORCHZONE
void initTee();
void destoryTee();
#endif

namespace core {

#define MAX_LAYERS_SEQUENCE (256)
#define INVALID_VALUE_U (0)
#define INVALID_VALUE (-1)
#define MAX_CONV_DIMENSIONS (8)

} // namespace end of core
} // namespace end of PyTZone


#endif
