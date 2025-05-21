#ifndef __PTZ_DEFS_H__
#define __PTZ_DEFS_H__

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define FLOAT_TA_SIZE (4) 
#define FLOAT64_TA_SIZE (8) 
#define UINT32_T_SIZE (4)
#define INT8_T_SIZE (1)
#define INT32_T_SIZE (4)
#define INT_TA_SIZE (4)
// #define FLOATTA float
// #define FLOATCA float   
// FLOATCA FLOATTA 用来对其 CA TA 参数的数据类型
typedef float FLOATTA;
typedef float FLOATCA;
// 专属于数据成员的类型
typedef double FLOAT64_TA;
typedef int32_t INT_TA;

#ifndef KYEE
#define KYEE (0)
#endif
// ===============================================
#ifndef MAX_LAYERS_SEQUENCE_TA
#define MAX_LAYERS_SEQUENCE_TA (256)
#endif
// #ifndef INVALID_VALUE
// #define INVALID_VALUE (0)
// #endif
#ifndef MAX_CONV_DIMENSIONS_TA
#define MAX_CONV_DIMENSIONS_TA (8)
#endif

// 内存对其边界
#define ALIGNMENT (4096)

// ===============================================

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
    // =========================================
    TSTATION_TYPE
};

// #endif // LIBTORCH_EXT

typedef enum LAYER_TYPE LAYER_TYPE_TA;
typedef enum PADDING_MODE PADDING_MODE_TA;
typedef enum DIMENSIONALITY DIMENSIONALITY_TA;
typedef enum ACTIVATION ACTIVATION_TA;
typedef enum STATION STATION_TA;

#endif