#ifndef __MAIN_H__
#define __MAIN_H__

#include <stdint.h>
// =====================================================

#define TA_SM3Init          1
#define TA_SM3Process       2
#define TA_SM3Done          3

#define TA_SM4Init          4
#define TA_SM4Crypto        5

#define TA_SM2KeyGet		6
#define TA_SM2GetZ			7
#define TA_SM2GetE			8
#define TA_SM2Sign			9
#define TA_SM2Verify		10
#define TA_SM2Encrypt		11
#define TA_SM2Decrypt		12
#define TA_SM2ExchangeKey	13

#define TA_STORAGE_WRITE      14
#define TA_STORAGE_READ       15
#define TA_STORAGE_DELETE     16

// =====================================================

// 与 hello 区分开 PyTZone 的项目相关的都申明在这里
#define MAKE_NETWORK_CMD_DEMO 99
#define MAKE_NETWORK_CMD 101
#define MAKE_LAYER_CMD 102   
#define MAKE_LAYER_EXT_CMD 103    
// 这里需要考虑到对于训练的场景中未必带有TEE环境
// 所以 TA 中网络结构构建过程需要独立执行
// #define UPDATE_PARAMS_CMD 5

#define FORWARD_CMD 104
#define FORWARD_RET_CMD 105
#define FORWARD_FETCH_CMD 106



#endif
