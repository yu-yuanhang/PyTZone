#ifndef __COMMON_TA_H__
#define __COMMON_TA_H__

#include <string.h>
#include <ptz_defs.h>

#ifndef CHECK
#define CHECK 1
#endif 

// 这里 DMSG 只有在 client 侧支持
// TA 中的日志输出系统我就懒得看了
// 后续再亲自补上一个日志系统来代替 printf (有空的情况下)
#if CHECK
#define ERROR_CHECK_RET_TA(ret, num, msg) {\
    if((ret) == (num)) {\
        printf(msg);\
        printf("\n");\
        return -1;\
    }\
}

#define ERROR_CHECK_TA(ret, num, msg) {\
    if((ret) == (num)) {\
        printf(msg);\
        printf("\n");\
        return;\
    }\
}
#else
#define ERROR_CHECK_RET_TA(ret, num, msg)
#define ERROR_CHECK_TA(ret, num, msg)
#endif

// static inline void setArr_int8_t(int8_t *dest, int8_t *src, uint32_t length) {
//     ERROR_CHECK_TA(dest, NULL, "setArr_int8_t() : dest == NULL");
//     ERROR_CHECK_TA(src, NULL, "setArr_int8_t() : src == NULL");
//     // 这个函数理论上是危险的 对于两个指针所指的实际空间大小没有进行检查
//     memcpy(dest, src, length * sizeof(int8_t));
// }

// static inline void setArr_size_t(size_t *dest, size_t *src, uint32_t length) {
//     ERROR_CHECK_TA(dest, NULL, "setArr_size_t() : dest == NULL");
//     ERROR_CHECK_TA(src, NULL, "setArr_size_t() : src == NULL");
//     // 这个函数理论上是危险的 对于两个指针所指的实际空间大小没有进行检查
//     memcpy(dest, src, length * sizeof(size_t));
// }

// static inline void setArr_uint32_t(uint32_t *dest, uint32_t *src, uint32_t length) {
//     ERROR_CHECK_TA(dest, NULL, "setArr_uint32_t() : dest == NULL");
//     ERROR_CHECK_TA(src, NULL, "setArr_uint32_t() : src == NULL");
//     // 这个函数理论上是危险的 对于两个指针所指的实际空间大小没有进行检查
//     memcpy(dest, src, length * sizeof(uint32_t));
// }


static inline void printBytes(const void *ptr, uint32_t length) {
    const unsigned char *bytePtr = (const unsigned char *)ptr;
    printf("Byte print for ptr : %p\n", bytePtr);
    for (uint32_t i = 0; i < length; ++i) {
        printf("0x%02x  ", bytePtr[i]);
    }
    printf("\n");
}
static inline uint32_t IntTUint32(INT_TA in) {
    return (in > 0) ? (in & 0xffffffff) : ((~in + 1) & 0xffffffff);
}


#endif