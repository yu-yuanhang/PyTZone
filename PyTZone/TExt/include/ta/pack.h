#ifndef __PACH_H__
#define __PACH_H__

#include "ptz_defs.h"
// ==============================================================

#ifndef BUFFER_SIZE
#define BUFFER_SIZE 1024
#endif

#define MAKE_NETWORK_LENGTH ((INT_TA_SIZE * 6) + (UINT32_T_SIZE * 1) + (FLOAT64_TA_SIZE * 1))
#define MAKE_LAYER_LENGTH ((INT_TA_SIZE * (6 + (MAX_CONV_DIMENSIONS_TA * 2))) + (UINT32_T_SIZE * 6) + INT32_T_SIZE)
#define MAKE_CONV_LENGTH ((INT_TA_SIZE * (3 + (MAX_CONV_DIMENSIONS_TA * 4))) + (UINT32_T_SIZE * 3) + INT32_T_SIZE)
#define MAKE_NORM_LENGTH ((INT_TA_SIZE) + (UINT32_T_SIZE * 3) + (FLOAT64_TA_SIZE * 2) + INT32_T_SIZE)
#define MAKE_ACTIV_LENGTH ((UINT32_T_SIZE * 2) + INT32_T_SIZE)
#define MAKE_STATION_LENGTH ((UINT32_T_SIZE * 4) + INT32_T_SIZE + (INT_TA_SIZE * 2))
#define MAKE_POOL_LENGTH ((INT_TA_SIZE) + (INT_TA_SIZE * (MAX_CONV_DIMENSIONS_TA * 4)) + (UINT32_T_SIZE * 5) + INT32_T_SIZE)
#define MAKE_FCONNECTED_LENGTH ((INT_TA_SIZE * 2) + (UINT32_T_SIZE * 2) + INT32_T_SIZE)

// enum 类型视作 uint32_t 

// static void pack(void *data, uint32_t offset, void *val, uint32_t length) {
//     unsigned char *tar = (unsigned char *)data + offset;
//     memcpy(tar, val, length);
// }

// static void unpack(void *data, uint32_t offset, void *val, uint32_t length) {
//     unsigned char *tar = (unsigned char *)data + offset;
//     memcpy(val, tar, length);
// }

#define PACK(data, offset, val, length) {\
    unsigned char *target = (unsigned char *)(data) + (offset); \
    memcpy(target, (val), (length)); \
}

#define UNPACK(data, offset, val, length) {\
    unsigned char *target = (unsigned char *)(data) + (offset); \
    memcpy((val), target, (length)); \
}

// aligned_alloc 标准库
// C11 标准实现 并不是所有编译器都支持
// 这里干脆自己重写了
static void *aligned_malloc(uint32_t size, size_t alignment) {
    // 检查对齐的值大于等于指针的大小 并且是 2 的幂
    if (alignment < sizeof(void *) || ((alignment & (alignment - 1) )!= 0))
        return NULL;
    void *ptr = NULL;
    ptr = malloc(size + alignment + sizeof(void *) + (alignment - 1));
    if (!ptr) {
        printf("malloc error : size = %ld\n", (size + alignment + sizeof(void *) + (alignment - 1)));
        return NULL;
    }
    void *aligned_ptr = (void *)(((uintptr_t)(ptr) + sizeof(void *) + alignment - 1) & (~(alignment - 1)));
    // 原始指针存储位置 
    ((void **)aligned_ptr)[-1] = ptr;
    return aligned_ptr;
}

static void aligned_free(void *ptr) {
    if(ptr) {
        void *original_ptr = ((void**)ptr)[-1];
        free(original_ptr);
    }
}


#endif
