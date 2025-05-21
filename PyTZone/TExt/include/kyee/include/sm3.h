#ifndef __SM3_H__
#define __SM3_H__

#include <stdint.h>
#include "kyee.h"

//保存中间计算状态的结构体
typedef struct TEE_Sm3Handle_s {
	uint8_t first_update_flag;						//whether first time to update message(1:yes, 0:no)
	uint8_t finish_flag;							//whether the whole message has been inputted(1:yes, 0:no)
	uint8_t hash_buffer[64];						//block buffer
	uint32_t total[2];								//total byte length of the whole message
		
#ifdef CONFIG_HASH_SUPPORT_MUL_THREAD
	uint32_t state[8];								//keep current hash iterator value for multiple thread
#endif
} TEE_sm3Handle_t;

void sm3_functest(struct tee_ctx *ctx);

#endif

