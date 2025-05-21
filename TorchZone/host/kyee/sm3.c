
#include <sm3.h>

static  uint8_t std_digest[32] = {
	0x66, 0xC7, 0xF0, 0xF4, 0x62, 0xEE, 0xED, 0xD9, 0xD1, 0xF2, 0xD4, 0x6B, 0xDC, 0x10, 0xE4, 0xE2,
	0x41, 0x67, 0xC4, 0x87, 0x5C, 0xF2, 0xF7, 0xA2, 0x29, 0x7D, 0xA0, 0x2B, 0x8F, 0x4B, 0xA8, 0xE0
};

/*
功能：		
	SM3算法初始化。
输入：		
	TEE_sm3Handle_t *context	保存中间计算状态的结构体
	uint32_t context_sz			context的字节数
输出：		
	无。
返回值：		
	TEE_SUCCESS：成功；其他值失败
*/
static TEEC_Result HOST_KYEE_SM3Init(struct tee_ctx *ctx,TEE_sm3Handle_t *context,uint32_t context_sz)
{
	TEEC_Operation op;
	uint32_t origin;
	TEEC_Result res;

	memset(&op, 0, sizeof(op));
	op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INOUT,
					 			  	 TEEC_NONE,
					 				 TEEC_NONE,
			 		 				 TEEC_NONE);

	op.params[0].tmpref.buffer = context;
	op.params[0].tmpref.size = context_sz;
	
	res = TEEC_InvokeCommand(&ctx->sess, TA_SM3Init,
                                 &op, &origin);
	if (res != TEEC_SUCCESS)
		errx(1, "TEEC_InvokeCommand(TEST_KYEE_SM3Init) failed 0x%x origin 0x%x",res, origin);

	return res;
}


/*
功能：		
	SM3算法消息输入处理。
输入：		
	TEE_sm3Handle_t *context		保存中间计算状态的结构体
	uint32_t context_sz				context的字节数
	const uint8_t *input			待输入的消息。
	uint32_t input_en				待输入的消息的字节长度。
输出：		
	无。
返回值：		
	TEE_SUCCESS：成功；其他值失败
*/
static TEEC_Result HOST_KYEE_SM3Process(struct tee_ctx *ctx,TEE_sm3Handle_t *context,
							uint32_t context_sz,
							const uint8_t *input, 
							uint32_t input_en)
{
	TEEC_Operation op;
	uint32_t origin;
	TEEC_Result res;

	memset(&op, 0, sizeof(op));
	op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INOUT,
					 				 TEEC_MEMREF_TEMP_INPUT,
				  	 				 TEEC_NONE,
					 				 TEEC_NONE);

	op.params[0].tmpref.buffer = context; 
	op.params[0].tmpref.size = context_sz; 
	op.params[1].tmpref.buffer = input; 
	op.params[1].tmpref.size = input_en; 
	

	res = TEEC_InvokeCommand(&ctx->sess, TA_SM3Process,
                                 &op, &origin);
	if (res != TEEC_SUCCESS)
		errx(1, "TEEC_InvokeCommand(TEST_KYEE_SM3Process) failed 0x%x origin 0x%x",res, origin);

	return res;
}


/*
功能：		
	SM3算法获取消息摘要。
输入：		
	TEE_sm3Handle_t *context		保存中间计算状态的结构体
	uint32_t context_sz				context的字节数
输出：		
	uint8_t digest[32]				SM3摘要，也即哈希值，32字节。
返回值：		
	TEE_SUCCESS：成功；其他值失败
*/
static TEEC_Result HOST_KYEE_SM3Done(struct tee_ctx *ctx,TEE_sm3Handle_t *context,uint32_t context_sz, uint8_t *digest)
{
	TEEC_Operation op;
	uint32_t origin;
	TEEC_Result res;

	memset(&op, 0, sizeof(op));
	op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INOUT,
					 				 TEEC_MEMREF_TEMP_OUTPUT,
					 				 TEEC_NONE,
					 				 TEEC_NONE);

	op.params[0].tmpref.buffer = context; 
	op.params[0].tmpref.size = context_sz; 
	op.params[1].tmpref.buffer = digest; 
	op.params[1].tmpref.size = 32; 
	

	res = TEEC_InvokeCommand(&ctx->sess, TA_SM3Done,
                                 &op, &origin);
	if (res != TEEC_SUCCESS)
		errx(1, "TEEC_InvokeCommand(TEST_KYEE_SM3Done) failed 0x%x origin 0x%x",res, origin);

	return res;
}

//void TEST_SM3Summary_functest(uint8_t *input, int inlen)
void sm3_functest(struct tee_ctx *ctx)
{
	TEEC_Result res;
	TEE_sm3Handle_t context;
	uint32_t context_sz = sizeof(context);
	uint8_t digest[32] = {0};
	uint8_t input[10]={0x61, 0x62, 0x63};
    int inlen = 3;

	res = HOST_KYEE_SM3Init(ctx,&context,context_sz);
	if(res != TEEC_SUCCESS) 
    {
	    goto out;
    }


	res = HOST_KYEE_SM3Process(ctx,&context,context_sz,input, inlen);
	if(res != TEEC_SUCCESS)
    {
        goto out;
    }   


	res = HOST_KYEE_SM3Done(ctx,&context,context_sz,digest);
	if(res != TEEC_SUCCESS)
	{	
		goto out;
	}

    if (memcmp(std_digest, digest, 32) == 0)
    {
        printf("SM3 cpu mode self-check sucessful\n");
    }
    else
    {
        printf("SM3 cpu mode self-check failure\n");
    }
    
out:
    return;
}


