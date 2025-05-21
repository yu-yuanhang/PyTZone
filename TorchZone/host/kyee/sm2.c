#include <sys/time.h>
#include <kyee.h>
#include <sm2.h>

static const uint8_t std_prikey[32] = {
	0x00, 0x00, 0x00, 0x17, 0x00, 0x00, 0x00, 0x41, 0x00, 0x00, 0x00, 0x92, 0x00, 0x00, 0x00, 0x8e, 
	0x00, 0x00, 0x00, 0xbc, 0x00, 0x00, 0x00, 0xb5, 0x00, 0x00, 0x00, 0xbc, 0x00, 0x00, 0x00, 0xf2, 
};

static const uint8_t std_pubkey[65] = {
	0x04, 0x46, 0x0b, 0xee, 0xae, 0xe7, 0x1a, 0x76, 0x14, 0xf4, 0x83, 0x24, 0xeb, 0x2a, 0x35, 0x91, 
	0xdd, 0xfe, 0x9c, 0xb8, 0xd8, 0xfd, 0x47, 0x05, 0x24, 0x4c, 0x1b, 0xc7, 0xab, 0x63, 0x7a, 0xe0, 
	0x51, 0x3d, 0x97, 0x7d, 0x75, 0xfe, 0xdc, 0xdf, 0x2d, 0xcd, 0x67, 0xf8, 0xb4, 0x03, 0xda, 0xc2, 
	0x9f, 0x07, 0x9e, 0x7e, 0x36, 0xc8, 0x40, 0x73, 0x8d, 0xc5, 0x2f, 0xd2, 0x15, 0x87, 0xc0, 0x77, 
	0x71, 
};

static const uint8_t std_Z[] = {
	0x6a, 0x6b, 0xf3, 0x6d, 0xa7, 0xda, 0xb4, 0xc8, 0xd3, 0xf5, 0x39, 0x80, 0x3a, 0x6d, 0xa2, 0xf1, 
	0x8e, 0x80, 0x71, 0x2b, 0x22, 0x6b, 0xff, 0x7b, 0x16, 0xd5, 0xc5, 0x99, 0x5c, 0x8b, 0x54, 0x01,
};
		
static const uint8_t std_E[] = {
	0xbc, 0x0b, 0xbf, 0x52, 0xb3, 0xf9, 0xa6, 0xce, 0xfc, 0x1b, 0xfa, 0x84, 0xd6, 0xe6, 0xb2, 0x53, 
	0x3d, 0x1a, 0x43, 0x3b, 0x0a, 0xed, 0xe2, 0xef, 0xab, 0xac, 0x89, 0x18, 0x2c, 0xa5, 0xd2, 0xb9, 
};
	
static const uint8_t std_signature[] = {
	0x85, 0x1b, 0x70, 0x88, 0xcd, 0x92, 0x12, 0xd2, 0x7f, 0xf0, 0xc6, 0xe1, 0x2d, 0x86, 0xe2, 0x00, 
	0xce, 0x09, 0x44, 0xe7, 0xab, 0xc4, 0xe2, 0x70, 0xf2, 0x2d, 0xb0, 0x3e, 0xe6, 0xd5, 0x02, 0xbe, 
	0x56, 0x37, 0xed, 0x5c, 0xf4, 0x0a, 0xb7, 0x5e, 0x8e, 0x7b, 0x2a, 0xc6, 0x3a, 0x5a, 0x6b, 0x80, 
	0xd7, 0xd1, 0xdf, 0x54, 0xc7, 0x7b, 0x26, 0x3d, 0xa6, 0x14, 0x2c, 0x83, 0x59, 0xb1, 0x50, 0x5c, 
};

static const uint8_t std_cipher1[] = {		
	0x04, 0xcb, 0x33, 0xcd, 0x05, 0xfa, 0x23, 0xc6, 0xe2, 0xb6, 0x15, 0xe1, 0x3f, 0x4e, 0x25, 0x3d,
	0x96, 0x49, 0x59, 0xc6, 0x1f, 0x15, 0x14, 0x06, 0x8a, 0xc3, 0x6a, 0xf8, 0x80, 0xc4, 0x61, 0x57,
	0xb7, 0x49, 0x4a, 0xc3, 0x3d, 0xa6, 0x53, 0x10, 0xb6, 0x44, 0x00, 0x88, 0xbe, 0x05, 0x3f, 0x29,
	0xcb, 0x29, 0x1f, 0xe9, 0x0e, 0xac, 0xf0, 0x45, 0xab, 0xab, 0x75, 0x47, 0x8f, 0x09, 0x8e, 0x54,
	0xa7, 0xe3, 0xbc, 0x4d, 0x96, 0x8a, 0x59, 0x02, 0x00, 0xb3, 0x9d, 0x70, 0xf0, 0x9a, 0x9e, 0xcc, 
	0xa4, 0x10, 0xf2, 0x3b, 0x91, 0x2e, 0x54, 0x91, 0x43, 0x4e, 0x2d, 0xa3, 0x66, 0x60, 0xfd, 0xe4, 
	0x7f, 0xce, 0x9a, 0xa3, 0xf1, 0x43, 0xd5, 0xb1, 0x6b, 0x06, 0xce, 0x5d, 0x6b, 0x4b, 0x1d, 0xa2, 
	0x4d, 0x99, 0x8a, 0xc4, 0xcb, 0xc2, 0xe4, 0x5c, 0x58, 0xcf, 0x00, 0xda, 0x09, 0xf7, 0x94, 0x5b, 
	0x30, 
};

static const uint8_t std_cipher2[] = {	
	0x04, 0xcb, 0x33, 0xcd, 0x05, 0xfa, 0x23, 0xc6, 0xe2, 0xb6, 0x15, 0xe1, 0x3f, 0x4e, 0x25, 0x3d, 
	0x96, 0x49, 0x59, 0xc6, 0x1f, 0x15, 0x14, 0x06, 0x8a, 0xc3, 0x6a, 0xf8, 0x80, 0xc4, 0x61, 0x57, 
	0xb7, 0x49, 0x4a, 0xc3, 0x3d, 0xa6, 0x53, 0x10, 0xb6, 0x44, 0x00, 0x88, 0xbe, 0x05, 0x3f, 0x29, 
	0xcb, 0x29, 0x1f, 0xe9, 0x0e, 0xac, 0xf0, 0x45, 0xab, 0xab, 0x75, 0x47, 0x8f, 0x09, 0x8e, 0x54, 
	0xa7, 0xce, 0x9a, 0xa3, 0xf1, 0x43, 0xd5, 0xb1, 0x6b, 0x06, 0xce, 0x5d, 0x6b, 0x4b, 0x1d, 0xa2, 
	0x4d, 0x99, 0x8a, 0xc4, 0xcb, 0xc2, 0xe4, 0x5c, 0x58, 0xcf, 0x00, 0xda, 0x09, 0xf7, 0x94, 0x5b, 
	0x30, 0xe3, 0xbc, 0x4d, 0x96, 0x8a, 0x59, 0x02, 0x00, 0xb3, 0x9d, 0x70, 0xf0, 0x9a, 0x9e, 0xcc, 
	0xa4, 0x10, 0xf2, 0x3b, 0x91, 0x2e, 0x54, 0x91, 0x43, 0x4e, 0x2d, 0xa3, 0x66, 0x60, 0xfd, 0xe4,
	0x7f, 
};

static uint8_t const std_key[16] = {
	0x6C, 0x89, 0x34, 0x73, 0x54, 0xDE, 0x24, 0x84, 0xC6, 0x0B, 0x4A, 0xB1, 0xFD, 0xE4, 0xC6, 0xE5,
};
	 
static uint8_t const std_S1_SB[32] = {
	 0xD3, 0xA0, 0xFE, 0x15, 0xDE, 0xE1, 0x85, 0xCE, 0xAE, 0x90, 0x7A, 0x6B, 0x59, 0x5C, 0xC3, 0x2A,
	 0x26, 0x6E, 0xD7, 0xB3, 0x36, 0x7E, 0x99, 0x83, 0xA8, 0x96, 0xDC, 0x32, 0xFA, 0x20, 0xF8, 0xEB
};
 
static uint8_t const std_S2_SA[32] = {
	0x18, 0xC7, 0x89, 0x4B, 0x38, 0x16, 0xDF, 0x16, 0xCF, 0x07, 0xB0, 0x5C, 0x5E, 0xC0, 0xBE, 0xF5,
	0xD6, 0x55, 0xD5, 0x8F, 0x77, 0x9C, 0xC1, 0xB4, 0x00, 0xA4, 0xF3, 0x88, 0x46, 0x44, 0xDB, 0x88
};

/*
功能：		
	SM2密钥对生成。
输入：		
	无
输出：		
	uint8_t priKey[32]			SM2私钥，32字节
	uint8_t pubKey[65]			SM2公钥，65字节。
返回值：		
	TEE_SUCCESS：成功；其他值失败.
*/
static TEEC_Result HOST_KYEE_SM2KeyGet(struct tee_ctx *ctx,uint8_t *priKey,uint8_t *pubKey)
{
	TEEC_Operation op;
	uint32_t origin;
	TEEC_Result res;

	memset(&op, 0, sizeof(op));
	op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_OUTPUT,
                                     TEEC_MEMREF_TEMP_OUTPUT,
                                     TEEC_NONE,
                                     TEEC_NONE);
	op.params[0].tmpref.buffer = priKey;
	op.params[0].tmpref.size = 32;
	op.params[1].tmpref.buffer = pubKey;
	op.params[1].tmpref.size = 65;

	res = TEEC_InvokeCommand(&ctx->sess, TA_SM2KeyGet,
                                 &op, &origin);
	if (res != TEEC_SUCCESS)
		errx(1, "TEEC_InvokeCommand(TEST_KYEE_SM2KeyGet) failed 0x%x origin 0x%x",res, origin);
	
	return res;
}

/*
功能：		
	获取Z值。
输入：		
	uint8_t *id						用户ID。
	uint32_t id_len					用户ID的字节长度，要小于8192。
	uint8_t pubKey[65]				用户的公钥。
输出：		
	uint8_t z[32]					计算得到的Z值，32字节。
返回值：		
	TEE_SUCCESS：成功；其他值失败，具体参考第二节的接口相关标识定义。
注意事项：	
	1.根据《SM2密码算法使用规范》，无特殊约定的情况下，用户标识 ID 的长度为 16 字节，
	其默认值从左至右依次如下：
		0x31,0x32,0x33,0x34,0x35,0x36,0x37,0x38,0x31,0x32,0x33,0x34,0x35,0x36,0x37,0x38。
	即没有用到用户标识的情况下，请使用该默认ID。
	2. 该接口实际是调用SM3接口计算hash。
*/
static TEEC_Result HOST_KYEE_SM2GetZ(struct tee_ctx *ctx,uint8_t *id,uint32_t id_len,uint8_t *pubKey,uint8_t *z)
{
	TEEC_Operation op;
	uint32_t origin;
	TEEC_Result res;

	memset(&op, 0, sizeof(op));
	op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT,
                                     TEEC_MEMREF_TEMP_INPUT,
                                     TEEC_MEMREF_TEMP_OUTPUT,
                                     TEEC_NONE);
	op.params[0].tmpref.buffer = id;
	op.params[0].tmpref.size = id_len;
	op.params[1].tmpref.buffer = pubKey;
	op.params[1].tmpref.size = 65;
	op.params[2].tmpref.buffer = z;
	op.params[2].tmpref.size = 32;

	res = TEEC_InvokeCommand(&ctx->sess, TA_SM2GetZ,
                                 &op, &origin);
	if (res != TEEC_SUCCESS)
		errx(1, "TEEC_InvokeCommand(TEST_KYEE_SM2GetZ) failed 0x%x origin 0x%x",
                        res, origin);
	return res;
}


/*
功能：		
	获取E值。
输入：		
	uint8_t *m					消息M。
	uint32_t m_len				消息M的字节长度。
	uint8_t z[32]				Z值，32字节。
输出：		
	uint8_t e[32]				计算得到的E值，32字节。
返回值：		
	TEE_SUCCESS：成功；其他值失败。
注意事项：	
	1.该接口实际是调用SM3接口计算hash。
*/
static TEEC_Result HOST_KYEE_SM2GetE(struct tee_ctx *ctx,uint8_t *m,uint32_t m_len,uint8_t *z,uint8_t *e)
{
	
	TEEC_Operation op;
	uint32_t origin;
	TEEC_Result res;


	memset(&op, 0, sizeof(op));
	op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT,
					 				 TEEC_MEMREF_TEMP_INPUT,
                                     TEEC_MEMREF_TEMP_OUTPUT,
                                     TEEC_NONE);

	
	op.params[0].tmpref.buffer = m;
	op.params[0].tmpref.size = m_len;
	op.params[1].tmpref.buffer = z;
	op.params[1].tmpref.size = 32;
	op.params[2].tmpref.buffer = e;
	op.params[2].tmpref.size = 32;
	
	res = TEEC_InvokeCommand(&ctx->sess, TA_SM2GetE,
                                 &op, &origin);
	if (res != TEEC_SUCCESS)
		errx(1, "TEEC_InvokeCommand(TEST_KYEE_SM2GetE) failed 0x%x origin 0x%x",
                        res, origin);

	return res;
}

/*
功能：		
	SM2签名。
输入：		
	uint8_t e[32]					待签名的E值，32字节
	uint8_t priKey[32]				签名者的私钥，32字节
输出：		
	uint8_t signature[64]			签名结果，64字节。
返回值：		
	TEE_SUCCESS：成功；其他值失败。
注意事项：	
	1.对消息进行签名，必须先调用sm2_getZ获得Z值，再调用sm2_getE获得E值。最终调用本接口得到最终签名结果。
*/
static TEEC_Result HOST_KYEE_SM2Sign(struct tee_ctx *ctx,uint8_t *e,uint8_t *priKey,uint8_t *signature)
{
	TEEC_Operation op;
	uint32_t origin;
	TEEC_Result res;

	memset(&op, 0, sizeof(op));
	op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT,
					 				 TEEC_MEMREF_TEMP_INPUT,
                                     TEEC_MEMREF_TEMP_OUTPUT,
                                     TEEC_NONE);

	op.params[0].tmpref.buffer = e;
	op.params[0].tmpref.size = 32;
	op.params[1].tmpref.buffer = priKey;
	op.params[1].tmpref.size = 32;
	op.params[2].tmpref.buffer = signature;
	op.params[2].tmpref.size = 64;
	

	res = TEEC_InvokeCommand(&ctx->sess, TA_SM2Sign,
                                 &op, &origin);
	if (res != TEEC_SUCCESS)
 		errx(1, "TEEC_InvokeCommand(TEST_KYEE_SM2Sign) failed 0x%x origin 0x%x",res, origin);

	return res;
}

/*
功能：		
	SM2签名验证。
输入：		
	uint8_t e[32]					待签名的E值，32字节
	uint8_t pubKey[65]				签名验证用的公钥，65字节
	uint8_t signature[64]		 	签名结果，64字节。
输出：		
	无
返回值：		
	TEE_SUCCESS：成功；其他值失败。
注意事项：	
	1.对消息进行签名验证，必须先调用sm2_getZ获得Z值，再调用sm2_getE获得E值。最终调用本接口得到验证结果。
*/
static TEEC_Result HOST_KYEE_SM2Verify(struct tee_ctx *ctx,uint8_t *e,uint8_t *pubKey,uint8_t *signature)
{
	TEEC_Operation op;
	uint32_t origin;
	TEEC_Result res;

	memset(&op, 0, sizeof(op));
	op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT,
					 				 TEEC_MEMREF_TEMP_INPUT,
                                     TEEC_MEMREF_TEMP_INPUT,
                                     TEEC_NONE);

	op.params[0].tmpref.buffer = e;
	op.params[0].tmpref.size = 32;
	op.params[1].tmpref.buffer = pubKey;
	op.params[1].tmpref.size = 65;
	op.params[2].tmpref.buffer = signature;
	op.params[2].tmpref.size = 64;
	
	res = TEEC_InvokeCommand(&ctx->sess, TA_SM2Verify,
                                 &op, &origin);
	if (res != TEEC_SUCCESS)
		errx(1, "TEEC_InvokeCommand(TEST_KYEE_SM2Verify) failed 0x%x origin 0x%x",res, origin);

	return res;
}

/*
功能：		
	SM2消息加密。
输入：		
	uint8_t *m					待加密的明文，MByteLen字节。
	uint32_t m_len				待加密的明文的字节长度，必须大于0。
	uint8_t pubKey[65]			加密用的公钥，65字节
输出：		
	uint8_t *c					加密得到的密文，注意其指向空间要保证至少是MByteLen+97字节。
	uint32_t *c_len				密文长度，正常情况下是MByteLen+97字节。
返回值：		
	TEE_SUCCESS：成功；其他值失败。
注意事项：	
	1.密文分三部分，C1是曲线上一个点，65字节；C2和明文一样长；C3是保证完整性的校验值，为SM3算法结果，32字节。
*/
static TEEC_Result HOST_KYEE_SM2Encrypt(struct tee_ctx *ctx,uint8_t *m,uint32_t m_len,
                            uint8_t *pubKey, int order, uint8_t *c, uint32_t *c_len)
{
	TEEC_Operation op;
	uint32_t origin;
	TEEC_Result res;

	memset(&op, 0, sizeof(op));
	op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT,
					 				 TEEC_MEMREF_TEMP_INPUT,
                                     TEEC_MEMREF_TEMP_OUTPUT,
					 				 TEEC_VALUE_INOUT);

	op.params[0].tmpref.buffer = m;
	op.params[0].tmpref.size = m_len;
	op.params[1].tmpref.buffer = pubKey;
	op.params[1].tmpref.size = 65;  // Length of pubkey
	op.params[2].tmpref.buffer = c;
	op.params[2].tmpref.size = *c_len;
	op.params[3].value.a = *c_len;
    op.params[3].value.b = order;
	
	res = TEEC_InvokeCommand(&ctx->sess, TA_SM2Encrypt,
                                 &op, &origin);
	if (res != TEEC_SUCCESS)
		errx(1, "TEEC_InvokeCommand(TEST_KYEE_SM2Encrypt) failed 0x%x origin 0x%x",res, origin);
	
	*c_len = op.params[3].value.a;
	
	return res;
}

/*
功能：		
	SM2消息解密。
输入：		
	uint8_t *c					密文，CByteLen字节。
	uint32_t c_len				密文长度，必须大于97字节。
	uint8_t priKey[32]			解密用到的私钥。
输出：		
	uint8_t *m					解密得到的明文。
	uint32_t *m_len				解密得到的明文的字节长度，正常应该是CByteLen-96。
返回值：		
	TEE_SUCCESS：成功；其他值失败，具体参考第二节的接口相关标识定义。
注意事项：	
	1.密文分三部分，C1是曲线上一个点，65字节；C2和明文一样长；C3是保证完整性的校验值，为SM3算法结果，32字节。
	2. 输入的密文C和其顺序标识order必须相符，否则解密失败。
*/
static TEEC_Result HOST_KYEE_SM2Decrypt(struct tee_ctx *ctx,uint8_t *c,uint32_t c_len,
                            uint8_t *priKey, int order, uint8_t *m,uint32_t *m_len)
{
	TEEC_Operation op;
	uint32_t origin;
	TEEC_Result res;

	memset(&op, 0, sizeof(op));
	op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT,
					 				 TEEC_MEMREF_TEMP_INPUT,
                                     TEEC_MEMREF_TEMP_OUTPUT,
					 				 TEEC_VALUE_INOUT);

	op.params[0].tmpref.buffer = c;
	op.params[0].tmpref.size = c_len;
	op.params[1].tmpref.buffer = priKey;
	op.params[1].tmpref.size = 32;  // Length of privkey
	op.params[2].tmpref.buffer = m;
	op.params[2].tmpref.size = *m_len;
	op.params[3].value.a = *m_len;
    op.params[3].value.b = order;
	

	res = TEEC_InvokeCommand(&ctx->sess, TA_SM2Decrypt,
                                 &op, &origin);
	if (res != TEEC_SUCCESS)
		errx(1, "TEEC_InvokeCommand(TEST_KYEE_SM2Decrypt) failed 0x%x origin 0x%x",res, origin);
	*m_len = op.params[3].value.a;

	return res;
}

/*
功能：		
	SM2密钥协商。

输入：		
	无
输出：		
	无

返回值：		
	TEE_SUCCESS：成功；其他值失败。

注意事项：	
	1.若协商成功，则双方协商出来的KA相同，发送方的S1和接收方的SB相同，发送方的SA和接收方的S2相同。
*/

TEEC_Result HOST_KYEE_SM2ExchangeKey(struct tee_ctx *ctx)
{

	TEEC_Operation op;
	uint32_t origin;
	TEEC_Result res;

	op.paramTypes = TEEC_PARAM_TYPES(TEEC_NONE,
					 				 TEEC_NONE,
                                     TEEC_NONE,
					 				 TEEC_NONE);
	
	res = TEEC_InvokeCommand(&ctx->sess, TA_SM2ExchangeKey,
                                 &op, &origin);
	if (res != TEEC_SUCCESS)
		errx(1, "TEEC_InvokeCommand(TEST_KYEE_SM2ExchangeKey) failed 0x%x origin 0x%x",res, origin);

	return res;
}

int sm2_sign_verify_test(struct tee_ctx *ctx)
{	
	int ret;
	uint8_t id[16] = {
		0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x38, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x38
	};	
	uint8_t prikey[32];		
	uint8_t pubkey[65];		
	uint8_t	Z[32],E[32];
	uint8_t msg[10] = {0x61, 0x62, 0x63};
	uint8_t signature[64];

  
	printf("SM2 sign & verify self-check start...\n");
	ret = HOST_KYEE_SM2KeyGet(ctx, prikey, pubkey);
	if(ret || memcmp(prikey, std_prikey, 32) || memcmp(pubkey, std_pubkey, 65)) {
		print_buf((char *)"prikey", prikey, 32);
		print_buf((char *)"pubkey", pubkey, 65);
		printf("get key failure \n");
		return SM2_VERIFY_FAILED;
	}
	
	ret = HOST_KYEE_SM2GetZ(ctx, id, 16, pubkey, Z);

	if(ret || memcmp(Z, std_Z, 32)) {
		printf("get Z failure\n");
		return SM2_VERIFY_FAILED;
	}
	
	ret = HOST_KYEE_SM2GetE(ctx, msg, 3, Z, E);
	if(ret || memcmp(E, std_E, 32)) {
		printf("get E failure\n");
		return SM2_VERIFY_FAILED;
	}

	ret = HOST_KYEE_SM2Sign(ctx, E, prikey, signature);
	if(ret || memcmp(signature, std_signature, 64)) {
		printf("sign failure ret = %d\n", ret);
		return SM2_VERIFY_FAILED;
	}
	
	ret = HOST_KYEE_SM2Verify(ctx, E, pubkey, signature);
	if(ret) {
		printf("verify failure\n");
		return SM2_VERIFY_FAILED;
	}
	
	printf("SM2 sign & verify self-check sucessful\n");
    
	return SM2_SUCCESS;
}

int sm2_encrypt_decrypt_test(struct tee_ctx *ctx)
{
	uint8_t ret;
	uint8_t prikey[32] = {
		0xCF, 0x7E, 0x4C, 0xC1, 0xCD, 0xD1, 0x8E, 0x22, 0xD5, 0x41, 0xD2, 0xBC, 0x92, 0x23, 0x53, 0x2D,
		0xA0, 0xD6, 0xC9, 0xEF, 0x06, 0x99, 0x3A, 0x8D, 0xBF, 0x77, 0x6B, 0x0A, 0x8C, 0xF2, 0xB1, 0x4E
	};
	uint8_t pubkey[65] = {
		0x04, 0xD1, 0x36, 0xD0, 0x2A, 0x7A, 0xDC, 0x1E, 0x24, 0x7C, 0x1F, 0x01, 0x4A, 0x30, 0xBC, 0x2A,
		0x7F, 0x80, 0x5C, 0x05, 0x92, 0xC3, 0x4A, 0x44, 0x39, 0xE5, 0x12, 0xBF, 0xA6, 0x55, 0xD4, 0xC2,
		0x22, 0x62, 0x4C, 0x03, 0x20, 0xC1, 0xFE, 0xAF, 0x0D, 0x66, 0x92, 0x4B, 0x35, 0xF7, 0xB2, 0xC3,
		0x75, 0x1D, 0x97, 0x7F, 0xAF, 0xD0, 0x32, 0xB0, 0x5E, 0x13, 0x1E, 0x05, 0xA4, 0xFD, 0x5B, 0x93,
		0xA3
	};
	uint8_t replain[32], cipher[132];
	uint32_t bytelen = 132; // Length of ciphertext buffer
    uint32_t bytelen2 = 32; // Length of relaintext buffer

	uint8_t  plain[32] = {	
		0x30, 0x82, 0x01, 0x94, 0xa0, 0x03, 0x02, 0x01, 0x02, 0x02, 0x09, 0x00, 0xe8, 0xd2, 0x3c, 0x1f, 
		0x2f, 0xcf, 0xd6, 0xa4, 0x30, 0x0a, 0x06, 0x08, 0x2a, 0x81, 0x1c, 0xcf, 0x55, 0x01, 0x83, 0x75, 
	}; 
		
	printf("SM2 encrypt & decrypt self-check start ...\n");
	
	//C1C3C2
	memset(cipher, 0, 132);
	ret = HOST_KYEE_SM2Encrypt(ctx, plain, 32, pubkey, SM2_C1C3C2, cipher, &bytelen);
	if(ret || memcmp(cipher, std_cipher1, bytelen)) {
		printf("sm2 encrypt failure\n");
		return SM2_DECRY_VERIFY_FAILED;
	}
	
	memset(replain, 0, 32);
	ret = HOST_KYEE_SM2Decrypt(ctx, cipher, bytelen, prikey, SM2_C1C3C2, replain, &bytelen2);
	if(ret || (32 != bytelen2)|| memcmp(plain, replain, bytelen2)) {
		printf("sm2 decrypt failure\n");
		return SM2_DECRY_VERIFY_FAILED;
	}
	
	//C1C2C3
	memset(cipher, 0, 132);
	ret = HOST_KYEE_SM2Encrypt(ctx, plain, 32, pubkey, SM2_C1C2C3, cipher, &bytelen);
	if(ret || memcmp(cipher, std_cipher2, bytelen)) {
		printf("sm2 encrypt failure\n");
		return SM2_DECRY_VERIFY_FAILED;
	}

	memset(replain, 0, 32);
	ret = HOST_KYEE_SM2Decrypt(ctx, cipher, bytelen, prikey, SM2_C1C2C3, replain, &bytelen2);
	if(ret || (32 != bytelen2)|| memcmp(plain, replain, bytelen2)) {
		printf("sm2 decrypt failure\n");
		return SM2_DECRY_VERIFY_FAILED;
	}
	
	printf("SM2 encrypt & decrypt self-check sucessful\n");
	return SM2_SUCCESS;
}

