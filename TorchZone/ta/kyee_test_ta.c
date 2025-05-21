#include "kyee_test_ta.h"

#if KYEE

//#include <tee_sm_internal.h>

//#define CFG_PHYTIUM2004_SMX

//调用SM2密钥对生成接口（KYEE_SM2KeyGet）
TEE_Result TA_KYEE_SM2KeyGet(uint32_t param_types,TEE_Param params[4])
{
	const uint32_t exp_param_types =
	TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_OUTPUT,
					TEE_PARAM_TYPE_MEMREF_OUTPUT,
					TEE_PARAM_TYPE_NONE,
					TEE_PARAM_TYPE_NONE);
	uint8_t *priKey;
	uint8_t *pubKey;


	if (param_types != exp_param_types)
		return TEE_ERROR_BAD_PARAMETERS;

	priKey = params[0].memref.buffer;
	pubKey = params[1].memref.buffer;

	return KYEE_SM2KeyGet(priKey, pubKey);
}

//调用获取Z值接口（KYEE_SM2GetZ）
TEE_Result TA_KYEE_SM2GetZ(uint32_t param_types,TEE_Param params[4])
{
	const uint32_t exp_param_types =
		TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
						TEE_PARAM_TYPE_MEMREF_INPUT,
						TEE_PARAM_TYPE_MEMREF_OUTPUT,
						TEE_PARAM_TYPE_NONE);
	uint8_t *id;
	uint32_t id_len;
	uint8_t *z;
	uint8_t *pubKey;


	if (param_types != exp_param_types)
		return TEE_ERROR_BAD_PARAMETERS;

	id = params[0].memref.buffer;
	id_len = params[0].memref.size;
	pubKey = params[1].memref.buffer;
	z = params[2].memref.buffer;

	return KYEE_SM2GetZ(id, id_len, pubKey, z);
}


//调用获取E值接口（KYEE_SM2GetE）
TEE_Result TA_KYEE_SM2GetE(uint32_t param_types,TEE_Param params[4])
{
	const uint32_t exp_param_types =
		TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
						TEE_PARAM_TYPE_MEMREF_INPUT,
						TEE_PARAM_TYPE_MEMREF_OUTPUT,
						TEE_PARAM_TYPE_NONE);
	uint8_t *m;
	uint32_t m_len;
	uint8_t *z;
	uint8_t *e;


	if (param_types != exp_param_types)
		return TEE_ERROR_BAD_PARAMETERS;

	m = params[0].memref.buffer;
	m_len = params[0].memref.size;
	z = params[1].memref.buffer;
	e = params[2].memref.buffer;

	return KYEE_SM2GetE(m, m_len, z, e);
}

//调用SM2签名接口（KYEE_SM2Sign）
TEE_Result TA_KYEE_SM2Sign(uint32_t param_types,TEE_Param params[4])
{
	const uint32_t exp_param_types =
	TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
					TEE_PARAM_TYPE_MEMREF_INPUT,
					TEE_PARAM_TYPE_MEMREF_OUTPUT,
					TEE_PARAM_TYPE_NONE);
	uint8_t *e;
	uint8_t *priKey;
	uint8_t *signature;


	if (param_types != exp_param_types)
		return TEE_ERROR_BAD_PARAMETERS;

	e = params[0].memref.buffer;
	priKey = params[1].memref.buffer;
	signature = params[2].memref.buffer;

	return KYEE_SM2Sign(e, priKey, signature);
}

//调用SM2签名验证接口（KYEE_SM2Verify）
TEE_Result TA_KYEE_SM2Verify(uint32_t param_types,TEE_Param params[4])
{
	const uint32_t exp_param_types =
	TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
					TEE_PARAM_TYPE_MEMREF_INPUT,
					TEE_PARAM_TYPE_MEMREF_INPUT,
					TEE_PARAM_TYPE_NONE);
	uint8_t *e;
	uint8_t *pubKey;
	uint8_t *signature;


	if (param_types != exp_param_types)
		return TEE_ERROR_BAD_PARAMETERS;

	e = params[0].memref.buffer;
	pubKey = params[1].memref.buffer;
	signature = params[2].memref.buffer;

	return KYEE_SM2Verify(e, pubKey, signature);
}

//调用SM2消息加密接口（KYEE_SM2Encrypt）
TEE_Result TA_KYEE_SM2Encrypt(uint32_t param_types,TEE_Param params[4])
{
	const uint32_t exp_param_types =
	TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
					TEE_PARAM_TYPE_MEMREF_INPUT,
					TEE_PARAM_TYPE_MEMREF_OUTPUT,
					TEE_PARAM_TYPE_VALUE_INOUT);
	uint8_t *m;
	uint32_t m_len;
	uint8_t *pubKey;
	uint8_t *c;
	uint32_t *c_len;
    int order;


	if (param_types != exp_param_types)
		return TEE_ERROR_BAD_PARAMETERS;

	m = params[0].memref.buffer;
	m_len = params[0].memref.size;
	pubKey = params[1].memref.buffer;
	c = params[2].memref.buffer;
	c_len = &params[3].value.a;
    order = params[3].value.b;

	return KYEE_SM2Encrypt(m, m_len, pubKey, order, c, c_len);
}


//调用SM2消息解密接口（KYEE_SM2Decrypt）
TEE_Result TA_KYEE_SM2Decrypt(uint32_t param_types,TEE_Param params[4])
{
	const uint32_t exp_param_types =
	TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
					TEE_PARAM_TYPE_MEMREF_INPUT,
					TEE_PARAM_TYPE_MEMREF_OUTPUT,
					TEE_PARAM_TYPE_VALUE_INOUT);
	uint8_t *c;
	uint32_t c_len;
	uint8_t *priKey;
	uint8_t *m;
	uint32_t *m_len;
    int order;

	if (param_types != exp_param_types)
		return TEE_ERROR_BAD_PARAMETERS;

	c = params[0].memref.buffer;
	c_len = params[0].memref.size;
	priKey = params[1].memref.buffer;
	m = params[2].memref.buffer;
	m_len = &params[3].value.a;
    order = params[3].value.b;

	return KYEE_SM2Decrypt(c, c_len, priKey, order, m, m_len);
}

//调用SM3算法初始化接口（KYEE_SM3Init）
TEE_Result TA_KYEE_SM3Init(uint32_t param_types,TEE_Param params[4])
{
    const uint32_t exp_param_types =
        TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INOUT,
                        TEE_PARAM_TYPE_NONE,
                        TEE_PARAM_TYPE_NONE,
                        TEE_PARAM_TYPE_NONE);
    TEE_sm3Handle_t *context;

    /*
     * Safely get the invocation parameters
     */
    if (param_types != exp_param_types)
        return TEE_ERROR_BAD_PARAMETERS;

    context = params[0].memref.buffer ;

    return KYEE_SM3Init(context);
}


//调用SM3算法消息输入处理接口（KYEE_SM3Process）
TEE_Result TA_KYEE_SM3Process(uint32_t param_types,TEE_Param params[4])
{
    const uint32_t exp_param_types =
        TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INOUT,
                        TEE_PARAM_TYPE_MEMREF_INPUT,
                        TEE_PARAM_TYPE_NONE,
                        TEE_PARAM_TYPE_NONE);
    TEE_sm3Handle_t *context;
    uint8_t *input;
    uint32_t input_len;

    /*
     * Safely get the invocation parameters
     */
    if (param_types != exp_param_types)
        return TEE_ERROR_BAD_PARAMETERS;

    context = params[0].memref.buffer ;
    input = params[1].memref.buffer ;
    input_len = params[1].memref.size ;

    return KYEE_SM3Process(context, input, input_len);    
}


//调用SM3算法获取消息摘要接口（KYEE_SM3Done）
TEE_Result TA_KYEE_SM3Done(uint32_t param_types,TEE_Param params[4])
{
    const uint32_t exp_param_types =
        TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INOUT,
                        TEE_PARAM_TYPE_MEMREF_OUTPUT,
                        TEE_PARAM_TYPE_NONE,
                        TEE_PARAM_TYPE_NONE);
    TEE_sm3Handle_t *context;
    uint8_t *digest;


    /*
     * Safely get the invocation parameters
     */
    if (param_types != exp_param_types)
        return TEE_ERROR_BAD_PARAMETERS;

    context = params[0].memref.buffer ;
    digest = params[1].memref.buffer ;

    return KYEE_SM3Done(context, digest);
}

//调用SM4算法初始化接口（KYEE_SM4Init），加解密前必须调用该API
TEE_Result TA_KYEE_SM4Init(uint32_t param_types,TEE_Param params[4])
{
	const uint32_t exp_param_types =
		TEE_PARAM_TYPES(TEE_PARAM_TYPE_VALUE_INPUT,
						TEE_PARAM_TYPE_MEMREF_INPUT,
						TEE_PARAM_TYPE_MEMREF_INPUT,
						TEE_PARAM_TYPE_NONE);
	sm4_mode_e mode;
	sm4_crypto_e crypto; 
	uint8_t *key;
	uint8_t *iv;
	/*
	 * Safely get the invocation parameters
	 */
	if (param_types != exp_param_types)
		return TEE_ERROR_BAD_PARAMETERS;

	mode = params[0].value.a ;
	crypto = params[0].value.b ;
	key = params[1].memref.buffer ;
	iv = params[2].memref.buffer ;


	return KYEE_SM4Init(mode, crypto, key, iv);
}


//调用SM4加解密接口（KYEE_SM4Crypto）
TEE_Result TA_KYEE_SM4Crypto(uint32_t param_types,TEE_Param params[4])
{
	const uint32_t exp_param_types =
		TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
				TEE_PARAM_TYPE_MEMREF_OUTPUT,
				TEE_PARAM_TYPE_NONE,
				TEE_PARAM_TYPE_NONE);
	uint8_t *in;
	uint32_t in_len;
	uint8_t *out;

	/*
	 * Safely get the invocation parameters
	 */
	if (param_types != exp_param_types)
		return TEE_ERROR_BAD_PARAMETERS;


	in = params[0].memref.buffer ;
	in_len = params[0].memref.size;
	out = params[1].memref.buffer ;

	return KYEE_SM4Crypto(in,out,in_len);
}

TEE_Result TA_KYEE_storage_write(uint32_t param_types, TEE_Param params[4])
{
	TEE_Result res;
	const uint32_t exp_param_types = TEE_PARAM_TYPES(
								TEE_PARAM_TYPE_VALUE_INPUT,
								TEE_PARAM_TYPE_MEMREF_INPUT,
								TEE_PARAM_TYPE_NONE,
								TEE_PARAM_TYPE_NONE);
	if (param_types != exp_param_types)
		return TEE_ERROR_BAD_PARAMETERS;

	res = KYEE_SecureStorage_Write(0,
						params[0].value.a,
						params[1].memref.buffer,
						params[1].memref.size);

	return res;
}

TEE_Result TA_KYEE_storage_read(uint32_t param_types, TEE_Param params[4])
{
	TEE_Result res;
	const uint32_t exp_param_types = TEE_PARAM_TYPES(
								TEE_PARAM_TYPE_VALUE_INPUT,
								TEE_PARAM_TYPE_MEMREF_OUTPUT,
								TEE_PARAM_TYPE_NONE,
								TEE_PARAM_TYPE_NONE);
	if (param_types != exp_param_types)
		return TEE_ERROR_BAD_PARAMETERS;

	res = KYEE_SecureStorage_Read(0,
							params[0].value.a,
							params[1].memref.buffer,
							params[1].memref.size);

	return res;
}

TEE_Result TA_KYEE_storage_delete(uint32_t param_types, TEE_Param params[4])
{
	TEE_Result res;
	const uint32_t exp_param_types = TEE_PARAM_TYPES(
								TEE_PARAM_TYPE_VALUE_INPUT,
								TEE_PARAM_TYPE_NONE,
								TEE_PARAM_TYPE_NONE,
								TEE_PARAM_TYPE_NONE);
	if (param_types != exp_param_types)
		return TEE_ERROR_BAD_PARAMETERS;
    
	return KYEE_SecureStorage_Delete(params[0].value.a, params[0].value.b);
}

#endif // KYEE