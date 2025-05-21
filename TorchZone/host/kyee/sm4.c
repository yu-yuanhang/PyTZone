
#include <sm4.h>

uint8_t std_in[48] = {
    0x81, 0x70, 0x99, 0x44, 0xE0, 0xCB, 0x2E, 0x1D, 0xB5, 0xB0, 0xA4, 0x77, 0xD1, 0xA8, 0x53, 0x9B,
    0x0A, 0x87, 0x86, 0xE3, 0x4E, 0xAA, 0xED, 0x99, 0x30, 0x3E, 0xA6, 0x97, 0x55, 0x95, 0xB2, 0x45,
    0x4D, 0x5D, 0x7F, 0x91, 0xEB, 0xBD, 0x4A, 0xCD, 0x72, 0x6C, 0x0E, 0x0E, 0x5E, 0x3E, 0xB5, 0x5E,
};
    
uint8_t std_key[16] = {
    0xE0, 0x70, 0x99, 0xF1, 0xBF, 0xAF, 0xFD, 0x7F, 0x24, 0x0C, 0xD7, 0x90, 0xCA, 0x4F, 0xE1, 0x34
};  
    
uint8_t std_iv[16] = {
    0xC7, 0x2B, 0x65, 0x91, 0xA0, 0xD7, 0xDE, 0x8F, 0x6B, 0x40, 0x72, 0x33, 0xAD, 0x35, 0x81, 0xD6
};

uint8_t std_ecb_out[48] = {
	0xCC, 0x62, 0x37, 0xA6, 0xA1, 0x35, 0x39, 0x75, 0xFF, 0xF5, 0xEE, 0x6A, 0xFD, 0xD7, 0x70, 0x15,
	0xE1, 0x32, 0x23, 0x1F, 0x18, 0xB8, 0xC9, 0x16, 0x07, 0x27, 0x9C, 0x6C, 0x7F, 0x8F, 0x7F, 0xF6,
	0xFD, 0xF1, 0xE4, 0x01, 0xEC, 0x7E, 0xD2, 0x60, 0xFD, 0xE7, 0x5C, 0xE5, 0xCF, 0x6E, 0xE7, 0x87,
	};
	
uint8_t std_cbc_out[48] = {
	0x60, 0x7A, 0xBE, 0xC9, 0xDA, 0xD7, 0x90, 0x73, 0xC7, 0x96, 0xDB, 0x34, 0x26, 0xFD, 0x2C, 0x2F,
	0x8E, 0x39, 0xC7, 0x0B, 0x60, 0xB2, 0x3D, 0xBE, 0xF3, 0xA9, 0xA5, 0x46, 0x65, 0x26, 0x41, 0xB7,
	0xAE, 0xC9, 0xC3, 0xAD, 0x8C, 0x9B, 0x95, 0x8D, 0x17, 0x53, 0x15, 0x35, 0x40, 0x2A, 0x8C, 0x6B,
	};
	
uint8_t std_cfb_out[48] = {
	0xC1, 0x27, 0x47, 0xC7, 0x44, 0x0C, 0x9A, 0x5C, 0x7D, 0x51, 0x26, 0x0D, 0x1B, 0xDB, 0x0D, 0x9D,
	0x52, 0x59, 0xAD, 0x56, 0x05, 0xBE, 0x92, 0xD2, 0xB7, 0x62, 0xF5, 0xD7, 0x53, 0xD3, 0x12, 0x2A,
	0x3C, 0x9A, 0x6E, 0x75, 0x80, 0xAB, 0x18, 0xE5, 0x72, 0x49, 0x9A, 0xD9, 0x80, 0x99, 0xC2, 0xE7,
	};
	
uint8_t std_ofb_out[48] = {
	0xC1, 0x27, 0x47, 0xC7, 0x44, 0x0C, 0x9A, 0x5C, 0x7D, 0x51, 0x26, 0x0D, 0x1B, 0xDB, 0x0D, 0x9D,
	0x0F, 0x0C, 0xAD, 0xA0, 0x2D, 0x18, 0x0B, 0x3C, 0x54, 0xA9, 0x87, 0x86, 0xBC, 0x6B, 0xF9, 0xFB,
	0x18, 0x68, 0x51, 0x1E, 0xB2, 0x53, 0x1D, 0xD5, 0x7F, 0x4B, 0xED, 0xB8, 0xCA, 0x8E, 0x81, 0xCE,
	};
	
uint8_t std_ctr_out[48] = {
	0xC1, 0x27, 0x47, 0xC7, 0x44, 0x0C, 0x9A, 0x5C, 0x7D, 0x51, 0x26, 0x0D, 0x1B, 0xDB, 0x0D, 0x9D,
	0xC3, 0x75, 0xCE, 0xBB, 0x63, 0x9A, 0x5B, 0x0C, 0xED, 0x64, 0x3F, 0x33, 0x80, 0x8F, 0x97, 0x40,
	0xB7, 0x5C, 0xA7, 0xFE, 0x2F, 0x7F, 0xFB, 0x20, 0x13, 0xEC, 0xDC, 0xBC, 0x96, 0xC8, 0x05, 0xF0,
	};

/*
功能：       
    SM4算法初始化，加解密前必须调用该API。
输入：       
    sm4_mode_e mode         工作模式
    sm4_crypto_e crypto     加密或者解密
    uint8_t *key            密钥，字节big-endian
    uint8_t *iv             IV或counter，字节big-endian
输出：       
    无
返回值：        
    TEE_SUCCESS：成功；其他值失败
*/
static TEEC_Result HOST_KYEE_SM4Init(struct tee_ctx *ctx,sm4_mode_e mode, sm4_crypto_e crypto, 
                            uint8_t *key, uint8_t *iv)
{
    TEEC_Operation op;
    uint32_t origin;
    TEEC_Result res;

    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(TEEC_VALUE_INPUT,
                                     TEEC_MEMREF_TEMP_INPUT,
                                     TEEC_MEMREF_TEMP_INPUT,
                                     TEEC_NONE);

    op.params[0].value.a = mode;
    op.params[0].value.b = crypto;
    op.params[1].tmpref.buffer = key; 
    op.params[1].tmpref.size = 16; 
    op.params[2].tmpref.buffer = iv; 
    op.params[2].tmpref.size = 16; 


    res = TEEC_InvokeCommand(&ctx->sess, TA_SM4Init,
                                 &op, &origin);
    if (res != TEEC_SUCCESS)
        errx(1, "TEEC_InvokeCommand(TEST_KYEE_SM4Init) failed 0x%x origin 0x%x",res, origin);

    //test_soc_rate();
    return res;
}


/*
功能：       
    SM4加解密。
输入：       
    uint8_t *in                 待加密的明文，或者待解密的密文，字节big-endian。
    uint32_t in_len         输入in的字节长度，必须是SM4算法分组长度(16)的正整数倍。
输出：   
    uint8_t *out                加密得到的密文，或者解密得到的明文，字节big-endian。
返回值：        
    TEE_SUCCESS：成功；其他值失败
*/
static TEEC_Result HOST_KYEE_SM4Crypto(struct tee_ctx *ctx,uint8_t *in, uint8_t *out,
                                            uint32_t in_len, int out_len)
{
    TEEC_Operation op;
    uint32_t origin;
    TEEC_Result res;                            
    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT,
                                     TEEC_MEMREF_TEMP_OUTPUT,
                                     TEEC_NONE,
                                     TEEC_NONE);

    op.params[0].tmpref.buffer = in; 
    op.params[0].tmpref.size = in_len;
    op.params[1].tmpref.buffer = out; 
    op.params[1].tmpref.size = out_len; 


    res = TEEC_InvokeCommand(&ctx->sess, TA_SM4Crypto,
                                 &op, &origin);
    if (res != TEEC_SUCCESS)
        errx(1, "TEEC_InvokeCommand(TEST_KYEE_SM4Crypto) failed 0x%x origin 0x%x",res, origin);                         
    return res;
}

char *mode_str[5] = {"ecb", "cbc", "cfb", "ofb", "ctr"};
int sm4_encrypt_decrypt(struct tee_ctx *ctx, sm4_mode_e mode, uint8_t wordAlign, uint8_t *std_plain,
              uint32_t byteLen, uint8_t *key, uint8_t *iv, const uint8_t *std_cipher)
{
    uint8_t key_buf[16];
    uint8_t iv_buf[16];
    uint8_t std_plain_buf[132];
    uint8_t std_cipher_buf[132];
    uint8_t cipher_buf[132];
    uint8_t replain_buf[132];
    uint8_t *cipher_, *replain_, *std_plain_, *std_cipher_, *key_, *iv_;
    //char *oper_mode[]={"ECB", "CBC", "CFB", "OFB", "CTR"};
    uint32_t block_byteLen, key_byteLen;

    block_byteLen = 16;
    key_byteLen = 16;

    if(wordAlign) {
        memcpy(std_plain_buf, std_plain, byteLen);
        memcpy(std_cipher_buf, std_cipher, byteLen);
        memcpy(key_buf, key, key_byteLen);
        if(SM4_MODE_ECB != mode)
            memcpy(iv_buf, iv, block_byteLen);
        
        cipher_     = cipher_buf;
        replain_    = replain_buf;
        std_plain_  = std_plain_buf;
        std_cipher_ = std_cipher_buf;
        key_        = key_buf;
        iv_         = iv_buf;
    }  else {
        memcpy(std_plain_buf + 1, std_plain, byteLen);
        memcpy(std_cipher_buf + 1, std_cipher, byteLen);
        memcpy(key_buf + 1, key, key_byteLen);
        if(SM4_MODE_ECB != mode)
            memcpy(iv_buf + 1, iv, block_byteLen);
        
        cipher_     = cipher_buf + 1;
        replain_    = replain_buf + 1;
        std_plain_  = std_plain_buf + 1;
        std_cipher_ = std_cipher_buf + 1;
        key_        = key_buf + 1;
        iv_         = iv_buf + 1;
    }
    
    //ENCRYPT
	HOST_KYEE_SM4Init(ctx, mode, SM4_CRYPTO_ENCRYPT, key_, iv_);
    HOST_KYEE_SM4Crypto(ctx, std_plain_, cipher_, byteLen, byteLen);
    
    //DECRYPT
    HOST_KYEE_SM4Init(ctx, mode, SM4_CRYPTO_DECRYPT, key_, iv_);
    HOST_KYEE_SM4Crypto(ctx, cipher_, replain_, byteLen, byteLen);

    if(memcmp(cipher_, std_cipher_, byteLen) || memcmp(replain_, std_plain_, byteLen)) {
        printf("SM4 cpu %s self-check failure!! \n", mode_str[mode]);
        return SM4_ERR_FAILURE;
    } else { 
        printf("SM4 cpu %s self-check success!! \n", mode_str[mode]);
    }

    return SM4_SUCCESS;
}

