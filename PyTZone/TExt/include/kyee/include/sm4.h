#ifndef __SM4_H__
#define __SM4_H__

#include <stdint.h>
#include "kyee.h"


//SM4 return code
enum SM4_RET_CODE
{
	SM4_SUCCESS = 0,
	SM4_BUFFER_NULL,
	SM4_CONFIG_INVALID,
	SM4_INPUT_INVALID,
	SM4_ERR_FAILURE
};

//SM4 Operation Mode
typedef enum 
{
    SM4_MODE_ECB                  = 0,   // ECB Mode
    SM4_MODE_CBC                     ,   // CBC Mode
    SM4_MODE_CFB                     ,   // CFB Mode
    SM4_MODE_OFB                     ,   // OFB Mode
    SM4_MODE_CTR                         // CTR Mode
} sm4_mode_e;


//SM4 Crypto Action
typedef enum {
    SM4_CRYPTO_ENCRYPT       = 0,   // encrypt
    SM4_CRYPTO_DECRYPT          ,   // decrypt
} sm4_crypto_e;


int sm4_encrypt_decrypt(struct tee_ctx *ctx, sm4_mode_e mode, uint8_t wordAlign, uint8_t *std_plain,
              uint32_t byteLen, uint8_t *key, uint8_t *iv, const uint8_t *std_cipher);

#endif // __SM4_H__


