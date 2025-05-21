#ifndef __SM2_H__
#define __SM2_H__

#include <stdint.h>

// SM2 ciphertext order
typedef enum {
	SM2_C1C3C2   = 0,
    SM2_C1C2C3,
} sm2_cipher_order_e;

//SM2 error code
enum SM2_RET_CODE
{
	SM2_SUCCESS = 0,
	SM2_BUFFER_NULL = 0x50,
	SM2_NOT_ON_CURVE,
	SM2_EXCHANGE_ROLE_INVALID,
	SM2_INPUT_INVALID,
	SM2_ZERO_ALL,
	SM2_INTEGER_TOO_BIG,
	SM2_VERIFY_FAILED,
	SM2_IN_OUT_SAME_BUFFER,
	SM2_DECRY_VERIFY_FAILED
};

int sm2_sign_verify_test(struct tee_ctx *ctx);
int sm2_encrypt_decrypt_test(struct tee_ctx *ctx);
int sm2_sign_speed_test(struct tee_ctx *ctx);
#endif

