/* To the the UUID (found the the TA's h-file(s)) */
//#include <kyee_test_ta.h>

// #include <stdlib.h>
// #include <sys/time.h>
// #include <unistd.h>
#include "kyee.h"
// #include "sm2.h"
// #include "sm3.h"
// #include "sm4.h"
// #include "storage.h"

// // sm4 standard values
// extern uint8_t std_in[48];
// extern uint8_t std_key[16];  
// extern uint8_t std_iv[16];
// extern uint8_t std_ecb_out[48];
// extern uint8_t std_cbc_out[48];
// extern uint8_t std_cfb_out[48];
// extern uint8_t std_ofb_out[48];
// extern uint8_t std_ctr_out[48];

void print_buf(char *name, uint8_t *buf, uint32_t len)
{
        int i = 0;
        printf("--------%s--------\n", name);
        for (i=0; i<len; i++)
        {
                printf("%x ", buf[i]);
                if ((i+1) % 16 == 0)
                        printf("\n");
        }
        printf("\n");
}

// void sm2_test(struct tee_ctx *ctx)
// {
//     sm2_sign_verify_test(ctx);

//     sm2_encrypt_decrypt_test(ctx);
        
//     return;
// }

// void sm3_test(struct tee_ctx *ctx)
// {
//     sm3_functest(ctx);
    
//     return;
// }

// void sm4_test(struct tee_ctx *ctx)
// {
//     sm4_encrypt_decrypt(ctx, SM4_MODE_ECB, 1, std_in, 48, std_key, NULL, std_ecb_out);
//     sm4_encrypt_decrypt(ctx, SM4_MODE_CBC, 1, std_in, 48, std_key, std_iv, std_cbc_out);
//     sm4_encrypt_decrypt(ctx, SM4_MODE_CFB, 1, std_in, 48, std_key, std_iv, std_cfb_out);
//     sm4_encrypt_decrypt(ctx, SM4_MODE_OFB, 1, std_in, 48, std_key, std_iv, std_ofb_out);
//     sm4_encrypt_decrypt(ctx, SM4_MODE_CTR, 1, std_in, 48, std_key, std_iv, std_ctr_out);

//     return;
// }

// void storage_test(struct tee_ctx *ctx)
// {
//     return storage_functest(ctx);
// }

// int main(int argc, char *argv[])
// {
//     struct tee_ctx ctx;

//     prepare_tee_session(&ctx);

//     sm2_test(&ctx);

//     sm3_test(&ctx);

//     storage_test(&ctx);

//     terminate_tee_session(&ctx);


//     return 0;
// }

