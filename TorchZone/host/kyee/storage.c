#include <unistd.h>
#include <storage.h>

static TEEC_Result HOST_KYEE_storage_write(struct tee_ctx *ctx, 
								int offset,  
								char *data,
								size_t data_len)
{
	TEEC_Operation op;
	uint32_t origin;
	TEEC_Result res;

	memset(&op, 0, sizeof(op));
	op.paramTypes = TEEC_PARAM_TYPES(TEEC_VALUE_INPUT, 
									TEEC_MEMREF_TEMP_INPUT,
							 		TEEC_NONE, TEEC_NONE);
	op.params[0].value.a = offset;   // offset

	op.params[1].tmpref.buffer = data;
	op.params[1].tmpref.size = data_len;

	res = TEEC_InvokeCommand(&ctx->sess, TA_STORAGE_WRITE, &op, &origin);
	if (res != TEEC_SUCCESS) {
		printf("write test failure: 0x%x / %u\n", res, origin);
	}

	return res;
}

static TEEC_Result HOST_KYEE_storage_read(struct tee_ctx *ctx, 
								int offset,  
								char *data,
								size_t data_len)
{
	TEEC_Operation op;
	uint32_t origin;
	TEEC_Result res;
	uint32_t reserve = 0;

	memset(&op, 0, sizeof(op));
	op.paramTypes = TEEC_PARAM_TYPES(TEEC_VALUE_INPUT, 
									TEEC_MEMREF_TEMP_OUTPUT,
							 		TEEC_NONE, TEEC_NONE);
	op.params[0].value.a = offset;   // offset

	op.params[1].tmpref.buffer = data;
	op.params[1].tmpref.size = data_len;

	res = TEEC_InvokeCommand(&ctx->sess, TA_STORAGE_READ, &op, &origin);
	switch (res) {
		case TEEC_SUCCESS:
		case TEEC_ERROR_SHORT_BUFFER:
		//case TEEC_ERROR_ITEM_NOT_FOUND:
			break;
        
		default:
			printf("read test failure: 0x%x / %u\n", res, origin);
            break;
	}

	return res;
}

TEEC_Result HOST_KYEE_storage_delete(struct tee_ctx *ctx, 
                                int offset, int size)
{
    TEEC_Result res = TEEC_SUCCESS;
	TEEC_Operation op;
	uint32_t origin;
    
	memset(&op, 0, sizeof(op));
	op.paramTypes = TEEC_PARAM_TYPES(TEEC_VALUE_INPUT, TEEC_NONE,
							 		TEEC_NONE, TEEC_NONE);
	op.params[0].value.a = offset;   // offset
	op.params[0].value.b = size;
	
	return TEEC_InvokeCommand(&ctx->sess, TA_STORAGE_DELETE, &op, &origin);
}                        

void storage_functest(struct tee_ctx *ctx)
{
    TEEC_Result res;
	char data[] = "1qaz2wsx3edc4rfv";
    int datalen = strlen(data);
    char r_data[512] = {0};
    int r_datalen = 512;
    uint32_t offset = 0x400000;
    
    res = HOST_KYEE_storage_write(ctx, offset, data, datalen);
    if (res != TEEC_SUCCESS)
    {
        printf("write failure\n");
        goto out;
    }
    else
    {
        printf("write success, data=(%s)\n", data);
    }
        
    HOST_KYEE_storage_read(ctx, offset, r_data, datalen);
    if (memcmp(r_data, data, datalen) == 0)
    {
		data[datalen] = '\0';
        printf("read success, data=(%s)\n", r_data);
    }
    else
    {
        printf("read failure, orgindata=(%s)\n", data);
		print_buf("read data:", r_data, datalen);
        goto out;
    }

    HOST_KYEE_storage_delete(ctx, offset, 1024*64);

out:
    return;
}
