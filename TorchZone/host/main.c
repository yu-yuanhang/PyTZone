/*
 * Copyright (c) 2016, Linaro Limited
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <err.h>
#include <stdio.h>
#include <string.h>

#include <stdlib.h>

/* OP-TEE TEE client API (built by optee_client) */
#include <tee_client_api.h>

/* For the UUID (found in the TA's h-file(s)) */
#include <hello_ta.h>

#include <main.h>
#include <time.h>
#include <head.h>

clock_t start, end;
double cpu_time;

// TEEC_Context ctx;
// TEEC_Session sess;

// TEEC_SharedMemory sm;

TEEC_UUID uuid = TA_HELLO_UUID;

// ===================================================================

void prepare_tee_session(TEEC_INVITATION_T *TEEC_INVITATION) {
	printf("prepare_tee_session()\n");
	uint32_t err_origin;
	TEEC_Result res;

	/* Initialize a context connecting us to the TEE */
	res = TEEC_InitializeContext(NULL, &(TEEC_INVITATION->ctx));
	if (res != TEEC_SUCCESS)
		errx(1, "TEEC_InitializeContext failed with code 0x%x", res);

	/*
	 * Open a session to the "hello world" TA, the TA will print "hello
	 * world!" in the log when the session is created.
	 */
	res = TEEC_OpenSession(&(TEEC_INVITATION->ctx), &(TEEC_INVITATION->sess), &uuid,
			       TEEC_LOGIN_PUBLIC, NULL, NULL, &err_origin);
	if (res != TEEC_SUCCESS)
		errx(1, "TEEC_Opensession failed with code 0x%x origin 0x%x",
			res, err_origin);

	// 置空 shareMem
	memset(&(TEEC_INVITATION->sm), 0, sizeof(TEEC_SharedMemory));
}

void alloc_shareMem(TEEC_INVITATION_T *TEEC_INVITATION, uint32_t size) {
	TEEC_Result res;
	
	TEEC_INVITATION->sm.size = size;
	TEEC_INVITATION->sm.flags = TEEC_MEM_INPUT | TEEC_MEM_OUTPUT;
	res = TEEC_AllocateSharedMemory(&(TEEC_INVITATION->ctx), &(TEEC_INVITATION->sm));
	if (res != TEEC_SUCCESS) {
		errx(1, "TEEC_AllocateSharedMemory failed with code 0x%x", res);
	}
	printf("print the share memory address : %p\n", TEEC_INVITATION->sm.buffer);
	memset(TEEC_INVITATION->sm.buffer, 0, size);

	// 释放共享内存
	// TEEC_ReleaseSharedMemory(&(TEEC_INVITATION->sm));	
}

void resetSM(TEEC_INVITATION_T *TEEC_INVITATION) {
	if (!TEEC_INVITATION->sm.size) memset(TEEC_INVITATION->sm.buffer, 0, TEEC_INVITATION->sm.size);
}

void terminate_tee_session(TEEC_INVITATION_T *TEEC_INVITATION) {
	printf("terminate_tee_session()\n");
	// if (!TEEC_INVITATION->sm.size) TEEC_ReleaseSharedMemory(&(TEEC_INVITATION->sm));	
	TEEC_CloseSession(&(TEEC_INVITATION->sess));
	TEEC_FinalizeContext(&(TEEC_INVITATION->ctx));
}

// ===================================================================
#if 0
int main(void)
{

    TEEC_Operation op;
    uint32_t err_origin;
    TEEC_Result res;

	prepare_tee_session();

	memset(&op, 0, sizeof(op));

#if 0
	// ===============================================================
	// op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_PARTIAL_INOUT, TEEC_NONE,
	// 				 TEEC_NONE, TEEC_NONE);
	op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT, TEEC_VALUE_INOUT,
					 TEEC_NONE, TEEC_NONE);
	

	printf("float 长度/B : %zu\n", sizeof(float));
	float *buffer = (float *)malloc(1024 * sizeof(float));
	// float buffer[1024];
	printf("buffer 长度/B : %zu\n", sizeof(buffer));
	// memset(buffer, 0, sizeof(buffer));
	memset(buffer, 0, sizeof(float) * 1024);
	// buffer[0] = 9.0;
	// buffer[1] = 8.0;
	// buffer[15] = 5.0;
	*(buffer) = 9.0;
	*(buffer + 1) = 8.0;
	*(buffer + 15) = 5.0;

    printBytes(buffer, 64);
	
	op.params[0].tmpref.buffer = (void *)buffer;
	// op.params[0].tmpref.size = sizeof(buffer);
	op.params[0].tmpref.size = sizeof(float) * 1024;

	op.params[1].value.a = 999;
#endif
#if 1
	// ===============================================================


	printf("float 长度/B : %zu\n", sizeof(float));
	float *buffer = (float *)malloc(1024 * sizeof(float));
	// float buffer[1024];
	// printf("buffer 长度/B : %zu\n", sizeof(buffer));
	// memset(buffer, 0, sizeof(buffer));
	memset(buffer, 0, sizeof(float) * 1024);
	// buffer[0] = 9.0;
	// buffer[1] = 8.0;
	// buffer[15] = 5.0;
	*(buffer) = 9.0;
	*(buffer + 1) = 8.0;
	*(buffer + 15) = 5.0;

    printBytes(buffer, 64);
	
	// op.params[0].tmpref.buffer = (void *)buffer;
	// op.params[0].tmpref.size = sizeof(buffer);
	// op.params[1].tmpref.buffer = (void *)buffer;
	// op.params[1].tmpref.size = size * sizeof(float);

	alloc_shareMem(sizeof(float) * 1024);

	// buffer[0] = 9.0;
	// buffer[1] = 8.0;
	// buffer[15] = 5.0;
	*((float *)sm.buffer) = 9.0;
	*((float *)sm.buffer + 1) = 8.0;
	*((float *)sm.buffer + 15) = 5.0;

	printBytes(sm.buffer, 64);

	// wp.buffer = (void *)buffer;
	// printBytes(wp.buffer, 64);
	memset(&op, 0, sizeof(op));
	op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_PARTIAL_INOUT, TEEC_VALUE_INPUT,
					 TEEC_NONE, TEEC_NONE);

	op.params[0].memref.parent = &sm;
	op.params[0].memref.offset = 0;
	op.params[0].memref.size = sizeof(float) * 1024;

	op.params[1].value.a = 999;


	// =============================================================
	// // 设置TEEC_Operation参数
	// memset(&op, 0, sizeof(op));
	// op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_PARTIAL_INOUT, TEEC_VALUE_INPUT,
	// 								TEEC_NONE, TEEC_NONE);

	// op.params[0].memref.parent = &wp;
	// op.params[0].memref.offset = 0;
	// op.params[0].memref.size = sizeof(float) * 1024;
	// op.params[1].value.a = 999;

	// 调用命令
	res = TEEC_InvokeCommand(&sess, MAKE_NETWORK_CMD_DEMO, &op, &err_origin);
	printBytes(sm.buffer, 64);
	// 检查参数设置
	printf("paramTypes: 0x%x\n", op.paramTypes);
	printf("memref.size: %zu\n", op.params[0].memref.size);
	printf("memref.offset: %zu\n", op.params[0].memref.offset);
	printf("Shared memory buffer address: %p\n", sm.buffer);
	printf("Shared memory size: %zu\n", sm.size);

	if (res != TEEC_SUCCESS) {
		printf("TEEC_InvokeCommand failed with code 0x%x origin 0x%x\n", res, err_origin);
		// 检查参数设置
		printf("paramTypes: 0x%x\n", op.paramTypes);
		printf("memref.size: %zu\n", op.params[0].memref.size);
		printf("memref.offset: %zu\n", op.params[0].memref.offset);
		printf("Shared memory buffer address: %p\n", sm.buffer);
		printf("Shared memory size: %zu\n", sm.size);
		errx(1, "TEEC_InvokeCommand failed");
	}

#endif
	free(buffer);
	/*start = clock();

	res = TEEC_InvokeCommand(&sess, MAKE_NETWORK_CMD_DEMO, &op,
				 &err_origin);

	end = clock();
	
	printf("finish\n");
	if (res != TEEC_SUCCESS)
	errx(1, "TEEC_InvokeCommand failed with code 0x%x origin 0x%x",
		res, err_origin);*/
	// ===============================================================
	terminate_tee_session();

	uint32_t arr[10];
    printf("sizeof arr = %ld\n", sizeof(arr));
	
	
	return 0;
}

#endif // main
