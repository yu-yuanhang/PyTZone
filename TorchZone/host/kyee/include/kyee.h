#ifndef __KYEE_H__
#define __KYEE_H__

#include <err.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
// #include <sys/types.h>
// #include <tee_client_api.h>

#include <head.h>

// 建立session通话的结构体
struct tee_ctx {
	TEEC_Context ctx;
	TEEC_Session sess;
};

// void prepare_tee_session(struct tee_ctx *ctx);
// void terminate_tee_session(struct tee_ctx *ctx);

void print_buf(char *name, uint8_t *buf, uint32_t len);

#endif

