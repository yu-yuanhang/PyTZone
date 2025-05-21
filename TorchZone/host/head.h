#ifndef __HEAD_H__
#define __HEAD_H__

#include <err.h>
#include <stdio.h>
#include <string.h>

#include <stdlib.h>

/* OP-TEE TEE client API (built by optee_client) */
#include <tee_client_api.h>

// ===================================================
#include <common_ta.h>
#include <hello_ta.h>
#include <main.h>
#include <pack.h>
#include <ptz_defs.h>

// ===================================================



// extern TEEC_Context ctx;
// extern TEEC_Session sess;
// extern TEEC_SharedMemory sm;

typedef struct TEEC_INVITATION_S {
    TEEC_Context ctx;
    TEEC_Session sess;
    TEEC_SharedMemory sm;
} TEEC_INVITATION_T;

extern TEEC_UUID uuid;

void prepare_tee_session(TEEC_INVITATION_T *TEEC_INVITATION);
void terminate_tee_session(TEEC_INVITATION_T *TEEC_INVITATION);

void alloc_shareMem(TEEC_INVITATION_T *TEEC_INVITATION, uint32_t size);
void resetSM(TEEC_INVITATION_T *TEEC_INVITATION);

#endif