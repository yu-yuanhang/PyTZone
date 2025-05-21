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

#include <tee_internal_api.h>
#include <tee_internal_api_extensions.h>

#include <hello_ta.h>
#include <main.h>
#include <torchzone.h>


#include "kyee_test_ta.h"

#include <tee_api.h>
/*
 * Called when the instance of the TA is created. This is the first call in
 * the TA.
 */
TEE_Result TA_CreateEntryPoint(void)
{
	DMSG("has been called");

	return TEE_SUCCESS;
}

/*
 * Called when the instance of the TA is destroyed if the TA has not
 * crashed or panicked. This is the last call in the TA.
 */
void TA_DestroyEntryPoint(void)
{
	DMSG("has been called");
}

/*
 * Called when a new session is opened to the TA. *sess_ctx can be updated
 * with a value to be able to identify this session in subsequent calls to the
 * TA. In this function you will normally do the global initialization for the
 * TA.
 */
TEE_Result TA_OpenSessionEntryPoint(uint32_t param_types,
		TEE_Param __maybe_unused params[4],
		void __maybe_unused **sess_ctx)
{
	uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_NONE,
						   TEE_PARAM_TYPE_NONE,
						   TEE_PARAM_TYPE_NONE,
						   TEE_PARAM_TYPE_NONE);

	DMSG("has been called");

	if (param_types != exp_param_types)
		return TEE_ERROR_BAD_PARAMETERS;

	/* Unused parameters */
	(void)&params;
	(void)&sess_ctx;

	/*
	 * The DMSG() macro is non-standard, TEE Internal API doesn't
	 * specify any means to logging from a TA.
	 */
	IMSG("Hello World!\n");

	/* If return value != TEE_SUCCESS the session will not be created. */
	return TEE_SUCCESS;
}

/*
 * Called when a session is closed, sess_ctx hold the value that was
 * assigned by TA_OpenSessionEntryPoint().
 */
void TA_CloseSessionEntryPoint(void __maybe_unused *sess_ctx)
{
	(void)&sess_ctx; /* Unused parameter */
	IMSG("Goodbye!\n");
}

static TEE_Result inc_value(uint32_t param_types,
	TEE_Param params[4])
{
	uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_VALUE_INOUT,
						   TEE_PARAM_TYPE_NONE,
						   TEE_PARAM_TYPE_NONE,
						   TEE_PARAM_TYPE_NONE);

	DMSG("inc_value has been called");

	if (param_types != exp_param_types)
		return TEE_ERROR_BAD_PARAMETERS;

	IMSG("Got value: %u from NW", params[0].value.a);
	params[0].value.a++;
	IMSG("Increase value to: %u", params[0].value.a);

	return TEE_SUCCESS;
}

static TEE_Result dec_value(uint32_t param_types,
	TEE_Param params[4])
{
	uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_VALUE_INOUT,
						   TEE_PARAM_TYPE_NONE,
						   TEE_PARAM_TYPE_NONE,
						   TEE_PARAM_TYPE_NONE);

	DMSG("dec_value has been called");

	if (param_types != exp_param_types)
		return TEE_ERROR_BAD_PARAMETERS;

	IMSG("Got value: %u from NW", params[0].value.a);
	params[0].value.a--;
	IMSG("Decrease value to: %u", params[0].value.a);

	return TEE_SUCCESS;
}
/*
 * Called when a TA is invoked. sess_ctx hold that value that was
 * assigned by TA_OpenSessionEntryPoint(). The rest of the paramters
 * comes from normal world.
 */
TEE_Result TA_InvokeCommandEntryPoint(void __maybe_unused *sess_ctx,
			uint32_t cmd_id,
			uint32_t param_types, TEE_Param params[4])
{
	(void)&sess_ctx; /* Unused parameter */

	switch (cmd_id) {
	// case TA_HELLO_CMD_INC_VALUE:
	// 	return inc_value(param_types, params);
	// case TA_HELLO_CMD_DEC_VALUE:
	// 	return dec_value(param_types, params);
	// ====================================================================
#if KYEE
	case TA_SM3Init:
		return TA_KYEE_SM3Init(param_types, params);
	case TA_SM3Process:
		return TA_KYEE_SM3Process(param_types, params);
    case TA_SM3Done:
        return TA_KYEE_SM3Done(param_types, params);
	// --------------------------- 
    case TA_SM4Init:
        return TA_KYEE_SM4Init(param_types, params);
    case TA_SM4Crypto:
        return TA_KYEE_SM4Crypto(param_types, params);
	// ---------------------------
    case TA_SM2KeyGet:
        return TA_KYEE_SM2KeyGet(param_types, params);
    case TA_SM2GetZ:
        return TA_KYEE_SM2GetZ(param_types, params);
    case TA_SM2GetE:
        return TA_KYEE_SM2GetE(param_types, params);
    case TA_SM2Sign:
        return TA_KYEE_SM2Sign(param_types, params);
    case TA_SM2Verify:
        return TA_KYEE_SM2Verify(param_types, params);
    case TA_SM2Encrypt:
        return TA_KYEE_SM2Encrypt(param_types, params);
    case TA_SM2Decrypt:
        return TA_KYEE_SM2Decrypt(param_types, params);
	// ---------------------------
    case TA_STORAGE_WRITE:
        return TA_KYEE_storage_write(param_types, params);
    case TA_STORAGE_READ:
        return TA_KYEE_storage_read(param_types, params);
    case TA_STORAGE_DELETE:
        return TA_KYEE_storage_delete(param_types, params);
    // TODO: 
    case TA_SM2ExchangeKey:
        return TEE_SUCCESS; 
#endif // KYEE
    // ====================================================================
    case MAKE_NETWORK_CMD_DEMO:
		return make_network_ta_demo(param_types, params);
	case MAKE_NETWORK_CMD:
		return make_network_ta(param_types, params);
	case MAKE_LAYER_CMD:
		return make_layer_ta(param_types, params);
	case MAKE_LAYER_EXT_CMD:
		return make_layer_ext_ta(param_types, params);
	case FORWARD_CMD:
		return forward_network_ta(param_types, params);
	case FORWARD_RET_CMD:
		return forward_ret_network_ta(param_types, params);
	case FORWARD_FETCH_CMD:
		return forwardFetch_network_ta(param_types, params);
    // ====================================================================
	default:
		return TEE_ERROR_BAD_PARAMETERS;
	}
}
