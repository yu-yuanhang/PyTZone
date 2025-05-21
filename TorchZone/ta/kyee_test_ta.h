#ifndef __KYEE_TEST_TA_H__
#define __KYEE_TEST_TA_H__

#ifndef KYEE
#define KYEE 0
#endif
#if KYEE

#include <ptz_defs.h>

#include <tee_internal_api.h>
#include <tee_internal_api_extensions.h>

#include <kyee_test_ta.h>
#include <tee_api.h>
// #include <tee_api_qspi.h>
// #include <tee_api_sm.h>

TEE_Result TA_KYEE_SM2KeyGet(uint32_t param_types,TEE_Param params[4]);
TEE_Result TA_KYEE_SM2GetZ(uint32_t param_types,TEE_Param params[4]);
TEE_Result TA_KYEE_SM2GetE(uint32_t param_types,TEE_Param params[4]);
TEE_Result TA_KYEE_SM2Sign(uint32_t param_types,TEE_Param params[4]);
TEE_Result TA_KYEE_SM2Verify(uint32_t param_types,TEE_Param params[4]);
TEE_Result TA_KYEE_SM2Encrypt(uint32_t param_types,TEE_Param params[4]);
TEE_Result TA_KYEE_SM2Decrypt(uint32_t param_types,TEE_Param params[4]);
TEE_Result TA_KYEE_SM3Init(uint32_t param_types,TEE_Param params[4]);
TEE_Result TA_KYEE_SM3Process(uint32_t param_types,TEE_Param params[4]);
TEE_Result TA_KYEE_SM3Done(uint32_t param_types,TEE_Param params[4]);
TEE_Result TA_KYEE_SM4Init(uint32_t param_types,TEE_Param params[4]);
TEE_Result TA_KYEE_SM4Crypto(uint32_t param_types,TEE_Param params[4]);
TEE_Result TA_KYEE_storage_write(uint32_t param_types, TEE_Param params[4]);
TEE_Result TA_KYEE_storage_read(uint32_t param_types, TEE_Param params[4]);
TEE_Result TA_KYEE_storage_delete(uint32_t param_types, TEE_Param params[4]);

#endif // KYEE


#endif /* __KYEE_TEST_TA_H__ */

