CFG_TEE_TA_LOG_LEVEL ?= 4
CPPFLAGS += -DCFG_TEE_TA_LOG_LEVEL=$(CFG_TEE_TA_LOG_LEVEL)
CPPFLAGS += -DCFG_PHYTIUM2004_SMX
CFG_TA_OPTEE_CORE_API_COMPAT_1_1=y

# CXXFLAGS += -Ofast
# CFLAGS += -Ofast

# The UUID for the Trusted Application
BINARY=20001120-2450-11e4-abe2-0002a5d5c51b

TA_DEV_KIT_DIR ?= ./../../ta/plat-phytiumd2000
-include $(TA_DEV_KIT_DIR)/mk/ta_dev_kit.mk

ifeq ($(wildcard $(TA_DEV_KIT_DIR)/mk/ta_dev_kit.mk), )
clean:
	@echo 'Note: $$(TA_DEV_KIT_DIR)/mk/ta_dev_kit.mk not found, cannot clean TA'
	@echo 'Note: TA_DEV_KIT_DIR=$(TA_DEV_KIT_DIR)'
endif
