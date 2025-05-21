global-incdirs-y += ./include
global-incdirs-y += ./torchzone
global-incdirs-y += ./torchzone/ops
global-incdirs-y += ./torchzone/station
global-incdirs-y += ./fdlibm_s

# srcs-y += hello_ta.c
# srcs-y += ./torchzone/torchzone.c

SRC_DIRS := ./ ./torchzone ./torchzone/ops ./torchzone/station ./fdlibm_s
# 查找所有 .c 文件
srcs-y := $(foreach dir,$(SRC_DIRS),$(wildcard $(dir)/*.c))

# To remove a certain compiler flag, add a line like this
#cflags-template_ta.c-y += -Wno-strict-prototypes
