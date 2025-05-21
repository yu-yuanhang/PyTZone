CC      ?= $(CROSS_COMPILE)gcc
LD      ?= $(CROSS_COMPILE)ld
AR      ?= $(CROSS_COMPILE)ar
NM      ?= $(CROSS_COMPILE)nm
OBJCOPY ?= $(CROSS_COMPILE)objcopy
OBJDUMP ?= $(CROSS_COMPILE)objdump
READELF ?= $(CROSS_COMPILE)readelf

DIRS = ./ ./C ./kyee

# 查找每个目录中的 .c 文件
SOURCES = $(wildcard $(addsuffix /*.c, $(DIRS)))

OBJS = $(SOURCES:.c=.o)

CFLAGS += -Wall -I../ta/include -I$(TEEC_EXPORT)/include 
CFLAGS += -I./../../client/include
CFLAGS += -I. -I./C -I./kyee/include

CFLAGS += -Wall -Wextra -Wno-declaration-after-statement

#Add/link other required libraries here
LDADD += -lteec -L$(TEEC_EXPORT)/lib

# BINARY = optee_example_hello_world
BINARY = libTorchZone.so

.PHONY: all
all: $(BINARY)

$(BINARY): $(OBJS)
	# $(CC) $(LDFLAGS) -o $@ $(OBJS) $(LDADD)
	$(CC) $(CFLAGS) -shared $(LDFLAGS) -o $@ $(OBJS) $(LDADD)

.PHONY: clean
clean:
	rm -f $(OBJS) $(BINARY)

%.o: %.c
	$(CC) $(CFLAGS) -fPIC -c $< -o $@

