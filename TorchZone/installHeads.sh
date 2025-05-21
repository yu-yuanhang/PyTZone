#!/bin/bash

# if [ "$#" -ne 2 ]; then
#     echo "Usage: $0 <source_directory> <destination_directory>"
#     exit 1
# fi
# SOURCE_DIR=$1
# DEST_DIR1=$2

SOURCE_DIR=/home/yyh/2.Programs/2.workplace/pytorch/2.demo/TorchZone
SOURCE_DIR_HOST="$SOURCE_DIR/host"
SOURCE_DIR_TA="$SOURCE_DIR/ta"
DEST_DIR1=/home/yyh/2.Programs/2.workplace/optee/optee/out-br/target/root/include
DEST_DIR2=/home/yyh/2.Programs/2.workplace/pytorch/2.demo/PyTZone/TExt/include

LIBTEE_HEAD=/home/yyh/2.Programs/2.workplace/optee/optee/optee_client/libteec/include

if [ -d "$DEST_DIR1" ]; then
    echo "Target directory '$DEST_DIR1' already exists. Deleting it..."
    rm -rf "$DEST_DIR1" 
fi
mkdir -p "$DEST_DIR1"
if [ -d "$DEST_DIR2" ]; then
    echo "Target directory '$DEST_DIR2' already exists. Deleting it..."
    rm -rf "$DEST_DIR2" 
fi
mkdir -p "$DEST_DIR2"

# 递归查找 .h 文件并复制
find "$SOURCE_DIR_HOST" -type f -name "*.h" | while read -r file; do
    # 获取原文件的相对路径
    RELATIVE_PATH=$(realpath --relative-to="$SOURCE_DIR_HOST" "$file")

    # 创建目标路径
    TARGET_PATH1="$DEST_DIR1/$(dirname "$RELATIVE_PATH")"
    mkdir -p "$TARGET_PATH1"
    TARGET_PATH2="$DEST_DIR2/$(dirname "$RELATIVE_PATH")"
    mkdir -p "$TARGET_PATH2"

    # 复制文件
    cp "$file" "$TARGET_PATH1/"
    echo "Copied: $file to $TARGET_PATH1/"
    cp "$file" "$TARGET_PATH2/"
    echo "Copied: $file to $TARGET_PATH2/"
done

echo "All header files copied successfully."

cp $LIBTEE_HEAD/* $DEST_DIR1

DIRECTORY_TA1="$DEST_DIR1/ta"
if [ ! -d "$DIRECTORY_TA1" ]; then
    echo "Directory does not exist. Creating it: $DIRECTORY_TA1"
    mkdir -p "$DIRECTORY_TA1" 
else
    echo "Directory already exists: $DIRECTORY_TA1"
fi
DIRECTORY_TA2="$DEST_DIR2/ta"
if [ ! -d "$DIRECTORY_TA2" ]; then
    echo "Directory does not exist. Creating it: $DIRECTORY_TA2"
    mkdir -p "$DIRECTORY_TA2" 
else
    echo "Directory already exists: $DIRECTORY_TA2"
fi

cp $SOURCE_DIR_TA/include/* $DIRECTORY_TA1
cp $SOURCE_DIR_TA/include/* $DIRECTORY_TA2
