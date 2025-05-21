#!/usr/bin/env python
# coding=utf-8

# python setup.py build_ext --inplace
# build_ext 指定构建 C 或 C++ 扩展模块
# --inplace 此选项表示将编译生成的扩展模块放置在源代码的相应目录中 而不是将它们安装到 Python 的全局包路径中

# python setup.py build_ext --build-lib=/path/to/output_directory
# --build-lib: 选项用于指定生成的库文件(如 .so 文件)的目标目录

# python setup.py build_ext --build-lib=/path/to/output_directory
# python setup.py build_ext  --build-lib=/home/yyh/2.Programs/3.workplace/3.pytorch/2.demo/cppExtension/PyTZone/build

# export LD_LIBRARY_PATH=/home/yyh/2.Programs/3.workplace/3.pytorch/2.demo/cppExtension/log4cpp/2.build/lib:$LD_LIBRARY_PATH

import glob
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension
import os

####################################################################

RunMgr_sources = glob.glob('./RunMgr/*.cc')
core_sources = glob.glob('./core/*.cc')
core_sources += glob.glob('./core/ops/*.cc')
log_sources = glob.glob('./Logger/*.cc')
# generic_sources = glob.glob('generic/*.cc')
# test_ca_sources = ['test_ca.cpp']

src_path = RunMgr_sources + core_sources + log_sources

print(src_path)
####################################################################

####################################################################

# 定义 C++ 扩展模块
setup(
    name = 'PyTZone',  # 对应的模块名
    ext_modules = [
        CppExtension(
            name = 'PyTZone',  # 生成的库名称
            sources = src_path,
            include_dirs = [
                os.path.abspath('.'),
                os.path.abspath('./RunMgr'),
                os.path.abspath('./generic'),  # 添加头文件目录
                os.path.abspath('./core'),
                os.path.abspath('./core/ops'),
                os.path.abspath('./core/utils'),
                os.path.abspath('./Logger'),
                os.path.abspath('./../log4cpp/2.build/include')
            ],
            # define_macros=[('PYBIND11_DETAILED_ERROR_MESSAGES', None)],
            extra_compile_args={
                'cxx': ['-g', '-O0',    # 用于调试的信息，可以设置为你需要的编译标志
                        # 关于 -fvisibility=hidden: pybind11 代码内部强制将所有内部代码的可见性设置为隐藏
                        # 我也不知道 反正就是报错 
                        # 有兴趣参考 https://pybind11.readthedocs.io/en/stable/faq.html#someclass-declared-with-greater-visibility-than-the-type-of-its-field-someclass-member-wattributes
                        '-fvisibility=hidden'],
                        # '-DPYBIND11_DETAILED_ERROR_MESSAGES'],  
            },
            libraries=['log4cpp'], # 链接外部库
            library_dirs=[
                '../log4cpp/2.build/lib',  # 等效于 find_library 中的路径
            ],
            extra_link_args=[]  # 如果需要添加额外的链接参数可以在此处定义
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

