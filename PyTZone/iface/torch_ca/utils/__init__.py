#!/usr/bin/env python
# coding=utf-8

from .load import weightShift
__all__ = ['weightShift']

import os
import torch

# 计算绝对路径，避免相对路径在不同工作目录下出错
_so_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "TExt", "build_cmake", "libPyTZone.so")
_so_path = os.path.abspath(_so_path)

if os.path.exists(_so_path):
    torch.classes.load_library(_so_path)
else:
    raise FileNotFoundError(f"Cannot find libPyTZone.so at: {_so_path}")


