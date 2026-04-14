# Deep Learning Module
# ====================
# 深度学习框架集成模块

from .framework import Framework, FrameworkType, get_framework
from .pytorch_backend import PyTorchBackend
from .tensorflow_backend import TensorFlowBackend

__all__ = [
    'Framework',
    'FrameworkType',
    'get_framework',
    'PyTorchBackend',
    'TensorFlowBackend'
]
