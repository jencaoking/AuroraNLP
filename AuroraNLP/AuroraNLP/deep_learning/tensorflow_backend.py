# TensorFlow Backend
# =================
# TensorFlow框架后端实现

import importlib
import os
from typing import Any, Optional
from .framework import Framework, FrameworkType


class TensorFlowBackend(Framework):
    """TensorFlow框架后端"""
    
    def __init__(self):
        self._tf = None
        self._available = None
        self._version = None
        self._device = None
        self._check_availability()
        self._setup_device()
    
    def _check_availability(self):
        """检查TensorFlow是否可用"""
        try:
            self._tf = importlib.import_module('tensorflow')
            self._version = self._tf.__version__
            self._available = True
        except ImportError:
            self._available = False
            self._version = "Not available"
    
    def _setup_device(self):
        """设置设备（CPU/GPU）"""
        if self.is_available():
            # 检查是否有可用的GPU
            gpus = self._tf.config.list_physical_devices('GPU')
            if gpus:
                self._device = '/GPU:0'
            else:
                self._device = '/CPU:0'
        else:
            self._device = None
    
    def is_available(self) -> bool:
        """检查TensorFlow是否可用"""
        return self._available
    
    def load_model(self, model_path: str, **kwargs) -> Any:
        """加载TensorFlow模型
        
        Args:
            model_path: 模型文件路径
            **kwargs: 额外参数
        
        Returns:
            加载的模型
        """
        if not self.is_available():
            raise RuntimeError("TensorFlow is not available")
        
        try:
            # 检查文件是否存在
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model path not found: {model_path}")
            
            # 检查模型文件格式
            if not os.path.isdir(model_path) and not model_path.endswith('.h5'):
                raise ValueError("TensorFlow model should be a directory or .h5 file")
            
            model = self._tf.keras.models.load_model(model_path, **kwargs)
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load TensorFlow model: {str(e)}")
    
    def save_model(self, model: Any, model_path: str, **kwargs) -> None:
        """保存TensorFlow模型
        
        Args:
            model: 模型对象
            model_path: 保存路径
            **kwargs: 额外参数
        """
        if not self.is_available():
            raise RuntimeError("TensorFlow is not available")
        
        try:
            model.save(model_path, **kwargs)
        except Exception as e:
            raise RuntimeError(f"Failed to save TensorFlow model: {str(e)}")
    
    def inference(self, model: Any, input_data: Any, **kwargs) -> Any:
        """TensorFlow模型推理
        
        Args:
            model: 模型对象
            input_data: 输入数据
            **kwargs: 额外参数
        
        Returns:
            推理结果
        """
        if not self.is_available():
            raise RuntimeError("TensorFlow is not available")
        
        try:
            # 使用指定设备执行推理
            with self._tf.device(self._device):
                # 执行推理
                output = model.predict(input_data, **kwargs)
            return output
        except Exception as e:
            raise RuntimeError(f"Failed to run TensorFlow inference: {str(e)}")
    
    def get_framework_type(self) -> FrameworkType:
        """获取框架类型"""
        return FrameworkType.TENSORFLOW
    
    def get_version(self) -> str:
        """获取TensorFlow版本"""
        return self._version
