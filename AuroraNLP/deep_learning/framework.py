# Framework Abstraction Layer
# ============================
# 深度学习框架抽象层，用于统一不同框架的接口

from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Dict, Any


class FrameworkType(Enum):
    """框架类型枚举"""
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    AUTO = "auto"


class Framework(ABC):
    """深度学习框架抽象基类"""
    
    @abstractmethod
    def is_available(self) -> bool:
        """检查框架是否可用"""
        pass
    
    @abstractmethod
    def load_model(self, model_path: str, **kwargs) -> Any:
        """加载模型"""
        pass
    
    @abstractmethod
    def save_model(self, model: Any, model_path: str, **kwargs) -> None:
        """保存模型"""
        pass
    
    @abstractmethod
    def inference(self, model: Any, input_data: Any, **kwargs) -> Any:
        """模型推理"""
        pass
    
    @abstractmethod
    def get_framework_type(self) -> FrameworkType:
        """获取框架类型"""
        pass
    
    @abstractmethod
    def get_version(self) -> str:
        """获取框架版本"""
        pass


def get_framework(framework_type: FrameworkType = FrameworkType.AUTO) -> Optional[Framework]:
    """获取可用的深度学习框架
    
    Args:
        framework_type: 框架类型，默认为自动检测
    
    Returns:
        可用的框架实例，如果没有可用框架则返回None
    """
    from .pytorch_backend import PyTorchBackend
    from .tensorflow_backend import TensorFlowBackend
    
    if framework_type == FrameworkType.PYTORCH:
        pt_backend = PyTorchBackend()
        return pt_backend if pt_backend.is_available() else None
    
    elif framework_type == FrameworkType.TENSORFLOW:
        tf_backend = TensorFlowBackend()
        return tf_backend if tf_backend.is_available() else None
    
    elif framework_type == FrameworkType.AUTO:
        # 优先检查PyTorch
        pt_backend = PyTorchBackend()
        if pt_backend.is_available():
            return pt_backend
        
        # 然后检查TensorFlow
        tf_backend = TensorFlowBackend()
        if tf_backend.is_available():
            return tf_backend
        
        return None
    
    return None
