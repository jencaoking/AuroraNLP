# PyTorch Backend
# ===============
# PyTorch框架后端实现

import importlib
from typing import Any, Optional
from .framework import Framework, FrameworkType


class PyTorchBackend(Framework):
    """PyTorch框架后端"""
    
    def __init__(self):
        self._torch = None
        self._available = None
        self._version = None
        self._device = None
        self._check_availability()
        self._setup_device()
    
    def _check_availability(self):
        """检查PyTorch是否可用"""
        try:
            self._torch = importlib.import_module('torch')
            self._version = self._torch.__version__
            self._available = True
        except ImportError:
            self._available = False
            self._version = "Not available"
    
    def _setup_device(self):
        """设置设备（CPU/GPU）"""
        if self.is_available():
            self._device = self._torch.device("cuda" if self._torch.cuda.is_available() else "cpu")
        else:
            self._device = None
    
    def is_available(self) -> bool:
        """检查PyTorch是否可用"""
        return self._available
    
    def load_model(self, model_path: str, **kwargs) -> Any:
        """加载PyTorch模型
        
        Args:
            model_path: 模型文件路径
            **kwargs: 额外参数
        
        Returns:
            加载的模型
        """
        if not self.is_available():
            raise RuntimeError("PyTorch is not available")
        
        try:
            model = self._torch.load(model_path, **kwargs)
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load PyTorch model: {str(e)}")
    
    def save_model(self, model: Any, model_path: str, **kwargs) -> None:
        """保存PyTorch模型
        
        Args:
            model: 模型对象
            model_path: 保存路径
            **kwargs: 额外参数
        """
        if not self.is_available():
            raise RuntimeError("PyTorch is not available")
        
        try:
            self._torch.save(model, model_path, **kwargs)
        except Exception as e:
            raise RuntimeError(f"Failed to save PyTorch model: {str(e)}")
    
    def inference(self, model: Any, input_data: Any, **kwargs) -> Any:
        """PyTorch模型推理
        
        Args:
            model: 模型对象
            input_data: 输入数据
            **kwargs: 额外参数
        
        Returns:
            推理结果
        """
        if not self.is_available():
            raise RuntimeError("PyTorch is not available")
        
        try:
            # 设置为评估模式
            model.eval()
            
            # 将模型移动到设备
            model = model.to(self._device)
            
            # 处理输入数据
            def move_to_device(data):
                if isinstance(data, self._torch.Tensor):
                    return data.to(self._device)
                elif isinstance(data, dict):
                    return {k: move_to_device(v) for k, v in data.items()}
                elif isinstance(data, (list, tuple)):
                    return type(data)(move_to_device(item) for item in data)
                else:
                    return data
            
            # 处理输入数据
            if isinstance(input_data, (list, tuple)):
                input_tensor = self._torch.tensor(input_data)
                input_tensor = move_to_device(input_tensor)
            elif not isinstance(input_data, self._torch.Tensor):
                input_tensor = self._torch.tensor(input_data)
                input_tensor = move_to_device(input_tensor)
            else:
                input_tensor = move_to_device(input_data)
            
            # 执行推理
            with self._torch.no_grad():
                output = model(input_tensor, **kwargs)
            
            # 将输出移回CPU（如果需要）
            if isinstance(output, self._torch.Tensor):
                output = output.cpu()
            
            return output
        except Exception as e:
            raise RuntimeError(f"Failed to run PyTorch inference: {str(e)}")
    
    def get_framework_type(self) -> FrameworkType:
        """获取框架类型"""
        return FrameworkType.PYTORCH
    
    def get_version(self) -> str:
        """获取PyTorch版本"""
        return self._version
