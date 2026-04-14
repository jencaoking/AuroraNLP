# BiLSTM-CRF Model
# =================
# 基于 PyTorch 的 BiLSTM-CRF 模型实现

import importlib
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from .framework import Framework, FrameworkType, get_framework


class BiLSTMCRF:
    """BiLSTM-CRF 模型"""
    
    def __init__(self, 
                 vocab_size: int, 
                 tagset_size: int, 
                 embedding_dim: int = 128, 
                 hidden_dim: int = 256, 
                 num_layers: int = 2, 
                 dropout: float = 0.5):
        """初始化 BiLSTM-CRF 模型
        
        Args:
            vocab_size: 词汇表大小
            tagset_size: 标签集大小
            embedding_dim: 词嵌入维度
            hidden_dim: LSTM 隐藏层维度
            num_layers: LSTM 层数
            dropout:  dropout 概率
        """
        self.vocab_size = vocab_size
        self.tagset_size = tagset_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.framework = get_framework()
        self.model = None
        self._torch = None
        self._crf = None
        
        if self.framework:
            self._setup_model()
    
    def _setup_model(self):
        """设置模型"""
        # 导入必要的库
        self._torch = importlib.import_module('torch')
        self._nn = importlib.import_module('torch.nn')
        
        # 尝试导入 pytorch-crf
        try:
            self._crf = importlib.import_module('torchcrf')
        except ImportError:
            # 如果没有安装 torchcrf，使用内置的 CRF 实现
            pass
        
        # 创建模型
        self.model = self._BiLSTMCRFModel(
            vocab_size=self.vocab_size,
            tagset_size=self.tagset_size,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            use_crf=(self._crf is not None)
        )
    
    def is_available(self) -> bool:
        """检查模型是否可用"""
        return self.framework is not None and self.model is not None
    
    def train(self, 
              train_data: List[Tuple[List[int], List[int]]], 
              val_data: Optional[List[Tuple[List[int], List[int]]]] = None,
              epochs: int = 10, 
              batch_size: int = 32, 
              learning_rate: float = 0.001,
              device: Optional[str] = None):
        """训练模型
        
        Args:
            train_data: 训练数据，格式为 [(输入序列, 标签序列), ...]
            val_data: 验证数据
            epochs: 训练轮数
            batch_size: 批量大小
            learning_rate: 学习率
            device: 设备，如 'cpu' 或 'cuda'
        """
        if not self.is_available():
            raise RuntimeError("Model is not available")
        
        # 设置设备
        if device:
            device_obj = self._torch.device(device)
        else:
            device_obj = self._torch.device("cuda" if self._torch.cuda.is_available() else "cpu")
        
        self.model.to(device_obj)
        
        # 定义优化器
        optimizer = self._torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # 训练循环
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            # 批量处理
            for i in range(0, len(train_data), batch_size):
                batch_data = train_data[i:i+batch_size]
                inputs, targets = zip(*batch_data)
                
                # 填充序列
                max_len = max(len(seq) for seq in inputs)
                padded_inputs = []
                padded_targets = []
                masks = []
                
                for seq, tag in zip(inputs, targets):
                    pad_len = max_len - len(seq)
                    padded_inputs.append(seq + [0] * pad_len)
                    padded_targets.append(tag + [0] * pad_len)
                    masks.append([1] * len(seq) + [0] * pad_len)
                
                # 转换为张量
                inputs_tensor = self._torch.tensor(padded_inputs, dtype=self._torch.long, device=device_obj)
                targets_tensor = self._torch.tensor(padded_targets, dtype=self._torch.long, device=device_obj)
                masks_tensor = self._torch.tensor(masks, dtype=self._torch.bool, device=device_obj)
                
                # 清零梯度
                optimizer.zero_grad()
                
                # 前向传播
                loss = self.model(inputs_tensor, targets_tensor, masks_tensor)
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # 计算平均损失
            avg_loss = total_loss / (len(train_data) // batch_size + 1)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            
            # 验证
            if val_data:
                self.model.eval()
                val_loss = 0
                
                with self._torch.no_grad():
                    for i in range(0, len(val_data), batch_size):
                        batch_data = val_data[i:i+batch_size]
                        inputs, targets = zip(*batch_data)
                        
                        # 填充序列
                        max_len = max(len(seq) for seq in inputs)
                        padded_inputs = []
                        padded_targets = []
                        masks = []
                        
                        for seq, tag in zip(inputs, targets):
                            pad_len = max_len - len(seq)
                            padded_inputs.append(seq + [0] * pad_len)
                            padded_targets.append(tag + [0] * pad_len)
                            masks.append([1] * len(seq) + [0] * pad_len)
                        
                        # 转换为张量
                        inputs_tensor = self._torch.tensor(padded_inputs, dtype=self._torch.long, device=device_obj)
                        targets_tensor = self._torch.tensor(padded_targets, dtype=self._torch.long, device=device_obj)
                        masks_tensor = self._torch.tensor(masks, dtype=self._torch.bool, device=device_obj)
                        
                        # 计算损失
                        loss = self.model(inputs_tensor, targets_tensor, masks_tensor)
                        val_loss += loss.item()
                
                avg_val_loss = val_loss / (len(val_data) // batch_size + 1)
                print(f"Validation Loss: {avg_val_loss:.4f}")
    
    def predict(self, input_data: List[List[int]]) -> List[List[int]]:
        """预测序列标签
        
        Args:
            input_data: 输入数据，格式为 [输入序列, ...]
            
        Returns:
            预测的标签序列
        """
        if not self.is_available():
            raise RuntimeError("Model is not available")
        
        self.model.eval()
        predictions = []
        
        # 获取模型所在设备
        device = next(self.model.parameters()).device
        
        with self._torch.no_grad():
            for seq in input_data:
                # 转换为张量并移动到模型所在设备
                input_tensor = self._torch.tensor([seq], dtype=self._torch.long, device=device)
                
                # 预测
                prediction = self.model.decode(input_tensor)
                predictions.append(prediction[0])
        
        return predictions
    
    def save(self, model_path: str):
        """保存模型
        
        Args:
            model_path: 模型保存路径
        """
        if not self.is_available():
            raise RuntimeError("Model is not available")
        
        self.framework.save_model(self.model, model_path)
    
    def load(self, model_path: str):
        """加载模型
        
        Args:
            model_path: 模型加载路径
        """
        if not self.framework:
            raise RuntimeError("No deep learning framework available")
        
        self.model = self.framework.load_model(model_path)
        
        # 导入必要的库
        self._torch = importlib.import_module('torch')
        self._nn = importlib.import_module('torch.nn')
        
        # 尝试导入 pytorch-crf
        try:
            self._crf = importlib.import_module('torchcrf')
        except ImportError:
            # 如果没有安装 torchcrf，使用内置的 CRF 实现
            pass
        
    class _BiLSTMCRFModel:
        """BiLSTM-CRF 模型内部类"""
        
        def __init__(self, vocab_size, tagset_size, embedding_dim, hidden_dim, num_layers, dropout, use_crf):
            nn = importlib.import_module('torch.nn')
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.lstm = nn.LSTM(
                embedding_dim, hidden_dim, num_layers=num_layers, bidirectional=True, dropout=dropout
            )
            self.hidden2tag = nn.Linear(hidden_dim * 2, tagset_size)
            self.use_crf = use_crf
            
            if use_crf:
                self.crf = importlib.import_module('torchcrf').CRF(tagset_size)
        
        def to(self, device):
            """移动模型到指定设备"""
            self.embedding = self.embedding.to(device)
            self.lstm = self.lstm.to(device)
            self.hidden2tag = self.hidden2tag.to(device)
            if self.use_crf:
                self.crf = self.crf.to(device)
            return self
        
        def train(self, mode=True):
            """设置训练模式"""
            self.embedding.train(mode)
            self.lstm.train(mode)
            self.hidden2tag.train(mode)
            return self
        
        def eval(self):
            """设置评估模式"""
            return self.train(False)
        
        def parameters(self):
            """获取模型参数"""
            params = []
            params.extend(self.embedding.parameters())
            params.extend(self.lstm.parameters())
            params.extend(self.hidden2tag.parameters())
            if self.use_crf:
                params.extend(self.crf.parameters())
            return params
        
        def __call__(self, *args, **kwargs):
            """使模型可调用"""
            return self.forward(*args, **kwargs)
        
        def forward(self, x, tags, mask):
            """前向传播
            
            Args:
                x: 输入序列
                tags: 标签序列
                mask: 掩码
                
            Returns:
                损失值
            """
            embedding = self.embedding(x)
            lstm_out, _ = self.lstm(embedding)
            tag_scores = self.hidden2tag(lstm_out)
            
            if self.use_crf:
                loss = -self.crf(tag_scores, tags, mask=mask)
            else:
                nn = importlib.import_module('torch.nn')
                loss_fn = nn.CrossEntropyLoss(ignore_index=0)
                loss = loss_fn(tag_scores.view(-1, tag_scores.shape[-1]), tags.view(-1))
            
            return loss
        
        def decode(self, x):
            """解码
            
            Args:
                x: 输入序列
                
            Returns:
                预测的标签序列
            """
            embedding = self.embedding(x)
            lstm_out, _ = self.lstm(embedding)
            tag_scores = self.hidden2tag(lstm_out)
            
            if self.use_crf:
                return self.crf.decode(tag_scores)
            else:
                return importlib.import_module('torch').argmax(tag_scores, dim=2).tolist()
