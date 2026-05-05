"""
BiLSTM-CRF 模型
==============
基于 PyTorch 的 BiLSTM-CRF 模型实现
"""
import importlib
import json
from typing import List, Tuple, Optional, Dict, Any
from collections import Counter
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .framework import Framework, FrameworkType, get_framework


class CRF:
    """
    纯 PyTorch 实现的 CRF 层，无需外部依赖
    """
    def __init__(self, num_tags: int):
        """
        初始化 CRF
        
        Args:
            num_tags: 标签数量
        """
        self.num_tags = num_tags
        
        # 转移矩阵 (num_tags + 2: 加上 START 和 END 标签)
        self.start_transitions = None
        self.transitions = None
        self.end_transitions = None
        self._params_initialized = False
        
        # 内部索引
        self.START_TAG = num_tags
        self.END_TAG = num_tags + 1
    
    def init_params(self, device=None):
        """
        初始化参数
        
        Args:
            device: 设备
        """
        torch = importlib.import_module('torch')
        nn = importlib.import_module('torch.nn')
        
        # 转移矩阵
        self.start_transitions = nn.Parameter(torch.empty(self.num_tags + 2))
        self.transitions = nn.Parameter(torch.empty(self.num_tags + 2, self.num_tags + 2))
        self.end_transitions = nn.Parameter(torch.empty(self.num_tags + 2))
        
        # 初始化参数
        nn.init.normal_(self.start_transitions, -0.1)
        nn.init.normal_(self.transitions, -0.1)
        nn.init.normal_(self.end_transitions, -0.1)
        
        self._params_initialized = True
    
    def parameters(self):
        """获取参数列表"""
        if not self._params_initialized:
            self.init_params()
        return [self.start_transitions, self.transitions, self.end_transitions]
    
    def to(self, device):
        """移动到设备"""
        if self._params_initialized:
            self.start_transitions = self.start_transitions.to(device)
            self.transitions = self.transitions.to(device)
            self.end_transitions = self.end_transitions.to(device)
        return self
    
    def __call__(self, emissions, tags=None, mask=None):
        """
        计算损失或解码
        
        Args:
            emissions: 发射分数 (batch_size, seq_len, num_tags)
            tags: 标签 (batch_size, seq_len)
            mask: 掩码 (batch_size, seq_len)
            
        Returns:
            损失值或解码结果
        """
        if tags is not None:
            return self.neg_log_likelihood(emissions, tags, mask)
        else:
            return self.decode(emissions, mask)
    
    def _validate_input(self, emissions, tags, mask):
        """验证输入"""
        torch = importlib.import_module('torch')
        
        if mask is None:
            mask = torch.ones_like(emissions[:, :, 0], dtype=torch.bool)
        
        if not self._params_initialized:
            self.init_params(device=emissions.device)
        
        return mask
    
    def neg_log_likelihood(self, emissions, tags, mask=None):
        """
        计算负对数似然
        
        Args:
            emissions: 发射分数 (batch_size, seq_len, num_tags)
            tags: 标签 (batch_size, seq_len)
            mask: 掩码 (batch_size, seq_len)
            
        Returns:
            损失值
        """
        torch = importlib.import_module('torch')
        mask = self._validate_input(emissions, tags, mask)
        
        batch_size, seq_len = emissions.shape[:2]
        
        # 计算 log Z
        log_z = self._forward_algorithm(emissions, mask)
        
        # 计算 gold 分数
        gold_score = self._score_sentence(emissions, tags, mask)
        
        # 负对数似然
        return (log_z - gold_score).mean()
    
    def decode(self, emissions, mask=None):
        """
        使用 Viterbi 算法解码
        
        Args:
            emissions: 发射分数 (batch_size, seq_len, num_tags)
            mask: 掩码 (batch_size, seq_len)
            
        Returns:
            最佳路径列表
        """
        torch = importlib.import_module('torch')
        
        if mask is None:
            mask = torch.ones_like(emissions[:, :, 0], dtype=torch.bool)
        
        batch_size, seq_len = emissions.shape[:2]
        
        best_paths = []
        
        for i in range(batch_size):
            seq_emissions = emissions[i]
            seq_mask = mask[i]
            seq_len_i = int(seq_mask.sum().item())
            
            if seq_len_i == 0:
                best_paths.append([])
                continue
            
            path = self._viterbi_decode(seq_emissions[:seq_len_i])
            best_paths.append(path)
        
        return best_paths
    
    def _forward_algorithm(self, emissions, mask):
        """
        前向算法计算 log Z
        
        Args:
            emissions: 发射分数
            mask: 掩码
            
        Returns:
            log Z
        """
        torch = importlib.import_module('torch')
        
        batch_size, seq_len = emissions.shape[:2]
        
        # 初始化
        alpha = self.start_transitions[:self.num_tags].unsqueeze(0) + emissions[:, 0]
        
        # 动态规划
        for t in range(1, seq_len):
            emit_score = emissions[:, t].unsqueeze(1)  # (batch_size, 1, num_tags)
            trans_score = self.transitions[:self.num_tags, :self.num_tags].unsqueeze(0)  # (1, num_tags, num_tags)
            next_tag_var = alpha.unsqueeze(2) + emit_score + trans_score  # (batch_size, num_tags, num_tags)
            
            next_tag_var = self._log_sum_exp(next_tag_var, dim=1)
            alpha = torch.where(mask[:, t].unsqueeze(1), next_tag_var, alpha)
        
        # 加上 END 转移
        alpha = alpha + self.end_transitions[:self.num_tags]
        
        # log Z
        return self._log_sum_exp(alpha, dim=1)
    
    def _score_sentence(self, emissions, tags, mask):
        """
        计算 gold 路径分数
        
        Args:
            emissions: 发射分数
            tags: 标签
            mask: 掩码
            
        Returns:
            分数
        """
        torch = importlib.import_module('torch')
        
        batch_size, seq_len = emissions.shape[:2]
        
        # 初始化
        score = self.start_transitions[tags[:, 0]]
        
        # 累积分数
        for t in range(1, seq_len):
            trans = self.transitions[tags[:, t-1], tags[:, t]]
            emit = emissions[range(batch_size), t, tags[:, t]]
            score += trans + emit
            score = torch.where(mask[:, t], score, score)
        
        # 加上 END 转移
        seq_lens = mask.sum(dim=1)
        last_tag_indices = seq_lens - 1
        last_tags = tags[range(batch_size), last_tag_indices]
        score += self.end_transitions[last_tags]
        
        return score
    
    def _viterbi_decode(self, emissions):
        """
        Viterbi 解码单序列
        
        Args:
            emissions: 发射分数 (seq_len, num_tags)
            
        Returns:
            最佳路径
        """
        torch = importlib.import_module('torch')
        
        seq_len = emissions.shape[0]
        
        # 初始化
        viterbi = []
        backpointers = []
        
        # 第一个时间步
        viterbi.append(self.start_transitions[:self.num_tags] + emissions[0])
        backpointers.append(torch.zeros(self.num_tags, dtype=torch.long))
        
        # 动态规划
        for t in range(1, seq_len):
            next_viterbi = []
            next_backpointer = []
            
            for v_idx in range(self.num_tags):
                # 计算到达 v 的所有可能路径
                trans = self.transitions[:self.num_tags, v_idx]
                emit = emissions[t, v_idx]
                next_viterbi_t = viterbi[t-1] + trans + emit
                
                # 最佳路径和对应的前一个标签
                best_idx = int(next_viterbi_t.argmax())
                next_viterbi.append(next_viterbi_t[best_idx])
                next_backpointer.append(best_idx)
            
            viterbi.append(torch.stack(next_viterbi))
            backpointers.append(torch.stack(next_backpointer))
        
        # 加上 END
        viterbi[-1] += self.end_transitions[:self.num_tags]
        
        # 找到最佳路径终点
        best_tag_id = int(viterbi[-1].argmax().item())
        best_path = [best_tag_id]
        
        # 回溯
        for t_idx in range(len(backpointers)-1, 0, -1):
            best_tag_id = int(backpointers[t_idx][best_tag_id].item())
            best_path.append(best_tag_id)
        
        # 反转路径
        best_path.reverse()
        return best_path
    
    def _log_sum_exp(self, vec, dim):
        """
        log sum exp
        
        Args:
            vec: 输入张量
            dim: 维度
            
        Returns:
            log(sum(exp(vec)))
        """
        torch = importlib.import_module('torch')
        
        max_val, _ = vec.max(dim=dim)
        return max_val + torch.log(torch.sum(torch.exp(vec - max_val.unsqueeze(dim)), dim=dim))


# 内部辅助类，不暴露在顶层
def _create_bilstm_crf_model_class():
    """延迟创建模型类，避免导入问题"""
    try:
        nn = importlib.import_module('torch.nn')
        
        class _BiLSTMCRFModel(nn.Module):
            """
            BiLSTM-CRF 模型内部类
            """
            def __init__(self, vocab_size, tagset_size, embedding_dim, hidden_dim, num_layers, dropout, use_crf, use_torchcrf):
                super().__init__()
                
                self.embedding = nn.Embedding(vocab_size, embedding_dim)
                self.lstm = nn.LSTM(
                    embedding_dim, hidden_dim, num_layers=num_layers, 
                    bidirectional=True, dropout=dropout if num_layers > 1 else 0.0,
                    batch_first=True
                )
                self.hidden2tag = nn.Linear(hidden_dim * 2, tagset_size)
                self.dropout_layer = nn.Dropout(dropout)
                self.use_crf = use_crf
                self.use_torchcrf = use_torchcrf
                
                if use_crf:
                    if use_torchcrf:
                        self.crf = importlib.import_module('torchcrf').CRF(tagset_size)
                    else:
                        self.crf = CRF(tagset_size)
                        self.crf.init_params()
            
            def forward(self, x, tags=None, mask=None):
                """前向传播"""
                torch = importlib.import_module('torch')
                nn = importlib.import_module('torch.nn')
                
                if mask is None:
                    mask = torch.ones_like(x, dtype=torch.bool)
                
                embedding = self.dropout_layer(self.embedding(x))
                lstm_out, _ = self.lstm(embedding)
                tag_scores = self.hidden2tag(self.dropout_layer(lstm_out))
                
                if self.use_crf:
                    if self.use_torchcrf:
                        loss = -self.crf(tag_scores, tags, mask=mask, reduction='mean')
                    else:
                        loss = self.crf(tag_scores, tags, mask)
                else:
                    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
                    loss = loss_fn(tag_scores.view(-1, tag_scores.shape[-1]), tags.view(-1))
                
                return loss
            
            def decode(self, x, mask=None):
                """解码"""
                torch = importlib.import_module('torch')
                
                if mask is None:
                    mask = torch.ones_like(x, dtype=torch.bool)
                
                embedding = self.embedding(x)
                lstm_out, _ = self.lstm(embedding)
                tag_scores = self.hidden2tag(lstm_out)
                
                if self.use_crf:
                    if self.use_torchcrf:
                        return self.crf.decode(tag_scores, mask=mask)
                    else:
                        return self.crf.decode(tag_scores, mask)
                else:
                    return torch.argmax(tag_scores, dim=2).tolist()
        
        return _BiLSTMCRFModel
    except ImportError:
        # 如果 torch 不可用，返回一个简单的占位类
        class _BiLSTMCRFModelStub:
            def __init__(self, *args, **kwargs):
                raise RuntimeError("PyTorch not available")
        return _BiLSTMCRFModelStub


class BiLSTMCRF:
    """BiLSTM-CRF 模型"""
    
    def __init__(self, 
                 vocab_size: int, 
                 tagset_size: int, 
                 embedding_dim: int = 128, 
                 hidden_dim: int = 256, 
                 num_layers: int = 2, 
                 dropout: float = 0.5,
                 use_crf: bool = True):
        """
        初始化 BiLSTM-CRF 模型
        
        Args:
            vocab_size: 词汇表大小
            tagset_size: 标签集大小
            embedding_dim: 词嵌入维度
            hidden_dim: LSTM 隐藏层维度
            num_layers: LSTM 层数
            dropout:  dropout 概率
            use_crf: 是否使用 CRF
        """
        self.vocab_size = vocab_size
        self.tagset_size = tagset_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_crf = use_crf
        
        self.framework = get_framework()
        self.model = None
        self._torch = None
        self._nn = None
        self._torchcrf = None
        
        if self.framework:
            self._setup_model()
    
    def _setup_model(self):
        """设置模型"""
        # 导入必要的库
        self._torch = importlib.import_module('torch')
        self._nn = importlib.import_module('torch.nn')
        
        # 尝试导入 pytorch-crf
        try:
            self._torchcrf = importlib.import_module('torchcrf')
        except ImportError:
            self._torchcrf = None
        
        # 创建模型
        model_class = _create_bilstm_crf_model_class()
        self.model = model_class(
            vocab_size=self.vocab_size,
            tagset_size=self.tagset_size,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            use_crf=self.use_crf,
            use_torchcrf=(self._torchcrf is not None)
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
              device: Optional[str] = None,
              verbose: bool = True) -> Dict[str, Any]:
        """
        训练模型
        
        Args:
            train_data: 训练数据，格式为 [(输入序列, 标签序列), ...]
            val_data: 验证数据
            epochs: 训练轮数
            batch_size: 批量大小
            learning_rate: 学习率
            device: 设备，如 'cpu' 或 'cuda'
            verbose: 是否打印训练信息
            
        Returns:
            训练指标字典
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
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            
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
                
                total_loss += float(loss.item())
            
            # 计算平均损失
            avg_loss = total_loss / (len(train_data) // batch_size + 1)
            train_losses.append(avg_loss)
            
            if verbose:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            
            # 验证
            if val_data:
                self.model.eval()
                val_loss = 0.0
                
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
                        val_loss += float(loss.item())
                
                avg_val_loss = val_loss / (len(val_data) // batch_size + 1)
                val_losses.append(avg_val_loss)
                
                if verbose:
                    print(f"Validation Loss: {avg_val_loss:.4f}")
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'final_train_loss': train_losses[-1] if train_losses else 0.0,
            'final_val_loss': val_losses[-1] if val_losses else 0.0
        }
    
    def predict(self, input_data: List[List[int]]) -> List[List[int]]:
        """
        预测序列标签
        
        Args:
            input_data: 输入数据，格式为 [输入序列, ...]
            
        Returns:
            预测的标签序列
        """
        if not self.is_available():
            raise RuntimeError("Model is not available")
        
        self.model.eval()
        
        # 获取模型所在设备
        device = next(self.model.parameters()).device
        
        if not input_data:
            return []
        
        # 批量处理
        max_len = max(len(seq) for seq in input_data)
        padded_inputs = []
        masks = []
        
        for seq in input_data:
            pad_len = max_len - len(seq)
            padded_inputs.append(seq + [0] * pad_len)
            masks.append([1] * len(seq) + [0] * pad_len)
        
        inputs_tensor = self._torch.tensor(padded_inputs, dtype=self._torch.long, device=device)
        masks_tensor = self._torch.tensor(masks, dtype=self._torch.bool, device=device)
        
        with self._torch.no_grad():
            predictions = self.model.decode(inputs_tensor, masks_tensor)
        
        # 去除填充部分
        result = []
        for i in range(len(input_data)):
            seq_len = len(input_data[i])
            result.append(predictions[i][:seq_len])
        
        return result
    
    def save(self, model_path: str):
        """保存模型"""
        if not self.is_available():
            raise RuntimeError("Model is not available")
        
        self.framework.save_model(self.model, model_path)
    
    def load(self, model_path: str):
        """加载模型"""
        if not self.framework:
            raise RuntimeError("No deep learning framework available")
        
        self.model = self.framework.load_model(model_path)
        
        # 导入必要的库
        self._torch = importlib.import_module('torch')
        self._nn = importlib.import_module('torch.nn')
        
        # 尝试导入 pytorch-crf
        try:
            self._torchcrf = importlib.import_module('torchcrf')
        except ImportError:
            self._torchcrf = None


class BiLSTMCRFSegmentor:
    """基于 BiLSTM-CRF 的分词器"""
    
    # 标签定义: B/M/E/S
    TAG_B = 0
    TAG_M = 1
    TAG_E = 2
    TAG_S = 3
    TAG_PAD = 4
    
    TAGS = ['B', 'M', 'E', 'S']
    
    def __init__(self, 
                 vocab_size: int = 5000,
                 embedding_dim: int = 128,
                 hidden_dim: int = 256,
                 num_layers: int = 2,
                 dropout: float = 0.5,
                 use_crf: bool = True):
        """初始化"""
        self.vocab_size = vocab_size
        self.vocab = {}
        self.inv_vocab = {}
        
        self._trained = False
        
        # 初始化模型
        self._model_class = BiLSTMCRF(
            vocab_size=vocab_size,
            tagset_size=4,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            use_crf=use_crf
        )
    
    @property
    def is_available(self):
        return self._model_class.is_available()
    
    @property
    def is_trained(self):
        return self._trained
    
    def _build_vocab(self, texts: List[str]):
        """构建词汇表"""
        counter = Counter()
        for text in texts:
            counter.update(text)
        
        chars = sorted(counter.keys(), key=lambda x: -counter[x])[:self.vocab_size-2]
        
        self.vocab = {'<PAD>': 0, '<UNK>': 1}
        for i, char in enumerate(chars, 2):
            self.vocab[char] = i
        
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
    
    def _text_to_ids(self, text: str) -> List[int]:
        return [self.vocab.get(c, 1) for c in text]
    
    def _tags_to_words(self, text: str, tags: List[int]) -> List[str]:
        words = []
        word = []
        
        for c, tag in zip(text, tags):
            if tag == self.TAG_B:
                if word:
                    words.append(''.join(word))
                    word = []
                word.append(c)
            elif tag == self.TAG_M:
                word.append(c)
            elif tag == self.TAG_E:
                word.append(c)
                words.append(''.join(word))
                word = []
            elif tag == self.TAG_S:
                if word:
                    words.append(''.join(word))
                    word = []
                words.append(c)
        
        if word:
            words.append(''.join(word))
        
        return words
    
    def train(self, 
              texts: List[str], 
              val_texts: Optional[List[str]] = None,
              epochs: int = 10, 
              batch_size: int = 32, 
              learning_rate: float = 0.001,
              device: Optional[str] = None,
              verbose: bool = True):
        """训练"""
        # 构建词汇表
        self._build_vocab(texts)
        
        # 准备训练数据
        train_data = []
        for text in texts:
            ids = self._text_to_ids(text)
            tags = self._generate_tags(text)
            train_data.append((ids, tags))
        
        # 准备验证数据
        val_data = None
        if val_texts:
            val_data = []
            for text in val_texts:
                ids = self._text_to_ids(text)
                tags = self._generate_tags(text)
                val_data.append((ids, tags))
        
        # 训练
        self._model_class.train(
            train_data,
            val_data,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device,
            verbose=verbose
        )
        
        self._trained = True
    
    def segment(self, text: str) -> List[str]:
        """分词"""
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        
        if not text:
            return []
        
        # 预测
        ids = self._text_to_ids(text)
        tags = self._model_class.predict([ids])[0]
        
        return self._tags_to_words(text, tags)
    
    def save(self, model_path: str):
        """保存模型"""
        # 保存词汇表
        vocab_path = model_path + '.vocab'
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump({'vocab': self.vocab, 'vocab_size': self.vocab_size}, f)
        
        self._model_class.save(model_path)
    
    def load(self, model_path: str):
        """加载模型"""
        vocab_path = model_path + '.vocab'
        with open(vocab_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.vocab = data['vocab']
            self.vocab_size = data['vocab_size']
            self.inv_vocab = {v: k for k, v in self.vocab.items()}
        
        self._model_class.load(model_path)
        self._trained = True
    
    def _generate_tags(self, text: str) -> List[int]:
        """生成标签（B/M/E/S）"""
        return [self.TAG_S for _ in text]
