# Pre-trained Model Integration
# ===========================
# 集成 HuggingFace Transformers 等开源预训练模型

import importlib
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum

from .framework import get_framework


class PreTrainedModelType(Enum):
    """预训练模型类型枚举"""
    BERT = "bert"
    BERT_CHINESE = "bert_chinese"
    MACBERT = "macbert"
    ROBERTA_CHINESE = "roberta_chinese"
    ALBERT = "albert"
    DISTILBERT = "distilbert"
    ELECTRA = "electra"
    XLNET = "xlnet"


class PreTrainedModelConfig:
    """预训练模型配置"""
    
    # 常用中文预训练模型
    MODEL_CONFIGS = {
        PreTrainedModelType.BERT_CHINESE: "bert-base-chinese",
        PreTrainedModelType.MACBERT: "hfl/chinese-macbert-base",
        PreTrainedModelType.ROBERTA_CHINESE: "hfl/chinese-roberta-wwm-ext",
        PreTrainedModelType.ALBERT: "voidful/albert_chinese_tiny",
        PreTrainedModelType.DISTILBERT: "distilbert-base-uncased-finetuned-sst-2-english"
    }
    
    def __init__(
        self,
        model_type: PreTrainedModelType = PreTrainedModelType.BERT_CHINESE,
        model_name_or_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
        max_seq_length: int = 512,
        use_auth_token: Optional[str] = None,
        num_labels: int = 4
    ):
        """初始化预训练模型配置
        
        Args:
            model_type: 预训练模型类型
            model_name_or_path: 自定义模型名称或路径
            cache_dir: 模型缓存目录
            max_seq_length: 最大序列长度
            use_auth_token: HuggingFace Hub 认证 token
            num_labels: 分类标签数
        """
        self.model_type = model_type
        self.model_name_or_path = model_name_or_path or self.MODEL_CONFIGS.get(model_type)
        self.cache_dir = cache_dir
        self.max_seq_length = max_seq_length
        self.use_auth_token = use_auth_token
        self.num_labels = num_labels


class PreTrainedModelBase:
    """预训练模型基类"""
    
    def __init__(self, config: PreTrainedModelConfig):
        """初始化预训练模型基类"""
        self.config = config
        self._model = None
        self._tokenizer = None
        self._framework = get_framework()
        self._loaded = False
        self._transformers_available = self._check_transformers_available()
    
    def _check_transformers_available(self) -> bool:
        """检查 transformers 库是否可用"""
        try:
            import transformers
            return True
        except ImportError:
            return False
    
    def is_available(self) -> bool:
        """检查预训练模型是否可用"""
        return self._transformers_available and self._framework is not None
    
    @property
    def is_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self._loaded
    
    def _load_pretrained_components(self):
        """加载预训练模型和分词器（需子类实现）"""
        raise NotImplementedError
    
    def load(self):
        """加载模型"""
        if not self.is_available():
            raise RuntimeError("Pre-trained models not available. Install transformers and torch/tf.")
        
        self._load_pretrained_components()
        self._loaded = True
    
    def encode(self, text: str, **kwargs) -> Any:
        """文本编码"""
        raise NotImplementedError
    
    def save(self, save_dir: str):
        """保存模型"""
        raise NotImplementedError


class PreTrainedBERT(PreTrainedModelBase):
    """BERT 预训练模型集成（基于 HuggingFace Transformers）"""
    
    def __init__(self, config: PreTrainedModelConfig):
        """初始化 BERT 预训练模型"""
        super().__init__(config)
        self._torch = None
        self._bert = None
        self._tokenizer_cls = None
    
    def _load_pretrained_components(self):
        """加载 BERT 预训练模型和分词器"""
        # 动态导入库
        self._torch = importlib.import_module('torch')
        transformers = importlib.import_module('transformers')
        
        # 选择模型类
        model_cls = None
        tokenizer_cls = None
        
        model_type = self.config.model_type
        if model_type in [PreTrainedModelType.BERT, 
                         PreTrainedModelType.BERT_CHINESE,
                         PreTrainedModelType.MACBERT,
                         PreTrainedModelType.ROBERTA_CHINESE]:
            model_cls = transformers.BertForTokenClassification
            tokenizer_cls = transformers.BertTokenizer
        elif model_type == PreTrainedModelType.ALBERT:
            model_cls = transformers.AlbertForTokenClassification
            tokenizer_cls = transformers.AlbertTokenizer
        elif model_type == PreTrainedModelType.DISTILBERT:
            model_cls = transformers.DistilBertForTokenClassification
            tokenizer_cls = transformers.DistilBertTokenizer
        elif model_type == PreTrainedModelType.ELECTRA:
            model_cls = transformers.ElectraForTokenClassification
            tokenizer_cls = transformers.ElectraTokenizer
        elif model_type == PreTrainedModelType.XLNET:
            model_cls = transformers.XLNetForTokenClassification
            tokenizer_cls = transformers.XLNetTokenizer
        
        # 加载模型和分词器
        self._tokenizer = tokenizer_cls.from_pretrained(
            self.config.model_name_or_path,
            cache_dir=self.config.cache_dir,
            use_auth_token=self.config.use_auth_token
        )
        
        self._model = model_cls.from_pretrained(
            self.config.model_name_or_path,
            cache_dir=self.config.cache_dir,
            use_auth_token=self.config.use_auth_token,
            num_labels=self.config.num_labels
        )
        
        # 获取设备
        device = self._torch.device("cuda" if self._torch.cuda.is_available() else "cpu")
        self._model.to(device)
    
    def encode(self, text: str, **kwargs) -> Dict[str, Any]:
        """文本编码"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # 参数配置
        max_length = kwargs.get("max_length", self.config.max_seq_length)
        padding = kwargs.get("padding", "max_length")
        truncation = kwargs.get("truncation", True)
        return_tensors = kwargs.get("return_tensors", "pt")
        
        return self._tokenizer(
            text,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors
        )
    
    def encode_batch(self, texts: List[str], **kwargs) -> Dict[str, Any]:
        """批量文本编码"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        max_length = kwargs.get("max_length", self.config.max_seq_length)
        padding = kwargs.get("padding", "max_length")
        truncation = kwargs.get("truncation", True)
        return_tensors = kwargs.get("return_tensors", "pt")
        
        return self._tokenizer(
            texts,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors
        )
    
    def predict_tags(self, text: str, **kwargs) -> List[int]:
        """预测标签"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        self._model.eval()
        
        # 编码
        inputs = self.encode(text, return_tensors="pt")
        
        # 获取设备
        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 预测
        with self._torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits
            predictions = self._torch.argmax(logits, dim=2)
            return predictions.squeeze().tolist()
    
    def predict_tags_batch(self, texts: List[str], **kwargs) -> List[List[int]]:
        """批量预测标签"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        self._model.eval()
        
        inputs = self.encode_batch(texts, return_tensors="pt")
        
        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with self._torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits
            predictions = self._torch.argmax(logits, dim=2)
            return predictions.tolist()
    
    def save(self, save_dir: str):
        """保存模型和分词器"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        self._model.save_pretrained(save_dir)
        self._tokenizer.save_pretrained(save_dir)


class BERTChineseSegmentor:
    """基于 BERT 的中文分词器"""
    
    def __init__(
        self,
        model_type: PreTrainedModelType = PreTrainedModelType.BERT_CHINESE,
        model_name_or_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
        max_seq_length: int = 512
    ):
        """初始化 BERT 分词器"""
        # 配置
        self.config = PreTrainedModelConfig(
            model_type=model_type,
            model_name_or_path=model_name_or_path,
            cache_dir=cache_dir,
            max_seq_length=max_seq_length,
            num_labels=4  # B/M/E/S 标签
        )
        
        # 加载模型
        self._bert_model = PreTrainedBERT(self.config)
        self._loaded = False
    
    def is_available(self) -> bool:
        """检查是否可用"""
        return self._bert_model.is_available()
    
    @property
    def is_loaded(self) -> bool:
        """检查是否已加载"""
        return self._bert_model.is_loaded
    
    def load(self):
        """加载模型"""
        self._bert_model.load()
        self._loaded = True
    
    def segment(self, text: str) -> List[str]:
        """中文分词"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        if not text:
            return []
        
        # 预测标签
        tags = self._bert_model.predict_tags(text)
        
        # 标签转词（处理 CLS 和 SEP）
        # HuggingFace 分词器会增加 [CLS] 和 [SEP]，我们跳过
        if len(tags) > len(text):
            tags = tags[1:-1]  # 去掉 CLS 和 SEP
        
        # 使用 B/M/E/S 解码
        return self._decode_bmes(text, tags)
    
    def _decode_bmes(self, text: str, tags: List[int]) -> List[str]:
        """将 B/M/E/S 标签解码为分词结果
        
        Args:
            text: 原文本
            tags: 标签序列
            
        Returns:
            分词结果
        """
        words = []
        word = []
        
        for char, tag_idx in zip(text, tags):
            tag = tag_idx % 4  # 防止越界
            
            if tag == 0:  # B
                if word:
                    words.append(''.join(word))
                word = [char]
            elif tag == 1:  # M
                word.append(char)
            elif tag == 2:  # E
                word.append(char)
                words.append(''.join(word))
                word = []
            elif tag == 3:  # S
                if word:
                    words.append(''.join(word))
                words.append(char)
                word = []
        
        if word:
            words.append(''.join(word))
        
        return words
    
    def segment_batch(self, texts: List[str]) -> List[List[str]]:
        """批量中文分词"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        results = []
        for text in texts:
            results.append(self.segment(text))
        
        return results


def get_available_pretrained_models() -> List[str]:
    """获取可用的预训练模型列表"""
    return [
        "bert-base-chinese",
        "hfl/chinese-macbert-base",
        "hfl/chinese-roberta-wwm-ext",
        "voidful/albert_chinese_tiny"
    ]


def create_bert_segmentor(
    model_name_or_path: str = "bert-base-chinese",
    cache_dir: Optional[str] = None
) -> BERTChineseSegmentor:
    """创建 BERT 分词器（便捷函数）"""
    config = PreTrainedModelConfig(model_name_or_path=model_name_or_path, cache_dir=cache_dir)
    segmentor = BERTChineseSegmentor(model_name_or_path=model_name_or_path)
    return segmentor
