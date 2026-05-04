# Pre-trained Model Integration - Extended
# =======================================
# 集成 HuggingFace Transformers 等开源预训练模型
# 包含轻量级模型支持

import importlib
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum

from .framework import get_framework


class PreTrainedModelType(Enum):
    """预训练模型类型枚举"""
    # 标准模型
    BERT = "bert"
    BERT_CHINESE = "bert_chinese"
    MACBERT = "macbert"
    ROBERTA_CHINESE = "roberta_chinese"
    
    # 轻量级模型
    ALBERT = "albert"
    ALBERT_TINY = "albert_tiny"
    ALBERT_SMALL = "albert_small"
    DISTILBERT = "distilbert"
    TINY_BERT = "tiny_bert"
    MINI_LM = "mini_lm"
    
    # 其他模型
    ELECTRA = "electra"
    XLNET = "xlnet"


class PreTrainedModelConfig:
    """预训练模型配置"""
    
    # 模型配置字典：模型类型 -> (模型ID, 参数数量, 模型大小)
    MODEL_CONFIGS = {
        # 标准中文模型
        PreTrainedModelType.BERT_CHINESE: {
            "model_name": "bert-base-chinese",
            "params": "102M",
            "size": "~400MB",
            "speed": "慢",
            "accuracy": "高"
        },
        PreTrainedModelType.MACBERT: {
            "model_name": "hfl/chinese-macbert-base",
            "params": "102M",
            "size": "~400MB",
            "speed": "慢",
            "accuracy": "高"
        },
        PreTrainedModelType.ROBERTA_CHINESE: {
            "model_name": "hfl/chinese-roberta-wwm-ext",
            "params": "102M",
            "size": "~400MB",
            "speed": "慢",
            "accuracy": "高"
        },
        
        # 轻量级模型
        PreTrainedModelType.ALBERT_TINY: {
            "model_name": "voidful/albert_chinese_tiny",
            "params": "4M",
            "size": "~16MB",
            "speed": "极快",
            "accuracy": "中"
        },
        PreTrainedModelType.ALBERT_SMALL: {
            "model_name": "albert/albert-small-v2",
            "params": "12M",
            "size": "~50MB",
            "speed": "快",
            "accuracy": "中高"
        },
        PreTrainedModelType.ALBERT: {
            "model_name": "albert-base-v2",
            "params": "12M",
            "size": "~50MB",
            "speed": "快",
            "accuracy": "高"
        },
        PreTrainedModelType.DISTILBERT: {
            "model_name": "distilbert-base-uncased",
            "params": "66M",
            "size": "~260MB",
            "speed": "较快",
            "accuracy": "中高"
        },
        PreTrainedModelType.TINY_BERT: {
            "model_name": "cointegrated/rubert-tiny2",
            "params": "29M",
            "size": "~110MB",
            "speed": "快",
            "accuracy": "中"
        },
        PreTrainedModelType.MINI_LM: {
            "model_name": "sentence-transformers/paraphrase-MiniLM-L3-v2",
            "params": "22M",
            "size": "~90MB",
            "speed": "快",
            "accuracy": "中高"
        },
    }
    
    def __init__(
        self,
        model_type: PreTrainedModelType = PreTrainedModelType.ALBERT_TINY,
        model_name_or_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
        max_seq_length: int = 128,  # 轻量级模型通常用较短序列
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
        self.model_name_or_path = model_name_or_path or self.MODEL_CONFIGS.get(model_type, {}).get("model_name")
        self.cache_dir = cache_dir
        self.max_seq_length = max_seq_length
        self.use_auth_token = use_auth_token
        self.num_labels = num_labels
    
    def get_model_info(self) -> Dict[str, str]:
        """获取模型信息"""
        return self.MODEL_CONFIGS.get(self.model_type, {})
    
    def is_lightweight(self) -> bool:
        """判断是否为轻量级模型"""
        lightweight_types = [
            PreTrainedModelType.ALBERT_TINY,
            PreTrainedModelType.ALBERT_SMALL,
            PreTrainedModelType.DISTILBERT,
            PreTrainedModelType.TINY_BERT,
            PreTrainedModelType.MINI_LM
        ]
        return self.model_type in lightweight_types


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
        elif model_type in [PreTrainedModelType.ALBERT,
                          PreTrainedModelType.ALBERT_TINY,
                          PreTrainedModelType.ALBERT_SMALL]:
            model_cls = transformers.AlbertForTokenClassification
            tokenizer_cls = transformers.AlbertTokenizerFast
        elif model_type == PreTrainedModelType.DISTILBERT:
            model_cls = transformers.DistilBertForTokenClassification
            tokenizer_cls = transformers.DistilBertTokenizer
        elif model_type == PreTrainedModelType.TINY_BERT:
            model_cls = transformers.BertForTokenClassification
            tokenizer_cls = transformers.BertTokenizer
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
        
        inputs = self.encode(text, return_tensors="pt")
        
        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
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
        self.config = PreTrainedModelConfig(
            model_type=model_type,
            model_name_or_path=model_name_or_path,
            cache_dir=cache_dir,
            max_seq_length=max_seq_length,
            num_labels=4  # B/M/E/S 标签
        )
        
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
        
        tags = self._bert_model.predict_tags(text)
        
        if len(tags) > len(text):
            tags = tags[1:-1]  # 去掉 CLS 和 SEP
        
        return self._decode_bmes(text, tags)
    
    def _decode_bmes(self, text: str, tags: List[int]) -> List[str]:
        """将 B/M/E/S 标签解码为分词结果"""
        words = []
        word = []
        
        for char, tag_idx in zip(text, tags):
            tag = tag_idx % 4
            
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


# ==================== 轻量级模型专有类 ====================

class LightweightSegmentor:
    """轻量级分词器工厂类"""
    
    @staticmethod
    def create_albert_tiny() -> BERTChineseSegmentor:
        """创建 ALBERT Tiny 分词器（4M参数，极轻量）"""
        return BERTChineseSegmentor(
            model_type=PreTrainedModelType.ALBERT_TINY,
            max_seq_length=128
        )
    
    @staticmethod
    def create_albert_small() -> BERTChineseSegmentor:
        """创建 ALBERT Small 分词器（12M参数，轻量）"""
        return BERTChineseSegmentor(
            model_type=PreTrainedModelType.ALBERT_SMALL,
            max_seq_length=128
        )
    
    @staticmethod
    def create_distilbert() -> BERTChineseSegmentor:
        """创建 DistilBERT 分词器（66M参数，较轻量）"""
        return BERTChineseSegmentor(
            model_type=PreTrainedModelType.DISTILBERT,
            max_seq_length=256
        )
    
    @staticmethod
    def create_tiny_bert() -> BERTChineseSegmentor:
        """创建 TinyBERT 分词器（29M参数，轻量）"""
        return BERTChineseSegmentor(
            model_type=PreTrainedModelType.TINY_BERT,
            max_seq_length=128
        )
    
    @staticmethod
    def create_by_name(model_name: str) -> Optional[BERTChineseSegmentor]:
        """根据模型名称创建分词器"""
        name_to_type = {
            "albert_tiny": PreTrainedModelType.ALBERT_TINY,
            "albert_small": PreTrainedModelType.ALBERT_SMALL,
            "distilbert": PreTrainedModelType.DISTILBERT,
            "tiny_bert": PreTrainedModelType.TINY_BERT,
            "bert_chinese": PreTrainedModelType.BERT_CHINESE,
            "macbert": PreTrainedModelType.MACBERT,
            "roberta": PreTrainedModelType.ROBERTA_CHINESE,
        }
        
        model_type = name_to_type.get(model_name.lower())
        if model_type:
            return BERTChineseSegmentor(model_type=model_type)
        return None


class ModelComparator:
    """模型比较工具"""
    
    def __init__(self):
        self.results = []
    
    def add_result(self, model_type: PreTrainedModelType, accuracy: float, speed: float, memory: float):
        """添加比较结果
        
        Args:
            model_type: 模型类型
            accuracy: 准确率 (0-100)
            speed: 速度 (tokens/second)
            memory: 内存占用 (MB)
        """
        info = PreTrainedModelConfig.MODEL_CONFIGS.get(model_type, {})
        self.results.append({
            "type": model_type,
            "name": info.get("model_name", ""),
            "params": info.get("params", ""),
            "size": info.get("size", ""),
            "accuracy": accuracy,
            "speed": speed,
            "memory": memory
        })
    
    def get_comparison_table(self) -> str:
        """获取比较表格"""
        if not self.results:
            return "No results yet."
        
        header = f"{'Model':<30} {'Params':<10} {'Size':<12} {'Accuracy':<12} {'Speed':<15} {'Memory':<10}"
        lines = [header, "-" * 100]
        
        for r in self.results:
            line = f"{r['name']:<30} {r['params']:<10} {r['size']:<12} {r['accuracy']:.1f}%{'':<6} {r['speed']:.1f} tok/s{'':<5} {r['memory']:.1f} MB"
            lines.append(line)
        
        return "\n".join(lines)
    
    def recommend_model(self, priority: str = "balanced") -> PreTrainedModelType:
        """推荐最适合的模型
        
        Args:
            priority: 优先级 ("speed", "accuracy", "balanced")
            
        Returns:
            推荐的模型类型
        """
        if priority == "speed":
            return PreTrainedModelType.ALBERT_TINY
        elif priority == "accuracy":
            return PreTrainedModelType.BERT_CHINESE
        else:  # balanced
            return PreTrainedModelType.ALBERT_SMALL


# ==================== 便捷函数 ====================

def get_available_pretrained_models() -> List[str]:
    """获取可用的预训练模型列表"""
    models = []
    for info in PreTrainedModelConfig.MODEL_CONFIGS.values():
        models.append(info["model_name"])
    return models


def get_lightweight_models() -> List[Tuple[str, str, str]]:
    """获取轻量级模型列表
    
    Returns:
        List of (model_type, model_name, description)
    """
    lightweight_types = [
        PreTrainedModelType.ALBERT_TINY,
        PreTrainedModelType.ALBERT_SMALL,
        PreTrainedModelType.DISTILBERT,
        PreTrainedModelType.TINY_BERT
    ]
    
    result = []
    for mt in lightweight_types:
        info = PreTrainedModelConfig.MODEL_CONFIGS.get(mt, {})
        if info:
            result.append((mt.value, info["model_name"], f"{info['params']}, {info['speed']}"))
    
    return result


def create_bert_segmentor(
    model_name_or_path: str = "bert-base-chinese",
    cache_dir: Optional[str] = None
) -> BERTChineseSegmentor:
    """创建 BERT 分词器（便捷函数）"""
    config = PreTrainedModelConfig(model_name_or_path=model_name_or_path, cache_dir=cache_dir)
    segmentor = BERTChineseSegmentor(model_name_or_path=model_name_or_path)
    return segmentor


def create_lightweight_segmentor(model_type: str = "albert_tiny") -> Optional[BERTChineseSegmentor]:
    """创建轻量级分词器（便捷函数）"""
    return LightweightSegmentor.create_by_name(model_type)


# ==================== BERT-NER: 命名实体识别（步骤 40） ====================

# 使用已定义的实体类型，保持与项目一致性
NER_ENTITY_TYPES = {
    'PER': '人名',
    'LOC': '地名',
    'ORG': '机构名',
    'TIME': '时间',
    'NUM': '数值',
    'MISC': '其他实体',
}

# 定义 NER 标签
NER_LABELS = ['O']
for entity_type in NER_ENTITY_TYPES.keys():
    NER_LABELS.extend([
        f'B-{entity_type}',
        f'I-{entity_type}',
    ])


class NEREntity:
    """实体识别结果"""
    def __init__(
        self,
        text: str,
        entity_type: str,
        start: int,
        end: int,
        confidence: float = 1.0
    ):
        self.text = text
        self.entity_type = entity_type
        self.start = start
        self.end = end
        self.confidence = confidence
    
    def __repr__(self):
        return f"NEREntity('{self.text}', {self.entity_type}, [{self.start}:{self.end}])"
    
    def to_dict(self):
        return {
            'text': self.text,
            'type': self.entity_type,
            'type_name': NER_ENTITY_TYPES.get(self.entity_type, '未知'),
            'start': self.start,
            'end': self.end,
            'confidence': self.confidence
        }


class BERTNER:
    """基于 BERT 的命名实体识别器"""
    
    def __init__(
        self,
        model_type: PreTrainedModelType = PreTrainedModelType.BERT_CHINESE,
        model_name_or_path: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """初始化 BERT-NER
        
        Args:
            model_type: 预训练模型类型
            model_name_or_path: 自定义模型名称或路径
            cache_dir: 模型缓存目录
        """
        self.config = PreTrainedModelConfig(
            model_type=model_type,
            model_name_or_path=model_name_or_path,
            cache_dir=cache_dir,
            num_labels=len(NER_LABELS)
        )
        
        self._model = PreTrainedBERT(self.config)
        self._loaded = False
    
    @property
    def is_available(self):
        """检查是否可用"""
        return self._model.is_available()
    
    @property
    def is_loaded(self):
        """检查是否已加载"""
        return self._model.is_loaded
    
    def load(self):
        """加载模型"""
        self._model.load()
        self._loaded = True
    
    def predict(self, text: str) -> List[NEREntity]:
        """命名实体识别
        
        Args:
            text: 输入文本
            
        Returns:
            实体列表
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # 预测标签
        labels = self._model.predict_tags(text)
        
        # 对齐标签和文本（处理 CLS 和 SEP）
        if len(labels) > len(text):
            labels = labels[1:-1]
        
        # 解析标签，提取实体
        entities = []
        current_entity = None
        
        for idx, (char, label) in enumerate(zip(text, labels)):
            if isinstance(label, int):
                label_str = NER_LABELS[label % len(NER_LABELS)]
            else:
                label_str = label
            
            if label_str.startswith('B-'):
                if current_entity:
                    entities.append(current_entity)
                entity_type = label_str[2:]
                current_entity = NEREntity(
                    text=char,
                    entity_type=entity_type,
                    start=idx,
                    end=idx+1,
                    confidence=0.8
                )
            elif label_str.startswith('I-'):
                if current_entity:
                    entity_type = label_str[2:]
                    if entity_type == current_entity.entity_type:
                        current_entity.text += char
                        current_entity.end = idx + 1
                    else:
                        entities.append(current_entity)
                        current_entity = NEREntity(
                            text=char,
                            entity_type=entity_type,
                            start=idx,
                            end=idx+1,
                            confidence=0.7
                        )
            else:
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
        
        if current_entity:
            entities.append(current_entity)
        
        return entities
    
    def predict_batch(self, texts: List[str]) -> List[List[NEREntity]]:
        """批量实体识别"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        return [self.predict(text) for text in texts]
    
    def save(self, save_dir: str):
        """保存模型"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        self._model.save(save_dir)


def create_bert_ner(
    model_name_or_path: str = "bert-base-chinese",
    cache_dir: Optional[str] = None
) -> BERTNER:
    """创建 BERT-NER（便捷函数）"""
    config = PreTrainedModelConfig(model_name_or_path=model_name_or_path, cache_dir=cache_dir)
    return BERTNER(model_name_or_path=model_name_or_path)


# ==================== BERT-POS: 词性标注（步骤 41） ====================

# 使用项目中已定义的词性标签，保持一致性
POS_LABELS = [
    'n', 'nr', 'ns', 'nt', 'nz', 
    'v', 'vd', 'vn', 
    'a', 'ad', 'an', 
    'd', 'm', 'q', 
    'r', 'p', 'c', 'u', 
    'xc', 'w', 'f', 's', 
    't', 'b', 'z', 
    'e', 'y', 'o', 
    'l', 'i', 'j', 
    'h', 'k', 'g', 'x'
]

POS_LABEL_NAMES = {
    'n': '名词',
    'nr': '人名',
    'ns': '地名',
    'nt': '机构团体',
    'nz': '其他专名',
    'v': '动词',
    'vd': '副动词',
    'vn': '名动词',
    'a': '形容词',
    'ad': '副形词',
    'an': '名形词',
    'd': '副词',
    'm': '数词',
    'q': '量词',
    'r': '代词',
    'p': '介词',
    'c': '连词',
    'u': '助词',
    'xc': '其他功能词',
    'w': '标点符号',
    'f': '方位词',
    's': '处所词',
    't': '时间词',
    'b': '区别词',
    'z': '状态词',
    'e': '叹词',
    'y': '语气词',
    'o': '拟声词',
    'l': '习用语',
    'i': '成语',
    'j': '简称',
    'h': '前缀',
    'k': '后缀',
    'g': '语素',
    'x': '非语素字',
}


class POSResult:
    """词性标注结果"""
    def __init__(
        self,
        word: str,
        pos_tag: str,
        confidence: float = 1.0
    ):
        self.word = word
        self.pos_tag = pos_tag
        self.confidence = confidence
    
    @property
    def pos_name(self):
        """词性名称"""
        return POS_LABEL_NAMES.get(self.pos_tag, '未知')
    
    def __repr__(self):
        return f"POSResult('{self.word}', {self.pos_tag}/{self.pos_name})"
    
    def to_dict(self):
        return {
            'word': self.word,
            'tag': self.pos_tag,
            'tag_name': self.pos_name,
            'confidence': self.confidence
        }


class BERTPOS:
    """基于 BERT 的词性标注器"""
    
    def __init__(
        self,
        model_type: PreTrainedModelType = PreTrainedModelType.BERT_CHINESE,
        model_name_or_path: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """初始化 BERT-POS
        
        Args:
            model_type: 预训练模型类型
            model_name_or_path: 自定义模型名称或路径
            cache_dir: 模型缓存目录
        """
        self.config = PreTrainedModelConfig(
            model_type=model_type,
            model_name_or_path=model_name_or_path,
            cache_dir=cache_dir,
            num_labels=len(POS_LABELS)
        )
        
        self._model = PreTrainedBERT(self.config)
        self._loaded = False
    
    @property
    def is_available(self):
        """检查是否可用"""
        return self._model.is_available()
    
    @property
    def is_loaded(self):
        """检查是否已加载"""
        return self._model.is_loaded
    
    def load(self):
        """加载模型"""
        self._model.load()
        self._loaded = True
    
    def tag(self, words: List[str]) -> List[POSResult]:
        """词性标注（单词级别）
        
        Args:
            words: 分词后的单词列表
            
        Returns:
            词性标注结果列表
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # 简单实现：对于每个单词，预测词性
        results = []
        for word in words:
            if not word:
                continue
            
            # 预测标签
            tags = self._model.predict_tags(word)
            
            # 选择最常见的标签
            if tags:
                label_idx = tags[len(tags)//2] % len(POS_LABELS)
                tag = POS_LABELS[label_idx]
            else:
                tag = 'n'
            
            results.append(POSResult(
                word=word,
                pos_tag=tag,
                confidence=0.75
            ))
        
        return results
    
    def tag_text(self, text: str, segmentor=None) -> List[POSResult]:
        """完整文本的词性标注（先分词，再标注）
        
        Args:
            text: 文本
            segmentor: 分词器（可选，默认使用双向最大匹配）
            
        Returns:
            词性标注结果列表
        """
        # 如果没有分词器，简单地按字分割
        if segmentor is None:
            words = list(text)
        else:
            words = segmentor.segment(text)
        
        return self.tag(words)
    
    def tag_batch(self, word_lists: List[List[str]]) -> List[List[POSResult]]:
        """批量词性标注"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        return [self.tag(words) for words in word_lists]
    
    def save(self, save_dir: str):
        """保存模型"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        self._model.save(save_dir)


def create_bert_pos(
    model_name_or_path: str = "bert-base-chinese",
    cache_dir: Optional[str] = None
) -> BERTPOS:
    """创建 BERT-POS（便捷函数）"""
    config = PreTrainedModelConfig(model_name_or_path=model_name_or_path, cache_dir=cache_dir)
    return BERTPOS(model_name_or_path=model_name_or_path)
