# Deep Learning Module
# ===================

from .framework import Framework, FrameworkType, get_framework
from .pytorch_backend import PyTorchBackend
from .tensorflow_backend import TensorFlowBackend
from .bilstm_crf import BiLSTMCRF
from .pretrained import (
    PreTrainedModelType,
    PreTrainedModelConfig,
    PreTrainedModelBase,
    PreTrainedBERT,
    BERTChineseSegmentor,
    LightweightSegmentor,
    ModelComparator,
    get_available_pretrained_models,
    get_lightweight_models,
    create_bert_segmentor,
    create_lightweight_segmentor,
    # BERT-NER (步骤 40)
    BERTNER,
    NEREntity,
    NER_ENTITY_TYPES,
    NER_LABELS,
    create_bert_ner,
    # BERT-POS (步骤 41)
    BERTPOS,
    POSResult,
    POS_LABELS,
    POS_LABEL_NAMES,
    create_bert_pos,
    # BERT-情感分析 (步骤 42)
    BERTSentiment,
    SentimentResult,
    create_bert_sentiment,
    # BERT-文本分类 (步骤 43)
    BERTClassifier,
    ClassificationResult,
    CLASSIFICATION_LABELS,
    create_bert_classifier,
    # 步骤 44: 模型微调接口
    FineTuningConfig,
    FineTuningTrainer,
    create_finetuning_config,
    # 步骤 45: 迁移学习框架
    FewShotLearningConfig,
    FewShotLearner,
    create_fewshot_learner,
    # 步骤 46: 知识蒸馏
    KnowledgeDistillationConfig,
    KnowledgeDistiller,
    create_knowledge_distiller,
    # 步骤 47: 模型量化
    QuantizationConfig,
    ModelQuantizer,
    create_quantizer,
    # 步骤 48: ONNX导出
    ONNXExportConfig,
    ONNXExporter,
    create_onnx_exporter,
    # 步骤 49: 模型热加载
    HotLoadConfig,
    HotModelLoader,
    create_hot_loader,
    # 步骤 50: 模型管理系统
    ModelVersion,
    ModelCacheConfig,
    ModelManager,
    create_model_manager
)

__all__ = [
    'Framework',
    'FrameworkType',
    'get_framework',
    'PyTorchBackend',
    'TensorFlowBackend',
    'BiLSTMCRF',
    'PreTrainedModelType',
    'PreTrainedModelConfig',
    'PreTrainedModelBase',
    'PreTrainedBERT',
    'BERTChineseSegmentor',
    'LightweightSegmentor',
    'ModelComparator',
    'get_available_pretrained_models',
    'get_lightweight_models',
    'create_bert_segmentor',
    'create_lightweight_segmentor',
    # BERT-NER
    'BERTNER',
    'NEREntity',
    'NER_ENTITY_TYPES',
    'NER_LABELS',
    'create_bert_ner',
    # BERT-POS
    'BERTPOS',
    'POSResult',
    'POS_LABELS',
    'POS_LABEL_NAMES',
    'create_bert_pos',
    # BERT-情感分析
    'BERTSentiment',
    'SentimentResult',
    'create_bert_sentiment',
    # BERT-文本分类
    'BERTClassifier',
    'ClassificationResult',
    'CLASSIFICATION_LABELS',
    'create_bert_classifier',
    # 步骤 44: 模型微调接口
    'FineTuningConfig',
    'FineTuningTrainer',
    'create_finetuning_config',
    # 步骤 45: 迁移学习框架
    'FewShotLearningConfig',
    'FewShotLearner',
    'create_fewshot_learner',
    # 步骤 46: 知识蒸馏
    'KnowledgeDistillationConfig',
    'KnowledgeDistiller',
    'create_knowledge_distiller',
    # 步骤 47: 模型量化
    'QuantizationConfig',
    'ModelQuantizer',
    'create_quantizer',
    # 步骤 48: ONNX导出
    'ONNXExportConfig',
    'ONNXExporter',
    'create_onnx_exporter',
    # 步骤 49: 模型热加载
    'HotLoadConfig',
    'HotModelLoader',
    'create_hot_loader',
    # 步骤 50: 模型管理系统
    'ModelVersion',
    'ModelCacheConfig',
    'ModelManager',
    'create_model_manager'
]
