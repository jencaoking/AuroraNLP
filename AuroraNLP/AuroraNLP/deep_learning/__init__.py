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
    create_bert_pos
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
    'create_bert_pos'
]
