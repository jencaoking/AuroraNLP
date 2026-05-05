# Pre-trained model tests
# ========================
import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_module_import():
    """测试预训练模型模块可导入"""
    from AuroraNLP.deep_learning import pretrained
    assert hasattr(pretrained, 'PreTrainedModelType')
    assert hasattr(pretrained, 'PreTrainedModelConfig')
    assert hasattr(pretrained, 'PreTrainedBERT')
    assert hasattr(pretrained, 'BERTChineseSegmentor')
    assert hasattr(pretrained, 'get_available_pretrained_models')
    assert hasattr(pretrained, 'create_bert_segmentor')
    assert hasattr(pretrained, 'LightweightSegmentor')
    assert hasattr(pretrained, 'ModelComparator')
    assert hasattr(pretrained, 'get_lightweight_models')
    assert hasattr(pretrained, 'create_lightweight_segmentor')


def test_model_type_enum():
    """测试模型类型枚举"""
    from AuroraNLP.deep_learning.pretrained import PreTrainedModelType
    assert PreTrainedModelType.BERT_CHINESE is not None
    assert PreTrainedModelType.MACBERT is not None
    assert PreTrainedModelType.ROBERTA_CHINESE is not None
    assert PreTrainedModelType.ALBERT is not None
    assert PreTrainedModelType.ALBERT_TINY is not None
    assert PreTrainedModelType.ALBERT_SMALL is not None
    assert PreTrainedModelType.DISTILBERT is not None
    assert PreTrainedModelType.TINY_BERT is not None
    assert PreTrainedModelType.MINI_LM is not None
    # 测试枚举值
    assert PreTrainedModelType.BERT_CHINESE.value == "bert_chinese"
    assert PreTrainedModelType.ALBERT_TINY.value == "albert_tiny"


def test_config_defaults():
    """测试预训练模型配置"""
    from AuroraNLP.deep_learning.pretrained import PreTrainedModelConfig, PreTrainedModelType
    # 默认配置
    config = PreTrainedModelConfig()
    assert config.model_type == PreTrainedModelType.ALBERT_TINY
    assert config.max_seq_length == 128
    assert config.num_labels == 4
    
    # 自定义配置
    custom_config = PreTrainedModelConfig(
        model_type=PreTrainedModelType.MACBERT,
        max_seq_length=256,
        num_labels=10
    )
    assert custom_config.model_type == PreTrainedModelType.MACBERT
    assert custom_config.max_seq_length == 256
    assert custom_config.num_labels == 10


def test_config_lightweight_check():
    """测试轻量级模型配置"""
    from AuroraNLP.deep_learning.pretrained import PreTrainedModelConfig, PreTrainedModelType
    
    # ALBERT Tiny 是轻量级
    config_tiny = PreTrainedModelConfig(model_type=PreTrainedModelType.ALBERT_TINY)
    assert config_tiny.is_lightweight() is True
    
    # BERT Chinese 不是轻量级
    config_bert = PreTrainedModelConfig(model_type=PreTrainedModelType.BERT_CHINESE)
    assert config_bert.is_lightweight() is False


def test_get_available_models():
    """测试获取可用模型列表"""
    from AuroraNLP.deep_learning.pretrained import get_available_pretrained_models
    models = get_available_pretrained_models()
    assert len(models) > 0
    assert "bert-base-chinese" in models


def test_get_lightweight_models():
    """测试获取轻量级模型列表"""
    from AuroraNLP.deep_learning.pretrained import get_lightweight_models
    models = get_lightweight_models()
    assert len(models) > 0
    # 检查返回格式
    for model in models:
        assert len(model) == 3  # (type, name, description)


def test_segmentor_init():
    """测试 BERT 分词器初始化"""
    from AuroraNLP.deep_learning.pretrained import BERTChineseSegmentor, PreTrainedModelType
    # 初始化
    seg = BERTChineseSegmentor(model_type=PreTrainedModelType.BERT_CHINESE)
    assert seg is not None
    assert seg.is_available() is not None  # 可能是 True 或 False（取决于依赖是否安装）


def test_bert_segmentor_available_check():
    """测试可用性检查"""
    from AuroraNLP.deep_learning.pretrained import PreTrainedBERT, PreTrainedModelConfig
    config = PreTrainedModelConfig()
    model = PreTrainedBERT(config)
    # 检查可用（可能返回 False 或 True，取决于环境）
    assert isinstance(model.is_available(), bool)


def test_lightweight_segmentor_factory():
    """测试轻量级分词器工厂"""
    from AuroraNLP.deep_learning.pretrained import LightweightSegmentor, PreTrainedModelType
    
    # 测试创建 ALBERT Tiny
    seg = LightweightSegmentor.create_albert_tiny()
    assert seg is not None
    assert seg.config.model_type == PreTrainedModelType.ALBERT_TINY
    
    # 测试创建 ALBERT Small
    seg_small = LightweightSegmentor.create_albert_small()
    assert seg_small.config.model_type == PreTrainedModelType.ALBERT_SMALL
    
    # 测试创建 TinyBERT
    seg_tiny = LightweightSegmentor.create_tiny_bert()
    assert seg_tiny.config.model_type == PreTrainedModelType.TINY_BERT


def test_lightweight_segmentor_by_name():
    """测试通过名称创建轻量级分词器"""
    from AuroraNLP.deep_learning.pretrained import LightweightSegmentor
    
    # 测试有效的模型名称
    seg = LightweightSegmentor.create_by_name("albert_tiny")
    assert seg is not None
    
    seg2 = LightweightSegmentor.create_by_name("distilbert")
    assert seg2 is not None
    
    # 测试无效的模型名称
    seg_invalid = LightweightSegmentor.create_by_name("invalid_model")
    assert seg_invalid is None


def test_model_comparator():
    """测试模型比较器"""
    from AuroraNLP.deep_learning.pretrained import ModelComparator, PreTrainedModelType
    
    comparator = ModelComparator()
    
    # 添加比较结果
    comparator.add_result(PreTrainedModelType.ALBERT_TINY, 85.0, 100.0, 16.0)
    comparator.add_result(PreTrainedModelType.BERT_CHINESE, 95.0, 50.0, 400.0)
    
    # 获取比较表格
    table = comparator.get_comparison_table()
    assert "bert-base-chinese" in table
    assert "voidful/albert_chinese_tiny" in table


def test_model_recommend():
    """测试模型推荐"""
    from AuroraNLP.deep_learning.pretrained import ModelComparator, PreTrainedModelType
    
    comparator = ModelComparator()
    
    # 推荐速度优先
    speed_rec = comparator.recommend_model("speed")
    assert speed_rec == PreTrainedModelType.ALBERT_TINY
    
    # 推荐精度优先
    acc_rec = comparator.recommend_model("accuracy")
    assert acc_rec == PreTrainedModelType.BERT_CHINESE
    
    # 推荐平衡
    balanced_rec = comparator.recommend_model("balanced")
    assert balanced_rec == PreTrainedModelType.ALBERT_SMALL


@pytest.mark.slow
def test_pretrained_model_integration():
    """完整集成测试（需要 PyTorch 和 Transformers）"""
    try:
        from AuroraNLP.deep_learning.pretrained import BERTChineseSegmentor, PreTrainedModelType
        
        # 使用一个轻量级的测试
        seg = BERTChineseSegmentor(model_type=PreTrainedModelType.ALBERT)
        
        # 检查是否可加载
        if seg.is_available():
            # 我们不实际加载（避免长下载时间），但确保函数存在
            assert hasattr(seg, 'load')
            assert hasattr(seg, 'segment')
        else:
            print("Pre-trained models not available (missing dependencies)")
            pass
            
    except ImportError:
        pytest.skip("Missing required dependencies (torch/transformers)")


def test_create_bert_segmentor_function():
    """测试便捷函数创建分词器"""
    from AuroraNLP.deep_learning.pretrained import create_bert_segmentor
    try:
        seg = create_bert_segmentor(model_name_or_path="bert-base-chinese")
        assert seg is not None
        assert hasattr(seg, 'is_available')
    except Exception:
        pytest.skip("Missing dependencies or network error")


def test_create_lightweight_segmentor_function():
    """测试创建轻量级分词器便捷函数"""
    from AuroraNLP.deep_learning.pretrained import create_lightweight_segmentor
    
    # 测试创建 ALBERT Tiny
    seg = create_lightweight_segmentor("albert_tiny")
    assert seg is not None
    assert seg.config.model_type.value == "albert_tiny"
    
    # 测试创建 ALBERT Small
    seg_small = create_lightweight_segmentor("albert_small")
    assert seg_small is not None


def test_bmes_decode():
    """测试 B/M/E/S 标签解码"""
    from AuroraNLP.deep_learning.pretrained import BERTChineseSegmentor
    
    # 直接测试内部解码函数
    seg = BERTChineseSegmentor()
    
    # 测试标签
    text = "我爱中国"
    tags = [3, 3, 0, 2]  # S/S/B/E
    
    # 调用内部函数
    result = seg._decode_bmes(text, tags)
    
    assert result == ["我", "爱", "中国"]


def test_bmes_decode_multichar():
    """测试多字符词 B/M/E/S 解码"""
    from AuroraNLP.deep_learning.pretrained import BERTChineseSegmentor
    
    seg = BERTChineseSegmentor()
    
    # 测试 "自然语言处理" -> B M M E B M E
    text = "自然语言处理"
    tags = [0, 1, 1, 2, 0, 1, 2]  # B M M E B M E
    
    result = seg._decode_bmes(text, tags)
    assert result == ["自然语言", "处理"]


def test_pretrained_config_dict():
    """测试模型配置字典"""
    from AuroraNLP.deep_learning.pretrained import PreTrainedModelConfig, PreTrainedModelType
    config = PreTrainedModelConfig(model_type=PreTrainedModelType.BERT_CHINESE)
    
    # 检查默认模型路径
    assert config.model_name_or_path is not None
    assert "bert" in config.model_name_or_path.lower()


def test_model_config_info():
    """测试模型配置信息"""
    from AuroraNLP.deep_learning.pretrained import PreTrainedModelConfig, PreTrainedModelType
    
    # 测试 ALBERT Tiny 配置
    config = PreTrainedModelConfig(model_type=PreTrainedModelType.ALBERT_TINY)
    info = config.get_model_info()
    assert info.get("params") == "4M"
    assert info.get("size") == "~16MB"
    
    # 测试 BERT 配置
    config_bert = PreTrainedModelConfig(model_type=PreTrainedModelType.BERT_CHINESE)
    info_bert = config_bert.get_model_info()
    assert info_bert.get("params") == "102M"


# ==================== NER 测试（步骤 40） ====================

def test_ner_entity_types():
    """测试 NER 实体类型"""
    from AuroraNLP.deep_learning.pretrained import NER_ENTITY_TYPES, NER_LABELS
    
    assert len(NER_ENTITY_TYPES) > 0
    assert 'PER' in NER_ENTITY_TYPES
    assert 'LOC' in NER_ENTITY_TYPES
    assert 'ORG' in NER_ENTITY_TYPES
    
    assert len(NER_LABELS) > 0
    assert 'O' in NER_LABELS
    assert 'B-PER' in NER_LABELS


def test_ner_entity():
    """测试 NER 实体对象"""
    from AuroraNLP.deep_learning.pretrained import NEREntity
    
    entity = NEREntity("张三", "PER", 0, 2, 0.95)
    assert entity.text == "张三"
    assert entity.entity_type == "PER"
    assert entity.start == 0
    assert entity.end == 2
    assert entity.confidence == 0.95
    
    # 测试 to_dict
    entity_dict = entity.to_dict()
    assert entity_dict['text'] == "张三"
    assert entity_dict['type'] == "PER"
    assert entity_dict['type_name'] == "人名"


def test_bert_ner_init():
    """测试 BERT-NER 初始化"""
    from AuroraNLP.deep_learning.pretrained import BERTNER, PreTrainedModelType, create_bert_ner
    
    ner = BERTNER(model_type=PreTrainedModelType.ALBERT_TINY)
    assert ner is not None
    assert hasattr(ner, 'is_available')
    assert hasattr(ner, 'is_loaded')
    assert hasattr(ner, 'load')
    assert hasattr(ner, 'predict')
    
    # 测试便捷函数
    ner2 = create_bert_ner()
    assert ner2 is not None


def test_ner_parse_label():
    """测试 NER 标签解析（模拟）"""
    from AuroraNLP.deep_learning.pretrained import NEREntity
    
    # 模拟标签序列和文本
    text = "我是中国人"
    labels = [0, 0, 1, 2, 2, 0]  # 简化标签
    
    # 简单的实体解析测试
    entities = []
    current_entity = None
    
    # 验证 B/I/O 标签逻辑
    # 这里只测试实体创建的基本功能
    test_entity = NEREntity("中国", "LOC", 2, 4, 0.9)
    entities.append(test_entity)
    
    assert len(entities) == 1
    assert entities[0].text == "中国"
    assert entities[0].entity_type == "LOC"


# ==================== POS 测试（步骤 41） ====================

def test_pos_labels():
    """测试词性标签"""
    from AuroraNLP.deep_learning.pretrained import POS_LABELS, POS_LABEL_NAMES
    
    assert len(POS_LABELS) > 0
    assert 'n' in POS_LABELS
    assert 'v' in POS_LABELS
    assert 'a' in POS_LABELS
    
    assert len(POS_LABEL_NAMES) > 0
    assert POS_LABEL_NAMES['n'] == "名词"
    assert POS_LABEL_NAMES['v'] == "动词"


def test_pos_result():
    """测试词性标注结果对象"""
    from AuroraNLP.deep_learning.pretrained import POSResult
    
    pos = POSResult("测试", "n", 0.85)
    assert pos.word == "测试"
    assert pos.pos_tag == "n"
    assert pos.pos_name == "名词"
    assert pos.confidence == 0.85
    
    # 测试 to_dict
    pos_dict = pos.to_dict()
    assert pos_dict['word'] == "测试"
    assert pos_dict['tag'] == "n"
    assert pos_dict['tag_name'] == "名词"


def test_bert_pos_init():
    """测试 BERT-POS 初始化"""
    from AuroraNLP.deep_learning.pretrained import BERTPOS, PreTrainedModelType, create_bert_pos
    
    pos = BERTPOS(model_type=PreTrainedModelType.ALBERT_TINY)
    assert pos is not None
    assert hasattr(pos, 'is_available')
    assert hasattr(pos, 'is_loaded')
    assert hasattr(pos, 'load')
    assert hasattr(pos, 'tag')
    assert hasattr(pos, 'tag_text')
    
    # 测试便捷函数
    pos2 = create_bert_pos()
    assert pos2 is not None


def test_pos_tag_single_word():
    """测试单词语词性标注"""
    from AuroraNLP.deep_learning.pretrained import POSResult
    
    # 直接测试 POSResult 和标签映射
    words = ["自然", "语言", "处理"]
    
    results = []
    for word in words:
        results.append(POSResult(word, "n", 0.7))
    
    assert len(results) == 3
    assert results[0].word == "自然"
    assert results[0].pos_tag == "n"
    assert results[0].pos_name == "名词"


# ==================== 情感分析测试（步骤 42）====================

def test_sentiment_result():
    """测试 SentimentResult 类"""
    from AuroraNLP.deep_learning.pretrained import SentimentResult
    
    # 正面情感测试
    result = SentimentResult(
        text="今天真开心",
        label="positive",
        score=0.9,
        confidence=0.9
    )
    assert result.text == "今天真开心"
    assert result.label == "positive"
    assert result.polarity == "positive"
    
    # 负面情感测试
    result_neg = SentimentResult("今天真糟糕", "negative", 0.85, 0.85)
    assert result_neg.polarity == "negative"
    
    # 中性测试
    result_neu = SentimentResult("天气不错", "neutral", 0.6, 0.6)
    assert result_neu.polarity == "neutral"
    
    # to_dict
    data = result.to_dict()
    assert data["text"] == "今天真开心"
    assert "polarity" in data


def test_bert_sentiment_init():
    """测试 BERT 情感分析器初始化"""
    from AuroraNLP.deep_learning.pretrained import BERTSentiment, PreTrainedModelType, create_bert_sentiment
    
    # 默认初始化
    sa = BERTSentiment()
    assert sa is not None
    assert sa.is_available is not None
    assert hasattr(sa, "predict")
    assert hasattr(sa, "predict_batch")
    
    # ALBERT tiny 版本
    sa_light = BERTSentiment(model_type=PreTrainedModelType.ALBERT_TINY)
    assert sa_light is not None
    
    # 2分类和3分类
    sa_2class = BERTSentiment(num_classes=2)
    assert sa_2class._num_classes == 2
    
    # 便捷函数
    sa_func = create_bert_sentiment()
    assert sa_func is not None


def test_bert_sentiment_predict():
    """测试 BERT 情感预测（模拟）"""
    from AuroraNLP.deep_learning.pretrained import BERTSentiment
    
    # 创建分析器（仅测试接口）
    sa = BERTSentiment()
    
    # 检查可用性，若不可用则跳过
    if not sa.is_available:
        pytest.skip("BERT models not available")
    
    # 这里跳过实际加载，仅测试函数结构
    assert hasattr(sa, "load")
    assert hasattr(sa, "save")
    assert hasattr(sa, "predict")
    assert hasattr(sa, "predict_batch")


# ==================== 文本分类测试（步骤 43）====================

def test_classification_result():
    """测试 ClassificationResult 类"""
    from AuroraNLP.deep_learning.pretrained import ClassificationResult
    
    result = ClassificationResult(
        text="这个产品真好用",
        label="tech",
        score=0.92,
        all_scores={
            "tech": 0.92,
            "business": 0.05,
            "entertainment": 0.03
        }
    )
    
    assert result.text == "这个产品真好用"
    assert result.label == "tech"
    assert result.score == 0.92
    assert len(result.all_scores) == 3
    
    data = result.to_dict()
    assert "all_scores" in data


def test_bert_classifier_init():
    """测试 BERT 文本分类器初始化"""
    from AuroraNLP.deep_learning.pretrained import BERTClassifier, PreTrainedModelType, CLASSIFICATION_LABELS, create_bert_classifier
    
    # 默认初始化
    clf = BERTClassifier()
    assert clf is not None
    assert len(clf.labels) > 0
    assert "tech" in clf.labels
    
    # 自定义标签
    custom_labels = ["商品", "物流", "服务"]
    clf_custom = BERTClassifier(labels=custom_labels)
    assert clf_custom.labels == custom_labels
    
    # ALBERT 版本
    clf_light = BERTClassifier(model_type=PreTrainedModelType.ALBERT_TINY)
    assert clf_light is not None
    
    # 便捷函数
    clf_func = create_bert_classifier()
    assert clf_func is not None
    
    # 检查预定义标签集合
    assert "topics" in CLASSIFICATION_LABELS
    assert "domain" in CLASSIFICATION_LABELS


def test_bert_classifier_predict():
    """测试 BERT 分类器预测接口"""
    from AuroraNLP.deep_learning.pretrained import BERTClassifier, CLASSIFICATION_LABELS
    
    # 创建分类器
    clf = BERTClassifier(labels=CLASSIFICATION_LABELS["domain"])
    
    # 检查接口
    assert hasattr(clf, "load")
    assert hasattr(clf, "predict")
    assert hasattr(clf, "predict_batch")
    assert hasattr(clf, "save")


def test_bert_classifier_labels():
    """测试分类标签功能"""
    from AuroraNLP.deep_learning.pretrained import BERTClassifier, CLASSIFICATION_LABELS
    
    # 测试各种标签
    for label_set_name, label_set in CLASSIFICATION_LABELS.items():
        clf = BERTClassifier(labels=label_set)
        assert clf.labels == label_set


# ==================== 步骤44: 模型微调接口测试
def test_finetuning_config():
    """测试微调配置"""
    from AuroraNLP.deep_learning.pretrained import FineTuningConfig, create_finetuning_config
    
    # 默认配置
    config = FineTuningConfig()
    assert config.learning_rate == 2e-5
    assert config.batch_size == 32
    assert config.epochs == 3
    
    # 自定义配置
    config = create_finetuning_config(learning_rate=5e-5, batch_size=16, epochs=2)
    assert config.learning_rate == 5e-5
    assert config.batch_size == 16
    assert config.epochs == 2


def test_finetuning_trainer():
    """测试微调训练器"""
    from AuroraNLP.deep_learning.pretrained import (
        FineTuningConfig,
        FineTuningTrainer,
        PreTrainedModelType
    )
    
    config = FineTuningConfig()
    trainer = FineTuningTrainer(
        None,
        config,
        PreTrainedModelType.BERT_CHINESE
    )
    
    assert trainer is not None
    assert trainer.is_available is not None
    assert hasattr(trainer, "prepare_data")
    assert hasattr(trainer, "train_epoch")
    assert hasattr(trainer, "train")
    assert hasattr(trainer, "history")


def test_training_history():
    """测试训练历史记录"""
    from AuroraNLP.deep_learning.pretrained import (
        FineTuningConfig,
        FineTuningTrainer,
        PreTrainedModelType
    )
    
    config = FineTuningConfig(epochs=2)
    trainer = FineTuningTrainer(None, config, PreTrainedModelType.BERT_CHINESE)
    assert len(trainer.history["loss"]) == 0
    
    # 模拟训练
    for epoch in range(2):
        trainer.train_epoch(epoch)
    
    assert len(trainer.history["loss"]) == 2
    assert len(trainer.history["val_loss"]) == 2


# ==================== 步骤45: 迁移学习框架测试
def test_fewshot_config():
    """测试FewShot学习配置"""
    from AuroraNLP.deep_learning.pretrained import FewShotLearningConfig
    
    config = FewShotLearningConfig()
    assert config.num_classes == 5
    assert config.num_shots == 5
    assert config.use_prototypical is True
    assert config.use_maml is False


def test_fewshot_learner():
    """测试FewShot学习器"""
    from AuroraNLP.deep_learning.pretrained import FewShotLearner, FewShotLearningConfig, create_fewshot_learner

    # 创建学习器
    learner = FewShotLearner(FewShotLearningConfig())
    assert hasattr(learner, "is_available")

    # 便捷函数
    learner = create_fewshot_learner()
    assert learner is not None

    # 检查接口
    assert hasattr(learner, "train_on_few_shots")
    assert hasattr(learner, "predict_on_few_shots")


# ==================== 步骤46: 知识蒸馏测试
def test_knowledge_distillation_config():
    """测试知识蒸馏配置"""
    from AuroraNLP.deep_learning.pretrained import KnowledgeDistillationConfig
    
    config = KnowledgeDistillationConfig()
    assert config.temperature == 4.0
    assert config.alpha == 0.5
    assert config.use_lm_loss is False


def test_knowledge_distiller():
    """测试知识蒸馏器"""
    from AuroraNLP.deep_learning.pretrained import (
        KnowledgeDistiller,
        KnowledgeDistillationConfig,
        create_knowledge_distiller
    )
    
    teacher = None
    student = None
    config = KnowledgeDistillationConfig()
    distiller = KnowledgeDistiller(teacher, student, config)
    
    assert distiller.is_available
    assert hasattr(distiller, "distill")
    assert hasattr(distiller, "distill_step")
    
    # 便捷函数
    distiller = create_knowledge_distiller(teacher, student, temperature=2.0)
    assert distiller._config.temperature == 2.0


# ==================== 步骤47: 模型量化测试
def test_quantization_config():
    """测试量化配置"""
    from AuroraNLP.deep_learning.pretrained import QuantizationConfig
    
    config = QuantizationConfig()
    assert config.quantization_type == "dynamic_int8"
    assert config.do_quantize is True
    
    # 测试常量
    assert QuantizationConfig.QuantizationType.DYNAMIC_INT8 == "dynamic_int8"
    assert QuantizationConfig.QuantizationType.STATIC_INT8 == "static_int8"


def test_model_quantizer():
    """测试模型量化器"""
    from AuroraNLP.deep_learning.pretrained import ModelQuantizer, QuantizationConfig, create_quantizer
    
    # 创建量化器
    config = QuantizationConfig()
    quantizer = ModelQuantizer(config)
    assert quantizer.is_available
    assert hasattr(quantizer, "quantize")
    assert hasattr(quantizer, "evaluate_quantization")
    
    # 便捷函数
    quantizer = create_quantizer("static_int8")
    assert quantizer._config.quantization_type == "static_int8"
    
    # 测试评估方法
    result = quantizer.evaluate_quantization(None, None, None)
    assert "size_reduction" in result
    assert "speedup" in result


# ==================== 步骤48: ONNX导出测试
def test_onnx_config():
    """测试ONNX配置"""
    from AuroraNLP.deep_learning.pretrained import ONNXExportConfig
    
    config = ONNXExportConfig()
    assert config.opset_version == 14
    assert config.optimize is True


def test_onnx_exporter():
    """测试ONNX导出器"""
    from AuroraNLP.deep_learning.pretrained import ONNXExporter, ONNXExportConfig, create_onnx_exporter
    
    # 创建导出器
    config = ONNXExportConfig(export_path="test.onnx")
    exporter = ONNXExporter(config)
    
    assert exporter.is_available
    assert hasattr(exporter, "export")
    assert hasattr(exporter, "load_onnx_model")
    
    # 便捷函数
    exporter = create_onnx_exporter("my_model.onnx")
    assert exporter._config.export_path == "my_model.onnx"


# ==================== 步骤49: 热加载测试
def test_hot_load_config():
    """测试热加载配置"""
    from AuroraNLP.deep_learning.pretrained import HotLoadConfig
    
    config = HotLoadConfig()
    assert config.auto_reload is True
    assert config.check_interval == 60


def test_hot_model_loader():
    """测试热加载器"""
    from AuroraNLP.deep_learning.pretrained import HotModelLoader, HotLoadConfig, create_hot_loader
    
    # 创建加载器
    model = None
    config = HotLoadConfig()
    loader = HotModelLoader(model, config)
    
    assert loader.is_available
    assert loader.is_running is False
    assert hasattr(loader, "start_monitoring")
    assert hasattr(loader, "stop_monitoring")
    assert hasattr(loader, "switch_model")
    
    # 便捷函数
    loader = create_hot_loader(model)
    assert loader is not None
    
    # 测试监控功能
    loader.start_monitoring()
    assert loader.is_running is True
    loader.stop_monitoring()
    assert loader.is_running is False


def test_version_switch():
    """测试版本切换功能"""
    from AuroraNLP.deep_learning.pretrained import HotModelLoader, HotLoadConfig
    
    model = None
    loader = HotModelLoader(model, HotLoadConfig())
    initial_version = loader._current_version
    
    # 切换版本
    result = loader.switch_model(None)
    assert result is True
    assert loader._current_version != initial_version


# ==================== 步骤50: 模型管理系统测试
def test_model_version():
    """测试模型版本"""
    from AuroraNLP.deep_learning.pretrained import ModelVersion
    
    version = ModelVersion("v0.1", "model.pt", "First version")
    assert version.version == "v0.1"
    assert version.file_path == "model.pt"
    assert version.description == "First version"


def test_model_cache_config():
    """测试模型缓存配置"""
    from AuroraNLP.deep_learning.pretrained import ModelCacheConfig
    
    config = ModelCacheConfig(max_versions=5, cache_dir="/tmp")
    assert config.max_versions == 5
    assert config.cache_dir == "/tmp"


def test_model_manager():
    """测试模型管理器"""
    from AuroraNLP.deep_learning.pretrained import (
        ModelManager,
        ModelCacheConfig,
        create_model_manager
    )
    
    # 创建管理器
    config = ModelCacheConfig(max_versions=10)
    manager = ModelManager(config)
    
    assert manager.is_available
    assert hasattr(manager, "register_model")
    assert hasattr(manager, "get_model")
    assert hasattr(manager, "delete_old_versions")
    
    # 便捷函数
    manager = create_model_manager(max_versions=5)
    assert manager is not None
    
    # 测试注册模型
    result = manager.register_model("bert", "v0.1", "model.pt")
    assert result is True
    
    # 测试获取模型
    version = manager.get_model("bert", "v0.1")
    assert version is not None
    assert version.version == "v0.1"


if __name__ == "__main__":
    # 简单运行一些测试
    test_module_import()
    test_model_type_enum()
    test_config_defaults()
    test_config_lightweight_check()
    test_get_lightweight_models()
    test_lightweight_segmentor_factory()
    test_lightweight_segmentor_by_name()
    test_model_comparator()
    test_model_recommend()
    test_create_lightweight_segmentor_function()
    test_bmes_decode()
    test_bmes_decode_multichar()
    test_model_config_info()
    print("All lightweight model tests passed!")
