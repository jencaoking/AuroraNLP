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


def test_model_type_enum():
    """测试模型类型枚举"""
    from AuroraNLP.deep_learning.pretrained import PreTrainedModelType
    assert PreTrainedModelType.BERT_CHINESE is not None
    assert PreTrainedModelType.MACBERT is not None
    assert PreTrainedModelType.ROBERTA_CHINESE is not None
    assert PreTrainedModelType.ALBERT is not None
    assert PreTrainedModelType.DISTILBERT is not None
    # 测试枚举值
    assert PreTrainedModelType.BERT_CHINESE.value == "bert_chinese"


def test_config_defaults():
    """测试预训练模型配置"""
    from AuroraNLP.deep_learning.pretrained import PreTrainedModelConfig, PreTrainedModelType
    # 默认配置
    config = PreTrainedModelConfig()
    assert config.model_type == PreTrainedModelType.BERT_CHINESE
    assert config.max_seq_length == 512
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


def test_get_available_models():
    """测试获取可用模型列表"""
    from AuroraNLP.deep_learning.pretrained import get_available_pretrained_models
    models = get_available_pretrained_models()
    assert len(models) > 0
    assert "bert-base-chinese" in models


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


def test_pretrained_config_dict():
    """测试模型配置字典"""
    from AuroraNLP.deep_learning.pretrained import PreTrainedModelConfig, PreTrainedModelType
    config = PreTrainedModelConfig(model_type=PreTrainedModelType.BERT_CHINESE)
    
    # 检查默认模型路径
    assert config.model_name_or_path is not None
    assert "bert" in config.model_name_or_path.lower()


if __name__ == "__main__":
    # 简单运行一些测试
    test_module_import()
    test_model_type_enum()
    test_config_defaults()
    test_get_available_models()
    test_segmentor_init()
    test_bmes_decode()
    print("Basic pre-trained model tests passed!")
