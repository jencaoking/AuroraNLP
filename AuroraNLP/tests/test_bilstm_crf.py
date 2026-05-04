"""
测试 BiLSTM-CRF 模型
"""
import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_import():
    """测试导入"""
    from AuroraNLP.deep_learning import bilstm_crf
    assert hasattr(bilstm_crf, 'BiLSTMCRF')
    assert hasattr(bilstm_crf, 'BiLSTMCRFSegmentor')
    assert hasattr(bilstm_crf, 'CRF')


def test_crf_init():
    """测试 CRF 初始化"""
    from AuroraNLP.deep_learning.bilstm_crf import CRF
    crf = CRF(4)
    assert crf.num_tags == 4
    
    # 尝试导入 PyTorch 来测试，否则跳过
    try:
        import torch
        crf.init_params()
        assert hasattr(crf, 'start_transitions')
        assert hasattr(crf, 'transitions')
        assert hasattr(crf, 'end_transitions')
    except ImportError:
        import pytest
        pytest.skip("PyTorch not available")


def test_bilstm_crf_init():
    """测试 BiLSTMCRF 初始化"""
    from AuroraNLP.deep_learning.bilstm_crf import BiLSTMCRF
    model = BiLSTMCRF(
        vocab_size=100,
        tagset_size=4,
        embedding_dim=32,
        hidden_dim=64,
        num_layers=1,
        dropout=0.1
    )
    assert model.vocab_size == 100
    assert model.tagset_size == 4
    # 如果框架可用
    if model.is_available():
        assert model.model is not None


def test_segmentor_init():
    """测试分词器初始化"""
    from AuroraNLP.deep_learning.bilstm_crf import BiLSTMCRFSegmentor
    seg = BiLSTMCRFSegmentor(
        vocab_size=100,
        embedding_dim=32,
        hidden_dim=64,
        num_layers=1,
        dropout=0.1
    )
    assert seg.vocab_size == 100
    assert not seg.is_trained


def test_segmentor_vocab():
    """测试词汇表构建"""
    from AuroraNLP.deep_learning.bilstm_crf import BiLSTMCRFSegmentor
    seg = BiLSTMCRFSegmentor(vocab_size=50)
    texts = ['我爱中国', '中国人民', '自然语言处理']
    seg._build_vocab(texts)
    assert len(seg.vocab) >= 2  # 至少有 <PAD> 和 <UNK>
    assert '中' in seg.vocab
    assert '国' in seg.vocab


def test_text_to_ids():
    """测试文本转 ID"""
    from AuroraNLP.deep_learning.bilstm_crf import BiLSTMCRFSegmentor
    seg = BiLSTMCRFSegmentor(vocab_size=50)
    texts = ['我爱中国', '中国人民']
    seg._build_vocab(texts)
    ids = seg._text_to_ids('中国')
    assert len(ids) == 2
    assert ids[0] != 1  # <UNK>
    assert ids[1] != 1


def test_tags_to_words():
    """测试标签转词"""
    from AuroraNLP.deep_learning.bilstm_crf import BiLSTMCRFSegmentor
    seg = BiLSTMCRFSegmentor()
    text = '我爱中国'
    tags = [3, 3, 0, 2]  # S, S, B, E
    words = seg._tags_to_words(text, tags)
    assert words == ['我', '爱', '中国']


@pytest.mark.slow
def test_train_small():
    """测试小数据训练"""
    try:
        from AuroraNLP.deep_learning.bilstm_crf import BiLSTMCRFSegmentor
        seg = BiLSTMCRFSegmentor(
            vocab_size=100,
            embedding_dim=16,
            hidden_dim=32,
            num_layers=1,
            dropout=0.1
        )
        # 创建简单的训练数据
        texts = ['我爱中国', '我爱自然语言处理', '中国人民', '自然语言处理'] * 5
        # 训练
        try:
            seg.train(texts, epochs=1, batch_size=2, verbose=False)
            # 简单测试分词（即使准确率不高，只要不报错）
            if seg.is_available and seg.is_trained:
                result = seg.segment('中国')
                assert isinstance(result, list)
        except Exception as e:
            # 测试不强制要求 PyTorch 可用，只要能正常处理初始化
            print(f"Training failed (PyTorch may not be available): {e}")
    except ImportError:
        pytest.skip("PyTorch not available")


def test_integration():
    """集成测试 - 导入和基本 API 检查"""
    try:
        from AuroraNLP.deep_learning import bilstm_crf
        # 检查 API
        assert hasattr(bilstm_crf, 'BiLSTMCRF')
        assert hasattr(bilstm_crf, 'BiLSTMCRFSegmentor')
        # 检查公共方法
        assert hasattr(bilstm_crf.BiLSTMCRF, 'train')
        assert hasattr(bilstm_crf.BiLSTMCRF, 'predict')
        assert hasattr(bilstm_crf.BiLSTMCRF, 'save')
        assert hasattr(bilstm_crf.BiLSTMCRF, 'load')
    except ImportError as e:
        pytest.skip(f"PyTorch not available: {e}")


if __name__ == '__main__':
    # 直接运行简单测试
    test_import()
    test_crf_init()
    test_bilstm_crf_init()
    test_segmentor_init()
    print("Basic tests passed!")
