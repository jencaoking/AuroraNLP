"""测试新词发现模块"""

import pytest

from AuroraNLP.segmentation.new_word_detector import (
    NewWordDetector,
    MutualInformation,
    EntropyCalculator,
)


@pytest.fixture
def sample_corpus():
    """创建重复多次的简单语料列表"""
    base = [
        "自然语言处理是人工智能的重要方向",
        "深度学习在自然语言处理中有广泛应用",
        "机器学习和深度学习都是人工智能的方法",
        "自然语言处理包括分词和词性标注",
        "人工智能改变了人们的生活方式",
        "深度学习模型需要大量训练数据",
        "自然语言处理技术不断发展进步",
        "人工智能在医疗领域有重要应用",
        "机器学习的算法不断优化改进",
        "深度学习和自然语言处理结合紧密",
    ]
    # 重复多次以获得足够的词频
    return base * 10


class TestNewWordDetectorInit:
    """测试新词检测器初始化"""

    def test_detector_init(self):
        """测试初始化"""
        detector = NewWordDetector()
        assert detector is not None
        assert detector.is_trained() is False
        assert detector.min_freq == 5
        assert detector.min_pmi == 1.0
        assert detector.min_entropy == 0.5


class TestNewWordDetectorTrain:
    """测试新词检测器训练"""

    def test_detector_train(self, sample_corpus):
        """测试训练"""
        detector = NewWordDetector(min_freq=3, min_pmi=0.5, min_entropy=0.3)
        detector.train(sample_corpus)
        assert detector.is_trained() is True
        stats = detector.get_statistics()
        assert stats['trained'] is True
        assert stats['total_chars'] > 0
        assert stats['unique_chars'] > 0


class TestNewWordDetectorDetect:
    """测试新词检测器检测"""

    def test_detector_detect(self, sample_corpus):
        """测试检测新词"""
        detector = NewWordDetector(min_freq=3, min_pmi=0.5, min_entropy=0.3)
        detector.train(sample_corpus)
        results = detector.detect(top_k=10)
        assert isinstance(results, list)
        for word, info in results:
            assert isinstance(word, str)
            assert isinstance(info, dict)
            assert 'frequency' in info
            assert 'pmi' in info
            assert 'left_entropy' in info
            assert 'right_entropy' in info
            assert 'avg_entropy' in info


class TestMutualInformation:
    """测试互信息计算"""

    def test_mutual_information(self, sample_corpus):
        """测试互信息计算"""
        mi = MutualInformation()
        mi.train(sample_corpus)
        assert mi.is_trained() is True
        # 测试两个连续字符的 PMI
        pmi = mi.calculate_pmi("自", "然")
        assert isinstance(pmi, float)
        # 测试词级别的 PMI
        word_pmi = mi.calculate_word_pmi("自然")
        assert isinstance(word_pmi, float)


class TestEntropyCalculator:
    """测试熵计算"""

    def test_entropy_calculator(self, sample_corpus):
        """测试熵计算"""
        ec = EntropyCalculator()
        ec.train(sample_corpus)
        assert ec.is_trained() is True
        # 测试左熵
        left_entropy = ec.get_left_entropy("然")
        assert isinstance(left_entropy, float)
        assert left_entropy >= 0.0
        # 测试右熵
        right_entropy = ec.get_right_entropy("然")
        assert isinstance(right_entropy, float)
        assert right_entropy >= 0.0
        # 测试平均熵
        avg_entropy = ec.get_avg_entropy("然")
        assert isinstance(avg_entropy, float)
        assert avg_entropy >= 0.0


class TestNewWordDetectorEdge:
    """测试新词检测器边界情况"""

    def test_invalid_word(self, sample_corpus):
        """测试无效词过滤"""
        detector = NewWordDetector(min_freq=3, min_pmi=0.5, min_entropy=0.3)
        detector.train(sample_corpus)
        results = detector.detect(top_k=100)
        for word, info in results:
            # 所有检测到的词应满足最小长度要求
            assert len(word) >= detector.min_word_len

    def test_set_thresholds(self):
        """测试设置阈值"""
        detector = NewWordDetector()
        detector.set_thresholds(min_freq=10, min_pmi=2.0, min_entropy=1.0)
        assert detector.min_freq == 10
        assert detector.min_pmi == 2.0
        assert detector.min_entropy == 1.0
