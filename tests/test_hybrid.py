"""Hybrid 混合分词测试"""
import pytest
from AuroraNLP.hybrid import (
    HybridStrategy,
    SegmenterType,
    SegmenterResult,
    HybridConfig,
    FusionContext,
    FusionStrategy,
    VoteFusionStrategy,
    WeightedFusionStrategy,
    CascadeFusionStrategy,
    ConfidenceFusionStrategy,
    FusionStrategyFactory,
    ConfidenceEstimator,
    TextClassifier,
    StrategySelector,
)


class TestHybridEnums:
    """测试混合分词枚举类型"""

    def test_hybrid_strategy_enum(self):
        """混合策略枚举"""
        assert HybridStrategy.VOTE.value == "vote"
        assert HybridStrategy.WEIGHTED.value == "weighted"
        assert HybridStrategy.CASCADE.value == "cascade"
        assert HybridStrategy.ADAPTIVE.value == "adaptive"
        assert HybridStrategy.CONFIDENCE.value == "confidence"

    def test_segmenter_type_enum(self):
        """分词器类型枚举"""
        assert SegmenterType.RULE_BASED.value == "rule_based"
        assert SegmenterType.STATISTICAL.value == "statistical"
        assert SegmenterType.DEEP_LEARNING.value == "deep_learning"
        assert SegmenterType.HYBRID.value == "hybrid"


class TestSegmenterResult:
    """测试 SegmenterResult"""

    def test_segmenter_result_creation(self):
        """分词结果对象创建"""
        result = SegmenterResult(words=["自然", "语言", "处理"])
        assert result.words == ["自然", "语言", "处理"]
        assert result.confidence == 1.0
        assert result.segmenter_type == SegmenterType.RULE_BASED
        assert result.segmenter_name == "rule_based"
        assert result.metadata == {}

    def test_segmenter_result_creation_with_options(self):
        """带参数的分词结果对象创建"""
        result = SegmenterResult(
            words=["中国", "人"],
            confidence=0.9,
            segmenter_type=SegmenterType.STATISTICAL,
            segmenter_name="hmm",
            metadata={"model": "v1"}
        )
        assert result.words == ["中国", "人"]
        assert result.confidence == 0.9
        assert result.segmenter_type == SegmenterType.STATISTICAL
        assert result.segmenter_name == "hmm"
        assert result.metadata == {"model": "v1"}

    def test_segmenter_result_properties(self):
        """分词结果属性（word_count, avg_word_length, single_char_ratio）"""
        result = SegmenterResult(words=["自然", "语言", "处理", "是", "好", "的", "技术"])
        assert result.word_count == 7
        # 2+2+2+1+1+1+2 = 11, avg = 11/7
        assert result.avg_word_length == pytest.approx(11 / 7, rel=1e-6)
        # "是", "好", "的" 是单字，共 3 个
        assert result.single_char_ratio == pytest.approx(3 / 7, rel=1e-6)

    def test_segmenter_result_properties_empty(self):
        """空分词结果属性"""
        result = SegmenterResult(words=[])
        assert result.word_count == 0
        assert result.avg_word_length == 0.0
        assert result.single_char_ratio == 0.0


class TestHybridConfig:
    """测试 HybridConfig"""

    def test_hybrid_config_creation(self):
        """配置对象创建"""
        config = HybridConfig()
        assert config.strategy == HybridStrategy.WEIGHTED
        assert isinstance(config.weights, dict)
        assert 'dict' in config.weights
        assert 'hmm' in config.weights
        assert 'crf' in config.weights
        assert 'perceptron' in config.weights
        assert isinstance(config.cascade_order, list)
        assert config.confidence_threshold == 0.7
        assert config.min_confidence == 0.3
        assert config.use_dict_fallback is True
        assert config.enable_cache is True
        assert config.cache_size == 1000

    def test_hybrid_config_validate(self):
        """配置验证"""
        config = HybridConfig()
        assert config.validate() is True


class TestFusionContext:
    """测试 FusionContext"""

    def test_fusion_context_creation(self):
        """融合上下文创建"""
        results = [
            SegmenterResult(words=["自然", "语言", "处理"], segmenter_name="dict"),
            SegmenterResult(words=["自然语言", "处理"], segmenter_name="hmm"),
        ]
        context = FusionContext(text="自然语言处理", results=results)
        assert context.text == "自然语言处理"
        assert len(context.results) == 2
        assert context.dictionary is None
        assert context.config is None

    def test_fusion_context_creation_with_config(self):
        """带配置的融合上下文创建"""
        config = HybridConfig()
        results = [SegmenterResult(words=["中国", "人"])]
        context = FusionContext(text="中国人", results=results, config=config)
        assert context.config is config


class TestFusionStrategies:
    """测试融合策略"""

    def test_vote_fusion_strategy_init(self):
        """投票融合策略初始化"""
        strategy = VoteFusionStrategy()
        assert strategy.get_name() == "vote"

    def test_weighted_fusion_strategy_init(self):
        """加权融合策略初始化"""
        strategy = WeightedFusionStrategy()
        assert strategy.get_name() == "weighted"

    def test_cascade_fusion_strategy_init(self):
        """级联融合策略初始化"""
        strategy = CascadeFusionStrategy()
        assert strategy.get_name() == "cascade"
        assert strategy.confidence_threshold == 0.7

    def test_cascade_fusion_strategy_init_custom_threshold(self):
        """自定义阈值的级联融合策略初始化"""
        strategy = CascadeFusionStrategy(confidence_threshold=0.5)
        assert strategy.confidence_threshold == 0.5

    def test_confidence_fusion_strategy_init(self):
        """置信度融合策略初始化"""
        strategy = ConfidenceFusionStrategy()
        assert strategy.get_name() == "confidence"


class TestFusionStrategyFactory:
    """测试策略工厂"""

    def test_fusion_strategy_factory(self):
        """策略工厂"""
        factory = FusionStrategyFactory()
        available = factory.get_available_strategies()
        assert 'vote' in available
        assert 'weighted' in available
        assert 'cascade' in available
        assert 'adaptive' in available
        assert 'confidence' in available

        vote_strategy = factory.get_strategy('vote')
        assert isinstance(vote_strategy, VoteFusionStrategy)

        weighted_strategy = factory.get_strategy('weighted')
        assert isinstance(weighted_strategy, WeightedFusionStrategy)

        cascade_strategy = factory.get_strategy('cascade')
        assert isinstance(cascade_strategy, CascadeFusionStrategy)

        confidence_strategy = factory.get_strategy('confidence')
        assert isinstance(confidence_strategy, ConfidenceFusionStrategy)

        unknown_strategy = factory.get_strategy('unknown')
        assert unknown_strategy is None


class TestTextClassifier:
    """测试文本分类器"""

    def test_text_classifier_init(self):
        """文本分类器初始化"""
        classifier = TextClassifier()
        features = classifier.extract_features("自然语言处理是人工智能的重要方向")
        assert isinstance(features, dict)
        assert 'length' in features
        assert 'chinese_ratio' in features
        assert 'english_ratio' in features
        assert 'digit_ratio' in features
        assert 'punct_ratio' in features
        assert features['length'] > 0


class TestStrategySelector:
    """测试策略选择器"""

    def test_strategy_selector_init(self):
        """策略选择器初始化"""
        selector = StrategySelector()
        features = {'chinese_ratio': 0.8, 'english_ratio': 0.1, 'digit_ratio': 0.0}
        results = [
            SegmenterResult(words=["自然", "语言", "处理"], segmenter_type=SegmenterType.STATISTICAL, segmenter_name="hmm"),
        ]
        selected = selector.select(features, results)
        assert isinstance(selected, str)
