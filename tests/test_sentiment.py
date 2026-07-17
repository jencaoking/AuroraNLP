"""测试情感分析模块"""

import pytest

from AuroraNLP.text_analysis.sentiment import (
    SentimentDictionary,
    SentimentAnalyzer,
    SentimentPolarity,
    SentimentIntensity,
    SentimentWord,
    SentimentResult,
)


class TestSentimentDictionaryInit:
    """测试情感词典初始化"""

    def test_sentiment_dict_init(self):
        """测试初始化空情感词典"""
        sd = SentimentDictionary(load_default=False)
        assert sd is not None
        assert sd.get_total_word_count() == 0

    def test_sentiment_dict_init_with_default(self):
        """测试初始化默认情感词典"""
        sd = SentimentDictionary(load_default=True)
        assert sd is not None
        assert sd.get_total_word_count() > 0


class TestSentimentDictionaryAdd:
    """测试情感词典添加功能"""

    def test_sentiment_dict_add_positive(self):
        """测试添加正面词"""
        sd = SentimentDictionary(load_default=False)
        sd.add_positive_word("开心", SentimentIntensity.MEDIUM, "joy")
        assert sd.is_positive("开心")
        assert sd.get_positive_word_count() == 1

    def test_sentiment_dict_add_negative(self):
        """测试添加负面词"""
        sd = SentimentDictionary(load_default=False)
        sd.add_negative_word("悲伤", SentimentIntensity.MEDIUM, "sadness")
        assert sd.is_negative("悲伤")
        assert sd.get_negative_word_count() == 1


class TestSentimentDictionaryAnalyze:
    """测试情感词典分析功能"""

    def test_sentiment_dict_analyze_positive(self):
        """测试分析正面文本"""
        sd = SentimentDictionary(load_default=True)
        result = sd.analyze("开心快乐", words=["开心", "快乐"])
        assert result.polarity == SentimentPolarity.POSITIVE
        assert result.score > 0

    def test_sentiment_dict_analyze_negative(self):
        """测试分析负面文本"""
        sd = SentimentDictionary(load_default=True)
        result = sd.analyze("愤怒悲伤", words=["愤怒", "悲伤"])
        assert result.polarity == SentimentPolarity.NEGATIVE
        assert result.score < 0

    def test_sentiment_dict_analyze_neutral(self):
        """测试分析中性文本"""
        sd = SentimentDictionary(load_default=False)
        result = sd.analyze("今天天气", words=["今天", "天气"])
        assert result.polarity == SentimentPolarity.NEUTRAL
        assert result.score == 0.0


class TestSentimentDictionarySpecial:
    """测试情感词典特殊功能"""

    def test_sentiment_dict_negation(self):
        """测试否定词处理（"不好"应为负面）"""
        sd = SentimentDictionary(load_default=True)
        result = sd.analyze("不好", words=["不", "好"])
        assert result.polarity == SentimentPolarity.NEGATIVE
        assert result.score < 0

    def test_sentiment_dict_degree(self):
        """测试程度副词处理（"非常好"应增强）"""
        sd = SentimentDictionary(load_default=True)
        result_normal = sd.analyze("好", words=["好"])
        result_degree = sd.analyze("非常好", words=["非常", "好"])
        # "非常好"的正面分数应该比单独的"好"更强
        assert result_degree.score > result_normal.score

    def test_sentiment_dict_is_positive(self):
        """测试判断正面词"""
        sd = SentimentDictionary(load_default=True)
        assert sd.is_positive("好") is True
        assert sd.is_positive("开心") is True
        assert sd.is_positive("愤怒") is False

    def test_sentiment_dict_is_negative(self):
        """测试判断负面词"""
        sd = SentimentDictionary(load_default=True)
        assert sd.is_negative("愤怒") is True
        assert sd.is_negative("悲伤") is True
        assert sd.is_negative("好") is False

    def test_sentiment_dict_get_word_score(self):
        """测试获取词分数"""
        sd = SentimentDictionary(load_default=True)
        score = sd.get_word_score("好")
        assert score > 0
        score_neg = sd.get_word_score("愤怒")
        assert score_neg < 0
        score_unknown = sd.get_word_score("未知词")
        assert score_unknown == 0.0

    def test_sentiment_dict_len(self):
        """测试情感词典长度"""
        sd = SentimentDictionary(load_default=False)
        assert len(sd) == 0
        sd.add_positive_word("好", SentimentIntensity.MEDIUM)
        sd.add_negative_word("坏", SentimentIntensity.MEDIUM)
        assert len(sd) == 2


class TestSentimentAnalyzer:
    """测试情感分析器"""

    def test_sentiment_analyzer_init(self):
        """测试分析器初始化"""
        analyzer = SentimentAnalyzer()
        assert analyzer is not None
        assert analyzer.get_dictionary() is not None

    def test_sentiment_analyzer_analyze(self):
        """测试分析器分析"""
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze("开心快乐", words=["开心", "快乐"])
        assert isinstance(result, SentimentResult)
        assert result.polarity == SentimentPolarity.POSITIVE
        assert result.score > 0
