"""
情感词典模块测试
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from AuroraNLP.sentiment import (
    SentimentDictionary,
    SentimentAnalyzer,
    SentimentPolarity,
    SentimentIntensity,
    SentimentWord,
    DegreeWord,
    NegationWord,
)


class TestSentimentDictionary(unittest.TestCase):
    """测试情感词典类"""
    
    def setUp(self):
        """测试前准备"""
        self.dictionary = SentimentDictionary(load_default=True)
    
    def test_initialization(self):
        """测试初始化"""
        self.assertTrue(self.dictionary.is_loaded())
        self.assertGreater(self.dictionary.get_total_word_count(), 0)
    
    def test_positive_words(self):
        """测试正面情感词"""
        # 测试基本正面词
        self.assertTrue(self.dictionary.is_positive("好"))
        self.assertTrue(self.dictionary.is_positive("开心"))
        self.assertTrue(self.dictionary.is_positive("喜欢"))
        self.assertTrue(self.dictionary.is_positive("优秀"))
        
        # 测试非正面词
        self.assertFalse(self.dictionary.is_positive("坏"))
        self.assertFalse(self.dictionary.is_positive("悲伤"))
    
    def test_negative_words(self):
        """测试负面情感词"""
        # 测试基本负面词
        self.assertTrue(self.dictionary.is_negative("愤怒"))
        self.assertTrue(self.dictionary.is_negative("悲伤"))
        self.assertTrue(self.dictionary.is_negative("害怕"))
        self.assertTrue(self.dictionary.is_negative("糟糕"))
        
        # 测试非负面词
        self.assertFalse(self.dictionary.is_negative("好"))
        self.assertFalse(self.dictionary.is_negative("开心"))
    
    def test_sentiment_word_detection(self):
        """测试情感词检测"""
        self.assertTrue(self.dictionary.is_sentiment_word("好"))
        self.assertTrue(self.dictionary.is_sentiment_word("愤怒"))
        self.assertFalse(self.dictionary.is_sentiment_word("桌子"))
        self.assertFalse(self.dictionary.is_sentiment_word("电脑"))
    
    def test_word_score(self):
        """测试情感分数"""
        # 正面词分数应为正
        self.assertGreater(self.dictionary.get_word_score("好"), 0)
        self.assertGreater(self.dictionary.get_word_score("优秀"), 0)
        
        # 负面词分数应为负
        self.assertLess(self.dictionary.get_word_score("愤怒"), 0)
        self.assertLess(self.dictionary.get_word_score("糟糕"), 0)
        
        # 非情感词分数应为0
        self.assertEqual(self.dictionary.get_word_score("桌子"), 0)
    
    def test_word_intensity(self):
        """测试情感强度"""
        intensity = self.dictionary.get_word_intensity("好")
        self.assertIsNotNone(intensity)
        self.assertEqual(intensity, SentimentIntensity.MEDIUM)
    
    def test_word_category(self):
        """测试情感类别"""
        self.assertEqual(self.dictionary.get_word_category("好"), "joy")
        self.assertEqual(self.dictionary.get_word_category("喜欢"), "love")
        self.assertEqual(self.dictionary.get_word_category("优秀"), "praise")
        self.assertEqual(self.dictionary.get_word_category("愤怒"), "anger")
    
    def test_negation_words(self):
        """测试否定词"""
        self.assertTrue(self.dictionary.is_negation_word("不"))
        self.assertTrue(self.dictionary.is_negation_word("没"))
        self.assertTrue(self.dictionary.is_negation_word("没有"))
        self.assertFalse(self.dictionary.is_negation_word("好"))
    
    def test_degree_words(self):
        """测试程度副词"""
        self.assertTrue(self.dictionary.is_degree_word("很"))
        self.assertTrue(self.dictionary.is_degree_word("非常"))
        self.assertTrue(self.dictionary.is_degree_word("稍微"))
        # 注意："好"既是情感词也是程度副词（如"好极了"中的"好"）
        self.assertTrue(self.dictionary.is_degree_word("好"))
    
    def test_get_degree(self):
        """测试获取程度值"""
        self.assertEqual(self.dictionary.get_degree("很"), 1.2)
        self.assertEqual(self.dictionary.get_degree("非常"), 1.4)
        self.assertEqual(self.dictionary.get_degree("稍微"), 0.3)
        self.assertEqual(self.dictionary.get_degree("不存在"), 1.0)
    
    def test_add_words(self):
        """测试添加词语"""
        # 添加正面词
        self.dictionary.add_positive_word("测试好词", SentimentIntensity.STRONG, "test")
        self.assertTrue(self.dictionary.is_positive("测试好词"))
        
        # 添加负面词
        self.dictionary.add_negative_word("测试坏词", SentimentIntensity.MEDIUM, "test")
        self.assertTrue(self.dictionary.is_negative("测试坏词"))
        
        # 添加否定词
        self.dictionary.add_negation_word("测试否定", 1.0)
        self.assertTrue(self.dictionary.is_negation_word("测试否定"))
        
        # 添加程度副词
        self.dictionary.add_degree_word("测试程度", 1.5, "test")
        self.assertTrue(self.dictionary.is_degree_word("测试程度"))
    
    def test_get_words_by_category(self):
        """测试按类别获取词语"""
        joy_words = self.dictionary.get_words_by_category("joy")
        self.assertIn("好", joy_words)
        self.assertIn("开心", joy_words)
        
        anger_words = self.dictionary.get_words_by_category("anger")
        self.assertIn("愤怒", anger_words)
    
    def test_get_words_by_intensity(self):
        """测试按强度获取词语"""
        strong_words = self.dictionary.get_words_by_intensity(SentimentIntensity.STRONG)
        self.assertGreater(len(strong_words), 0)
    
    def test_word_count(self):
        """测试词数统计"""
        positive_count = self.dictionary.get_positive_word_count()
        negative_count = self.dictionary.get_negative_word_count()
        total_count = self.dictionary.get_total_word_count()
        
        self.assertGreater(positive_count, 0)
        self.assertGreater(negative_count, 0)
        self.assertEqual(total_count, positive_count + negative_count)
    
    def test_contains(self):
        """测试contains操作"""
        self.assertIn("好", self.dictionary)
        self.assertIn("愤怒", self.dictionary)
        self.assertNotIn("桌子", self.dictionary)
    
    def test_len(self):
        """测试len操作"""
        self.assertEqual(len(self.dictionary), self.dictionary.get_total_word_count())


class TestSentimentAnalysis(unittest.TestCase):
    """测试情感分析功能"""
    
    def setUp(self):
        """测试前准备"""
        self.dictionary = SentimentDictionary(load_default=True)
    
    def test_analyze_positive(self):
        """测试正面情感分析"""
        result = self.dictionary.analyze("这个产品很好", ["这个", "产品", "很", "好"])
        
        self.assertEqual(result.polarity, SentimentPolarity.POSITIVE)
        self.assertGreater(result.score, 0)
        self.assertGreater(len(result.positive_words), 0)
    
    def test_analyze_negative(self):
        """测试负面情感分析"""
        result = self.dictionary.analyze("这个产品很糟糕", ["这个", "产品", "很", "糟糕"])
        
        self.assertEqual(result.polarity, SentimentPolarity.NEGATIVE)
        self.assertLess(result.score, 0)
        self.assertGreater(len(result.negative_words), 0)
    
    def test_analyze_with_negation(self):
        """测试带否定词的情感分析"""
        # 否定正面词
        result = self.dictionary.analyze("不好", ["不", "好"])
        self.assertLess(result.score, 0)
        
        # 否定负面词
        result = self.dictionary.analyze("不糟糕", ["不", "糟糕"])
        self.assertGreater(result.score, 0)
    
    def test_analyze_with_degree(self):
        """测试带程度副词的情感分析"""
        # 正常正面
        result1 = self.dictionary.analyze("好", ["好"])
        # 程度加强
        result2 = self.dictionary.analyze("非常好", ["非常", "好"])
        self.assertGreater(abs(result2.score), abs(result1.score))
    
    def test_confidence(self):
        """测试置信度"""
        result = self.dictionary.analyze("好", ["好"])
        self.assertGreater(result.confidence, 0)
    
    def test_intensity(self):
        """测试情感强度"""
        result = self.dictionary.analyze("好", ["好"])
        self.assertGreater(result.intensity, 0)


class TestSentimentAnalyzer(unittest.TestCase):
    """测试情感分析器类"""
    
    def setUp(self):
        """测试前准备"""
        self.analyzer = SentimentAnalyzer()
    
    def test_initialization(self):
        """测试初始化"""
        self.assertIsNotNone(self.analyzer.get_dictionary())
        self.assertTrue(self.analyzer.get_dictionary().is_loaded())
    
    def test_analyze(self):
        """测试分析功能"""
        result = self.analyzer.analyze("好")
        self.assertIsNotNone(result)
        self.assertEqual(result.polarity, SentimentPolarity.POSITIVE)
    
    def test_custom_dictionary(self):
        """测试自定义词典"""
        dictionary = SentimentDictionary(load_default=True)
        analyzer = SentimentAnalyzer(dictionary)
        self.assertEqual(analyzer.get_dictionary(), dictionary)


if __name__ == '__main__':
    unittest.main()