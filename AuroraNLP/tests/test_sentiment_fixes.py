"""
情感词典修复验证测试

验证否定词和程度副词处理的改进
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from AuroraNLP.sentiment import (
    SentimentDictionary,
    SentimentPolarity,
)


class TestSentimentFixes(unittest.TestCase):
    """测试情感词典修复"""
    
    def setUp(self):
        """测试前准备"""
        self.dictionary = SentimentDictionary(load_default=True)
    
    def test_negation_combinations(self):
        """测试否定词组合"""
        # 测试简单否定
        result1 = self.dictionary.analyze("好", ["好"])
        result2 = self.dictionary.analyze("不好", ["不", "好"])
        self.assertGreater(result1.score, 0)
        self.assertLess(result2.score, 0)
        
        # 测试否定词组合（不是很好）
        result3 = self.dictionary.analyze("不是很好", ["不", "是", "很", "好"])
        # 应该被识别为否定，分数为负
        self.assertLess(result3.score, 0)
        
        # 测试双重否定（不不是很好）
        result4 = self.dictionary.analyze("不不是很好", ["不", "不", "是", "很", "好"])
        # 双重否定应该被识别为肯定，分数为正
        self.assertGreater(result4.score, 0)
    
    def test_negation_distance(self):
        """测试否定词距离"""
        # 测试否定词与情感词之间有间隔词
        test_cases = [
            ("不 是 好", ["不", "是", "好"], "否定1个词间隔"),
            ("不 很 好", ["不", "很", "好"], "否定+程度+情感词"),
            ("不 太 好", ["不", "太", "好"], "否定+程度+情感词"),
        ]
        
        for text, words, desc in test_cases:
            with self.subTest(desc=desc):
                result = self.dictionary.analyze(text, words)
                # 应该被识别为否定
                self.assertLess(result.score, 0, f"'{text}' 应该被识别为否定")
    
    def test_degree_complex_sentences(self):
        """测试复杂句式中的程度副词"""
        # 测试程度副词在不同位置
        test_cases = [
            ("好", ["好"], "基础情况"),
            ("很好", ["很", "好"], "程度副词在前"),
            ("非常好", ["非常", "好"], "程度副词在前"),
            ("好极了", ["好", "极", "了"], "程度副词在后"),
            ("非常很好", ["非常", "很", "好"], "多个程度副词"),
            ("非常非常好", ["非常", "非常", "好"], "重复程度副词"),
        ]
        
        # 基础分数
        base_result = self.dictionary.analyze("好", ["好"])
        base_score = abs(base_result.score)
        
        for text, words, desc in test_cases:
            with self.subTest(desc=desc):
                result = self.dictionary.analyze(text, words)
                current_score = abs(result.score)
                
                # 带程度副词的分数应该大于基础分数
                if "很" in words or "非常" in words or "极" in words:
                    self.assertGreater(current_score, base_score, 
                                     f"'{text}' 应该增强情感强度")
    
    def test_complex_sentences(self):
        """测试复杂句式"""
        test_cases = [
            # 否定+程度+情感词
            ("不是很好", ["不", "是", "很", "好"], SentimentPolarity.NEGATIVE),
            ("不太好", ["不", "太", "好"], SentimentPolarity.NEGATIVE),
            ("不非常好", ["不", "非常", "好"], SentimentPolarity.NEGATIVE),
            
            # 双重否定
            ("不是不好", ["不", "是", "不", "好"], SentimentPolarity.POSITIVE),
            ("不不太好", ["不", "不", "太", "好"], SentimentPolarity.POSITIVE),
            
            # 多重程度副词
            ("非常很好", ["非常", "很", "好"], SentimentPolarity.POSITIVE),
            ("非常非常好", ["非常", "非常", "好"], SentimentPolarity.POSITIVE),
        ]
        
        for text, words, expected_polarity in test_cases:
            with self.subTest(text=text):
                result = self.dictionary.analyze(text, words)
                self.assertEqual(result.polarity, expected_polarity, 
                                f"'{text}' 应该被识别为 {expected_polarity.value}")
    
    def test_edge_cases(self):
        """测试边界情况"""
        # 测试超过3个词的间隔
        result1 = self.dictionary.analyze("我 不 认为 这 好", ["我", "不", "认为", "这", "好"])
        # 否定词与情感词间隔2个词，在3个词范围内，应该被识别为否定
        self.assertLess(result1.score, 0)
        
        # 测试多个否定词
        result2 = self.dictionary.analyze("不 不 不 好", ["不", "不", "不", "好"])
        # 奇数个否定词，应该被识别为否定
        self.assertLess(result2.score, 0)
        
        # 测试多个程度副词
        result3 = self.dictionary.analyze("非常 很 特别 好", ["非常", "很", "特别", "好"])
        # 多个程度副词应该叠加
        base_result = self.dictionary.analyze("好", ["好"])
        self.assertGreater(abs(result3.score), abs(base_result.score))


if __name__ == '__main__':
    unittest.main()