"""
情感词典演示

展示 AuroraNLP 情感词典模块的基本功能
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from AuroraNLP.sentiment import (
    SentimentDictionary,
    SentimentAnalyzer,
    SentimentPolarity,
    SentimentIntensity,
)


def demo_basic_usage():
    """演示基本用法"""
    print("=" * 60)
    print("情感词典基本用法演示")
    print("=" * 60)
    
    # 创建情感词典
    dictionary = SentimentDictionary()
    
    print(f"\n词典统计:")
    print(f"  正面情感词数量: {dictionary.get_positive_word_count()}")
    print(f"  负面情感词数量: {dictionary.get_negative_word_count()}")
    print(f"  情感词总数: {dictionary.get_total_word_count()}")
    print(f"  否定词数量: {dictionary.get_negation_word_count()}")
    print(f"  程度副词数量: {dictionary.get_degree_word_count()}")


def demo_word_queries():
    """演示词语查询"""
    print("\n" + "=" * 60)
    print("情感词查询演示")
    print("=" * 60)
    
    dictionary = SentimentDictionary()
    
    # 测试词语
    test_words = ["好", "优秀", "开心", "愤怒", "悲伤", "糟糕", "桌子"]
    
    print("\n词语情感分析:")
    for word in test_words:
        is_positive = dictionary.is_positive(word)
        is_negative = dictionary.is_negative(word)
        is_sentiment = dictionary.is_sentiment_word(word)
        score = dictionary.get_word_score(word)
        category = dictionary.get_word_category(word)
        intensity = dictionary.get_word_intensity(word)
        
        print(f"\n  '{word}':")
        print(f"    是情感词: {is_sentiment}")
        print(f"    是正面词: {is_positive}")
        print(f"    是负面词: {is_negative}")
        print(f"    情感分数: {score}")
        print(f"    情感类别: {category or 'N/A'}")
        print(f"    情感强度: {intensity.name if intensity else 'N/A'}")


def demo_category_queries():
    """演示类别查询"""
    print("\n" + "=" * 60)
    print("按类别查询情感词")
    print("=" * 60)
    
    dictionary = SentimentDictionary()
    
    # 正面类别
    print("\n正面情感类别:")
    for code, name in dictionary.POSITIVE_CATEGORIES.items():
        words = dictionary.get_words_by_category(code)
        print(f"  {name}({code}): {', '.join(words[:5])}{'...' if len(words) > 5 else ''}")
    
    # 负面类别
    print("\n负面情感类别:")
    for code, name in dictionary.NEGATIVE_CATEGORIES.items():
        words = dictionary.get_words_by_category(code)
        if words:
            print(f"  {name}({code}): {', '.join(words[:5])}{'...' if len(words) > 5 else ''}")


def demo_intensity_queries():
    """演示强度查询"""
    print("\n" + "=" * 60)
    print("按强度查询情感词")
    print("=" * 60)
    
    dictionary = SentimentDictionary()
    
    for intensity in SentimentIntensity:
        words = dictionary.get_words_by_intensity(intensity)
        print(f"\n  {intensity.name} (强度{intensity.value}): {len(words)}个词")
        if words:
            print(f"    示例: {', '.join(words[:5])}")


def demo_sentiment_analysis():
    """演示情感分析"""
    print("\n" + "=" * 60)
    print("文本情感分析演示")
    print("=" * 60)
    
    dictionary = SentimentDictionary()
    
    # 测试文本
    test_cases = [
        ("这个产品很好", ["这个", "产品", "很", "好"]),
        ("我非常开心", ["我", "非常", "开心"]),
        ("这太糟糕了", ["这", "太", "糟糕", "了"]),
        ("不喜欢", ["不", "喜欢"]),
        ("不难过", ["不", "难过"]),
        ("非常好", ["非常", "好"]),
    ]
    
    for text, words in test_cases:
        result = dictionary.analyze(text, words)
        
        print(f"\n  文本: '{text}'")
        print(f"    情感极性: {result.polarity.value}")
        print(f"    情感分数: {result.score:.3f}")
        print(f"    情感强度: {result.intensity:.3f}")
        print(f"    置信度: {result.confidence:.3f}")
        
        if result.positive_words:
            print(f"    正面词: {result.positive_words}")
        if result.negative_words:
            print(f"    负面词: {result.negative_words}")


def demo_sentiment_analyzer():
    """演示情感分析器"""
    print("\n" + "=" * 60)
    print("情感分析器演示")
    print("=" * 60)
    
    analyzer = SentimentAnalyzer()
    
    # 简单分析（不分词）
    print("\n  简单分析:")
    test_texts = ["好", "开心", "愤怒", "糟糕"]
    for text in test_texts:
        result = analyzer.analyze(text)
        print(f"    '{text}': {result.polarity.value} (分数: {result.score:.3f})")


def demo_degree_and_negation():
    """演示程度副词和否定词"""
    print("\n" + "=" * 60)
    print("程度副词和否定词演示")
    print("=" * 60)
    
    dictionary = SentimentDictionary()
    
    # 程度副词
    print("\n  程度副词示例:")
    degree_words = ["很", "非常", "稍微", "比较", "极", "太"]
    for word in degree_words:
        if dictionary.is_degree_word(word):
            degree = dictionary.get_degree(word)
            print(f"    '{word}': 程度系数 = {degree}")
    
    # 否定词
    print("\n  否定词示例:")
    negation_words = ["不", "没", "没有", "无", "非"]
    for word in negation_words:
        if dictionary.is_negation_word(word):
            strength = dictionary.get_negation_strength(word)
            print(f"    '{word}': 否定强度 = {strength}")


if __name__ == "__main__":
    demo_basic_usage()
    demo_word_queries()
    demo_category_queries()
    demo_intensity_queries()
    demo_sentiment_analysis()
    demo_sentiment_analyzer()
    demo_degree_and_negation()
    
    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)
