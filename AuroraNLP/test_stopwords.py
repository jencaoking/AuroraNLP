#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试停用词分级功能
"""

from AuroraNLP.stopwords import StopWords


def test_default_stopwords():
    """测试默认停用词加载"""
    print("=== 测试默认停用词加载 ===")
    stopwords = StopWords()
    print(f"默认停用词数量: {len(stopwords)}")
    print(f"示例停用词: {list(stopwords.get_stopwords())[:10]}")
    print(f"'的'是否是停用词: {'的' in stopwords}")
    print(f"'测试'是否是停用词: {'测试' in stopwords}")
    print()


def test_scenario_stopwords():
    """测试场景化停用词加载"""
    print("=== 测试场景化停用词加载 ===")
    
    # 测试新闻分析场景
    news_stopwords = StopWords(scenario='news_analysis')
    print(f"新闻分析场景停用词数量: {len(news_stopwords)}")
    print(f"'报道'是否是停用词: {'报道' in news_stopwords}")
    print(f"'新闻'是否是停用词: {'新闻' in news_stopwords}")
    
    # 测试医疗研究场景
    medical_stopwords = StopWords(scenario='medical_research')
    print(f"医疗研究场景停用词数量: {len(medical_stopwords)}")
    print(f"'医院'是否是停用词: {'医院' in medical_stopwords}")
    print(f"'医生'是否是停用词: {'医生' in medical_stopwords}")
    
    # 测试法律文档场景
    legal_stopwords = StopWords(scenario='legal_document')
    print(f"法律文档场景停用词数量: {len(legal_stopwords)}")
    print(f"'法律'是否是停用词: {'法律' in legal_stopwords}")
    print(f"'法规'是否是停用词: {'法规' in legal_stopwords}")
    
    # 测试电商分析场景
    ecommerce_stopwords = StopWords(scenario='ecommerce_analysis')
    print(f"电商分析场景停用词数量: {len(ecommerce_stopwords)}")
    print(f"'电商'是否是停用词: {'电商' in ecommerce_stopwords}")
    print(f"'电子商务'是否是停用词: {'电子商务' in ecommerce_stopwords}")
    print()


def test_filter_function():
    """测试停用词过滤功能"""
    print("=== 测试停用词过滤功能 ===")
    
    # 测试通用场景
    general_stopwords = StopWords(scenario='general')
    test_words = ['这', '是', '一个', '测试', '句子', '用于', '测试', '停用词', '过滤']
    filtered_words = general_stopwords.filter(test_words)
    print(f"原始词: {test_words}")
    print(f"过滤后: {filtered_words}")
    
    # 测试新闻场景
    news_stopwords = StopWords(scenario='news_analysis')
    news_words = ['据', '报道', '这', '是', '一条', '重要', '新闻']
    filtered_news = news_stopwords.filter(news_words)
    print(f"原始新闻词: {news_words}")
    print(f"过滤后: {filtered_news}")
    print()


def test_load_methods():
    """测试不同加载方法"""
    print("=== 测试不同加载方法 ===")
    
    # 测试加载通用停用词
    stopwords = StopWords(load_default=False)
    print(f"初始停用词数量: {len(stopwords)}")
    
    stopwords.load_common_stopwords()
    print(f"加载通用停用词后: {len(stopwords)}")
    
    # 测试加载领域停用词
    stopwords.load_domain_stopwords('news')
    print(f"加载新闻领域停用词后: {len(stopwords)}")
    
    stopwords.load_domain_stopwords('medical')
    print(f"加载医疗领域停用词后: {len(stopwords)}")
    print()


def test_scenarios_info():
    """测试场景信息"""
    print("=== 测试场景信息 ===")
    stopwords = StopWords(load_default=False)
    scenarios = stopwords.get_scenarios()
    print(f"可用场景数量: {len(scenarios)}")
    for name, info in scenarios.items():
        print(f"场景: {name} - {info['name']}")
        print(f"  描述: {info['description']}")
    print()


if __name__ == '__main__':
    test_default_stopwords()
    test_scenario_stopwords()
    test_filter_function()
    test_load_methods()
    test_scenarios_info()
    print("所有测试完成！")
