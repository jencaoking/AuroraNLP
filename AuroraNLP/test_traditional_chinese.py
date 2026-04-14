#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试繁体中文处理功能
"""

from AuroraNLP.segmentor import Segmentor


def test_traditional_chinese():
    """测试繁体中文处理"""
    print("=== 测试繁体中文处理 ===")
    
    # 创建分词器
    segmentor = Segmentor()
    
    # 测试1: 繁简转换
    print("\n1. 测试繁简转换:")
    simplified_text = "我爱中文自然语言处理"
    traditional_text = segmentor.simplified_to_traditional(simplified_text)
    print(f"简体: {simplified_text}")
    print(f"繁体: {traditional_text}")
    
    # 测试2: 地区变体
    print("\n2. 测试地区变体:")
    tw_text = segmentor.simplified_to_traditional(simplified_text, region='tw')
    hk_text = segmentor.simplified_to_traditional(simplified_text, region='hk')
    mo_text = segmentor.simplified_to_traditional(simplified_text, region='mo')
    print(f"台湾变体: {tw_text}")
    print(f"香港变体: {hk_text}")
    print(f"澳门变体: {mo_text}")
    
    # 测试3: 繁体中文分词
    print("\n3. 测试繁体中文分词:")
    traditional_input = "我愛中文自然語言處理"
    segmented = segmentor.segment_traditional(traditional_input)
    print(f"输入: {traditional_input}")
    print(f"分词结果: {segmented}")
    
    # 测试4: 混合文本处理
    print("\n4. 测试混合文本处理:")
    mixed_text = "我爱中文自然語言處理"
    segmented_mixed = segmentor.segment_with_traditional(mixed_text)
    print(f"输入: {mixed_text}")
    print(f"分词结果: {segmented_mixed}")
    
    # 测试5: 语言变体检测
    print("\n5. 测试语言变体检测:")
    tw_sample = "臺灣總統行政院"
    hk_sample = "香港特首立法會"
    mo_sample = "澳門特首大三巴"
    
    tw_detected = segmentor.detect_language_variant(tw_sample)
    hk_detected = segmentor.detect_language_variant(hk_sample)
    mo_detected = segmentor.detect_language_variant(mo_sample)
    
    print(f"台湾样本检测结果: {tw_detected}")
    print(f"香港样本检测结果: {hk_detected}")
    print(f"澳门样本检测结果: {mo_detected}")
    
    # 测试6: 繁体中文词典
    print("\n6. 测试繁体中文词典:")
    segmentor.add_traditional_word("自然語言")
    segmentor.add_traditional_word("處理")
    
    test_text = "自然語言處理是一門重要的學科"
    segmented_dict = segmentor.segment_traditional(test_text)
    print(f"输入: {test_text}")
    print(f"分词结果: {segmented_dict}")
    
    print("\n=== 测试完成 ===")


if __name__ == "__main__":
    test_traditional_chinese()
