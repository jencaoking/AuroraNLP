#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试地区代码参数验证功能
"""

from AuroraNLP.segmentor import Segmentor


def test_region_validation():
    """测试地区代码参数验证"""
    print("=== 测试地区代码参数验证 ===")
    
    # 创建分词器
    segmentor = Segmentor()
    
    # 测试1: 有效地区代码
    print("\n1. 测试有效地区代码:")
    try:
        result = segmentor.simplified_to_traditional("我爱中文", region="tw")
        print("台湾地区代码测试通过:", result)
    except Exception as e:
        print("台湾地区代码测试失败:", e)
    
    try:
        result = segmentor.simplified_to_traditional("我爱中文", region="hk")
        print("香港地区代码测试通过:", result)
    except Exception as e:
        print("香港地区代码测试失败:", e)
    
    try:
        result = segmentor.simplified_to_traditional("我爱中文", region="mo")
        print("澳门地区代码测试通过:", result)
    except Exception as e:
        print("澳门地区代码测试失败:", e)
    
    # 测试2: 无效地区代码
    print("\n2. 测试无效地区代码:")
    test_cases = ["cn", "us", "jp", "", 123, None]
    
    for invalid_region in test_cases:
        try:
            segmentor.simplified_to_traditional("我爱中文", region=invalid_region)
            print(f"无效地区代码 '{invalid_region}' 测试失败: 未抛出异常")
        except ValueError as e:
            if "无效的地区代码" in str(e):
                print(f"无效地区代码 '{invalid_region}' 测试通过: 正确抛出异常")
            else:
                print(f"无效地区代码 '{invalid_region}' 测试失败: 异常信息不正确")
        except Exception as e:
            print(f"无效地区代码 '{invalid_region}' 测试失败: 抛出了意外异常: {e}")
    
    # 测试3: 其他方法的地区代码验证
    print("\n3. 测试其他方法的地区代码验证:")
    
    # 测试traditional_to_simplified
    try:
        segmentor.traditional_to_simplified("我愛中文", region="invalid")
        print("traditional_to_simplified 测试失败: 未抛出异常")
    except ValueError as e:
        print("traditional_to_simplified 测试通过: 正确抛出异常")
    
    # 测试segment_traditional
    try:
        segmentor.segment_traditional("我愛中文", region="invalid")
        print("segment_traditional 测试失败: 未抛出异常")
    except ValueError as e:
        print("segment_traditional 测试通过: 正确抛出异常")
    
    # 测试segment_with_traditional
    try:
        segmentor.segment_with_traditional("我愛中文", region="invalid")
        print("segment_with_traditional 测试失败: 未抛出异常")
    except ValueError as e:
        print("segment_with_traditional 测试通过: 正确抛出异常")
    
    print("\n=== 测试完成 ===")


if __name__ == "__main__":
    test_region_validation()
