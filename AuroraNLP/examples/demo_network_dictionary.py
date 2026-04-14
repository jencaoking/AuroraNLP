#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
网络新词库示例

展示如何使用 AuroraNLP 的网络新词库功能，包括：
1. 创建网络词典
2. 更新网络热词
3. 清理过期词汇
4. 查看网络词典统计信息
5. 在分词中使用网络词典
"""

import sys
sys.path.insert(0, '.')
from AuroraNLP import Segmentor, DictionaryService, NetworkDictionary


def demo_network_dictionary():
    """演示网络新词库功能"""
    print("=== AuroraNLP 网络新词库示例 ===")
    print()
    
    # 1. 创建词典服务
    dict_service = DictionaryService()
    
    # 2. 创建网络词典
    print("1. 创建网络词典...")
    network_dict = dict_service.create_network_dictionary(
        priority=50,  # 优先级
        update_interval=60 * 60,  # 1小时更新一次
        expiry_days=7  # 7天过期
    )
    print(f"网络词典创建成功，当前词汇量: {len(network_dict)}")
    print()
    
    # 3. 手动更新网络热词
    print("2. 手动更新网络热词...")
    added_count = dict_service.update_network_hotwords()
    print(f"更新完成，新增 {added_count} 个网络热词")
    print(f"更新后词汇量: {len(network_dict)}")
    print()
    
    # 4. 查看网络词典统计信息
    print("3. 查看网络词典统计信息...")
    stats = dict_service.get_network_dictionary_statistics()
    print(f"总词汇数: {stats.get('total_words', 0)}")
    print(f"最近7天新词: {stats.get('recent_words', 0)}")
    print(f"过期词汇数: {stats.get('expired_words', 0)}")
    print(f"上次更新时间: {stats.get('last_update', 0)}")
    print()
    
    # 5. 清理过期词汇
    print("4. 清理过期词汇...")
    expired_count = dict_service.cleanup_expired_network_words()
    print(f"清理完成，移除 {expired_count} 个过期词汇")
    print(f"清理后词汇量: {len(network_dict)}")
    print()
    
    # 6. 在分词中使用网络词典
    print("5. 在分词中使用网络词典...")
    segmentor = Segmentor()
    
    # 示例文本，包含网络热词
    test_text = """ 
    最近ChatGPT很火，大家都在讨论人工智能和机器学习，
    还有元宇宙、NFT这些新概念，
    年轻人喜欢用yyds、绝绝子这样的网络用语。
    """
    
    print("原始文本:")
    print(test_text)
    print()
    
    print("分词结果:")
    words = segmentor.segment(test_text)
    print(" ".join(words))
    print()
    
    # 7. 查看网络词典中的词汇
    print("6. 查看网络词典中的部分词汇...")
    network_words = network_dict.get_recent_words(days=7)
    print(f"最近7天的网络热词 ({len(network_words)}个):")
    for i, word in enumerate(network_words[:10]):  # 只显示前10个
        print(f"  {i+1}. {word}")
    if len(network_words) > 10:
        print(f"  ... 等{len(network_words)-10}个更多词汇")
    print()
    
    # 8. 测试网络词典的时效性管理
    print("7. 测试网络词典的时效性管理...")
    # 模拟添加一个过期词汇
    import time
    expired_word = "过期网络词"
    network_dict.add_word(expired_word, timestamp=int(time.time()) - 8 * 24 * 60 * 60)  # 8天前的词汇
    print(f"添加测试过期词汇: {expired_word}")
    
    # 再次清理过期词汇
    expired_count = dict_service.cleanup_expired_network_words()
    print(f"清理完成，移除 {expired_count} 个过期词汇")
    print(f"最终词汇量: {len(network_dict)}")
    print()
    
    print("=== 示例完成 ===")


if __name__ == "__main__":
    demo_network_dictionary()
