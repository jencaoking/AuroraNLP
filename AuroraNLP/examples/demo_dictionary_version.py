#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
词典版本管理示例

演示如何使用词典版本管理功能，包括：
1. 创建支持版本控制的词典
2. 添加/删除词语
3. 提交版本
4. 查看版本历史
5. 回滚到之前的版本
6. 切换到指定版本
"""

import sys
import os

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from AuroraNLP import (
    VersionedDictionary,
    VersionedUserDictionary,
    DictionaryVersionManager
)

def demo_versioned_dictionary():
    """演示版本控制词典"""
    print("=== 版本控制词典演示 ===")
    
    # 创建版本控制词典
    versioned_dict = VersionedDictionary(load_default=False)
    
    # 添加一些词语
    versioned_dict.add_word("人工智能", "n", 1.0, 10)
    versioned_dict.add_word("机器学习", "n", 1.0, 10)
    versioned_dict.add_word("深度学习", "n", 1.0, 10)
    
    # 提交第一个版本
    version_id1 = versioned_dict.commit("初始词典版本", "admin")
    print(f"提交版本 1: {version_id1}")
    
    # 添加更多词语
    versioned_dict.add_word("自然语言处理", "n", 1.0, 10)
    versioned_dict.add_word("计算机视觉", "n", 1.0, 10)
    
    # 提交第二个版本
    version_id2 = versioned_dict.commit("添加AI相关术语", "admin")
    print(f"提交版本 2: {version_id2}")
    
    # 删除一个词语
    versioned_dict.remove_word("机器学习")
    
    # 提交第三个版本
    version_id3 = versioned_dict.commit("删除机器学习", "admin")
    print(f"提交版本 3: {version_id3}")
    
    # 查看版本历史
    print("\n版本历史:")
    history = versioned_dict.get_version_history()
    for i, version in enumerate(history):
        print(f"版本 {i+1}: {version.version_id}")
        print(f"  消息: {version.message}")
        print(f"  作者: {version.author}")
        print(f"  时间: {version.timestamp_str}")
        print()
    
    # 查看当前版本
    current_version = versioned_dict.get_current_version()
    print(f"当前版本: {current_version.version_id}")
    print(f"当前词典内容: {versioned_dict.get_words()}")
    
    # 回滚到版本 1
    print("\n回滚到版本 1...")
    rolled_back_version = versioned_dict.rollback(2)  # 回滚2步
    print(f"回滚到版本: {rolled_back_version}")
    print(f"回滚后词典内容: {versioned_dict.get_words()}")
    
    # 切换到版本 2
    print("\n切换到版本 2...")
    versioned_dict.checkout(version_id2)
    print(f"切换后词典内容: {versioned_dict.get_words()}")

def demo_versioned_user_dictionary():
    """演示版本控制用户词典"""
    print("\n=== 版本控制用户词典演示 ===")
    
    # 创建版本控制用户词典
    versioned_user_dict = VersionedUserDictionary(name="my_user_dict")
    
    # 添加一些词语
    versioned_user_dict.add_word("自定义术语1", "n")
    versioned_user_dict.add_word("自定义术语2", "n")
    
    # 提交第一个版本
    version_id1 = versioned_user_dict.commit("初始用户词典", "user")
    print(f"提交版本 1: {version_id1}")
    
    # 添加更多词语
    versioned_user_dict.add_word("自定义术语3", "n")
    
    # 提交第二个版本
    version_id2 = versioned_user_dict.commit("添加新术语", "user")
    print(f"提交版本 2: {version_id2}")
    
    # 查看版本历史
    print("\n版本历史:")
    history = versioned_user_dict.get_version_history()
    for i, version in enumerate(history):
        print(f"版本 {i+1}: {version.version_id}")
        print(f"  消息: {version.message}")
        print(f"  作者: {version.author}")
        print(f"  时间: {version.timestamp_str}")
        print()

def demo_version_manager():
    """演示版本管理器"""
    print("\n=== 版本管理器演示 ===")
    
    # 创建版本管理器
    version_manager = DictionaryVersionManager()
    
    # 创建一个普通词典
    from AuroraNLP import Dictionary
    dict1 = Dictionary(load_default=False)
    dict1.add_word("测试词1", "n")
    dict1.add_word("测试词2", "n")
    
    # 提交版本
    version_id = version_manager.commit(dict1, "测试版本", "admin")
    print(f"提交版本: {version_id}")
    
    # 查看版本历史
    print("\n版本历史:")
    history = version_manager.get_version_history()
    for i, version in enumerate(history):
        print(f"版本 {i+1}: {version.version_id}")
        print(f"  消息: {version.message}")
        print(f"  作者: {version.author}")
        print(f"  时间: {version.timestamp_str}")
        print()

if __name__ == "__main__":
    demo_versioned_dictionary()
    demo_versioned_user_dictionary()
    demo_version_manager()