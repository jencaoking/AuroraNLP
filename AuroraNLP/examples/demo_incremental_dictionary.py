#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
词典增量更新示例

演示如何使用 AuroraNLP 的词典增量更新功能，包括：
1. 热更新支持
2. 无需重启服务
3. 更新通知机制
"""

import os
import sys
import time

# 添加父目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from AuroraNLP import (
    IncrementalDictionary,
    IncrementalUserDictionary,
    HotUpdateDictionaryManager,
    DictionaryObserver,
    DictionaryUpdateEvent
)


class CustomObserver(DictionaryObserver):
    """自定义词典观察者"""
    def on_dictionary_update(self, event: DictionaryUpdateEvent) -> None:
        """词典更新回调"""
        print(f"[观察者] 词典 {event.dictionary_name} 发生更新:")
        print(f"  - 更新类型: {event.update_type}")
        print(f"  - 涉及词语: {event.words}")
        print(f"  - 更新时间: {event.timestamp_str}")
        print()


def demo_basic_incremental():
    """演示基本的增量更新功能"""
    print("=== 基本增量更新演示 ===")
    
    # 创建支持增量更新的词典
    inc_dict = IncrementalDictionary(name="test_dict")
    
    # 添加观察者
    observer = CustomObserver()
    inc_dict.add_observer(observer)
    
    # 添加词语
    print("添加词语 '人工智能'...")
    inc_dict.add_word("人工智能", "n", 10.0, 10)
    
    # 更新词语
    print("更新词语 '人工智能' 的权重...")
    inc_dict.update_word("人工智能", weight=20.0)
    
    # 删除词语
    print("删除词语 '人工智能'...")
    inc_dict.remove_word("人工智能")
    
    print()


def demo_hot_update():
    """演示热更新功能"""
    print("=== 热更新功能演示 ===")
    
    # 创建支持热更新的词典管理器
    hot_manager = HotUpdateDictionaryManager()
    
    # 创建增量词典
    inc_dict = IncrementalDictionary(name="hot_dict")
    hot_manager.register_dictionary(inc_dict)
    
    # 添加观察者
    observer = CustomObserver()
    inc_dict.add_observer(observer)
    
    # 手动触发更新
    print("手动触发词典更新...")
    words = [
        {"word": "机器学习", "pos_tag": "n", "weight": 15.0, "action": "add"},
        {"word": "深度学习", "pos_tag": "n", "weight": 18.0, "action": "add"},
        {"word": "自然语言处理", "pos_tag": "n", "weight": 20.0, "action": "add"}
    ]
    
    updated_count = hot_manager.update_dictionary("hot_dict", words)
    print(f"成功更新 {updated_count} 个词语")
    
    # 检查词典内容
    print("\n词典当前包含的词语:")
    for word in inc_dict.get_words():
        found, pos_tag, weight, priority = inc_dict.search_with_info(word)
        print(f"  - {word}: {pos_tag}, 权重: {weight}, 优先级: {priority}")
    
    print()


def demo_file_monitoring():
    """演示文件监控功能"""
    print("=== 文件监控功能演示 ===")
    
    # 创建支持热更新的词典管理器
    hot_manager = HotUpdateDictionaryManager()
    
    # 创建增量词典
    inc_dict = IncrementalDictionary(name="file_dict")
    hot_manager.register_dictionary(inc_dict)
    
    # 添加观察者
    observer = CustomObserver()
    inc_dict.add_observer(observer)
    
    # 创建临时词典文件
    import os
    temp_file = "temp_dict.txt"
    with open(temp_file, 'w', encoding='utf-8') as f:
        f.write("云计算 n 10.0 5\n")
        f.write("大数据 n 12.0 5\n")
    
    # 添加监控文件
    hot_manager.add_monitored_file(temp_file, inc_dict)
    
    # 启动文件监控
    print("启动文件监控...")
    hot_manager.start_file_monitoring()
    
    # 等待2秒
    time.sleep(2)
    
    # 修改文件
    print("\n修改词典文件，添加新词语...")
    with open(temp_file, 'a', encoding='utf-8') as f:
        f.write("物联网 n 15.0 5\n")
        f.write("区块链 n 18.0 5\n")
    
    # 等待文件监控检测到变化
    time.sleep(7)
    
    # 停止文件监控
    hot_manager.stop_file_monitoring()
    
    # 清理临时文件
    if os.path.exists(temp_file):
        os.remove(temp_file)
    
    print()


def demo_user_dictionary():
    """演示用户词典的增量更新"""
    print("=== 用户词典增量更新演示 ===")
    
    # 创建支持增量更新的用户词典
    inc_user_dict = IncrementalUserDictionary(name="user_dict")
    
    # 添加观察者
    observer = CustomObserver()
    inc_user_dict.add_observer(observer)
    
    # 添加用户词语
    print("添加用户词语...")
    inc_user_dict.add_word("自定义词汇", "n")
    inc_user_dict.add_word("专业术语", "n")
    
    # 更新用户词语
    print("更新用户词语权重...")
    inc_user_dict.update_word("自定义词汇", weight=50.0)
    
    # 检查词典内容
    print("\n用户词典当前包含的词语:")
    for word in inc_user_dict.get_words():
        found, pos_tag, weight, priority = inc_user_dict.search_with_info(word)
        print(f"  - {word}: {pos_tag}, 权重: {weight}, 优先级: {priority}")
    
    print()


if __name__ == "__main__":
    print("AuroraNLP 词典增量更新功能演示")
    print("=" * 60)
    
    # 运行各个演示
    demo_basic_incremental()
    demo_hot_update()
    demo_file_monitoring()
    demo_user_dictionary()
    
    print("演示完成！")
