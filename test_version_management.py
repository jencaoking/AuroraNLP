#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试词典版本管理功能
"""

import sys
import os

# 添加AuroraNLP父目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from AuroraNLP.AuroraNLP import VersionedDictionary

def test_version_management():
    """测试版本管理功能"""
    print("=== 测试词典版本管理功能 ===")
    
    # 创建支持版本控制的词典
    dict_with_version = VersionedDictionary(load_default=False)
    
    # 初始状态
    print("\n1. 初始状态")
    print(f"当前版本: {dict_with_version.get_current_version()}")
    print(f"词典词语数量: {len(dict_with_version.get_words())}")
    
    # 第一次提交：添加词语
    print("\n2. 第一次提交：添加词语")
    dict_with_version.add_word("测试", "v", 1.0, 10)
    dict_with_version.add_word("版本", "n", 1.0, 10)
    dict_with_version.add_word("管理", "n", 1.0, 10)
    version1_id = dict_with_version.commit("添加测试词语")
    print(f"提交成功，版本ID: {version1_id}")
    print(f"当前版本: {dict_with_version.get_current_version().version_id}")
    print(f"词典词语数量: {len(dict_with_version.get_words())}")
    
    # 第二次提交：修改词语
    print("\n3. 第二次提交：修改词语")
    dict_with_version.add_word("测试", "v", 2.0, 20)  # 修改权重和优先级
    dict_with_version.add_word("功能", "n", 1.0, 10)  # 新增词语
    version2_id = dict_with_version.commit("修改测试词语，添加功能词语")
    print(f"提交成功，版本ID: {version2_id}")
    print(f"当前版本: {dict_with_version.get_current_version().version_id}")
    print(f"词典词语数量: {len(dict_with_version.get_words())}")
    
    # 第三次提交：删除词语
    print("\n4. 第三次提交：删除词语")
    dict_with_version.remove_word("管理")
    version3_id = dict_with_version.commit("删除管理词语")
    print(f"提交成功，版本ID: {version3_id}")
    print(f"当前版本: {dict_with_version.get_current_version().version_id}")
    print(f"词典词语数量: {len(dict_with_version.get_words())}")
    
    # 查看版本历史
    print("\n5. 版本历史")
    history = dict_with_version.get_version_history()
    for i, version in enumerate(history):
        print(f"版本 {i+1}: {version.version_id} - {version.message} ({version.timestamp_str})")
    
    # 测试版本恢复
    print("\n6. 测试版本恢复")
    print(f"恢复到版本 {version1_id}")
    dict_with_version.checkout(version1_id)
    print(f"当前版本: {dict_with_version.get_current_version().version_id}")
    print(f"词典词语数量: {len(dict_with_version.get_words())}")
    print(f"词典词语: {dict_with_version.get_words()}")
    
    # 测试版本差异比较
    print("\n7. 测试版本差异比较")
    diff_result = dict_with_version.version_manager.diff(version1_id, version3_id)
    print(f"版本 {version1_id} 和 {version3_id} 的差异:")
    print(f"新增词语: {list(diff_result['changes']['added'].keys())}")
    print(f"删除词语: {list(diff_result['changes']['deleted'].keys())}")
    print(f"修改词语: {list(diff_result['changes']['modified'].keys())}")
    
    # 测试回滚功能
    print("\n8. 测试回滚功能")
    print("回滚到上一个版本")
    rolled_back_version = dict_with_version.rollback(1)
    print(f"回滚成功，当前版本: {rolled_back_version}")
    print(f"词典词语数量: {len(dict_with_version.get_words())}")
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    test_version_management()
