#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练语料库构建示例
"""

from AuroraNLP import CorpusBuilder, CorpusManager
import os
import sys

# 获取包内的 data 目录路径
data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "AuroraNLP", "data")
os.makedirs(data_dir, exist_ok=True)


def demo_corpus_builder():
    """演示语料库构建功能"""
    print("=== 语料库构建示例 ===")
    
    # 创建语料库构建器
    builder = CorpusBuilder(data_dir)
    
    # 示例1: 构建单个语料库
    print("\n1. 构建单个语料库:")
    
    # 假设我们有一个简单的测试语料文件
    test_corpus_path = os.path.join(data_dir, "test_corpus.txt")
    
    # 确保测试文件存在
    if not os.path.exists(test_corpus_path):
        # 创建测试语料文件
        test_content = "这 是 一个 测试 句子。\n我们 在 测试 语料库 构建。\n"
        os.makedirs(os.path.dirname(test_corpus_path), exist_ok=True)
        with open(test_corpus_path, 'w', encoding='utf-8') as f:
            f.write(test_content)
        print(f"创建测试语料文件: {test_corpus_path}")
    
    # 构建语料库配置
    corpus_config = {
        "output_path": os.path.join(data_dir, "train_corpus.txt"),
        "corpora": [
            {"type": "custom", "path": test_corpus_path}
        ],
        "format": "segmented"
    }
    
    # 构建语料库
    result_path = builder.build_corpus(corpus_config)
    print(f"语料库构建完成: {result_path}")
    
    # 计算统计信息
    stats = builder.statistics(result_path)
    print("语料库统计信息:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # 分割语料库
    print("\n2. 分割语料库:")
    train_path, val_path, test_path = builder.split_corpus(result_path, 0.8, 0.1)
    print(f"训练集: {train_path}")
    print(f"验证集: {val_path}")
    print(f"测试集: {test_path}")


def demo_corpus_manager():
    """演示语料库管理器功能"""
    print("\n=== 语料库管理器示例 ===")
    
    # 创建语料库管理器
    manager = CorpusManager(data_dir)
    
    # 注册语料库
    print("\n1. 注册语料库:")
    
    # 注册测试语料库
    test_corpus_path = os.path.join(data_dir, "test_corpus.txt")
    manager.register_corpus(
        name="test_corpus",
        corpus_type="custom",
        path=test_corpus_path,
        description="测试语料库"
    )
    
    # 列出所有语料库
    print("\n2. 列出所有语料库:")
    corpora = manager.list_corpora()
    for corpus in corpora:
        print(f"  - {corpus['name']} ({corpus['type']}): {corpus['description']}")
    
    # 构建组合语料库
    print("\n3. 构建组合语料库:")
    combined_path = manager.build_combined_corpus(
        name="combined",
        corpus_names=["test_corpus"],
        output_format="segmented"
    )
    print(f"组合语料库构建完成: {combined_path}")
    
    # 获取语料库信息
    print("\n4. 获取语料库信息:")
    corpus_info = manager.get_corpus("combined")
    if corpus_info:
        print(f"语料库: {corpus_info['name']}")
        print(f"类型: {corpus_info['type']}")
        print(f"路径: {corpus_info['path']}")
        print(f"描述: {corpus_info['description']}")
    
    # 移除语料库
    print("\n5. 移除语料库:")
    manager.remove_corpus("test_corpus")
    print("语料库 test_corpus 已移除")


if __name__ == "__main__":
    demo_corpus_builder()
    demo_corpus_manager()
    print("\n示例执行完成！")
