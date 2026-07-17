#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BiLSTM-CRF 模型使用示例
========================

本示例展示如何使用 AuroraNLP 的 BiLSTM-CRF 模型进行序列标注任务。
"""

import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from AuroraNLP import BiLSTMCRF


def create_sample_data():
    """创建示例数据"""
    # 简单的序列标注示例：命名实体识别
    # 标签: 0=O, 1=B-PER, 2=I-PER, 3=B-LOC, 4=I-LOC
    
    # 词汇表（简化版）
    vocab = {
        '我': 1, '是': 2, '中国': 3, '人': 4, '来自': 5, '北京': 6, '张三': 7, '李四': 8, '上海': 9
    }
    
    # 训练数据：(输入序列, 标签序列)
    train_data = [
        ([1, 2, 3, 4], [0, 0, 3, 4]),  # 我是中国人 -> O O B-LOC I-LOC
        ([1, 5, 6], [0, 0, 3]),          # 我来自北京 -> O O B-LOC
        ([7, 2, 5, 9], [1, 0, 0, 3]),    # 张三来自上海 -> B-PER O O B-LOC
        ([8, 2, 3, 4], [1, 0, 3, 4])     # 李四是中国人 -> B-PER O B-LOC I-LOC
    ]
    
    # 验证数据
    val_data = [
        ([1, 5, 9], [0, 0, 3]),          # 我来自上海 -> O O B-LOC
        ([7, 2, 5, 6], [1, 0, 0, 3])     # 张三来自北京 -> B-PER O O B-LOC
    ]
    
    return vocab, train_data, val_data


def main():
    """主函数"""
    print("BiLSTM-CRF 模型使用示例")
    print("=" * 50)
    
    # 创建示例数据
    vocab, train_data, val_data = create_sample_data()
    
    # 词汇表大小和标签集大小
    vocab_size = len(vocab) + 1  # +1 用于 padding
    tagset_size = 5  # 0=O, 1=B-PER, 2=I-PER, 3=B-LOC, 4=I-LOC
    
    # 创建 BiLSTM-CRF 模型
    model = BiLSTMCRF(
        vocab_size=vocab_size,
        tagset_size=tagset_size,
        embedding_dim=64,
        hidden_dim=128,
        num_layers=2,
        dropout=0.5
    )
    
    # 检查模型是否可用
    if not model.is_available():
        print("错误: 深度学习框架不可用，请安装 PyTorch")
        return
    
    print("模型创建成功，开始训练...")
    
    # 训练模型
    model.train(
        train_data=train_data,
        val_data=val_data,
        epochs=10,
        batch_size=2,
        learning_rate=0.001
    )
    
    print("训练完成，开始预测...")
    
    # 测试数据
    test_data = [
        [1, 5, 6],  # 我来自北京
        [8, 2, 5, 9]  # 李四来自上海
    ]
    
    # 预测
    predictions = model.predict(test_data)
    
    # 标签映射
    tag_map = {0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-LOC', 4: 'I-LOC'}
    
    # 打印预测结果
    print("预测结果:")
    for i, pred in enumerate(predictions):
        print(f"输入序列 {i+1}: {test_data[i]}")
        print(f"预测标签: {[tag_map[t] for t in pred]}")
        print()
    
    # 保存模型
    model_path = "bilstm_crf_model.pth"
    model.save(model_path)
    print(f"模型已保存到: {model_path}")
    
    # 加载模型
    new_model = BiLSTMCRF(
        vocab_size=vocab_size,
        tagset_size=tagset_size
    )
    new_model.load(model_path)
    print("模型加载成功")
    
    # 使用加载的模型进行预测
    loaded_predictions = new_model.predict(test_data)
    print("加载模型的预测结果:")
    for i, pred in enumerate(loaded_predictions):
        print(f"输入序列 {i+1}: {test_data[i]}")
        print(f"预测标签: {[tag_map[t] for t in pred]}")
        print()


if __name__ == "__main__":
    main()
