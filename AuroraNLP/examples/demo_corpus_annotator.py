#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
语料自动标注示例

展示如何使用 CorpusAnnotator 进行半监督标注、主动学习和标注质量评估
"""

from AuroraNLP import (
    CorpusAnnotator, 
    AnnotationManager, 
    ActiveLearningStrategy, 
    AnnotationQualityEvaluator,
    Segmentor
)
import json
import os

def demo_semi_supervised_annotation():
    """演示半监督标注功能"""
    print("=== 半监督标注演示 ===")
    
    # 创建标注器
    annotator = CorpusAnnotator()
    
    # 添加一些未标注数据
    test_data = [
        "这是一个测试句子",
        "语料自动标注功能非常强大",
        "AuroraNLP 是一个专业的 NLP 工具包",
        "今天天气真好",
        "机器学习正在改变世界"
    ]
    
    for text in test_data:
        annotator.unannotated_data.append(text)
    
    # 创建一个简单的模型（使用分词器作为示例）
    class SimpleModel:
        def __init__(self):
            self.segmentor = Segmentor()
        
        def analyze(self, text):
            result = self.segmentor.segment(text)
            # 模拟置信度
            confidence = 0.8 + (len(text) / 100)
            
            class Result:
                def __init__(self, words, confidence):
                    self.words = words
                    self.confidence = confidence
                
                def to_dict(self):
                    return {"words": self.words, "confidence": self.confidence}
            
            return Result(result, confidence)
    
    model = SimpleModel()
    
    # 执行半监督标注
    annotated_samples = annotator.semi_supervised_annotate(model, sample_size=5, confidence_threshold=0.8)
    
    print(f"半监督标注完成，标注了 {len(annotated_samples)} 个样本")
    for sample in annotated_samples:
        print(f"文本: {sample['text']}")
        print(f"标注: {sample['annotation']}")
        print(f"置信度: {sample['confidence']}")
        print()
    
    # 查看统计信息
    stats = annotator.get_statistics()
    print("统计信息:")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print()

def demo_active_learning():
    """演示主动学习功能"""
    print("=== 主动学习演示 ===")
    
    # 创建标注器
    annotator = CorpusAnnotator()
    
    # 添加一些未标注数据
    test_data = [
        "这是一个常见的句子",
        "语料自动标注功能需要更多样例",
        "AuroraNLP 支持多种 NLP 任务",
        "今天的天气真的非常好",
        "机器学习和深度学习正在快速发展",
        "自然语言处理是人工智能的重要分支",
        "分词是 NLP 的基础任务",
        "命名实体识别可以识别人名、地名等实体",
        "词性标注可以标记词语的语法类别",
        "依存句法分析可以分析句子的结构"
    ]
    
    for text in test_data:
        annotator.unannotated_data.append(text)
    
    # 创建一个简单的模型
    class SimpleModel:
        def __init__(self):
            self.segmentor = Segmentor()
        
        def analyze(self, text):
            result = self.segmentor.segment(text)
            # 模拟置信度（长度越长，置信度越低）
            confidence = max(0.5, 1.0 - (len(text) / 100))
            
            class Result:
                def __init__(self, words, confidence):
                    self.words = words
                    self.confidence = confidence
                
                def to_dict(self):
                    return {"words": self.words, "confidence": self.confidence}
            
            return Result(result, confidence)
    
    model = SimpleModel()
    
    # 使用不确定性采样
    uncertain_samples = annotator.active_learning_sample(model, sample_size=3, strategy='uncertainty')
    print("基于不确定性采样的样本:")
    for sample in uncertain_samples:
        print(f"- {sample}")
    print()
    
    # 使用多样性采样
    diverse_samples = annotator.active_learning_sample(model, sample_size=3, strategy='diversity')
    print("基于多样性采样的样本:")
    for sample in diverse_samples:
        print(f"- {sample}")
    print()

def demo_annotation_quality():
    """演示标注质量评估功能"""
    print("=== 标注质量评估演示 ===")
    
    # 创建标注器
    annotator = CorpusAnnotator()
    
    # 添加一些标注数据
    annotated_data = [
        {
            "text": "这是一个测试句子",
            "annotation": {"words": ["这是", "一个", "测试", "句子"], "confidence": 0.9},
            "confidence": 0.9,
            "method": "semi_supervised"
        },
        {
            "text": "语料自动标注功能",
            "annotation": {"words": ["语料", "自动", "标注", "功能"], "confidence": 0.85},
            "confidence": 0.85,
            "method": "semi_supervised"
        },
        {
            "text": "AuroraNLP 工具包",
            "annotation": {"words": ["AuroraNLP", "工具包"], "confidence": 0.95},
            "confidence": 0.95,
            "method": "manual"
        }
    ]
    
    annotator.annotated_data = annotated_data
    annotator.unannotated_data = ["未标注数据1", "未标注数据2"]
    
    # 评估标注质量
    quality_metrics = annotator.evaluate_annotation_quality()
    print("标注质量评估结果:")
    print(json.dumps(quality_metrics, indent=2, ensure_ascii=False))
    print()
    
    # 获取质量报告
    quality_report = annotator.get_quality_report()
    print("质量报告:")
    print(json.dumps(quality_report, indent=2, ensure_ascii=False))
    print()

def demo_annotation_manager():
    """演示标注管理器功能"""
    print("=== 标注管理器演示 ===")
    
    # 创建标注管理器
    manager = AnnotationManager()
    
    # 创建多个标注器
    annotator1 = manager.create_annotator("segmentation")
    annotator2 = manager.create_annotator("ner")
    
    # 为第一个标注器添加数据
    annotator1.annotated_data = [
        {
            "text": "这是分词测试",
            "annotation": {"words": ["这是", "分词", "测试"], "confidence": 0.9},
            "confidence": 0.9,
            "method": "semi_supervised"
        }
    ]
    
    # 为第二个标注器添加数据
    annotator2.annotated_data = [
        {
            "text": "张三在北京大学工作",
            "annotation": {"entities": [{"text": "张三", "type": "PER"}, {"text": "北京大学", "type": "ORG"}], "confidence": 0.85},
            "confidence": 0.85,
            "method": "semi_supervised"
        }
    ]
    
    # 评估所有标注器
    reports = manager.evaluate_all_annotators()
    print("所有标注器评估结果:")
    for name, report in reports.items():
        print(f"{name}:")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        print()
    
    # 获取全局报告
    global_report = manager.get_global_report()
    print("全局评估报告:")
    print(json.dumps(global_report, indent=2, ensure_ascii=False))
    print()

def demo_active_learning_strategies():
    """演示主动学习策略"""
    print("=== 主动学习策略演示 ===")
    
    # 创建测试数据
    test_data = [
        "这是一个短句子",
        "这是一个稍微长一点的句子",
        "这是一个包含更多内容的较长句子，用于测试多样性",
        "机器学习是人工智能的重要分支",
        "自然语言处理涉及很多复杂的技术"
    ]
    
    # 创建一个简单的模型
    class SimpleModel:
        def __init__(self):
            self.segmentor = Segmentor()
        
        def analyze(self, text):
            result = self.segmentor.segment(text)
            # 模拟置信度
            confidence = max(0.5, 1.0 - (len(text) / 200))
            
            class Result:
                def __init__(self, words, confidence):
                    self.words = words
                    self.confidence = confidence
                
                def to_dict(self):
                    return {"words": self.words, "confidence": self.confidence}
            
            return Result(result, confidence)
    
    model = SimpleModel()
    
    # 使用不确定性采样
    uncertain_samples = ActiveLearningStrategy.uncertainty_sampling(model, test_data, n=2)
    print("不确定性采样结果:")
    for sample in uncertain_samples:
        print(f"- {sample}")
    print()
    
    # 使用多样性采样
    diverse_samples = ActiveLearningStrategy.diversity_sampling(test_data, n=2)
    print("多样性采样结果:")
    for sample in diverse_samples:
        print(f"- {sample}")
    print()
    
    # 使用组合采样
    combined_samples = ActiveLearningStrategy.combination_sampling(model, test_data, n=2, alpha=0.5)
    print("组合采样结果:")
    for sample in combined_samples:
        print(f"- {sample}")
    print()

def main():
    """主函数"""
    print("AuroraNLP 语料自动标注功能演示")
    print("=" * 60)
    
    try:
        demo_semi_supervised_annotation()
        demo_active_learning()
        demo_annotation_quality()
        demo_annotation_manager()
        demo_active_learning_strategies()
        
        print("演示完成！")
    except Exception as e:
        print(f"演示过程中出现错误: {e}")

if __name__ == "__main__":
    main()