#!/usr/bin/env python
"""
AuroraNLP - 功能完整性验证脚本

验证所有模块是否都能正常导入和使用
"""

import sys
import os

def main():
    print("=" * 60)
    print("AuroraNLP 功能完整性验证")
    print("=" * 60)
    
    # 1. 测试核心模块
    print("\n1. 核心模块导入检查...")
    
    modules_to_check = [
        "AuroraNLP.dictionary.trie",
        "AuroraNLP.dictionary.dictionary",
        "AuroraNLP.core.tokenizer",
        "AuroraNLP.dictionary.stopwords",
        "AuroraNLP.segmentation.segmentor",
        "AuroraNLP.segmentation.hmm",
        "AuroraNLP.segmentation.crf",
        "AuroraNLP.segmentation.perceptron",
        "AuroraNLP.segmentation.ngram",
        "AuroraNLP.segmentation.lattice",
        "AuroraNLP.ner.ner",
        "AuroraNLP.parsing.pos_tagger",
        "AuroraNLP.text_analysis.sentiment",
        "AuroraNLP.text_analysis.keyword_extractor",
        "AuroraNLP.text_analysis.similarity",
        "AuroraNLP.segmentation.ambiguity",
        "AuroraNLP.segmentation.new_word_detector",
        "AuroraNLP.dictionary.domain_dictionary",
        "AuroraNLP.segmentation.hybrid",
        # 深度学习模块
        "AuroraNLP.deep_learning",
        "AuroraNLP.deep_learning.framework",
        "AuroraNLP.deep_learning.pytorch_backend",
        "AuroraNLP.deep_learning.tensorflow_backend",
        "AuroraNLP.deep_learning.bilstm_crf",
        "AuroraNLP.deep_learning.pretrained"
    ]
    
    success_count = 0
    fail_count = 0
    
    for module in modules_to_check:
        try:
            __import__(module)
            print(f"  ✓ {module}")
            success_count += 1
        except Exception as e:
            print(f"  ✗ {module} - 失败: {type(e).__name__}")
            fail_count += 1
    
    # 2. 测试分段器
    print("\n2. 分段器基本功能检查...")
    
    try:
        from AuroraNLP.segmentation.segmentor import Segmentor
        seg = Segmentor(load_default_dict=False, load_default_stopwords=False)
        
        # 添加一些测试词
        test_dict = ["我", "爱", "中国", "自然", "语言", "处理"]
        for word in test_dict:
            seg.add_word(word)
        
        text = "我爱中国自然语言处理"
        words = seg.segment(text)
        print(f"  ✓ 分词测试: {text} → {words}")
        
        # 基本方法测试
        print(f"  ✓ 模式设置: {seg.set_mode('forward')}")
        words_forward = seg.segment(text)
        print(f"    → 正向分词: {words_forward}")
        
        success_count += 1
    except Exception as e:
        print(f"  ✗ 分段器测试失败: {type(e).__name__}")
        import traceback
        print(f"    详情: {traceback.format_exc()}")
        fail_count += 1
    
    # 3. 测试深度学习模块（可选）
    print("\n3. 深度学习模块功能检查...")
    
    try:
        from AuroraNLP.deep_learning import (
            PreTrainedModelType,
            PreTrainedModelConfig,
            LightweightSegmentor,
            BERTNER,
            BERTPOS
        )
        
        config = PreTrainedModelConfig(model_type=PreTrainedModelType.ALBERT_TINY)
        print(f"  ✓ 模型配置创建: {config.model_type}")
        
        # 创建轻量分词器
        try:
            light_seg = LightweightSegmentor.create_albert_tiny()
            print(f"  ✓ 轻量分词器创建")
        except Exception:
            print(f"  · 轻量分词器初始化失败（可能依赖未安装，属于可选功能）")
        
        # 创建 NER
        try:
            ner = BERTNER(model_type=PreTrainedModelType.ALBERT_TINY)
            print(f"  ✓ BERT-NER 初始化")
        except Exception:
            print(f"  · BERT-NER 初始化失败（可选功能）")
        
        # 创建 POS
        try:
            pos = BERTPOS(model_type=PreTrainedModelType.ALBERT_TINY)
            print(f"  ✓ BERT-POS 初始化")
        except Exception:
            print(f"  · BERT-POS 初始化失败（可选功能）")
        
        success_count += 1
        
    except Exception as e:
        print(f"  ✗ 深度学习模块初始化失败: {type(e).__name__}")
        import traceback
        print(f"    详情: {traceback.format_exc()}")
        fail_count += 1
    
    # 4. 验证统计模块
    print("\n4. 统计模块检查...")
    
    try:
        from AuroraNLP.segmentation.hmm import HMMSegmentor
        from AuroraNLP.segmentation.crf import CRFModel
        from AuroraNLP.segmentation.perceptron import PerceptronSegmentor
        from AuroraNLP.segmentation.ngram import NGramModel
        
        print("  ✓ HMM/CRF/Perceptron/NGram 导入成功")
        success_count += 1
        
    except Exception as e:
        print(f"  ✗ 统计模块导入失败: {type(e).__name__}")
        fail_count += 1
    
    # 5. 高级功能模块
    print("\n5. 高级功能模块检查...")
    
    try:
        from AuroraNLP.text_analysis.sentiment import SentimentDictionary
        from AuroraNLP.text_analysis.keyword_extractor import KeywordExtractor
        from AuroraNLP.text_analysis.similarity import Similarity
        
        sentiment = SentimentDictionary(load_default=False)
        extractor = KeywordExtractor()
        
        print("  ✓ 情感/关键词/相似度模块导入成功")
        success_count += 1
        
    except Exception as e:
        print(f"  ✗ 高级功能模块导入失败: {type(e).__name__}")
        fail_count += 1
    
    # 6. 汇总结果
    print("\n" + "=" * 60)
    print("验证结果")
    print("=" * 60)
    print(f"  成功: {success_count}")
    print(f"  失败: {fail_count}")
    print("=" * 60)
    
    if fail_count == 0:
        print("\n🎉 所有核心功能验证通过！")
        return 0
    elif success_count > fail_count:
        print("\n⚠️  部分功能验证失败，但核心功能正常")
        return 1
    else:
        print("\n❌ 大量功能验证失败，需要检查")
        return 2


if __name__ == "__main__":
    sys.exit(main())
