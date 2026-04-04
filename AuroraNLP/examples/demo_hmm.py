"""
HMM 中文分词演示示例
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from AuroraNLP import Segmentor


def main():
    print("=" * 60)
    print("HMM 隐马尔可夫模型中文分词演示")
    print("=" * 60)
    
    seg = Segmentor()
    
    corpus_path = os.path.join(
        os.path.dirname(__file__),
        '..',
        'AuroraNLP',
        'data',
        'train_corpus.txt'
    )
    
    print(f"\n正在从语料库训练 HMM 模型...")
    print(f"语料库路径: {corpus_path}")
    
    if os.path.exists(corpus_path):
        seg.train_hmm_from_file(corpus_path)
        print("✓ HMM 模型训练完成！")
    else:
        print("语料库文件不存在，使用默认语料训练...")
        default_corpus = [
            ['我', '爱', '自然语言处理'],
            ['今天', '天气', '很好'],
            ['我们', '在', '学习', '中文', '分词'],
            ['北京', '是', '中国', '的', '首都'],
            ['这个', '项目', '很', '有趣']
        ]
        seg.train_hmm(default_corpus)
        print("✓ HMM 模型训练完成！")
    
    model_info = seg.get_hmm_model_info()
    print(f"\n模型信息:")
    print(f"  - 总状态数: {model_info['total_states']}")
    print(f"  - 各状态计数: {model_info['state_counts']}")
    print(f"  - 词汇表大小: {model_info['vocabulary_sizes']}")
    
    test_sentences = [
        "我爱自然语言处理",
        "今天天气很好",
        "我们在学习中文分词",
        "深度学习改变世界",
        "人工智能正在快速发展"
    ]
    
    print("\n" + "=" * 60)
    print("分词结果对比")
    print("=" * 60)
    
    for sentence in test_sentences:
        print(f"\n原句: {sentence}")
        
        print("  词典分词 (双向最大匹配): ", end="")
        words_dict = seg.segment(sentence, mode='bidirectional')
        print(" / ".join(words_dict))
        
        print("  HMM 分词:                ", end="")
        words_hmm = seg.segment(sentence, mode='hmm')
        print(" / ".join(words_hmm))
        
        print("  HMM 状态标注:            ", end="")
        states = seg.segment_with_hmm_states(sentence)
        state_str = " ".join([f"{char}({state})" for char, state in states])
        print(state_str)
    
    print("\n" + "=" * 60)
    print("HMM 模型优势")
    print("=" * 60)
    print("✓ 能够识别未登录词 (OOV)")
    print("✓ 基于统计模型，泛化能力强")
    print("✓ 可以处理歧义切分问题")
    print("✓ 不依赖大规模词典")
    
    model_path = "hmm_model.pkl"
    seg.save_hmm_model(model_path)
    print(f"\n✓ 模型已保存到: {model_path}")
    
    seg2 = Segmentor()
    seg2.load_hmm_model(model_path)
    print(f"✓ 模型已加载，可以继续使用")
    
    test_text = "自然语言处理很有趣"
    words = seg2.segment(test_text, mode='hmm')
    print(f"\n加载模型后分词测试: {' / '.join(words)}")
    
    if os.path.exists(model_path):
        os.remove(model_path)
        print(f"\n✓ 清理临时模型文件")


if __name__ == '__main__':
    main()
