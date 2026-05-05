#!/usr/bin/env python3
"""
基础分词示例
演示 AuroraNLP 的各种分词功能
"""

from AuroraNLP import Segmentor


def basic_segmentation():
    """基础分词"""
    print("=" * 60)
    print("1. 基础分词")
    print("=" * 60)
    
    seg = Segmentor()
    text = "中文分词是自然语言处理的基础任务"
    words = seg.segment(text)
    print(f"原文: {text}")
    print(f"分词: {' / '.join(words)}")
    print()


def pos_tagging():
    """词性标注"""
    print("=" * 60)
    print("2. 词性标注")
    print("=" * 60)
    
    seg = Segmentor()
    text = "我喜欢吃苹果"
    words_with_pos = seg.segment_with_pos(text)
    
    print(f"原文: {text}")
    print("词性标注结果:")
    for word, pos in words_with_pos:
        print(f"  {word}: {pos}")
    print()


def different_strategies():
    """不同的分词策略"""
    print("=" * 60)
    print("3. 不同的分词策略")
    print("=" * 60)
    
    text = "人工智能正在改变世界"
    
    strategies = [
        ("默认", {}),
        ("HMM", {"use_hmm": True}),
        ("CRF", {"use_crf": True}),
        ("感知机", {"use_perceptron": True}),
        ("词格", {"use_lattice": True}),
        ("混合（推荐）", {"use_hybrid": True}),
    ]
    
    for name, kwargs in strategies:
        seg = Segmentor(**kwargs)
        words = seg.segment(text)
        print(f"{name}: {' / '.join(words)}")
    print()


def user_dictionary():
    """用户词典"""
    print("=" * 60)
    print("4. 用户词典")
    print("=" * 60)
    
    from AuroraNLP import UserDictionary
    
    text = "这是一个自定义词的测试"
    
    # 不使用用户词典
    seg1 = Segmentor()
    words1 = seg1.segment(text)
    print(f"不使用用户词典: {' / '.join(words1)}")
    
    # 使用用户词典
    user_dict = UserDictionary()
    user_dict.add_word("自定义词", "n", 10.0)
    
    seg2 = Segmentor(dictionary=user_dict)
    words2 = seg2.segment(text)
    print(f"使用用户词典: {' / '.join(words2)}")
    print()


def domain_dictionary():
    """领域词典"""
    print("=" * 60)
    print("5. 领域词典")
    print("=" * 60)
    
    from AuroraNLP import DomainDictionaryManager
    
    text = "患者出现发热咳嗽症状，医生建议做CT检查"
    
    manager = DomainDictionaryManager()
    manager.load_domain("medical")
    
    seg = Segmentor(dictionary=manager.combined_dictionary)
    words = seg.segment(text)
    print(f"原文: {text}")
    print(f"医疗领域分词: {' / '.join(words)}")
    print()


def traditional_chinese():
    """繁体中文"""
    print("=" * 60)
    print("6. 繁体中文")
    print("=" * 60)
    
    from AuroraNLP import TraditionalChineseConverter
    
    converter = TraditionalChineseConverter()
    
    traditional = "這是一段繁體中文"
    simplified = converter.to_simplified(traditional)
    back_to_traditional = converter.to_traditional(simplified)
    
    print(f"繁体: {traditional}")
    print(f"简体: {simplified}")
    print(f"转回繁体: {back_to_traditional}")
    print()


def main():
    print("\n")
    basic_segmentation()
    pos_tagging()
    different_strategies()
    user_dictionary()
    domain_dictionary()
    traditional_chinese()
    print("\n")


if __name__ == "__main__":
    main()
