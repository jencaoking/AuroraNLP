"""
最短路径分词示例

演示如何使用Dijkstra算法进行最短路径分词
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from AuroraNLP import Segmentor
from AuroraNLP.lattice import LatticeSegmentor
from AuroraNLP.dictionary import Dictionary


def demo_basic_shortest_path():
    """基础最短路径分词示例"""
    print("=" * 60)
    print("基础最短路径分词示例")
    print("=" * 60)
    
    dict_obj = Dictionary(load_default=False)
    dict_obj.add_word("研究", "v")
    dict_obj.add_word("研究生", "n")
    dict_obj.add_word("生命", "n")
    dict_obj.add_word("的", "u")
    dict_obj.add_word("起源", "n")
    
    segmentor = LatticeSegmentor(dict_obj)
    
    text = "研究生命的起源"
    print(f"\n原文: {text}")
    
    result = segmentor.segment(text)
    print(f"分词结果: {' / '.join(result)}")
    
    lattice = segmentor.build_lattice(text)
    print(f"\n词格统计:")
    stats = lattice.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n词格可视化:")
    print(lattice.visualize())


def demo_shortest_path_vs_other_methods():
    """对比最短路径与其他分词方法"""
    print("\n" + "=" * 60)
    print("最短路径分词 vs 其他分词方法对比")
    print("=" * 60)
    
    segmentor = Segmentor(load_default_dict=False)
    segmentor.add_word("研究", "v")
    segmentor.add_word("研究生", "n")
    segmentor.add_word("生命", "n")
    segmentor.add_word("的", "u")
    segmentor.add_word("起源", "n")
    segmentor.add_word("我", "r")
    segmentor.add_word("喜欢", "v")
    segmentor.add_word("学习", "v")
    
    test_cases = [
        "研究生命的起源",
        "我喜欢学习",
        "研究生"
    ]
    
    for text in test_cases:
        print(f"\n原文: {text}")
        
        forward = segmentor.segment(text, mode='forward')
        backward = segmentor.segment(text, mode='backward')
        bidirectional = segmentor.segment(text, mode='bidirectional')
        shortest = segmentor.segment(text, mode='lattice')
        
        print(f"  正向最大匹配: {' / '.join(forward)}")
        print(f"  逆向最大匹配: {' / '.join(backward)}")
        print(f"  双向最大匹配: {' / '.join(bidirectional)}")
        print(f"  最短路径分词: {' / '.join(shortest)}")


def demo_ambiguity_resolution():
    """歧义消解示例"""
    print("\n" + "=" * 60)
    print("歧义消解示例")
    print("=" * 60)
    
    segmentor = Segmentor(load_default_dict=False)
    segmentor.add_word("南京市", "ns")
    segmentor.add_word("南京", "ns")
    segmentor.add_word("市长", "n")
    segmentor.add_word("长江", "ns")
    segmentor.add_word("大桥", "n")
    segmentor.add_word("长江大桥", "n")
    
    text = "南京市长江大桥"
    print(f"\n原文: {text}")
    
    ambiguities = segmentor.detect_lattice_ambiguity(text)
    print(f"\n检测到 {len(ambiguities)} 处歧义:")
    for i, amb in enumerate(ambiguities, 1):
        print(f"  歧义 {i}:")
        print(f"    类型: {amb['type']}")
        print(f"    位置: {amb['position']}")
        print(f"    候选词: {amb['words']}")
    
    all_segs = segmentor.get_all_lattice_segmentations(text, max_results=5)
    print(f"\n所有可能的分词结果 (共 {len(all_segs)} 种):")
    for i, seg in enumerate(all_segs, 1):
        print(f"  {i}. {' / '.join(seg)}")
    
    result = segmentor.segment(text, mode='lattice')
    print(f"\n最短路径分词结果: {' / '.join(result)}")


def demo_k_best_paths():
    """K-best路径示例"""
    print("\n" + "=" * 60)
    print("K-best路径示例")
    print("=" * 60)
    
    segmentor = Segmentor(load_default_dict=False)
    segmentor.add_word("研究", "v")
    segmentor.add_word("研究生", "n")
    segmentor.add_word("生命", "n")
    segmentor.add_word("的", "u")
    segmentor.add_word("起源", "n")
    
    text = "研究生命的起源"
    print(f"\n原文: {text}")
    
    k_best = segmentor.find_k_best_paths(text, k=3)
    print(f"\n前 {len(k_best)} 条最优路径:")
    for i, path in enumerate(k_best, 1):
        print(f"  {i}. {' / '.join(path)}")


def demo_with_pos_tags():
    """带词性标注的最短路径分词"""
    print("\n" + "=" * 60)
    print("带词性标注的最短路径分词")
    print("=" * 60)
    
    segmentor = Segmentor(load_default_dict=False)
    segmentor.add_word("我", "r")
    segmentor.add_word("喜欢", "v")
    segmentor.add_word("研究", "v")
    segmentor.add_word("研究生", "n")
    segmentor.add_word("学习", "v")
    
    text = "我喜欢研究生学习"
    print(f"\n原文: {text}")
    
    result = segmentor.segment_with_lattice_pos(text)
    print(f"\n分词结果 (带词性):")
    for word, pos in result:
        pos_name = segmentor.get_pos_tag_name(pos) if pos else "未知"
        print(f"  {word:6s} [{pos:4s}] {pos_name}")


if __name__ == "__main__":
    demo_basic_shortest_path()
    demo_shortest_path_vs_other_methods()
    demo_ambiguity_resolution()
    demo_k_best_paths()
    demo_with_pos_tags()
    
    print("\n" + "=" * 60)
    print("示例完成!")
    print("=" * 60)
