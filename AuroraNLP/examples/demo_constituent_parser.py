#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
成分句法分析演示

演示PCFG和CKY算法的使用方法
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from AuroraNLP.constituent_parser import (
    ConstituentTree,
    ConstituentNode,
    GrammarRule,
    PCFG,
    CKYParser,
    ConstituentParser,
    create_sample_constituent_trees,
    CONSTITUENT_LABELS,
    POS_LABELS,
)


def demo_tree_parsing():
    print("=" * 60)
    print("1. 成分树解析演示")
    print("=" * 60)
    
    tree_str = """
    (S
        (NP (NR 我))
        (VP (VV 爱)
            (NP (NR 中国))))
    """
    
    tree = ConstituentTree.from_penn_treebank(tree_str)
    
    print(f"解析树: {tree}")
    print(f"根节点标签: {tree.root.label}")
    print(f"树的高度: {tree.get_height()}")
    print(f"句子词数: {len(tree)}")
    print(f"句子文本: {tree.get_text()}")
    print(f"词序列: {tree.get_words()}")
    
    print("\nPenn Treebank格式:")
    print(tree.to_penn_treebank())
    
    print("\nLisp格式:")
    print(tree.to_lisp_string())
    
    print("\n词性标注:")
    pos_tags = tree.get_pos_tags()
    for pos, word in pos_tags:
        print(f"  {word}: {pos} ({POS_LABELS.get(pos, '未知')})")


def demo_tree_operations():
    print("\n" + "=" * 60)
    print("2. 成分树操作演示")
    print("=" * 60)
    
    tree_str = """
    (S
        (NP (DP (DT 这))
            (NP (NN 本书)))
        (VP (ADVP (AD 很))
            (VP (VA 有趣))))
    """
    
    tree = ConstituentTree.from_penn_treebank(tree_str)
    
    print(f"句子: {tree.get_text()}")
    
    print("\n名词短语 (NP):")
    for np in tree.get_noun_phrases():
        print(f"  [{np.start}:{np.end}] {np.get_text()}")
    
    print("\n动词短语 (VP):")
    for vp in tree.get_verb_phrases():
        print(f"  [{vp.start}:{vp.end}] {vp.get_text()}")
    
    print("\n所有短语:")
    for label in ['NP', 'VP', 'ADVP', 'DP']:
        phrases = tree.get_phrases(label)
        if phrases:
            print(f"  {label} ({CONSTITUENT_LABELS.get(label, '未知')}):")
            for phrase in phrases:
                print(f"    [{phrase.start}:{phrase.end}] {phrase.get_text()}")


def demo_rule_extraction():
    print("\n" + "=" * 60)
    print("3. 文法规则提取演示")
    print("=" * 60)
    
    tree_str = """
    (S
        (NP (NR 我))
        (VP (VV 爱)
            (NP (NR 中国))))
    """
    
    tree = ConstituentTree.from_penn_treebank(tree_str)
    rules = tree.extract_rules()
    
    print(f"从树中提取了 {len(rules)} 条规则:")
    for rule in rules:
        print(f"  {rule.lhs} -> {' '.join(rule.rhs)}")


def demo_pcfg_training():
    print("\n" + "=" * 60)
    print("4. PCFG训练演示")
    print("=" * 60)
    
    trees = create_sample_constituent_trees()
    print(f"创建了 {len(trees)} 棵示例树")
    
    pcfg = PCFG()
    pcfg.train(trees, smooth=0.1)
    
    print(f"\nPCFG模型信息:")
    info = pcfg.get_model_info()
    print(f"  规则数量: {info['num_rules']}")
    print(f"  非终结符数量: {info['num_non_terminals']}")
    print(f"  终结符数量: {info['num_terminals']}")
    print(f"  起始符号: {info['start_symbol']}")
    
    print(f"\n部分规则示例:")
    count = 0
    for lhs, rules in pcfg.rules.items():
        for rule in rules[:3]:
            print(f"  {rule}")
            count += 1
            if count >= 10:
                break
        if count >= 10:
            break


def demo_cky_parsing():
    print("\n" + "=" * 60)
    print("5. CKY解析演示")
    print("=" * 60)
    
    trees = create_sample_constituent_trees()
    
    pcfg = PCFG()
    pcfg.train(trees, smooth=0.1)
    
    cky = CKYParser(pcfg)
    
    test_sentences = [
        ['我', '爱', '中国'],
        ['他', '吃', '了', '一个', '苹果'],
        ['我们', '学习', '中文'],
    ]
    
    for words in test_sentences:
        print(f"\n解析句子: {''.join(words)}")
        tree = cky.parse(words)
        if tree:
            print(f"解析成功!")
            print(f"  树结构: {tree.to_lisp_string()}")
            print(f"  概率: {pcfg.compute_probability(tree):.6f}")
        else:
            print("  解析失败")


def demo_constituent_parser():
    print("\n" + "=" * 60)
    print("6. ConstituentParser完整演示")
    print("=" * 60)
    
    parser = ConstituentParser(start_symbol='S')
    
    trees = create_sample_constituent_trees()
    parser.train(trees, smooth=0.1)
    
    print(f"模型信息: {parser.get_model_info()}")
    
    test_sentences = [
        ['我', '爱', '中国'],
        ['北京', '是', '中国', '首都'],
        ['她', '买', '了', '三', '本', '书'],
    ]
    
    for words in test_sentences:
        print(f"\n解析: {''.join(words)}")
        
        tree = parser.parse(words)
        if tree:
            print(f"最佳解析:")
            print(f"  {tree.to_lisp_string()}")
            
            nps = tree.get_noun_phrases()
            vps = tree.get_verb_phrases()
            print(f"  名词短语: {[np.get_text() for np in nps]}")
            print(f"  动词短语: {[vp.get_text() for vp in vps]}")
        
        k_best = parser.parse_k_best(words, k=3)
        if k_best:
            print(f"  前{len(k_best)}个解析:")
            for i, (t, prob) in enumerate(k_best):
                print(f"    {i+1}. {t.to_lisp_string()} (log_prob={prob:.2f})")


def demo_node_operations():
    print("\n" + "=" * 60)
    print("7. 节点操作演示")
    print("=" * 60)
    
    tree_str = """
    (S
        (NP (NR 小明))
        (VP (PP (P 在)
                (NP (NN 学校)))
            (VP (VV 读书))))
    """
    
    tree = ConstituentTree.from_penn_treebank(tree_str)
    
    print(f"句子: {tree.get_text()}")
    print(f"树高度: {tree.get_height()}")
    
    print("\n遍历所有节点:")
    for node in tree:
        indent = "  " * (node.get_span()[0])
        if node.is_terminal():
            print(f"{indent}[{node.label}] {node.word}")
        else:
            print(f"{indent}[{node.label}] ({CONSTITUENT_LABELS.get(node.label, '未知')})")
    
    print("\n查找特定位置的节点:")
    node = tree.root.find_node_at(1, 3)
    if node:
        print(f"  位置[1:3]的节点: {node.label} -> {node.get_text()}")


def main():
    print("成分句法分析演示程序")
    print("=" * 60)
    
    demo_tree_parsing()
    demo_tree_operations()
    demo_rule_extraction()
    demo_pcfg_training()
    demo_cky_parsing()
    demo_constituent_parser()
    demo_node_operations()
    
    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()
