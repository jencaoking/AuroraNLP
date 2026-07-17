"""测试成分句法分析模块"""

import pytest

from AuroraNLP.parsing.constituent_parser import (
    ConstituentParser,
    ConstituentTree,
    ConstituentNode,
    PCFG,
    CKYParser,
    CONSTITUENT_LABELS,
    POS_LABELS,
    DEFAULT_NON_TERMINALS,
    DEFAULT_TERMINALS,
    GrammarRule,
)


class TestConstituentParser:
    """ConstituentParser 测试"""

    def test_const_parser_init(self):
        """测试初始化"""
        parser = ConstituentParser()
        assert parser is not None
        assert not parser.is_trained()


class TestConstants:
    """常量测试"""

    def test_constituent_labels_defined(self):
        """测试 CONSTITUENT_LABELS 常量存在"""
        assert isinstance(CONSTITUENT_LABELS, dict)
        assert len(CONSTITUENT_LABELS) > 0
        assert "S" in CONSTITUENT_LABELS
        assert "NP" in CONSTITUENT_LABELS
        assert "VP" in CONSTITUENT_LABELS

    def test_pos_labels_defined(self):
        """测试 POS_LABELS 常量存在"""
        assert isinstance(POS_LABELS, dict)
        assert len(POS_LABELS) > 0
        assert "NN" in POS_LABELS
        assert "VV" in POS_LABELS

    def test_default_non_terminals_defined(self):
        """测试 DEFAULT_NON_TERMINALS 常量存在"""
        assert isinstance(DEFAULT_NON_TERMINALS, list)
        assert len(DEFAULT_NON_TERMINALS) > 0
        assert "S" in DEFAULT_NON_TERMINALS

    def test_default_terminals_defined(self):
        """测试 DEFAULT_TERMINALS 常量存在"""
        assert isinstance(DEFAULT_TERMINALS, list)
        assert len(DEFAULT_TERMINALS) > 0
        assert "NN" in DEFAULT_TERMINALS


class TestConstituentNode:
    """ConstituentNode 测试"""

    def test_constituent_node_creation(self):
        """测试 ConstituentNode 对象创建"""
        node = ConstituentNode(label="NP", start=0, end=2)
        assert node.label == "NP"
        assert node.start == 0
        assert node.end == 2
        assert node.is_terminal()

        word_node = ConstituentNode(label="我", word="我", start=0, end=1)
        assert word_node.label == "我"
        assert word_node.word == "我"
        assert word_node.is_terminal()


class TestConstituentTree:
    """ConstituentTree 测试"""

    def test_constituent_tree_creation(self):
        """测试 ConstituentTree 对象创建"""
        tree = ConstituentTree()
        assert tree is not None
        assert len(tree) == 0

        word_node = ConstituentNode(label="我", word="我", start=0, end=1)
        pos_node = ConstituentNode(label="PN", children=[word_node], start=0, end=1)
        tree = ConstituentTree(root=pos_node)
        assert tree.root is not None
        assert tree.root.label == "PN"


class TestGrammarRule:
    """GrammarRule 测试"""

    def test_grammar_rule_creation(self):
        """测试 GrammarRule 对象创建"""
        rule = GrammarRule(lhs="NP", rhs=("DT", "NN"), probability=0.8, count=10)
        assert rule.lhs == "NP"
        assert rule.rhs == ("DT", "NN")
        assert rule.probability == 0.8
        assert rule.count == 10
        assert rule.is_binary()
        assert not rule.is_unary()
