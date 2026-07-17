"""测试依存句法分析模块"""

import pytest

from AuroraNLP.dependency_parser import (
    DependencyParser,
    DependencyTree,
    DependencyNode,
    DependencyArc,
    ParserState,
    ArcEagerOracle,
    DEPENDENCY_RELATIONS,
    DEFAULT_RELATIONS,
    Transition,
)


class TestDependencyParser:
    """DependencyParser 测试"""

    def test_dep_parser_init(self):
        """测试初始化"""
        parser = DependencyParser()
        assert parser is not None
        assert not parser.is_trained()
        assert len(parser.relations) > 0


class TestConstants:
    """常量测试"""

    def test_dep_relations_defined(self):
        """测试 DEPENDENCY_RELATIONS 常量存在"""
        assert isinstance(DEPENDENCY_RELATIONS, dict)
        assert len(DEPENDENCY_RELATIONS) > 0
        assert "root" in DEPENDENCY_RELATIONS
        assert "nsubj" in DEPENDENCY_RELATIONS
        assert "obj" in DEPENDENCY_RELATIONS

    def test_default_relations_defined(self):
        """测试 DEFAULT_RELATIONS 常量存在"""
        assert isinstance(DEFAULT_RELATIONS, list)
        assert len(DEFAULT_RELATIONS) > 0
        assert "root" in DEFAULT_RELATIONS


class TestDependencyNode:
    """DependencyNode 测试"""

    def test_dependency_node_creation(self):
        """测试 DependencyNode 对象创建"""
        node = DependencyNode(
            id=1,
            form="我",
            pos="r",
            head=2,
            deprel="nsubj",
        )
        assert node.id == 1
        assert node.form == "我"
        assert node.pos == "r"
        assert node.head == 2
        assert node.deprel == "nsubj"


class TestDependencyArc:
    """DependencyArc 测试"""

    def test_dependency_arc_creation(self):
        """测试 DependencyArc 对象创建"""
        arc = DependencyArc(head=2, dependent=1, relation="nsubj")
        assert arc.head == 2
        assert arc.dependent == 1
        assert arc.relation == "nsubj"


class TestDependencyTree:
    """DependencyTree 测试"""

    def test_dependency_tree_creation(self):
        """测试 DependencyTree 对象创建"""
        tree = DependencyTree()
        assert tree is not None
        assert len(tree) == 0

        node1 = DependencyNode(id=1, form="我", pos="r", head=2, deprel="nsubj")
        node2 = DependencyNode(id=2, form="爱", pos="v", head=0, deprel="root")
        tree.add_node(node1)
        tree.add_node(node2)
        assert len(tree) == 2


class TestParserState:
    """ParserState 测试"""

    def test_parser_state_creation(self):
        """测试 ParserState 对象创建"""
        nodes = [
            DependencyNode(id=0, form="我", pos="r"),
            DependencyNode(id=1, form="爱", pos="v"),
            DependencyNode(id=2, form="中国", pos="ns"),
        ]
        state = ParserState(nodes)
        assert state is not None
        assert len(state.buffer) == 3
        assert len(state.stack) == 0
        assert len(state.arcs) == 0


class TestTransition:
    """Transition 枚举测试"""

    def test_transition_enum(self):
        """测试 Transition 枚举"""
        assert hasattr(Transition, "SHIFT")
        assert hasattr(Transition, "LEFT_ARC")
        assert hasattr(Transition, "RIGHT_ARC")
        assert hasattr(Transition, "REDUCE")
