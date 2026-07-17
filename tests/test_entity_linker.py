"""测试实体链接模块"""

import pytest

from AuroraNLP.ner.entity_linker import (
    EntityLinker,
    KnowledgeBase,
    KnowledgeEntity,
    LinkedEntity,
    EntityNormalizer,
    create_sample_knowledge_base,
    create_sample_normalizer,
)
from AuroraNLP.ner.ner import Entity


class TestEntityLinker:
    """EntityLinker 测试"""

    def test_entity_linker_init(self):
        """测试初始化"""
        linker = EntityLinker()
        assert linker is not None
        assert linker.knowledge_base is not None
        assert linker.normalizer is not None


class TestKnowledgeBase:
    """KnowledgeBase 测试"""

    def test_knowledge_base_creation(self):
        """测试知识库创建"""
        kb = KnowledgeBase(name="test")
        assert kb is not None
        assert kb.name == "test"
        assert kb.get_entity_count() == 0

        kb.add_entity(
            canonical_name="测试实体",
            entity_type="ORG",
            aliases=["测试"],
        )
        assert kb.get_entity_count() == 1


class TestKnowledgeEntity:
    """KnowledgeEntity 测试"""

    def test_knowledge_entity_creation(self):
        """测试知识实体对象创建"""
        entity = KnowledgeEntity(
            entity_id="KB_000001",
            canonical_name="阿里巴巴集团",
            entity_type="ORG",
            aliases=["阿里巴巴", "阿里"],
            description="中国领先的电子商务公司",
            confidence=1.0,
        )
        assert entity.entity_id == "KB_000001"
        assert entity.canonical_name == "阿里巴巴集团"
        assert entity.entity_type == "ORG"
        assert "阿里巴巴" in entity.aliases
        assert entity.description == "中国领先的电子商务公司"
        assert entity.confidence == 1.0


class TestLinkedEntity:
    """LinkedEntity 测试"""

    def test_linked_entity_creation(self):
        """测试链接实体对象创建"""
        ner_entity = Entity(
            text="阿里巴巴",
            entity_type="ORG",
            start=0,
            end=4,
        )
        linked = LinkedEntity(
            entity=ner_entity,
            is_linked=False,
            confidence=0.0,
        )
        assert linked.entity.text == "阿里巴巴"
        assert linked.entity.entity_type == "ORG"
        assert not linked.is_linked
        assert linked.confidence == 0.0


class TestSampleFactories:
    """示例工厂函数测试"""

    def test_create_sample_knowledge_base(self):
        """测试创建示例知识库"""
        kb = create_sample_knowledge_base()
        assert kb is not None
        assert kb.name == "sample"
        assert kb.get_entity_count() > 0

    def test_create_sample_normalizer(self):
        """测试创建示例归一化器"""
        normalizer = create_sample_normalizer()
        assert normalizer is not None


class TestEntityLinking:
    """实体链接功能测试"""

    def test_entity_linker_link(self):
        """测试实体链接（使用示例知识库）"""
        kb = create_sample_knowledge_base()
        linker = EntityLinker(knowledge_base=kb)

        ner_entity = Entity(
            text="阿里巴巴",
            entity_type="ORG",
            start=0,
            end=4,
        )

        linked = linker.link_entity(ner_entity)
        assert linked is not None
        assert linked.is_linked
        assert linked.knowledge_entity is not None
        assert linked.confidence > 0
