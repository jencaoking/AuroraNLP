"""
Unit tests for Entity Linking module.
"""

import unittest
import os
import tempfile
import json

from AuroraNLP.entity_linker import (
    KnowledgeEntity,
    LinkedEntity,
    KnowledgeBase,
    EntityNormalizer,
    EntityLinker,
    create_sample_knowledge_base,
    create_sample_normalizer,
)
from AuroraNLP.ner import Entity


class TestKnowledgeEntity(unittest.TestCase):
    
    def test_create_entity(self):
        entity = KnowledgeEntity(
            entity_id="TEST_001",
            canonical_name="测试实体",
            entity_type="ORG",
            aliases=["测试", "Test"],
            attributes={"key": "value"},
            description="这是一个测试实体"
        )
        
        self.assertEqual(entity.entity_id, "TEST_001")
        self.assertEqual(entity.canonical_name, "测试实体")
        self.assertEqual(entity.entity_type, "ORG")
        self.assertEqual(len(entity.aliases), 2)
        self.assertEqual(entity.attributes["key"], "value")
    
    def test_add_alias(self):
        entity = KnowledgeEntity(
            entity_id="TEST_001",
            canonical_name="测试实体",
            entity_type="ORG"
        )
        
        entity.add_alias("别名1")
        entity.add_alias("别名2")
        
        self.assertEqual(len(entity.aliases), 2)
        self.assertIn("别名1", entity.aliases)
        self.assertIn("别名2", entity.aliases)
    
    def test_add_duplicate_alias(self):
        entity = KnowledgeEntity(
            entity_id="TEST_001",
            canonical_name="测试实体",
            entity_type="ORG"
        )
        
        entity.add_alias("别名")
        entity.add_alias("别名")
        
        self.assertEqual(len(entity.aliases), 1)
    
    def test_remove_alias(self):
        entity = KnowledgeEntity(
            entity_id="TEST_001",
            canonical_name="测试实体",
            entity_type="ORG",
            aliases=["别名1", "别名2"]
        )
        
        entity.remove_alias("别名1")
        
        self.assertEqual(len(entity.aliases), 1)
        self.assertNotIn("别名1", entity.aliases)
    
    def test_get_all_names(self):
        entity = KnowledgeEntity(
            entity_id="TEST_001",
            canonical_name="测试实体",
            entity_type="ORG",
            aliases=["别名1", "别名2"]
        )
        
        names = entity.get_all_names()
        
        self.assertEqual(len(names), 3)
        self.assertIn("测试实体", names)
        self.assertIn("别名1", names)
        self.assertIn("别名2", names)
    
    def test_matches(self):
        entity = KnowledgeEntity(
            entity_id="TEST_001",
            canonical_name="测试实体",
            entity_type="ORG",
            aliases=["别名", "Test"]
        )
        
        self.assertTrue(entity.matches("测试实体"))
        self.assertTrue(entity.matches("别名"))
        self.assertTrue(entity.matches("Test"))
        self.assertFalse(entity.matches("其他"))
        
        self.assertTrue(entity.matches("test", case_sensitive=False))
        self.assertFalse(entity.matches("test", case_sensitive=True))
    
    def test_to_dict_and_from_dict(self):
        entity = KnowledgeEntity(
            entity_id="TEST_001",
            canonical_name="测试实体",
            entity_type="ORG",
            aliases=["别名"],
            attributes={"key": "value"},
            description="描述",
            confidence=0.9,
            source="test",
            external_ids={"wikidata": "Q123"}
        )
        
        data = entity.to_dict()
        restored = KnowledgeEntity.from_dict(data)
        
        self.assertEqual(restored.entity_id, entity.entity_id)
        self.assertEqual(restored.canonical_name, entity.canonical_name)
        self.assertEqual(restored.entity_type, entity.entity_type)
        self.assertEqual(restored.aliases, entity.aliases)
        self.assertEqual(restored.attributes, entity.attributes)
        self.assertEqual(restored.description, entity.description)
        self.assertEqual(restored.confidence, entity.confidence)
        self.assertEqual(restored.source, entity.source)
        self.assertEqual(restored.external_ids, entity.external_ids)


class TestKnowledgeBase(unittest.TestCase):
    
    def setUp(self):
        self.kb = KnowledgeBase(name="test_kb")
    
    def test_add_entity(self):
        entity = self.kb.add_entity(
            canonical_name="阿里巴巴",
            entity_type="ORG",
            aliases=["阿里", "Alibaba"]
        )
        
        self.assertIsNotNone(entity.entity_id)
        self.assertEqual(entity.canonical_name, "阿里巴巴")
        self.assertEqual(len(self.kb), 1)
    
    def test_get_entity_by_id(self):
        added = self.kb.add_entity(
            canonical_name="测试实体",
            entity_type="ORG"
        )
        
        retrieved = self.kb.get_entity_by_id(added.entity_id)
        
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.canonical_name, "测试实体")
    
    def test_get_entities_by_name(self):
        self.kb.add_entity(
            canonical_name="阿里巴巴",
            entity_type="ORG",
            aliases=["阿里"]
        )
        
        entities = self.kb.get_entities_by_name("阿里巴巴")
        self.assertEqual(len(entities), 1)
        
        entities = self.kb.get_entities_by_name("阿里")
        self.assertEqual(len(entities), 1)
        
        entities = self.kb.get_entities_by_name("不存在")
        self.assertEqual(len(entities), 0)
    
    def test_get_entities_by_type(self):
        self.kb.add_entity(canonical_name="阿里巴巴", entity_type="ORG")
        self.kb.add_entity(canonical_name="腾讯", entity_type="ORG")
        self.kb.add_entity(canonical_name="北京", entity_type="LOC")
        
        orgs = self.kb.get_entities_by_type("ORG")
        locs = self.kb.get_entities_by_type("LOC")
        
        self.assertEqual(len(orgs), 2)
        self.assertEqual(len(locs), 1)
    
    def test_search_entities(self):
        self.kb.add_entity(
            canonical_name="阿里巴巴集团",
            entity_type="ORG",
            aliases=["阿里"]
        )
        
        results = self.kb.search_entities("阿里巴巴集团")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][1], 1.0)
        
        results = self.kb.search_entities("阿里")
        self.assertEqual(len(results), 1)
        self.assertGreater(results[0][1], 0.5)
    
    def test_fuzzy_search(self):
        self.kb.add_entity(
            canonical_name="阿里巴巴",
            entity_type="ORG"
        )
        
        results = self.kb.fuzzy_search("阿里巴", threshold=0.5)
        self.assertGreater(len(results), 0)
        
        results = self.kb.fuzzy_search("完全不相关", threshold=0.5)
        self.assertEqual(len(results), 0)
    
    def test_remove_entity(self):
        entity = self.kb.add_entity(
            canonical_name="测试实体",
            entity_type="ORG"
        )
        
        self.assertEqual(len(self.kb), 1)
        
        result = self.kb.remove_entity(entity.entity_id)
        self.assertTrue(result)
        self.assertEqual(len(self.kb), 0)
        
        result = self.kb.remove_entity("不存在的ID")
        self.assertFalse(result)
    
    def test_update_entity(self):
        entity = self.kb.add_entity(
            canonical_name="测试实体",
            entity_type="ORG"
        )
        
        updated = self.kb.update_entity(
            entity.entity_id,
            canonical_name="更新后的名称",
            aliases=["新别名"],
            attributes={"key": "value"}
        )
        
        self.assertIsNotNone(updated)
        self.assertEqual(updated.canonical_name, "更新后的名称")
        self.assertIn("新别名", updated.aliases)
        self.assertEqual(updated.attributes["key"], "value")
    
    def test_save_and_load(self):
        self.kb.add_entity(
            canonical_name="阿里巴巴",
            entity_type="ORG",
            aliases=["阿里"]
        )
        self.kb.add_entity(
            canonical_name="北京",
            entity_type="LOC"
        )
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            filepath = f.name
        
        try:
            self.kb.save(filepath)
            
            new_kb = KnowledgeBase()
            new_kb.load(filepath)
            
            self.assertEqual(len(new_kb), 2)
            self.assertEqual(new_kb.name, self.kb.name)
            
            entities = new_kb.get_entities_by_name("阿里巴巴")
            self.assertEqual(len(entities), 1)
        finally:
            os.unlink(filepath)
    
    def test_merge(self):
        kb1 = KnowledgeBase(name="kb1")
        kb1.add_entity(canonical_name="实体1", entity_type="ORG")
        
        kb2 = KnowledgeBase(name="kb2")
        kb2.add_entity(canonical_name="实体2", entity_type="LOC")
        
        kb1.merge(kb2)
        
        self.assertEqual(len(kb1), 2)
    
    def test_get_entity_count_by_type(self):
        self.kb.add_entity(canonical_name="实体1", entity_type="ORG")
        self.kb.add_entity(canonical_name="实体2", entity_type="ORG")
        self.kb.add_entity(canonical_name="实体3", entity_type="LOC")
        
        counts = self.kb.get_entity_count_by_type()
        
        self.assertEqual(counts["ORG"], 2)
        self.assertEqual(counts["LOC"], 1)


class TestEntityNormalizer(unittest.TestCase):
    
    def setUp(self):
        self.normalizer = EntityNormalizer()
    
    def test_add_alias(self):
        self.normalizer.add_alias("阿里", "阿里巴巴集团")
        
        result = self.normalizer.normalize("阿里")
        self.assertEqual(result, "阿里巴巴集团")
    
    def test_add_aliases(self):
        self.normalizer.add_aliases({
            "阿里": "阿里巴巴集团",
            "腾讯": "腾讯控股有限公司"
        })
        
        self.assertEqual(self.normalizer.normalize("阿里"), "阿里巴巴集团")
        self.assertEqual(self.normalizer.normalize("腾讯"), "腾讯控股有限公司")
    
    def test_add_abbreviation(self):
        self.normalizer.add_abbreviation("THU", "清华大学")
        
        result = self.normalizer.normalize("THU")
        self.assertEqual(result, "清华大学")
    
    def test_normalization_rules(self):
        self.normalizer.add_normalization_rule(r'\s+', '')
        
        result = self.normalizer.normalize("测试 实体")
        self.assertEqual(result, "测试实体")
    
    def test_type_specific_rules(self):
        self.normalizer.add_type_specific_rule('ORG', r'有限公司$', '集团')
        
        result = self.normalizer.normalize("测试有限公司", entity_type="ORG")
        self.assertEqual(result, "测试集团")
        
        result = self.normalizer.normalize("测试有限公司", entity_type="LOC")
        self.assertEqual(result, "测试有限公司")
    
    def test_get_canonical(self):
        self.normalizer.add_alias("阿里", "阿里巴巴集团")
        
        result = self.normalizer.get_canonical("阿里")
        self.assertEqual(result, "阿里巴巴集团")
        
        result = self.normalizer.get_canonical("不存在")
        self.assertIsNone(result)
    
    def test_save_and_load(self):
        self.normalizer.add_alias("阿里", "阿里巴巴集团")
        self.normalizer.add_abbreviation("THU", "清华大学")
        self.normalizer.add_normalization_rule(r'\s+', '')
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            filepath = f.name
        
        try:
            self.normalizer.save(filepath)
            
            new_normalizer = EntityNormalizer()
            new_normalizer.load(filepath)
            
            self.assertEqual(new_normalizer.normalize("阿里"), "阿里巴巴集团")
            self.assertEqual(new_normalizer.normalize("THU"), "清华大学")
        finally:
            os.unlink(filepath)


class TestLinkedEntity(unittest.TestCase):
    
    def test_create_linked_entity(self):
        entity = Entity(text="阿里巴巴", entity_type="ORG", start=0, end=4)
        ke = KnowledgeEntity(
            entity_id="KB_001",
            canonical_name="阿里巴巴集团",
            entity_type="ORG",
            attributes={"industry": "电子商务"}
        )
        
        linked = LinkedEntity(
            entity=entity,
            knowledge_entity=ke,
            confidence=0.95,
            is_linked=True,
            linking_method="exact"
        )
        
        self.assertTrue(linked.is_linked)
        self.assertEqual(linked.get_canonical_name(), "阿里巴巴集团")
        self.assertEqual(linked.get_entity_id(), "KB_001")
        self.assertEqual(linked.get_attribute("industry"), "电子商务")
    
    def test_unlinked_entity(self):
        entity = Entity(text="未知实体", entity_type="ORG", start=0, end=4)
        
        linked = LinkedEntity(
            entity=entity,
            confidence=0.0,
            is_linked=False
        )
        
        self.assertFalse(linked.is_linked)
        self.assertEqual(linked.get_canonical_name(), "未知实体")
        self.assertIsNone(linked.get_entity_id())
    
    def test_to_dict(self):
        entity = Entity(text="阿里巴巴", entity_type="ORG", start=0, end=4)
        ke = KnowledgeEntity(
            entity_id="KB_001",
            canonical_name="阿里巴巴集团",
            entity_type="ORG"
        )
        
        linked = LinkedEntity(
            entity=entity,
            knowledge_entity=ke,
            confidence=0.95,
            is_linked=True
        )
        
        data = linked.to_dict()
        
        self.assertEqual(data["text"], "阿里巴巴")
        self.assertEqual(data["type"], "ORG")
        self.assertEqual(data["canonical_name"], "阿里巴巴集团")
        self.assertTrue(data["is_linked"])


class TestEntityLinker(unittest.TestCase):
    
    def setUp(self):
        self.kb = create_sample_knowledge_base()
        self.normalizer = create_sample_normalizer()
        self.linker = EntityLinker(
            knowledge_base=self.kb,
            normalizer=self.normalizer
        )
    
    def test_link_entity_exact_match(self):
        entity = Entity(text="阿里巴巴", entity_type="ORG", start=0, end=4)
        
        linked = self.linker.link_entity(entity)
        
        self.assertTrue(linked.is_linked)
        self.assertEqual(linked.get_canonical_name(), "阿里巴巴集团")
        self.assertGreater(linked.confidence, 0.9)
    
    def test_link_entity_alias(self):
        entity = Entity(text="阿里", entity_type="ORG", start=0, end=2)
        
        linked = self.linker.link_entity(entity)
        
        self.assertTrue(linked.is_linked)
        self.assertEqual(linked.get_canonical_name(), "阿里巴巴集团")
    
    def test_link_entity_no_match(self):
        entity = Entity(text="完全不存在的实体", entity_type="ORG", start=0, end=8)
        
        linked = self.linker.link_entity(entity)
        
        self.assertFalse(linked.is_linked)
    
    def test_link_entities(self):
        entities = [
            Entity(text="阿里巴巴", entity_type="ORG", start=0, end=4),
            Entity(text="北京", entity_type="LOC", start=5, end=7),
        ]
        
        linked = self.linker.link_entities(entities)
        
        self.assertEqual(len(linked), 2)
        self.assertTrue(linked[0].is_linked)
        self.assertTrue(linked[1].is_linked)
    
    def test_link_with_context(self):
        entity = Entity(text="清华", entity_type="ORG", start=0, end=2)
        
        linked = self.linker.link_entity(entity, context="清华大学是中国著名的高等学府")
        
        self.assertTrue(linked.is_linked)
    
    def test_get_linked_entities_by_type(self):
        entities = [
            Entity(text="阿里巴巴", entity_type="ORG", start=0, end=4),
            Entity(text="北京", entity_type="LOC", start=5, end=7),
        ]
        
        linked = self.linker.link_entities(entities)
        
        orgs = self.linker.get_linked_entities_by_type(linked, "ORG")
        locs = self.linker.get_linked_entities_by_type(linked, "LOC")
        
        self.assertEqual(len(orgs), 1)
        self.assertEqual(len(locs), 1)
    
    def test_get_unlinked_entities(self):
        entities = [
            Entity(text="阿里巴巴", entity_type="ORG", start=0, end=4),
            Entity(text="未知实体", entity_type="ORG", start=5, end=9),
        ]
        
        linked = self.linker.link_entities(entities)
        
        unlinked = self.linker.get_unlinked_entities(linked)
        
        self.assertEqual(len(unlinked), 1)
        self.assertEqual(unlinked[0].entity.text, "未知实体")
    
    def test_add_knowledge_entry(self):
        self.linker.add_knowledge_entry(
            canonical_name="新实体",
            entity_type="ORG",
            aliases=["新实体别名"]
        )
        
        entity = Entity(text="新实体", entity_type="ORG", start=0, end=3)
        linked = self.linker.link_entity(entity)
        
        self.assertTrue(linked.is_linked)
    
    def test_annotate_text(self):
        entities = [
            Entity(text="阿里巴巴", entity_type="ORG", start=0, end=4),
        ]
        
        linked = self.linker.link_entities(entities, "阿里巴巴在北京")
        
        result = []
        text = "阿里巴巴在北京"
        last_end = 0
        for le in sorted(linked, key=lambda x: x.entity.start):
            result.append(text[last_end:le.entity.start])
            if le.is_linked:
                result.append(f"[{le.entity.text}→{le.knowledge_entity.canonical_name}/{le.knowledge_entity.entity_type}]")
            else:
                result.append(f"[{le.entity.text}/{le.entity.entity_type}(未链接)]")
            last_end = le.entity.end
        result.append(text[last_end:])
        annotated = ''.join(result)
        
        self.assertIn("阿里巴巴集团", annotated)
    
    def test_get_statistics(self):
        entities = [
            Entity(text="阿里巴巴", entity_type="ORG", start=0, end=4),
            Entity(text="未知实体", entity_type="ORG", start=5, end=9),
        ]
        
        linked = self.linker.link_entities(entities)
        stats = self.linker.get_statistics(linked)
        
        self.assertEqual(stats["total_entities"], 2)
        self.assertEqual(stats["linked_entities"], 1)
        self.assertEqual(stats["unlinked_entities"], 1)
        self.assertEqual(stats["link_rate"], 0.5)
    
    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.linker.save(tmpdir)
            
            new_linker = EntityLinker()
            new_linker.load(tmpdir)
            
            self.assertEqual(len(new_linker.knowledge_base), len(self.kb))
            
            entity = Entity(text="阿里巴巴", entity_type="ORG", start=0, end=4)
            linked = new_linker.link_entity(entity)
            
            self.assertTrue(linked.is_linked)


class TestSampleFunctions(unittest.TestCase):
    
    def test_create_sample_knowledge_base(self):
        kb = create_sample_knowledge_base()
        
        self.assertGreater(len(kb), 0)
        
        orgs = kb.get_entities_by_type("ORG")
        locs = kb.get_entities_by_type("LOC")
        
        self.assertGreater(len(orgs), 0)
        self.assertGreater(len(locs), 0)
    
    def test_create_sample_normalizer(self):
        normalizer = create_sample_normalizer()
        
        self.assertEqual(normalizer.normalize("阿里"), "阿里巴巴集团")
        self.assertEqual(normalizer.normalize("腾讯"), "腾讯控股有限公司")
        self.assertEqual(normalizer.normalize("清华"), "清华大学")


if __name__ == '__main__':
    unittest.main()
