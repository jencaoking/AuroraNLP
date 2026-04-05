import unittest
import tempfile
import os
from AuroraNLP.ner import (
    NER_ENTITY_TYPES,
    NER_TAGS,
    Entity,
    NERFeatureTemplate,
    CRFNERModel,
    NERRecognizer,
    create_sample_ner_corpus,
    NestedEntity,
    EntityHierarchy,
    NESTING_RULES,
    NestedNERRecognizer,
    create_nested_ner_corpus,
)


class TestEntity(unittest.TestCase):
    def test_entity_creation(self):
        entity = Entity("张三", "PER", 0, 2)
        self.assertEqual(entity.text, "张三")
        self.assertEqual(entity.entity_type, "PER")
        self.assertEqual(entity.start, 0)
        self.assertEqual(entity.end, 2)
        self.assertEqual(entity.length, 2)
    
    def test_entity_to_dict(self):
        entity = Entity("北京", "LOC", 5, 7, confidence=0.95)
        result = entity.to_dict()
        self.assertEqual(result['text'], "北京")
        self.assertEqual(result['entity_type'], "LOC")
        self.assertEqual(result['type_name'], "地名")
        self.assertEqual(result['start'], 5)
        self.assertEqual(result['end'], 7)
        self.assertEqual(result['confidence'], 0.95)
    
    def test_entity_equality(self):
        entity1 = Entity("张三", "PER", 0, 2)
        entity2 = Entity("张三", "PER", 0, 2)
        entity3 = Entity("李四", "PER", 0, 2)
        self.assertEqual(entity1, entity2)
        self.assertNotEqual(entity1, entity3)
    
    def test_entity_repr(self):
        entity = Entity("张三", "PER", 0, 2)
        repr_str = repr(entity)
        self.assertIn("张三", repr_str)
        self.assertIn("PER", repr_str)


class TestNERFeatureTemplate(unittest.TestCase):
    def setUp(self):
        self.template = NERFeatureTemplate()
    
    def test_add_unigram_feature(self):
        self.template.add_unigram_feature("char", 0)
        self.assertEqual(len(self.template.feature_functions), 1)
        self.assertIn("unigram_char_0", self.template.feature_names)
    
    def test_add_bigram_feature(self):
        self.template.add_bigram_feature("bigram", 0)
        self.assertEqual(len(self.template.feature_functions), 1)
        self.assertIn("bigram_bigram_0", self.template.feature_names)
    
    def test_extract_features(self):
        self.template.add_unigram_feature("char", 0)
        self.template.add_transition_feature()
        
        chars = list("张三")
        features = self.template.extract_features(chars, 0, '', 'B-PER')
        
        self.assertIsInstance(features, list)
        self.assertTrue(len(features) > 0)


class TestCRFNERModel(unittest.TestCase):
    def setUp(self):
        self.model = CRFNERModel()
        self.sample_corpus = [
            (list("张三在北京"), ['B-PER', 'E-PER', 'O', 'B-LOC', 'E-LOC']),
            (list("李四在上海"), ['B-PER', 'E-PER', 'O', 'B-LOC', 'E-LOC']),
            (list("王五去广州"), ['B-PER', 'E-PER', 'O', 'B-LOC', 'E-LOC']),
        ]
    
    def test_model_initialization(self):
        self.assertIsInstance(self.model.tags, list)
        self.assertIn('O', self.model.tags)
        self.assertIn('B-PER', self.model.tags)
    
    def test_model_train(self):
        self.model.train(self.sample_corpus, max_iter=10, verbose=False)
        self.assertTrue(self.model.is_trained())
    
    def test_model_predict(self):
        self.model.train(self.sample_corpus, max_iter=10, verbose=False)
        tags = self.model.predict("张三在广州")
        self.assertEqual(len(tags), 5)
        self.assertTrue(all(tag in self.model.tags for tag in tags))
    
    def test_model_predict_entities(self):
        self.model.train(self.sample_corpus, max_iter=10, verbose=False)
        entities = self.model.predict_entities("张三在广州")
        self.assertIsInstance(entities, list)
    
    def test_model_save_load(self):
        self.model.train(self.sample_corpus, max_iter=10, verbose=False)
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            temp_path = f.name
        
        try:
            self.model.save_model(temp_path)
            
            new_model = CRFNERModel()
            new_model.load_model(temp_path)
            
            self.assertTrue(new_model.is_trained())
            self.assertEqual(new_model.tags, self.model.tags)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_model_info(self):
        self.model.train(self.sample_corpus, max_iter=10, verbose=False)
        info = self.model.get_model_info()
        
        self.assertTrue(info['trained'])
        self.assertIn('num_tags', info)
        self.assertIn('tags', info)


class TestNERRecognizer(unittest.TestCase):
    def setUp(self):
        self.recognizer = NERRecognizer()
        self.sample_corpus = create_sample_ner_corpus()
    
    def test_recognizer_initialization(self):
        self.assertFalse(self.recognizer.is_trained())
    
    def test_recognizer_train(self):
        self.recognizer.train(self.sample_corpus, max_iter=10, verbose=False)
        self.assertTrue(self.recognizer.is_trained())
    
    def test_recognizer_recognize(self):
        self.recognizer.train(self.sample_corpus, max_iter=20, verbose=False)
        entities = self.recognizer.recognize("张三在北京清华大学学习")
        
        self.assertIsInstance(entities, list)
        for entity in entities:
            self.assertIsInstance(entity, Entity)
    
    def test_recognizer_get_entities_by_type(self):
        self.recognizer.train(self.sample_corpus, max_iter=20, verbose=False)
        
        persons = self.recognizer.get_persons("张三在北京学习")
        self.assertIsInstance(persons, list)
        
        locations = self.recognizer.get_locations("张三在北京学习")
        self.assertIsInstance(locations, list)
    
    def test_recognizer_annotate_text(self):
        self.recognizer.train(self.sample_corpus, max_iter=20, verbose=False)
        annotated = self.recognizer.annotate_text("张三在北京学习")
        
        self.assertIsInstance(annotated, str)
        self.assertIn("张三", annotated)
    
    def test_recognizer_save_load(self):
        self.recognizer.train(self.sample_corpus, max_iter=10, verbose=False)
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            temp_path = f.name
        
        try:
            self.recognizer.save_model(temp_path)
            
            new_recognizer = NERRecognizer()
            new_recognizer.load_model(temp_path)
            
            self.assertTrue(new_recognizer.is_trained())
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_batch_recognize(self):
        self.recognizer.train(self.sample_corpus, max_iter=20, verbose=False)
        
        texts = ["张三在北京", "李四在上海"]
        results = self.recognizer.batch_recognize(texts)
        
        self.assertEqual(len(results), 2)
        for entities in results:
            self.assertIsInstance(entities, list)


class TestNERConstants(unittest.TestCase):
    def test_ner_entity_types(self):
        self.assertIn('PER', NER_ENTITY_TYPES)
        self.assertIn('LOC', NER_ENTITY_TYPES)
        self.assertIn('ORG', NER_ENTITY_TYPES)
    
    def test_ner_tags(self):
        self.assertIn('O', NER_TAGS)
        self.assertIn('B-PER', NER_TAGS)
        self.assertIn('I-PER', NER_TAGS)
        self.assertIn('E-PER', NER_TAGS)
        self.assertIn('S-PER', NER_TAGS)


class TestSampleCorpus(unittest.TestCase):
    def test_create_sample_ner_corpus(self):
        corpus = create_sample_ner_corpus()
        
        self.assertIsInstance(corpus, list)
        self.assertTrue(len(corpus) > 0)
        
        for text, entities in corpus:
            self.assertIsInstance(text, str)
            self.assertIsInstance(entities, list)
            for entity in entities:
                self.assertIsInstance(entity, Entity)


class TestNestedEntity(unittest.TestCase):
    def test_nested_entity_creation(self):
        entity = NestedEntity("北京协和医院", "ORG", 0, 6, level=0)
        self.assertEqual(entity.text, "北京协和医院")
        self.assertEqual(entity.entity_type, "ORG")
        self.assertEqual(entity.start, 0)
        self.assertEqual(entity.end, 6)
        self.assertEqual(entity.level, 0)
        self.assertEqual(len(entity.children), 0)
        self.assertIsNone(entity.parent)
    
    def test_nested_entity_add_child(self):
        parent = NestedEntity("北京协和医院", "ORG", 0, 6, level=0)
        child = NestedEntity("北京", "LOC", 0, 2, level=1)
        
        parent.add_child(child)
        
        self.assertEqual(len(parent.children), 1)
        self.assertEqual(parent.children[0], child)
        self.assertEqual(child.parent, parent)
        self.assertEqual(child.level, 1)
    
    def test_nested_entity_remove_child(self):
        parent = NestedEntity("北京协和医院", "ORG", 0, 6, level=0)
        child = NestedEntity("北京", "LOC", 0, 2, level=1)
        
        parent.add_child(child)
        parent.remove_child(child)
        
        self.assertEqual(len(parent.children), 0)
        self.assertIsNone(child.parent)
    
    def test_nested_entity_get_all_children(self):
        grandparent = NestedEntity("广东省深圳市南山区", "LOC", 0, 8, level=0)
        parent = NestedEntity("深圳市南山区", "LOC", 3, 8, level=1)
        child = NestedEntity("南山区", "LOC", 6, 9, level=2)
        
        grandparent.add_child(parent)
        parent.add_child(child)
        
        all_children = grandparent.get_all_children()
        self.assertEqual(len(all_children), 2)
        self.assertIn(parent, all_children)
        self.assertIn(child, all_children)
    
    def test_nested_entity_contains(self):
        entity1 = NestedEntity("北京协和医院", "ORG", 0, 6, level=0)
        entity2 = NestedEntity("北京", "LOC", 0, 2, level=1)
        entity3 = NestedEntity("上海", "LOC", 10, 12, level=0)
        
        self.assertTrue(entity1.contains(entity2))
        self.assertFalse(entity1.contains(entity3))
    
    def test_nested_entity_to_dict(self):
        parent = NestedEntity("北京协和医院", "ORG", 0, 6, level=0)
        child = NestedEntity("北京", "LOC", 0, 2, level=1)
        parent.add_child(child)
        
        result = parent.to_dict()
        
        self.assertEqual(result['text'], "北京协和医院")
        self.assertEqual(result['entity_type'], "ORG")
        self.assertEqual(result['level'], 0)
        self.assertTrue(result['is_nested'])
        self.assertIn('children', result)
        self.assertEqual(len(result['children']), 1)


class TestEntityHierarchy(unittest.TestCase):
    def test_hierarchy_creation(self):
        hierarchy = EntityHierarchy()
        self.assertEqual(len(hierarchy.root_entities), 0)
        self.assertEqual(len(hierarchy.all_entities), 0)
    
    def test_hierarchy_add_entity(self):
        hierarchy = EntityHierarchy()
        entity = NestedEntity("北京协和医院", "ORG", 0, 6, level=0)
        
        hierarchy.add_entity(entity)
        
        self.assertEqual(len(hierarchy.all_entities), 1)
        self.assertEqual(len(hierarchy.root_entities), 1)
    
    def test_hierarchy_build_nested_structure(self):
        hierarchy = EntityHierarchy()
        entities = [
            NestedEntity("北京协和医院", "ORG", 0, 6, level=0),
            NestedEntity("北京", "LOC", 0, 2, level=1),
        ]
        
        hierarchy.build_from_entities(entities)
        
        self.assertEqual(len(hierarchy.all_entities), 2)
        self.assertEqual(len(hierarchy.root_entities), 1)
        self.assertEqual(len(hierarchy.root_entities[0].children), 1)
    
    def test_hierarchy_get_entities_by_type(self):
        hierarchy = EntityHierarchy()
        entities = [
            NestedEntity("北京协和医院", "ORG", 0, 6, level=0),
            NestedEntity("北京", "LOC", 0, 2, level=1),
            NestedEntity("上海", "LOC", 10, 12, level=0),
        ]
        
        hierarchy.build_from_entities(entities)
        
        loc_entities = hierarchy.get_entities_by_type("LOC")
        self.assertEqual(len(loc_entities), 2)
        
        org_entities = hierarchy.get_entities_by_type("ORG")
        self.assertEqual(len(org_entities), 1)
    
    def test_hierarchy_get_max_depth(self):
        hierarchy = EntityHierarchy()
        entities = [
            NestedEntity("广东省深圳市南山区", "LOC", 0, 9, level=0),
            NestedEntity("深圳市南山区", "LOC", 3, 9, level=1),
            NestedEntity("南山区", "LOC", 6, 9, level=2),
        ]
        
        hierarchy.build_from_entities(entities)
        
        self.assertEqual(hierarchy.get_max_depth(), 2)


class TestNestedNERRecognizer(unittest.TestCase):
    def setUp(self):
        self.recognizer = NestedNERRecognizer()
        self.corpus = create_nested_ner_corpus()
    
    def test_recognizer_initialization(self):
        self.assertFalse(self.recognizer.is_trained())
        self.assertIsNotNone(self.recognizer.nesting_rules)
    
    def test_recognizer_train(self):
        self.recognizer.train(self.corpus, max_iter=10, verbose=False)
        self.assertTrue(self.recognizer.is_trained())
    
    def test_recognizer_recognize_nested(self):
        self.recognizer.train(self.corpus, max_iter=30, verbose=False)
        
        hierarchy = self.recognizer.recognize_nested("北京协和医院是一家著名医院", max_levels=3)
        
        self.assertIsInstance(hierarchy, EntityHierarchy)
    
    def test_recognizer_recognize(self):
        self.recognizer.train(self.corpus, max_iter=20, verbose=False)
        
        entities = self.recognizer.recognize("北京协和医院是一家著名医院", max_levels=3)
        
        self.assertIsInstance(entities, list)
        for entity in entities:
            self.assertIsInstance(entity, NestedEntity)
    
    def test_recognizer_get_nested_entities(self):
        self.recognizer.train(self.corpus, max_iter=20, verbose=False)
        
        nested_entities = self.recognizer.get_nested_entities("北京协和医院是一家著名医院", max_levels=3)
        
        self.assertIsInstance(nested_entities, list)
    
    def test_recognizer_annotate_nested(self):
        self.recognizer.train(self.corpus, max_iter=20, verbose=False)
        
        annotated = self.recognizer.annotate_nested("北京协和医院是一家著名医院", max_levels=3)
        
        self.assertIsInstance(annotated, str)
        self.assertIn("北京协和医院", annotated)
    
    def test_nesting_rules(self):
        self.assertTrue(self.recognizer.can_nest('ORG', 'LOC'))
        self.assertTrue(self.recognizer.can_nest('LOC', 'LOC'))
        self.assertFalse(self.recognizer.can_nest('NUM', 'ORG'))
    
    def test_set_nesting_rule(self):
        self.recognizer.set_nesting_rule('ORG', ['LOC', 'PER'])
        
        self.assertTrue(self.recognizer.can_nest('ORG', 'LOC'))
        self.assertTrue(self.recognizer.can_nest('ORG', 'PER'))
        self.assertFalse(self.recognizer.can_nest('ORG', 'TIME'))
    
    def test_recognizer_save_load(self):
        self.recognizer.train(self.corpus, max_iter=10, verbose=False)
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            temp_path = f.name
        
        try:
            self.recognizer.save_model(temp_path)
            
            new_recognizer = NestedNERRecognizer()
            new_recognizer.load_model(temp_path)
            
            self.assertTrue(new_recognizer.is_trained())
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestNestedCorpus(unittest.TestCase):
    def test_create_nested_ner_corpus(self):
        corpus = create_nested_ner_corpus()
        
        self.assertIsInstance(corpus, list)
        self.assertTrue(len(corpus) > 0)
        
        for text, entities in corpus:
            self.assertIsInstance(text, str)
            self.assertIsInstance(entities, list)
            for entity in entities:
                self.assertIsInstance(entity, Entity)


if __name__ == '__main__':
    unittest.main()
