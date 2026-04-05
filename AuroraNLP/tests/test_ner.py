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
        self.assertEqual(result['type'], "LOC")
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


if __name__ == '__main__':
    unittest.main()
