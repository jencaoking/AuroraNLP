import pytest
import os
import tempfile
from AuroraNLP.crf import CRFModel, CRFSegmentor, CRFFeatureTemplate


class TestCRFFeatureTemplate:
    def test_feature_template_creation(self):
        template = CRFFeatureTemplate()
        assert len(template.feature_functions) == 0
        assert len(template.feature_names) == 0
    
    def test_add_unigram_feature(self):
        template = CRFFeatureTemplate()
        template.add_unigram_feature('word', 0)
        
        assert len(template.feature_functions) == 1
        assert 'unigram_word_0' in template.feature_names
    
    def test_add_bigram_feature(self):
        template = CRFFeatureTemplate()
        template.add_bigram_feature('bigram', 0)
        
        assert len(template.feature_functions) == 1
        assert 'bigram_bigram_0' in template.feature_names
    
    def test_add_transition_feature(self):
        template = CRFFeatureTemplate()
        template.add_transition_feature()
        
        assert len(template.feature_functions) == 1
        assert 'transition' in template.feature_names
    
    def test_extract_features(self):
        template = CRFFeatureTemplate()
        template.add_unigram_feature('word', 0)
        template.add_transition_feature()
        
        tokens = ['我', '爱', '中国']
        features = template.extract_features(tokens, 0, '', 'B')
        
        assert isinstance(features, list)
        assert len(features) > 0
        assert any('word:我|B' in f for f in features)


class TestCRFModel:
    def test_crf_model_creation(self):
        model = CRFModel()
        assert not model.is_trained()
        assert len(model.tags) == 0
    
    def test_crf_model_creation_with_tags(self):
        tags = ['B', 'M', 'E', 'S']
        model = CRFModel(tags=tags)
        assert model.tags == tags
    
    def test_crf_basic_training(self):
        model = CRFModel(tags=['B', 'M', 'E', 'S'])
        
        corpus = [
            (['我', '爱', '中', '国'], ['S', 'S', 'B', 'E']),
            (['他', '是', '学', '生'], ['S', 'S', 'B', 'E'])
        ]
        
        model.train(corpus, max_iter=10, verbose=False)
        
        assert model.is_trained()
        assert len(model.weights) > 0
    
    def test_crf_predict(self):
        model = CRFModel(tags=['B', 'M', 'E', 'S'])
        
        corpus = [
            (['我', '爱', '中', '国'], ['S', 'S', 'B', 'E']),
            (['他', '是', '学', '生'], ['S', 'S', 'B', 'E'])
        ]
        
        model.train(corpus, max_iter=10, verbose=False)
        
        tokens = ['我', '爱', '北', '京']
        tags = model.predict(tokens)
        
        assert len(tags) == len(tokens)
        assert all(tag in ['B', 'M', 'E', 'S'] for tag in tags)
    
    def test_crf_viterbi(self):
        model = CRFModel(tags=['B', 'M', 'E', 'S'])
        
        corpus = [
            (['我', '爱', '中', '国'], ['S', 'S', 'B', 'E'])
        ]
        
        model.train(corpus, max_iter=10, verbose=False)
        
        tokens = ['我', '爱', '中', '国']
        tags = model.viterbi(tokens)
        
        assert len(tags) == len(tokens)
    
    def test_crf_save_and_load(self):
        model = CRFModel(tags=['B', 'M', 'E', 'S'])
        
        corpus = [
            (['我', '爱', '中', '国'], ['S', 'S', 'B', 'E'])
        ]
        
        model.train(corpus, max_iter=10, verbose=False)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            temp_path = f.name
        
        try:
            model.save_model(temp_path)
            
            new_model = CRFModel()
            new_model.load_model(temp_path)
            
            assert new_model.is_trained()
            assert new_model.tags == model.tags
            assert len(new_model.weights) > 0
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_crf_get_model_info(self):
        model = CRFModel(tags=['B', 'M', 'E', 'S'])
        
        info = model.get_model_info()
        assert info['trained'] == False
        
        corpus = [
            (['我', '爱', '中', '国'], ['S', 'S', 'B', 'E'])
        ]
        
        model.train(corpus, max_iter=10, verbose=False)
        
        info = model.get_model_info()
        assert info['trained'] == True
        assert 'num_tags' in info
        assert 'num_features' in info
    
    def test_crf_empty_input(self):
        model = CRFModel(tags=['B', 'M', 'E', 'S'])
        
        corpus = [
            (['我', '爱'], ['S', 'S'])
        ]
        
        model.train(corpus, max_iter=10, verbose=False)
        
        tags = model.predict([])
        assert tags == []
    
    def test_crf_train_empty_corpus(self):
        model = CRFModel(tags=['B', 'M', 'E', 'S'])
        
        with pytest.raises(ValueError):
            model.train([], max_iter=10, verbose=False)
    
    def test_crf_predict_without_training(self):
        model = CRFModel(tags=['B', 'M', 'E', 'S'])
        
        with pytest.raises(RuntimeError):
            model.predict(['我', '爱'])


class TestCRFSegmentor:
    def test_crf_segmentor_creation(self):
        segmentor = CRFSegmentor()
        assert not segmentor.is_trained()
    
    def test_crf_segmentor_basic_training(self):
        segmentor = CRFSegmentor()
        
        corpus = [
            ['我', '爱', '中国'],
            ['他', '是', '学生']
        ]
        
        segmentor.train(corpus, max_iter=10, verbose=False)
        
        assert segmentor.is_trained()
    
    def test_crf_segmentor_segment(self):
        segmentor = CRFSegmentor()
        
        corpus = [
            ['我', '爱', '中国'],
            ['他', '是', '学生']
        ]
        
        segmentor.train(corpus, max_iter=10, verbose=False)
        
        text = '我爱北京'
        words = segmentor.segment(text)
        
        assert isinstance(words, list)
        assert len(words) > 0
        assert ''.join(words) == text
    
    def test_crf_segmentor_segment_with_states(self):
        segmentor = CRFSegmentor()
        
        corpus = [
            ['我', '爱', '中国']
        ]
        
        segmentor.train(corpus, max_iter=10, verbose=False)
        
        text = '我爱中国'
        result = segmentor.segment_with_states(text)
        
        assert isinstance(result, list)
        assert len(result) == len(text)
        assert all(isinstance(item, tuple) and len(item) == 2 for item in result)
    
    def test_crf_segmentor_save_and_load(self):
        segmentor = CRFSegmentor()
        
        corpus = [
            ['我', '爱', '中国']
        ]
        
        segmentor.train(corpus, max_iter=10, verbose=False)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            temp_path = f.name
        
        try:
            segmentor.save_model(temp_path)
            
            new_segmentor = CRFSegmentor()
            new_segmentor.load_model(temp_path)
            
            assert new_segmentor.is_trained()
            
            text = '我爱中国'
            words1 = segmentor.segment(text)
            words2 = new_segmentor.segment(text)
            
            assert words1 == words2
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_crf_segmentor_get_model_info(self):
        segmentor = CRFSegmentor()
        
        info = segmentor.get_model_info()
        assert info['trained'] == False
        
        corpus = [
            ['我', '爱', '中国']
        ]
        
        segmentor.train(corpus, max_iter=10, verbose=False)
        
        info = segmentor.get_model_info()
        assert info['trained'] == True
    
    def test_crf_segmentor_empty_input(self):
        segmentor = CRFSegmentor()
        
        corpus = [
            ['我', '爱']
        ]
        
        segmentor.train(corpus, max_iter=10, verbose=False)
        
        words = segmentor.segment('')
        assert words == []
    
    def test_crf_segmentor_segment_without_training(self):
        segmentor = CRFSegmentor()
        
        with pytest.raises(RuntimeError):
            segmentor.segment('我爱中国')
    
    def test_crf_segmentor_single_char(self):
        segmentor = CRFSegmentor()
        
        corpus = [
            ['我', '爱', '你']
        ]
        
        segmentor.train(corpus, max_iter=10, verbose=False)
        
        words = segmentor.segment('我')
        assert words == ['我']
    
    def test_crf_segmentor_long_text(self):
        segmentor = CRFSegmentor()
        
        corpus = [
            ['我', '爱', '中国'],
            ['北京', '是', '首都']
        ]
        
        segmentor.train(corpus, max_iter=10, verbose=False)
        
        text = '我爱中国北京是首都'
        words = segmentor.segment(text)
        
        assert isinstance(words, list)
        assert len(words) > 0
        assert ''.join(words) == text


class TestSegmentorWithCRF:
    def test_segmentor_crf_mode(self):
        from AuroraNLP import Segmentor
        
        segmentor = Segmentor()
        
        corpus = [
            ['我', '爱', '中国']
        ]
        
        segmentor.train_crf(corpus, max_iter=10, verbose=False)
        
        assert segmentor.is_crf_trained()
    
    def test_segmentor_segment_with_crf(self):
        from AuroraNLP import Segmentor
        
        segmentor = Segmentor()
        
        corpus = [
            ['我', '爱', '中国']
        ]
        
        segmentor.train_crf(corpus, max_iter=10, verbose=False)
        
        text = '我爱北京'
        words = segmentor.segment(text, mode='crf')
        
        assert isinstance(words, list)
        assert len(words) > 0
    
    def test_segmentor_set_crf_mode(self):
        from AuroraNLP import Segmentor
        
        segmentor = Segmentor()
        segmentor.set_mode('crf')
        
        assert segmentor.mode == 'crf'
    
    def test_segmentor_crf_pos_tagging_error(self):
        from AuroraNLP import Segmentor
        
        segmentor = Segmentor()
        
        corpus = [
            ['我', '爱', '中国']
        ]
        
        segmentor.train_crf(corpus, max_iter=10, verbose=False)
        
        with pytest.raises(ValueError):
            segmentor.segment_with_pos('我爱中国', mode='crf')
    
    def test_segmentor_crf_from_file(self):
        from AuroraNLP import Segmentor
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8', suffix='.txt') as f:
            f.write('我 爱 中国\n')
            f.write('他 是 学生\n')
            temp_path = f.name
        
        try:
            segmentor = Segmentor()
            segmentor.train_crf_from_file(temp_path)
            
            assert segmentor.is_crf_trained()
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_segmentor_crf_save_and_load(self):
        from AuroraNLP import Segmentor
        
        segmentor = Segmentor()
        
        corpus = [
            ['我', '爱', '中国']
        ]
        
        segmentor.train_crf(corpus, max_iter=10, verbose=False)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            temp_path = f.name
        
        try:
            segmentor.save_crf_model(temp_path)
            
            new_segmentor = Segmentor()
            new_segmentor.load_crf_model(temp_path)
            
            assert new_segmentor.is_crf_trained()
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_segmentor_crf_model_info(self):
        from AuroraNLP import Segmentor
        
        segmentor = Segmentor()
        
        corpus = [
            ['我', '爱', '中国']
        ]
        
        segmentor.train_crf(corpus, max_iter=10, verbose=False)
        
        info = segmentor.get_crf_model_info()
        
        assert 'trained' in info
        assert info['trained'] == True
