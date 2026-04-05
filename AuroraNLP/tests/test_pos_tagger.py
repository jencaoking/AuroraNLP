import pytest
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from AuroraNLP import HMMPOSTagger, CRFPOSTagger, POS_TAGS, DEFAULT_TAGS


class TestHMMPOSTagger:
    
    def test_hmm_pos_basic_training(self):
        tagger = HMMPOSTagger()
        
        corpus = [
            (['我', '爱', '自然语言处理'], ['r', 'v', 'n']),
            (['今天', '天气', '很好'], ['t', 'n', 'a']),
            (['他', '在', '学习'], ['r', 'p', 'v'])
        ]
        
        tagger.train(corpus)
        
        assert tagger.is_trained()
        
        model_info = tagger.get_model_info()
        assert model_info['trained'] == True
        assert model_info['total_tags'] > 0
    
    def test_hmm_pos_tag(self):
        tagger = HMMPOSTagger()
        
        corpus = [
            (['我', '爱', '编程'], ['r', 'v', 'v']),
            (['今天', '天气', '很好'], ['t', 'n', 'a']),
            (['他', '喜欢', '学习'], ['r', 'v', 'v'])
        ]
        
        tagger.train(corpus)
        
        words = ['我', '爱', '编程']
        tags = tagger.tag(words)
        
        assert len(tags) == len(words)
        assert all(isinstance(t, str) for t in tags)
    
    def test_hmm_pos_tag_sentence(self):
        tagger = HMMPOSTagger()
        
        corpus = [
            (['我', '爱', '编程'], ['r', 'v', 'v']),
            (['今天', '天气', '很好'], ['t', 'n', 'a'])
        ]
        
        tagger.train(corpus)
        
        words = ['我', '爱', '编程']
        result = tagger.tag_sentence(words)
        
        assert len(result) == len(words)
        assert all(isinstance(item, tuple) and len(item) == 2 for item in result)
        assert all(isinstance(item[0], str) and isinstance(item[1], str) for item in result)
    
    def test_hmm_pos_viterbi(self):
        tagger = HMMPOSTagger()
        
        corpus = [
            (['我', '爱', '自然语言处理'], ['r', 'v', 'n']),
            (['北京', '是', '中国', '的', '首都'], ['ns', 'v', 'ns', 'u', 'n'])
        ]
        
        tagger.train(corpus)
        
        words = ['我', '爱', '自然语言处理']
        tags = tagger.viterbi(words)
        
        assert len(tags) == len(words)
    
    def test_hmm_pos_save_and_load(self):
        tagger = HMMPOSTagger()
        
        corpus = [
            (['我', '爱', '编程'], ['r', 'v', 'v']),
            (['今天', '天气', '很好'], ['t', 'n', 'a'])
        ]
        
        tagger.train(corpus)
        
        model_path = 'test_hmm_pos_model.pkl'
        tagger.save_model(model_path)
        
        assert os.path.exists(model_path)
        
        tagger2 = HMMPOSTagger()
        tagger2.load_model(model_path)
        
        assert tagger2.is_trained()
        
        words = ['我', '爱', '编程']
        tags1 = tagger.tag(words)
        tags2 = tagger2.tag(words)
        
        assert tags1 == tags2
        
        if os.path.exists(model_path):
            os.remove(model_path)
    
    def test_hmm_pos_empty_input(self):
        tagger = HMMPOSTagger()
        
        corpus = [(['我', '爱', '编程'], ['r', 'v', 'v'])]
        tagger.train(corpus)
        
        tags = tagger.tag([])
        assert tags == []
    
    def test_hmm_pos_model_info(self):
        tagger = HMMPOSTagger()
        
        info = tagger.get_model_info()
        assert info['trained'] == False
        
        corpus = [(['我', '爱', '编程'], ['r', 'v', 'v'])]
        tagger.train(corpus)
        
        info = tagger.get_model_info()
        assert info['trained'] == True
        assert 'num_tags' in info
        assert 'tags' in info
        assert 'total_tags' in info
    
    def test_hmm_pos_unseen_words(self):
        tagger = HMMPOSTagger()
        
        corpus = [
            (['我', '爱', '编程'], ['r', 'v', 'v']),
            (['今天', '天气', '很好'], ['t', 'n', 'a'])
        ]
        
        tagger.train(corpus)
        
        words = ['他', '喜欢', '学习']
        tags = tagger.tag(words)
        
        assert len(tags) == len(words)


class TestCRFPOSTagger:
    
    def test_crf_pos_basic_training(self):
        tagger = CRFPOSTagger()
        
        corpus = [
            (['我', '爱', '自然语言处理'], ['r', 'v', 'n']),
            (['今天', '天气', '很好'], ['t', 'n', 'a']),
            (['他', '在', '学习'], ['r', 'p', 'v'])
        ]
        
        tagger.train(corpus, max_iter=10, verbose=False)
        
        assert tagger.is_trained()
        
        model_info = tagger.get_model_info()
        assert model_info['trained'] == True
        assert model_info['num_tags'] > 0
    
    def test_crf_pos_tag(self):
        tagger = CRFPOSTagger()
        
        corpus = [
            (['我', '爱', '编程'], ['r', 'v', 'v']),
            (['今天', '天气', '很好'], ['t', 'n', 'a']),
            (['他', '喜欢', '学习'], ['r', 'v', 'v'])
        ]
        
        tagger.train(corpus, max_iter=10, verbose=False)
        
        words = ['我', '爱', '编程']
        tags = tagger.tag(words)
        
        assert len(tags) == len(words)
        assert all(isinstance(t, str) for t in tags)
    
    def test_crf_pos_tag_sentence(self):
        tagger = CRFPOSTagger()
        
        corpus = [
            (['我', '爱', '编程'], ['r', 'v', 'v']),
            (['今天', '天气', '很好'], ['t', 'n', 'a'])
        ]
        
        tagger.train(corpus, max_iter=10, verbose=False)
        
        words = ['我', '爱', '编程']
        result = tagger.tag_sentence(words)
        
        assert len(result) == len(words)
        assert all(isinstance(item, tuple) and len(item) == 2 for item in result)
    
    def test_crf_pos_viterbi(self):
        tagger = CRFPOSTagger()
        
        corpus = [
            (['我', '爱', '自然语言处理'], ['r', 'v', 'n']),
            (['北京', '是', '中国', '的', '首都'], ['ns', 'v', 'ns', 'u', 'n'])
        ]
        
        tagger.train(corpus, max_iter=10, verbose=False)
        
        words = ['我', '爱', '自然语言处理']
        tags = tagger.viterbi(words)
        
        assert len(tags) == len(words)
    
    def test_crf_pos_save_and_load(self):
        tagger = CRFPOSTagger()
        
        corpus = [
            (['我', '爱', '编程'], ['r', 'v', 'v']),
            (['今天', '天气', '很好'], ['t', 'n', 'a'])
        ]
        
        tagger.train(corpus, max_iter=10, verbose=False)
        
        model_path = 'test_crf_pos_model.pkl'
        tagger.save_model(model_path)
        
        assert os.path.exists(model_path)
        
        tagger2 = CRFPOSTagger()
        tagger2.load_model(model_path)
        
        assert tagger2.is_trained()
        
        words = ['我', '爱', '编程']
        tags1 = tagger.tag(words)
        tags2 = tagger2.tag(words)
        
        assert tags1 == tags2
        
        if os.path.exists(model_path):
            os.remove(model_path)
    
    def test_crf_pos_empty_input(self):
        tagger = CRFPOSTagger()
        
        corpus = [(['我', '爱', '编程'], ['r', 'v', 'v'])]
        tagger.train(corpus, max_iter=10, verbose=False)
        
        tags = tagger.tag([])
        assert tags == []
    
    def test_crf_pos_model_info(self):
        tagger = CRFPOSTagger()
        
        info = tagger.get_model_info()
        assert info['trained'] == False
        
        corpus = [(['我', '爱', '编程'], ['r', 'v', 'v'])]
        tagger.train(corpus, max_iter=10, verbose=False)
        
        info = tagger.get_model_info()
        assert info['trained'] == True
        assert 'num_tags' in info
        assert 'tags' in info
        assert 'num_features' in info
    
    def test_crf_pos_unseen_words(self):
        tagger = CRFPOSTagger()
        
        corpus = [
            (['我', '爱', '编程'], ['r', 'v', 'v']),
            (['今天', '天气', '很好'], ['t', 'n', 'a'])
        ]
        
        tagger.train(corpus, max_iter=10, verbose=False)
        
        words = ['他', '喜欢', '学习']
        tags = tagger.tag(words)
        
        assert len(tags) == len(words)


class TestPOSTags:
    
    def test_pos_tags_dict(self):
        assert isinstance(POS_TAGS, dict)
        assert len(POS_TAGS) > 0
        assert 'n' in POS_TAGS
        assert 'v' in POS_TAGS
        assert 'a' in POS_TAGS
    
    def test_default_tags_list(self):
        assert isinstance(DEFAULT_TAGS, list)
        assert len(DEFAULT_TAGS) > 0
        assert 'n' in DEFAULT_TAGS
        assert 'v' in DEFAULT_TAGS


class TestTrainFromFile:
    
    def test_train_pos_from_file(self):
        corpus_path = os.path.join(
            os.path.dirname(__file__), 
            '..', 
            'AuroraNLP', 
            'data', 
            'pos_corpus.txt'
        )
        
        if os.path.exists(corpus_path):
            tagger = HMMPOSTagger()
            from AuroraNLP.pos_tagger import train_pos_from_file
            train_pos_from_file(tagger, corpus_path)
            
            assert tagger.is_trained()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
