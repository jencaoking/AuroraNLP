import pytest
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from AuroraNLP import Segmentor, HMMSegmentor


class TestHMM:
    
    def test_hmm_basic_training(self):
        hmm = HMMSegmentor()
        
        corpus = [
            ['我', '爱', '自然语言处理'],
            ['今天', '天气', '很好'],
            ['我们', '在', '学习', '中文', '分词']
        ]
        
        hmm.train(corpus)
        
        assert hmm.is_trained()
        
        model_info = hmm.get_model_info()
        assert model_info['trained'] == True
        assert model_info['total_states'] > 0
    
    def test_hmm_viterbi(self):
        hmm = HMMSegmentor()
        
        corpus = [
            ['我', '爱', '自然语言处理'],
            ['今天', '天气', '很好']
        ]
        
        hmm.train(corpus)
        
        text = "我爱自然语言处理"
        states = hmm.viterbi(text)
        
        assert len(states) == len(text)
        assert all(s in ['B', 'M', 'E', 'S'] for s in states)
    
    def test_hmm_segment(self):
        hmm = HMMSegmentor()
        
        corpus = [
            ['我', '爱', '自然语言处理'],
            ['今天', '天气', '很好'],
            ['北京', '是', '中国', '的', '首都']
        ]
        
        hmm.train(corpus)
        
        text = "我爱自然语言处理"
        words = hmm.segment(text)
        
        assert isinstance(words, list)
        assert len(words) > 0
        assert all(isinstance(w, str) for w in words)
        assert ''.join(words) == text
    
    def test_hmm_segment_with_states(self):
        hmm = HMMSegmentor()
        
        corpus = [
            ['我', '爱', '编程'],
            ['今天', '天气', '很好']
        ]
        
        hmm.train(corpus)
        
        text = "我爱编程"
        result = hmm.segment_with_states(text)
        
        assert len(result) == len(text)
        assert all(len(item) == 2 for item in result)
        assert all(isinstance(item[0], str) and isinstance(item[1], str) for item in result)
    
    def test_hmm_save_and_load(self):
        hmm = HMMSegmentor()
        
        corpus = [
            ['我', '爱', '自然语言处理'],
            ['今天', '天气', '很好']
        ]
        
        hmm.train(corpus)
        
        model_path = 'test_hmm_model.pkl'
        hmm.save_model(model_path)
        
        assert os.path.exists(model_path)
        
        hmm2 = HMMSegmentor()
        hmm2.load_model(model_path)
        
        assert hmm2.is_trained()
        
        text = "我爱自然语言处理"
        words1 = hmm.segment(text)
        words2 = hmm2.segment(text)
        
        assert words1 == words2
        
        if os.path.exists(model_path):
            os.remove(model_path)
    
    def test_segmentor_with_hmm(self):
        seg = Segmentor()
        
        corpus = [
            ['我', '爱', '自然语言处理'],
            ['今天', '天气', '很好']
        ]
        
        seg.train_hmm(corpus)
        
        assert seg.is_hmm_trained()
        
        text = "我爱自然语言处理"
        words = seg.segment(text, mode='hmm')
        
        assert isinstance(words, list)
        assert len(words) > 0
    
    def test_segmentor_hmm_mode(self):
        seg = Segmentor()
        
        corpus = [
            ['我', '爱', '编程'],
            ['今天', '天气', '很好']
        ]
        
        seg.train_hmm(corpus)
        
        text = "我爱编程"
        words = seg.segment(text, mode='hmm')
        
        assert isinstance(words, list)
        assert ''.join(words) == text
    
    def test_segmentor_train_from_file(self):
        seg = Segmentor()
        
        corpus_path = os.path.join(
            os.path.dirname(__file__), 
            '..', 
            'AuroraNLP', 
            'data', 
            'train_corpus.txt'
        )
        
        if os.path.exists(corpus_path):
            seg.train_hmm_from_file(corpus_path)
            assert seg.is_hmm_trained()
            
            text = "我爱自然语言处理"
            words = seg.segment(text, mode='hmm')
            assert isinstance(words, list)
    
    def test_hmm_unseen_text(self):
        hmm = HMMSegmentor()
        
        corpus = [
            ['我', '爱', '编程'],
            ['今天', '天气', '很好']
        ]
        
        hmm.train(corpus)
        
        text = "他喜欢学习"
        words = hmm.segment(text)
        
        assert isinstance(words, list)
        assert ''.join(words) == text
    
    def test_hmm_empty_input(self):
        hmm = HMMSegmentor()
        
        corpus = [['我', '爱', '编程']]
        hmm.train(corpus)
        
        words = hmm.segment('')
        assert words == []
        
        states = hmm.viterbi('')
        assert states == []
    
    def test_hmm_single_char(self):
        hmm = HMMSegmentor()
        
        corpus = [['我', '爱', '你']]
        hmm.train(corpus)
        
        words = hmm.segment('我')
        assert words == ['我']
    
    def test_hmm_model_info(self):
        hmm = HMMSegmentor()
        
        info = hmm.get_model_info()
        assert info['trained'] == False
        
        corpus = [['我', '爱', '编程']]
        hmm.train(corpus)
        
        info = hmm.get_model_info()
        assert info['trained'] == True
        assert 'total_states' in info
        assert 'state_counts' in info
        assert 'vocabulary_sizes' in info


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
