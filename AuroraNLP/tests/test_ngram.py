import pytest
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from AuroraNLP import NGramModel, BigramModel, TrigramModel


class TestNGramModel:
    
    def test_ngram_basic_training(self):
        model = NGramModel(n=2)
        
        corpus = [
            ['我', '爱', '自然语言处理'],
            ['今天', '天气', '很好'],
            ['我们', '在', '学习', '中文', '分词']
        ]
        
        model.train(corpus)
        
        assert model.is_trained()
        
        info = model.get_model_info()
        assert info['trained'] == True
        assert info['n'] == 2
        assert info['vocabulary_size'] > 0
    
    def test_bigram_model(self):
        model = BigramModel()
        
        corpus = [
            ['我', '爱', '编程'],
            ['我', '爱', '学习']
        ]
        
        model.train(corpus)
        
        prob = model.probability('爱', ['我'])
        assert prob > 0
        
        prob2 = model.probability('编程', ['爱'])
        assert prob2 > 0
    
    def test_trigram_model(self):
        model = TrigramModel()
        
        corpus = [
            ['我', '爱', '自然语言处理'],
            ['我', '爱', '编程']
        ]
        
        model.train(corpus)
        
        prob = model.probability('自然语言处理', ['我', '爱'])
        assert prob > 0
    
    def test_sentence_probability(self):
        model = BigramModel()
        
        corpus = [
            ['我', '爱', '编程'],
            ['今天', '天气', '很好']
        ]
        
        model.train(corpus)
        
        prob1 = model.sentence_probability(['我', '爱', '编程'])
        prob2 = model.sentence_probability(['我', '爱', '学习'])
        
        assert isinstance(prob1, float)
        assert isinstance(prob2, float)
    
    def test_perplexity(self):
        model = BigramModel()
        
        corpus = [
            ['我', '爱', '编程'],
            ['今天', '天气', '很好']
        ]
        
        model.train(corpus)
        
        test_sentences = [['我', '爱', '学习']]
        perplexity = model.perplexity(test_sentences)
        
        assert perplexity > 0
        assert isinstance(perplexity, float)
    
    def test_predict_next_word(self):
        model = BigramModel()
        
        corpus = [
            ['我', '爱', '编程'],
            ['我', '爱', '学习'],
            ['我', '爱', '自然语言处理']
        ]
        
        model.train(corpus)
        
        predictions = model.predict_next_word(['我', '爱'], top_k=3)
        
        assert isinstance(predictions, list)
        assert len(predictions) <= 3
        assert all(isinstance(p, tuple) and len(p) == 2 for p in predictions)
    
    def test_disambiguate(self):
        model = BigramModel()
        
        corpus = [
            ['我', '爱', '编程'],
            ['今天', '天气', '很好'],
            ['研究', '自然语言处理']
        ]
        
        model.train(corpus)
        
        candidates = [
            ['爱', '编程'],
            ['天气', '很好']
        ]
        
        best, score = model.disambiguate(candidates, context_before=['我'])
        
        assert isinstance(best, list)
        assert isinstance(score, float)
    
    def test_smoothing_laplace(self):
        model = BigramModel()
        
        corpus = [['我', '爱', '编程']]
        model.train(corpus, smoothing='laplace', alpha=1.0)
        
        prob = model.probability('未知词', ['我'])
        assert prob > 0
    
    def test_save_and_load(self):
        model = BigramModel()
        
        corpus = [
            ['我', '爱', '自然语言处理'],
            ['今天', '天气', '很好']
        ]
        
        model.train(corpus)
        
        model_path = 'test_ngram_model.pkl'
        model.save_model(model_path)
        
        assert os.path.exists(model_path)
        
        model2 = BigramModel()
        model2.load_model(model_path)
        
        assert model2.is_trained()
        
        prob1 = model.probability('爱', ['我'])
        prob2 = model2.probability('爱', ['我'])
        
        assert prob1 == prob2
        
        if os.path.exists(model_path):
            os.remove(model_path)
    
    def test_empty_input(self):
        model = BigramModel()
        
        corpus = [['我', '爱', '编程']]
        model.train(corpus)
        
        prob = model.sentence_probability([])
        assert prob == 0.0
    
    def test_unseen_context(self):
        model = BigramModel()
        
        corpus = [['我', '爱', '编程']]
        model.train(corpus, smoothing='laplace')
        
        prob = model.probability('编程', ['未知词'])
        assert prob > 0
    
    def test_model_info(self):
        model = NGramModel(n=3)
        
        info = model.get_model_info()
        assert info['trained'] == False
        
        corpus = [['我', '爱', '编程']]
        model.train(corpus)
        
        info = model.get_model_info()
        assert info['trained'] == True
        assert 'vocabulary_size' in info
        assert 'total_tokens' in info
        assert 'ngram_types' in info
    
    def test_invalid_n(self):
        with pytest.raises(ValueError):
            NGramModel(n=0)
        
        with pytest.raises(ValueError):
            NGramModel(n=-1)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
