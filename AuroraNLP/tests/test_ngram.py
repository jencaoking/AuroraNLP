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


class TestBigramModelEnhanced:
    
    def test_bigram_count(self):
        model = BigramModel()
        
        corpus = [
            ['我', '爱', '编程'],
            ['我', '爱', '学习'],
            ['我', '爱', '编程']
        ]
        
        model.train(corpus)
        
        count = model.get_bigram_count('我', '爱')
        assert count == 3
        
        count2 = model.get_bigram_count('爱', '编程')
        assert count2 == 2
        
        count3 = model.get_bigram_count('爱', '学习')
        assert count3 == 1
    
    def test_unigram_count(self):
        model = BigramModel()
        
        corpus = [
            ['我', '爱', '编程'],
            ['我', '爱', '学习']
        ]
        
        model.train(corpus)
        
        count = model.get_unigram_count('我')
        assert count == 2
        
        count2 = model.get_unigram_count('爱')
        assert count2 == 2
        
        count3 = model.get_unigram_count('编程')
        assert count3 == 1
    
    def test_bigram_frequency(self):
        model = BigramModel()
        
        corpus = [
            ['我', '爱', '编程'],
            ['我', '爱', '学习']
        ]
        
        model.train(corpus)
        
        freq = model.get_bigram_frequency('我', '爱')
        assert 0 < freq <= 1
        
        freq2 = model.get_bigram_frequency('不', '存在')
        assert freq2 == 0.0
    
    def test_conditional_probability(self):
        model = BigramModel()
        
        corpus = [
            ['我', '爱', '编程'],
            ['我', '爱', '学习'],
            ['我', '喜欢', '编程']
        ]
        
        model.train(corpus)
        
        prob = model.conditional_probability('我', '爱')
        assert 0 < prob <= 1
        
        prob2 = model.conditional_probability('我', '喜欢')
        assert 0 < prob2 <= 1
    
    def test_joint_probability(self):
        model = BigramModel()
        
        corpus = [
            ['我', '爱', '编程'],
            ['我', '爱', '学习']
        ]
        
        model.train(corpus)
        
        prob = model.joint_probability('我', '爱')
        assert 0 < prob <= 1
    
    def test_pmi(self):
        model = BigramModel()
        
        corpus = [
            ['我', '爱', '编程'],
            ['我', '爱', '编程'],
            ['我', '爱', '编程'],
            ['今天', '天气', '很好']
        ]
        
        model.train(corpus)
        
        pmi = model.pmi('我', '爱')
        assert isinstance(pmi, float)
        
        pmi2 = model.pmi('爱', '编程')
        assert isinstance(pmi2, float)
    
    def test_pmi_normalized(self):
        model = BigramModel()
        
        corpus = [
            ['我', '爱', '编程'],
            ['我', '爱', '学习']
        ]
        
        model.train(corpus)
        
        pmi_norm = model.pmi_normalized('我', '爱')
        assert isinstance(pmi_norm, float)
        assert -1 <= pmi_norm <= 1 or pmi_norm == 0.0
    
    def test_extract_collocations(self):
        model = BigramModel()
        
        corpus = [
            ['自然', '语言', '处理'],
            ['自然', '语言', '处理'],
            ['自然', '语言', '处理'],
            ['机器', '学习'],
            ['机器', '学习'],
            ['今天', '天气', '很好']
        ]
        
        model.train(corpus)
        
        collocations = model.extract_collocations(min_freq=2, min_pmi=0.0, top_k=5)
        
        assert isinstance(collocations, list)
        assert len(collocations) <= 5
        assert all(len(c) == 4 for c in collocations)
        assert all(isinstance(c[2], int) for c in collocations)
        assert all(isinstance(c[3], float) for c in collocations)
    
    def test_get_frequent_bigrams(self):
        model = BigramModel()
        
        corpus = [
            ['我', '爱', '编程'],
            ['我', '爱', '编程'],
            ['我', '爱', '学习']
        ]
        
        model.train(corpus)
        
        bigrams = model.get_frequent_bigrams(min_freq=2, top_k=10)
        
        assert isinstance(bigrams, list)
        assert all(len(b) == 3 for b in bigrams)
        assert all(b[2] >= 2 for b in bigrams)
    
    def test_get_word_collocations_after(self):
        model = BigramModel()
        
        corpus = [
            ['我', '爱', '编程'],
            ['我', '爱', '学习'],
            ['我', '喜欢', '编程']
        ]
        
        model.train(corpus)
        
        collocations = model.get_word_collocations('我', position='after', top_k=5)
        
        assert isinstance(collocations, list)
        assert all(len(c) == 3 for c in collocations)
        assert all(isinstance(c[1], int) for c in collocations)
        assert all(isinstance(c[2], float) for c in collocations)
    
    def test_get_word_collocations_before(self):
        model = BigramModel()
        
        corpus = [
            ['我', '爱', '编程'],
            ['他', '爱', '学习']
        ]
        
        model.train(corpus)
        
        collocations = model.get_word_collocations('爱', position='before', top_k=5)
        
        assert isinstance(collocations, list)
        assert all(len(c) == 3 for c in collocations)
    
    def test_get_word_collocations_invalid_position(self):
        model = BigramModel()
        
        corpus = [['我', '爱', '编程']]
        model.train(corpus)
        
        with pytest.raises(ValueError):
            model.get_word_collocations('爱', position='invalid')
    
    def test_bigram_model_info(self):
        model = BigramModel()
        
        corpus = [
            ['我', '爱', '编程'],
            ['我', '爱', '学习']
        ]
        
        model.train(corpus)
        
        info = model.get_model_info()
        
        assert info['trained'] == True
        assert 'total_bigrams' in info
        assert 'unique_bigrams' in info
        assert 'vocabulary_size' in info
        assert info['total_bigrams'] > 0
    
    def test_bigram_save_and_load(self):
        model = BigramModel()
        
        corpus = [
            ['我', '爱', '编程'],
            ['我', '爱', '学习']
        ]
        
        model.train(corpus)
        
        model_path = 'test_bigram_model.pkl'
        model.save_model(model_path)
        
        assert os.path.exists(model_path)
        
        bigram_path = model_path.replace('.pkl', '_bigram.pkl')
        assert os.path.exists(bigram_path)
        
        model2 = BigramModel()
        model2.load_model(model_path)
        
        assert model2.is_trained()
        
        count1 = model.get_bigram_count('我', '爱')
        count2 = model2.get_bigram_count('我', '爱')
        assert count1 == count2
        
        if os.path.exists(model_path):
            os.remove(model_path)
        if os.path.exists(bigram_path):
            os.remove(bigram_path)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
