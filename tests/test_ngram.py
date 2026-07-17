"""N-gram 模型测试"""
import pytest
from AuroraNLP.segmentation.ngram import NGramModel, BigramModel, TrigramModel


class TestNGramInit:
    """NGramModel 初始化测试"""

    def test_ngram_init(self):
        """测试 NGramModel 初始化 (n=2)"""
        model = NGramModel(n=2)
        assert model.n == 2
        assert model.is_trained() is False

    def test_ngram_invalid_n(self):
        """测试 n < 1 时抛异常"""
        with pytest.raises(ValueError, match="n must be at least 1"):
            NGramModel(n=0)
        with pytest.raises(ValueError, match="n must be at least 1"):
            NGramModel(n=-1)


class TestNGramTrain:
    """NGramModel 训练测试"""

    def test_ngram_train(self, sample_corpus):
        """测试 NGramModel 训练"""
        model = NGramModel(n=2)
        model.train(sample_corpus)
        assert model.is_trained() is True
        assert model.total_tokens > 0
        assert len(model.vocabulary) > 0


class TestNGramProbability:
    """NGramModel 概率计算测试"""

    def test_ngram_probability(self, sample_corpus):
        """测试 n-gram 概率计算"""
        model = NGramModel(n=2)
        model.train(sample_corpus)
        prob = model.probability("中国", ["我", "爱"])
        assert isinstance(prob, float)
        assert 0.0 <= prob <= 1.0


class TestNGramSentenceProbability:
    """NGramModel 句子概率测试"""

    def test_ngram_sentence_probability(self, sample_corpus):
        """测试句子概率计算"""
        model = NGramModel(n=2)
        model.train(sample_corpus)
        log_prob = model.sentence_probability(["我", "爱", "中国"])
        assert isinstance(log_prob, float)
        assert log_prob <= 0.0


class TestNGramNotTrained:
    """NGramModel 未训练异常测试"""

    def test_ngram_not_trained(self):
        """测试未训练时抛异常"""
        model = NGramModel(n=2)
        with pytest.raises(RuntimeError, match="not been trained"):
            model.probability("中国", ["我", "爱"])


class TestBigramModel:
    """BigramModel 测试"""

    def test_bigram_init(self):
        """测试 BigramModel 初始化"""
        model = BigramModel()
        assert model.n == 2
        assert model.is_trained() is False

    def test_bigram_train(self, sample_corpus):
        """测试 BigramModel 训练"""
        model = BigramModel()
        model.train(sample_corpus)
        assert model.is_trained() is True
        assert model.total_tokens > 0
        assert model._total_bigrams > 0


class TestTrigramModel:
    """TrigramModel 测试"""

    def test_trigram_init(self):
        """测试 TrigramModel 初始化"""
        model = TrigramModel()
        assert model.n == 3
        assert model.is_trained() is False

    def test_trigram_train(self, sample_corpus):
        """测试 TrigramModel 训练"""
        model = TrigramModel()
        model.train(sample_corpus)
        assert model.is_trained() is True
        assert model.total_tokens > 0
