"""测试文本相似度模块"""

import pytest

from AuroraNLP.text_analysis.similarity import Similarity


class MockSegmentor:
    """简单的 Mock 分词器，按字分词"""

    def segment(self, text):
        return list(text)


@pytest.fixture
def mock_segmentor():
    """创建 Mock 分词器"""
    return MockSegmentor()


@pytest.fixture
def similarity():
    """创建相似度计算器"""
    return Similarity()


class TestSimilarityMethods:
    """测试相似度计算方法"""

    def test_cosine_similarity(self, similarity, mock_segmentor):
        """测试余弦相似度"""
        score = similarity.cosine_similarity("自然语言处理", "自然语言分析", mock_segmentor)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_jaccard_similarity(self, similarity, mock_segmentor):
        """测试 Jaccard 相似度"""
        score = similarity.jaccard_similarity("自然语言处理", "自然语言分析", mock_segmentor)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_dice_similarity(self, similarity, mock_segmentor):
        """测试 Dice 相似度"""
        score = similarity.dice_similarity("自然语言处理", "自然语言分析", mock_segmentor)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_overlap_similarity(self, similarity, mock_segmentor):
        """测试重叠相似度"""
        score = similarity.overlap_similarity("自然语言处理", "自然语言分析", mock_segmentor)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_edit_distance(self, similarity):
        """测试编辑距离"""
        dist = similarity.edit_distance("abc", "abc")
        assert dist == 0
        dist2 = similarity.edit_distance("abc", "abd")
        assert dist2 == 1
        dist3 = similarity.edit_distance("", "abc")
        assert dist3 == 3

    def test_edit_similarity(self, similarity):
        """测试编辑距离相似度"""
        score = similarity.edit_similarity("abc", "abc")
        assert score == 1.0
        score2 = similarity.edit_similarity("abc", "abd")
        assert 0.0 < score2 < 1.0

    def test_batch_similarity(self, similarity, mock_segmentor):
        """测试批量相似度"""
        query = "自然语言处理"
        documents = ["深度学习", "自然语言分析", "机器学习"]
        results = similarity.batch_similarity(query, documents, mock_segmentor, method='cosine')
        assert isinstance(results, list)
        assert len(results) == 3
        # 结果应按分数降序排列
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)


class TestSimilarityEdge:
    """测试相似度边界情况"""

    def test_empty_text(self, similarity, mock_segmentor):
        """测试空文本返回0"""
        score = similarity.cosine_similarity("", "自然语言", mock_segmentor)
        assert score == 0.0
        score2 = similarity.jaccard_similarity("", "自然语言", mock_segmentor)
        assert score2 == 0.0

    def test_identical_text(self, similarity, mock_segmentor):
        """测试相同文本相似度为1"""
        score = similarity.jaccard_similarity("自然语言处理", "自然语言处理", mock_segmentor)
        assert score == 1.0
        score2 = similarity.dice_similarity("自然语言处理", "自然语言处理", mock_segmentor)
        assert score2 == 1.0
        score3 = similarity.overlap_similarity("自然语言处理", "自然语言处理", mock_segmentor)
        assert score3 == 1.0

    def test_invalid_method(self, similarity, mock_segmentor):
        """测试无效方法抛异常"""
        with pytest.raises(ValueError):
            similarity.batch_similarity("文本", ["文档"], mock_segmentor, method='invalid')
