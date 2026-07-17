"""BatchProcessor 批量处理器测试"""
import pytest
from AuroraNLP.core.batch_processor import BatchProcessor


class TestBatchProcessor:
    """测试 BatchProcessor 批量处理功能"""

    def test_batch_processor_init(self, sample_segmentor):
        """初始化"""
        bp = BatchProcessor(sample_segmentor)
        assert bp.segmentor is sample_segmentor
        assert bp.batch_size == 100

    def test_batch_processor_init_custom_batch_size(self, sample_segmentor):
        """自定义批次大小初始化"""
        bp = BatchProcessor(sample_segmentor, batch_size=50)
        assert bp.batch_size == 50

    def test_segment_batch(self, sample_segmentor):
        """批量分词"""
        bp = BatchProcessor(sample_segmentor)
        texts = ["中国人在北京", "自然语言处理", "我爱学习"]
        results = bp.segment_batch(texts)
        assert isinstance(results, list)
        assert len(results) == 3
        for i, result in enumerate(results):
            assert isinstance(result, list)
            assert "".join(result) == texts[i]

    def test_segment_batch_iter(self, sample_segmentor):
        """迭代批量分词"""
        bp = BatchProcessor(sample_segmentor, batch_size=2)
        texts = ["中国人在北京", "自然语言处理", "我爱学习"]
        results = list(bp.segment_batch_iter(texts))
        assert isinstance(results, list)
        assert len(results) == 3
        for i, result in enumerate(results):
            assert isinstance(result, list)
            assert "".join(result) == texts[i]

    def test_segment_with_pos_batch(self, sample_segmentor):
        """批量带词性分词"""
        bp = BatchProcessor(sample_segmentor)
        texts = ["中国人在北京", "自然语言处理"]
        results = bp.segment_with_pos_batch(texts)
        assert isinstance(results, list)
        assert len(results) == 2
        for result in results:
            assert isinstance(result, list)
            assert all(isinstance(item, tuple) and len(item) == 2 for item in result)

    def test_segment_without_stopwords_batch(self, sample_segmentor):
        """批量去停用词分词"""
        sample_segmentor.add_stopword("的")
        sample_segmentor.add_stopword("在")
        bp = BatchProcessor(sample_segmentor)
        texts = ["我在中国的工作", "北京是中国的首都"]
        results = bp.segment_without_stopwords_batch(texts)
        assert isinstance(results, list)
        assert len(results) == 2
        for result in results:
            assert isinstance(result, list)
            assert "的" not in result
            assert "在" not in result

    def test_extract_keywords_batch(self, sample_segmentor):
        """批量关键词提取"""
        bp = BatchProcessor(sample_segmentor)
        texts = [
            "自然语言处理是人工智能的重要方向，自然语言处理技术发展迅速",
            "机器学习是人工智能的核心技术"
        ]
        results = bp.extract_keywords_batch(texts, top_k=5, method='freq', use_stopwords=False)
        assert isinstance(results, list)
        assert len(results) == 2
        for result in results:
            assert isinstance(result, list)
            assert len(result) <= 5
            assert all(isinstance(item, tuple) and len(item) == 2 for item in result)

    def test_compute_similarity_matrix(self, sample_segmentor):
        """相似度矩阵计算"""
        bp = BatchProcessor(sample_segmentor)
        texts = ["自然语言处理", "自然语言处理", "机器学习"]
        matrix = bp.compute_similarity_matrix(texts, method='jaccard', use_stopwords=False)
        assert isinstance(matrix, list)
        assert len(matrix) == 3
        for row in matrix:
            assert isinstance(row, list)
            assert len(row) == 3
            assert all(isinstance(v, float) for v in row)
        # 对角线应为 1.0
        assert matrix[0][0] == 1.0
        assert matrix[1][1] == 1.0
        assert matrix[2][2] == 1.0
        # 矩阵应对称
        assert matrix[0][1] == matrix[1][0]
        assert matrix[0][2] == matrix[2][0]

    def test_empty_batch(self, sample_segmentor):
        """空批次"""
        bp = BatchProcessor(sample_segmentor)
        results = bp.segment_batch([])
        assert results == []
        results_iter = list(bp.segment_batch_iter([]))
        assert results_iter == []
