"""测试关键词提取模块"""

import pytest

from AuroraNLP.keyword_extractor import KeywordExtractor


class MockSegmentor:
    """简单的 Mock 分词器，按字分词"""

    def segment(self, text):
        return list(text)


@pytest.fixture
def mock_segmentor():
    """创建 Mock 分词器"""
    return MockSegmentor()


@pytest.fixture
def extractor():
    """创建关键词提取器"""
    return KeywordExtractor()


@pytest.fixture
def trained_extractor(extractor, mock_segmentor):
    """创建已构建 IDF 语料的提取器"""
    documents = [
        "自然语言处理是人工智能的重要方向",
        "深度学习在自然语言处理中有广泛应用",
        "机器学习和深度学习都是人工智能的方法",
        "自然语言处理包括分词和词性标注",
        "人工智能改变了人们的生活方式",
    ]
    extractor.build_idf_corpus(documents, mock_segmentor)
    return extractor


class TestKeywordExtractorInit:
    """测试关键词提取器初始化"""

    def test_init(self, extractor):
        """测试初始化"""
        assert extractor is not None
        assert extractor.get_document_count() == 0


class TestKeywordExtractorExtract:
    """测试关键词提取功能"""

    def test_extract_keywords_tfidf(self, trained_extractor, mock_segmentor):
        """测试 TF-IDF 提取"""
        text = "自然语言处理和深度学习是人工智能的重要方向"
        keywords = trained_extractor.extract_keywords_tfidf(text, mock_segmentor, top_k=5)
        assert isinstance(keywords, list)
        assert len(keywords) <= 5
        for kw, score in keywords:
            assert isinstance(kw, str)
            assert isinstance(score, float)
            assert score >= 0

    def test_extract_keywords_freq(self, extractor, mock_segmentor):
        """测试词频提取"""
        text = "自然语言自然语言处理"
        keywords = extractor.extract_keywords_freq(text, mock_segmentor, top_k=5)
        assert isinstance(keywords, list)
        assert len(keywords) > 0
        for kw, freq in keywords:
            assert isinstance(kw, str)
            assert isinstance(freq, int)
            assert freq >= 1

    def test_extract_keywords_textrank(self, extractor, mock_segmentor):
        """测试 TextRank 提取"""
        text = "自然语言处理是人工智能的重要方向深度学习在自然语言处理中有广泛应用"
        keywords = extractor.extract_keywords_textrank(text, mock_segmentor, top_k=5)
        assert isinstance(keywords, list)
        assert len(keywords) <= 5
        for kw, score in keywords:
            assert isinstance(kw, str)
            assert isinstance(score, float)


class TestKeywordExtractorIDF:
    """测试 IDF 语料构建"""

    def test_build_idf_corpus(self, extractor, mock_segmentor):
        """测试构建 IDF 语料"""
        documents = [
            "自然语言处理",
            "深度学习",
            "机器学习",
        ]
        extractor.build_idf_corpus(documents, mock_segmentor)
        assert extractor.get_document_count() == 3

    def test_get_document_count(self, extractor, mock_segmentor):
        """测试获取文档数"""
        assert extractor.get_document_count() == 0
        documents = ["文档一", "文档二", "文档三", "文档四"]
        extractor.build_idf_corpus(documents, mock_segmentor)
        assert extractor.get_document_count() == 4


class TestKeywordExtractorEdge:
    """测试关键词提取边界情况"""

    def test_extract_with_stopwords(self, extractor, mock_segmentor):
        """测试带停用词提取"""
        text = "自然语言的处理的"
        stopwords = {"的"}
        keywords = extractor.extract_keywords_tfidf(text, mock_segmentor, top_k=5, stopwords=stopwords)
        for kw, score in keywords:
            assert kw not in stopwords

    def test_empty_text(self, extractor, mock_segmentor):
        """测试空文本"""
        keywords = extractor.extract_keywords_tfidf("", mock_segmentor, top_k=5)
        assert keywords == []
        keywords_freq = extractor.extract_keywords_freq("", mock_segmentor, top_k=5)
        assert keywords_freq == []
