"""Managers 管理器测试"""
import pytest
from AuroraNLP.managers import (
    DictionaryService,
    StopWordsManager,
    KeywordExtractorManager,
    SimilarityManager,
    MLSegmentorManager,
    LatticeSegmentorManager,
    AmbiguityDetectorManager,
    NewWordDetectorManager,
    HybridSegmentorManager,
)
from AuroraNLP.dictionary import Dictionary


class TestDictionaryService:
    """测试 DictionaryService"""

    def test_dictionary_service_init(self):
        """词典服务初始化"""
        service = DictionaryService(load_default_dict=False)
        assert service is not None
        assert service.dictionary is not None

    def test_dictionary_service_init_with_dict(self):
        """带词典的初始化"""
        d = Dictionary(load_default=False)
        d.add_word("测试", "n", 1.0)
        service = DictionaryService(dictionary=d, load_default_dict=False)
        assert service is not None
        assert service.get_dictionary_size() >= 1


class TestStopWordsManager:
    """测试 StopWordsManager"""

    def test_stopwords_manager_init(self):
        """停用词管理器初始化"""
        manager = StopWordsManager(load_default_stopwords=False)
        assert manager is not None
        assert manager.stopwords is not None


class TestKeywordExtractorManager:
    """测试 KeywordExtractorManager"""

    def test_keyword_extractor_manager_init(self):
        """关键词管理器初始化"""
        manager = KeywordExtractorManager()
        assert manager is not None
        assert manager.keyword_extractor is not None


class TestSimilarityManager:
    """测试 SimilarityManager"""

    def test_similarity_manager_init(self):
        """相似度管理器初始化"""
        manager = SimilarityManager()
        assert manager is not None
        assert manager.similarity is not None


class TestMLSegmentorManager:
    """测试 MLSegmentorManager"""

    def test_ml_segmentor_manager_init(self):
        """机器学习分词管理器初始化"""
        manager = MLSegmentorManager()
        assert manager is not None
        assert manager.hmm_segmentor is not None
        assert manager.crf_segmentor is not None
        assert manager.perceptron_segmentor is not None
        assert manager.use_hmm is False
        assert manager.use_crf is False
        assert manager.use_perceptron is False


class TestLatticeSegmentorManager:
    """测试 LatticeSegmentorManager"""

    def test_lattice_segmentor_manager_init(self):
        """词格管理器初始化"""
        d = Dictionary(load_default=False)
        manager = LatticeSegmentorManager(dictionary=d)
        assert manager is not None
        assert manager.lattice_segmentor is not None
        assert manager.use_lattice is False


class TestAmbiguityDetectorManager:
    """测试 AmbiguityDetectorManager"""

    def test_ambiguity_detector_manager_init(self):
        """歧义检测管理器初始化"""
        d = Dictionary(load_default=False)
        manager = AmbiguityDetectorManager(dictionary=d)
        assert manager is not None
        assert manager.ambiguity_detector is not None


class TestNewWordDetectorManager:
    """测试 NewWordDetectorManager"""

    def test_new_word_detector_manager_init(self):
        """新词检测管理器初始化"""
        manager = NewWordDetectorManager()
        assert manager is not None
        assert manager.new_word_detector is not None


class TestHybridSegmentorManager:
    """测试 HybridSegmentorManager"""

    def test_hybrid_segmentor_manager_init(self):
        """混合分词管理器初始化"""
        d = Dictionary(load_default=False)
        manager = HybridSegmentorManager(dictionary=d)
        assert manager is not None
        assert manager._dictionary is d
        assert manager.use_hybrid is False
