import pytest
import tempfile
import os
from AuroraNLP.segmentor import Segmentor
from AuroraNLP.dictionary import Dictionary
from AuroraNLP.stopwords import StopWords
from AuroraNLP.keyword_extractor import KeywordExtractor
from AuroraNLP.similarity import Similarity
from AuroraNLP.trie import Trie
from AuroraNLP.batch_processor import BatchProcessor
from AuroraNLP.benchmark import PerformanceBenchmark


class TestStopWords:
    def test_load_default_stopwords(self):
        sw = StopWords(load_default=True)
        assert len(sw) > 0

    def test_add_stopword(self):
        sw = StopWords(load_default=False)
        sw.add_stopword("测试词")
        assert sw.is_stopword("测试词") == True

    def test_remove_stopword(self):
        sw = StopWords(load_default=False)
        sw.add_stopword("测试词")
        result = sw.remove_stopword("测试词")
        assert result == True
        assert sw.is_stopword("测试词") == False

    def test_filter(self):
        sw = StopWords(load_default=False)
        sw.add_stopword("的")
        sw.add_stopword("是")
        words = ["我", "的", "名字", "是", "张三"]
        result = sw.filter(words)
        assert "的" not in result
        assert "是" not in result
        assert "我" in result

    def test_filter_with_pos(self):
        sw = StopWords(load_default=False)
        sw.add_stopword("的")
        words_with_pos = [("我", "r"), ("的", "u"), ("名字", "n")]
        result = sw.filter_with_pos(words_with_pos)
        assert len(result) == 2
        assert ("的", "u") not in result

    def test_save_and_load(self):
        sw = StopWords(load_default=False)
        sw.add_stopword("测试词1")
        sw.add_stopword("测试词2")

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            temp_path = f.name

        try:
            sw.save_stopwords(temp_path)
            sw2 = StopWords(load_default=False)
            sw2.load_stopwords(temp_path)
            assert sw2.is_stopword("测试词1")
            assert sw2.is_stopword("测试词2")
        finally:
            os.unlink(temp_path)


class TestKeywordExtractor:
    def test_extract_keywords_freq(self):
        ke = KeywordExtractor()
        d = Dictionary(load_default=False)
        d.add_word("人工智能")
        d.add_word("机器学习")
        d.add_word("深度学习")
        d.add_word("技术")

        seg = Segmentor(dictionary=d, load_default_dict=False, load_default_stopwords=False)

        text = "人工智能和机器学习是重要的技术深度学习也是重要的技术"
        result = ke.extract_keywords_freq(text, seg, top_k=5)

        assert len(result) <= 5
        assert all(isinstance(item, tuple) for item in result)

    def test_extract_keywords_tfidf(self):
        ke = KeywordExtractor()
        d = Dictionary(load_default=False)
        d.add_word("人工智能")
        d.add_word("机器学习")
        d.add_word("技术")

        seg = Segmentor(dictionary=d, load_default_dict=False, load_default_stopwords=False)

        text = "人工智能和机器学习是重要的技术"
        result = ke.extract_keywords_tfidf(text, seg, top_k=5)

        assert isinstance(result, list)
        assert all(isinstance(item, tuple) for item in result)

    def test_extract_keywords_textrank(self):
        ke = KeywordExtractor()
        d = Dictionary(load_default=False)
        d.add_word("人工智能")
        d.add_word("机器学习")
        d.add_word("技术")

        seg = Segmentor(dictionary=d, load_default_dict=False, load_default_stopwords=False)

        text = "人工智能和机器学习是重要的技术"
        result = ke.extract_keywords_textrank(text, seg, top_k=5)

        assert isinstance(result, list)


class TestSimilarity:
    def test_cosine_similarity(self):
        sim = Similarity()
        d = Dictionary(load_default=False)
        d.add_word("今天")
        d.add_word("天气")
        d.add_word("很好")

        seg = Segmentor(dictionary=d, load_default_dict=False, load_default_stopwords=False)

        text1 = "今天天气很好"
        text2 = "今天天气很好"
        result = sim.cosine_similarity(text1, text2, seg)

        assert result >= 0.0
        assert result <= 1.0

    def test_jaccard_similarity(self):
        sim = Similarity()
        d = Dictionary(load_default=False)
        d.add_word("今天")
        d.add_word("天气")
        d.add_word("很好")

        seg = Segmentor(dictionary=d, load_default_dict=False, load_default_stopwords=False)

        text1 = "今天天气很好"
        text2 = "今天天气"
        result = sim.jaccard_similarity(text1, text2, seg)

        assert result >= 0.0
        assert result <= 1.0

    def test_edit_distance(self):
        sim = Similarity()
        result = sim.edit_distance("kitten", "sitting")
        assert result == 3

    def test_edit_similarity(self):
        sim = Similarity()
        result = sim.edit_similarity("hello", "hello")
        assert result == 1.0

    def test_batch_similarity(self):
        sim = Similarity()
        d = Dictionary(load_default=False)
        d.add_word("人工智能")
        d.add_word("机器学习")

        seg = Segmentor(dictionary=d, load_default_dict=False, load_default_stopwords=False)

        query = "人工智能"
        documents = ["机器学习", "人工智能技术"]
        result = sim.batch_similarity(query, documents, seg)

        assert len(result) == 2
        assert all(isinstance(item, tuple) for item in result)


class TestSegmentorNewFeatures:
    def test_segment_without_stopwords(self):
        seg = Segmentor(load_default_dict=True, load_default_stopwords=True)
        seg.add_stopword("的")

        text = "今天的天气很好"
        result = seg.segment_without_stopwords(text)

        assert "的" not in result

    def test_extract_keywords(self):
        seg = Segmentor(load_default_dict=True, load_default_stopwords=True)

        text = "人工智能和机器学习是重要的技术领域"
        result = seg.extract_keywords(text, top_k=5, method='freq')

        assert isinstance(result, list)
        assert len(result) <= 5

    def test_compute_similarity(self):
        seg = Segmentor(load_default_dict=True, load_default_stopwords=True)

        text1 = "今天天气很好"
        text2 = "今天天气不错"
        result = seg.compute_similarity(text1, text2, method='cosine')

        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_add_and_remove_stopword(self):
        seg = Segmentor(load_default_dict=False, load_default_stopwords=False)

        seg.add_stopword("测试词")
        assert seg.is_stopword("测试词") == True

        seg.remove_stopword("测试词")
        assert seg.is_stopword("测试词") == False


class TestBatchProcessor:
    def test_segment_batch(self):
        seg = Segmentor(load_default_dict=True, load_default_stopwords=False)
        bp = BatchProcessor(seg)

        texts = ["今天天气很好", "明天天气也不错"]
        result = bp.segment_batch(texts)

        assert len(result) == 2
        assert all(isinstance(r, list) for r in result)

    def test_segment_batch_iter(self):
        seg = Segmentor(load_default_dict=True, load_default_stopwords=False)
        bp = BatchProcessor(seg)

        texts = ["今天天气很好", "明天天气也不错"]
        result = list(bp.segment_batch_iter(texts))

        assert len(result) == 2

    def test_extract_keywords_batch(self):
        seg = Segmentor(load_default_dict=True, load_default_stopwords=True)
        bp = BatchProcessor(seg)

        texts = ["人工智能技术", "机器学习算法"]
        result = bp.extract_keywords_batch(texts, top_k=3)

        assert len(result) == 2
        assert all(isinstance(r, list) for r in result)

    def test_segment_large_text(self):
        seg = Segmentor(load_default_dict=True, load_default_stopwords=False)
        bp = BatchProcessor(seg)

        text = "今天天气很好" * 1000
        result = bp.segment_large_text(text, chunk_size=100)

        assert isinstance(result, list)
        assert len(result) > 0


class TestPerformanceBenchmark:
    def test_benchmark_segment(self):
        seg = Segmentor(load_default_dict=True, load_default_stopwords=False)
        bench = PerformanceBenchmark(seg)

        result = bench.benchmark_segment("今天天气很好", iterations=10)

        assert result.iterations == 10
        assert result.total_time > 0
        assert result.avg_time > 0

    def test_run_full_benchmark(self):
        seg = Segmentor(load_default_dict=True, load_default_stopwords=False)
        bench = PerformanceBenchmark(seg)

        test_texts = ["今天天气很好", "明天天气也不错"]
        results = bench.run_full_benchmark(test_texts, iterations=10)

        assert isinstance(results, dict)
        assert 'segment_forward' in results
        assert 'segment_backward' in results
        assert 'segment_bidirectional' in results

    def test_format_result(self):
        from AuroraNLP.benchmark import BenchmarkResult

        result = BenchmarkResult(
            operation="test",
            total_time=1.0,
            iterations=100,
            avg_time=0.01,
            min_time=0.005,
            max_time=0.02,
            ops_per_second=100.0
        )

        formatted = PerformanceBenchmark.format_result(result)
        assert "test" in formatted
        assert "100" in formatted


class TestIntegration:
    def test_full_pipeline(self):
        seg = Segmentor()

        text = "人工智能和机器学习是当今最重要的技术领域，深度学习算法在图像识别和自然语言处理方面取得了突破性进展。"

        words = seg.segment(text)
        assert isinstance(words, list)

        words_no_stop = seg.segment_without_stopwords(text)
        assert len(words_no_stop) <= len(words)

        keywords = seg.extract_keywords(text, top_k=5)
        assert len(keywords) <= 5

        pos_result = seg.segment_with_pos(text)
        assert all(isinstance(item, tuple) and len(item) == 2 for item in pos_result)

        entities = seg.recognize_entities(text)
        assert isinstance(entities, list)

    def test_similarity_workflow(self):
        seg = Segmentor()

        docs = [
            "人工智能技术发展迅速",
            "机器学习算法不断优化",
            "深度学习应用广泛"
        ]

        seg.build_similarity_corpus(docs)

        query = "人工智能和机器学习"
        similar = seg.batch_similarity(query, docs)

        assert len(similar) == 3
        assert all(score >= 0 for _, score in similar)

    def test_keyword_extraction_with_corpus(self):
        seg = Segmentor()

        corpus = [
            "人工智能是计算机科学的一个分支",
            "机器学习是人工智能的核心技术",
            "深度学习是机器学习的重要方法"
        ]

        seg.build_keyword_corpus(corpus)

        text = "人工智能和机器学习技术"
        keywords = seg.extract_keywords(text, method='tfidf')

        assert isinstance(keywords, list)
