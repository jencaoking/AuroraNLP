"""Segmentor 主类测试"""
import pytest
from AuroraNLP.segmentation.segmentor import Segmentor


class TestSegmentorSegment:
    """测试 Segmentor 分词功能"""

    def test_segment_default(self, sample_segmentor):
        """默认分词"""
        seg = sample_segmentor
        result = seg.segment("中国人在北京")
        assert isinstance(result, list)
        assert "".join(result) == "中国人在北京"

    def test_segment_forward(self, sample_segmentor):
        """正向分词"""
        seg = sample_segmentor
        result = seg.segment("中国人在北京", mode='forward')
        assert isinstance(result, list)
        assert "".join(result) == "中国人在北京"

    def test_segment_backward(self, sample_segmentor):
        """逆向分词"""
        seg = sample_segmentor
        result = seg.segment("中国人在北京", mode='backward')
        assert isinstance(result, list)
        assert "".join(result) == "中国人在北京"

    def test_segment_bidirectional(self, sample_segmentor):
        """双向分词"""
        seg = sample_segmentor
        result = seg.segment("中国人在北京", mode='bidirectional')
        assert isinstance(result, list)
        assert "".join(result) == "中国人在北京"

    def test_segment_with_pos(self, sample_segmentor):
        """带词性分词"""
        seg = sample_segmentor
        result = seg.segment_with_pos("中国人在北京")
        assert isinstance(result, list)
        assert all(isinstance(item, tuple) and len(item) == 2 for item in result)
        words = [item[0] for item in result]
        assert "".join(words) == "中国人在北京"

    def test_segment_without_stopwords(self, sample_segmentor):
        """去停用词分词"""
        seg = sample_segmentor
        seg.add_stopword("的")
        seg.add_stopword("在")
        result = seg.segment_without_stopwords("我在中国的工作")
        assert isinstance(result, list)
        assert "的" not in result
        assert "在" not in result

    def test_add_word(self, sample_segmentor):
        """添加词"""
        seg = sample_segmentor
        seg.add_word("人工智能", "n", 1.5)
        result = seg.segment("人工智能很好")
        assert isinstance(result, list)
        assert "".join(result) == "人工智能很好"

    def test_remove_word(self, sample_segmentor):
        """删除词"""
        seg = sample_segmentor
        seg.add_word("临时词", "n", 1.0)
        assert seg.remove_word("临时词") is True
        assert seg.remove_word("不存在") is False

    def test_add_stopword(self, sample_segmentor):
        """添加停用词"""
        seg = sample_segmentor
        seg.add_stopword("测试停用词")
        assert seg.is_stopword("测试停用词") is True

    def test_set_mode(self, sample_segmentor):
        """设置分词模式"""
        seg = sample_segmentor
        seg.set_mode('forward')
        assert seg.mode == 'forward'
        seg.set_mode('backward')
        assert seg.mode == 'backward'
        seg.set_mode('bidirectional')
        assert seg.mode == 'bidirectional'

    def test_invalid_mode(self, sample_segmentor):
        """无效模式抛异常"""
        seg = sample_segmentor
        with pytest.raises(ValueError):
            seg.set_mode('invalid_mode')
        with pytest.raises(ValueError):
            seg.segment("测试", mode='invalid_mode')

    def test_get_dictionary_size(self, sample_segmentor):
        """获取词典大小"""
        seg = sample_segmentor
        size = seg.get_dictionary_size()
        assert isinstance(size, int)
        assert size > 0

    def test_empty_text(self, sample_segmentor):
        """空文本"""
        seg = sample_segmentor
        result = seg.segment("")
        assert result == []

    def test_extract_keywords(self, sample_segmentor):
        """关键词提取"""
        seg = sample_segmentor
        text = "自然语言处理是人工智能的重要方向，自然语言处理技术发展迅速"
        result = seg.extract_keywords(text, top_k=5, method='freq', use_stopwords=False)
        assert isinstance(result, list)
        assert len(result) <= 5
        assert all(isinstance(item, tuple) and len(item) == 2 for item in result)

    def test_compute_similarity(self, sample_segmentor):
        """相似度计算"""
        seg = sample_segmentor
        text1 = "自然语言处理"
        text2 = "自然语言处理"
        similarity = seg.compute_similarity(text1, text2, method='jaccard', use_stopwords=False)
        assert isinstance(similarity, float)
        assert similarity > 0.0
