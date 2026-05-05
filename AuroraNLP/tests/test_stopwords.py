"""停用词测试"""
import os
import pytest
from AuroraNLP.stopwords import StopWords


class TestStopWords:
    """测试 StopWords"""

    def test_add_and_check(self, sample_stopwords):
        """添加停用词"""
        sw = sample_stopwords
        sw.add_stopword("也")
        assert sw.is_stopword("也") is True

    def test_remove_stopword(self, sample_stopwords):
        """删除停用词"""
        sw = sample_stopwords
        assert sw.is_stopword("的") is True
        result = sw.remove_stopword("的")
        assert result is True
        assert sw.is_stopword("的") is False
        # 删除不存在的停用词
        result = sw.remove_stopword("不存在")
        assert result is False

    def test_filter(self, sample_stopwords):
        """过滤停用词"""
        sw = sample_stopwords
        words = ["我", "爱", "中国", "的", "人民"]
        filtered = sw.filter(words)
        assert "我" not in filtered
        assert "的" not in filtered
        assert "爱" in filtered
        assert "中国" in filtered
        assert "人民" in filtered

    def test_filter_with_pos(self, sample_stopwords):
        """带词性过滤"""
        sw = sample_stopwords
        words_with_pos = [("我", "r"), ("爱", "v"), ("中国", "ns"), ("的", "u")]
        filtered = sw.filter_with_pos(words_with_pos)
        words = [item[0] for item in filtered]
        assert "我" not in words
        assert "的" not in words
        assert "爱" in words
        assert "中国" in words

    def test_is_stopword(self, sample_stopwords):
        """检查是否停用词"""
        sw = sample_stopwords
        assert sw.is_stopword("的") is True
        assert sw.is_stopword("了") is True
        assert sw.is_stopword("在") is True
        assert sw.is_stopword("是") is True
        assert sw.is_stopword("我") is True
        assert sw.is_stopword("中国") is False

    def test_get_stopwords(self, sample_stopwords):
        """获取停用词集合"""
        sw = sample_stopwords
        stopwords = sw.get_stopwords()
        assert isinstance(stopwords, set)
        assert "的" in stopwords
        assert "了" in stopwords
        assert "在" in stopwords

    def test_len(self, sample_stopwords):
        """长度"""
        sw = sample_stopwords
        assert len(sw) == 5

    def test_contains(self, sample_stopwords):
        """in 操作符"""
        sw = sample_stopwords
        assert "的" in sw
        assert "了" in sw
        assert "中国" not in sw

    def test_save_and_load(self, sample_stopwords, tmp_path):
        """保存和加载"""
        sw = sample_stopwords
        save_path = str(tmp_path / "test_stopwords.txt")
        sw.save_stopwords(save_path)

        # 加载到新对象
        sw2 = StopWords(load_default=False)
        sw2.load_stopwords(save_path)
        assert sw2.is_stopword("的") is True
        assert sw2.is_stopword("了") is True
        assert sw2.is_stopword("在") is True
        assert len(sw2) == len(sw)

    def test_load_stopwords_from_file(self, tmp_stopwords_file):
        """从文件加载"""
        sw = StopWords(load_default=False)
        sw.load_stopwords(tmp_stopwords_file)
        assert sw.is_stopword("的") is True
        assert sw.is_stopword("了") is True
        assert sw.is_stopword("在") is True
        assert sw.is_stopword("是") is True
        assert sw.is_stopword("我") is True
        assert len(sw) == 5
