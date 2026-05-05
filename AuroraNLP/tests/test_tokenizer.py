"""分词算法测试"""
import pytest
from AuroraNLP.tokenizer import (
    forward_max_match,
    backward_max_match,
    bidirectional_max_match,
    forward_max_match_with_pos,
    backward_max_match_with_pos,
    bidirectional_max_match_with_pos,
    forward_max_match_weighted,
    backward_max_match_weighted,
    bidirectional_max_match_weighted,
)


class TestForwardMaxMatch:
    """正向最大匹配测试"""

    def test_forward_max_match(self, sample_dictionary):
        """正向最大匹配"""
        result = forward_max_match("中国人在北京", sample_dictionary)
        assert isinstance(result, list)
        assert "中国人" in result
        assert "北京" in result
        # 拼接结果应等于原文
        assert "".join(result) == "中国人在北京"

    def test_forward_max_match_with_pos(self, sample_dictionary):
        """正向带词性"""
        result = forward_max_match_with_pos("中国人在北京", sample_dictionary)
        assert isinstance(result, list)
        assert all(isinstance(item, tuple) and len(item) == 2 for item in result)
        words = [item[0] for item in result]
        assert "中国人" in words
        assert "北京" in words
        assert "".join(words) == "中国人在北京"

    def test_forward_max_match_weighted(self, sample_dictionary):
        """正向加权匹配"""
        result = forward_max_match_weighted("自然语言处理很好", sample_dictionary)
        assert isinstance(result, list)
        assert "".join(result) == "自然语言处理很好"
        # 加权匹配应优先选择权重更高的词
        assert "自然语言处理" in result


class TestBackwardMaxMatch:
    """逆向最大匹配测试"""

    def test_backward_max_match(self, sample_dictionary):
        """逆向最大匹配"""
        result = backward_max_match("中国人在北京", sample_dictionary)
        assert isinstance(result, list)
        assert "".join(result) == "中国人在北京"

    def test_backward_max_match_with_pos(self, sample_dictionary):
        """逆向带词性"""
        result = backward_max_match_with_pos("中国人在北京", sample_dictionary)
        assert isinstance(result, list)
        assert all(isinstance(item, tuple) and len(item) == 2 for item in result)
        words = [item[0] for item in result]
        assert "".join(words) == "中国人在北京"

    def test_backward_max_match_weighted(self, sample_dictionary):
        """逆向加权匹配"""
        result = backward_max_match_weighted("自然语言处理很好", sample_dictionary)
        assert isinstance(result, list)
        assert "".join(result) == "自然语言处理很好"


class TestBidirectionalMaxMatch:
    """双向最大匹配测试"""

    def test_bidirectional_max_match(self, sample_dictionary):
        """双向最大匹配"""
        result = bidirectional_max_match("中国人在北京", sample_dictionary)
        assert isinstance(result, list)
        assert "".join(result) == "中国人在北京"

    def test_bidirectional_max_match_with_pos(self, sample_dictionary):
        """双向带词性"""
        result = bidirectional_max_match_with_pos("中国人在北京", sample_dictionary)
        assert isinstance(result, list)
        assert all(isinstance(item, tuple) and len(item) == 2 for item in result)
        words = [item[0] for item in result]
        assert "".join(words) == "中国人在北京"

    def test_bidirectional_max_match_weighted(self, sample_dictionary):
        """双向加权匹配"""
        result = bidirectional_max_match_weighted("自然语言处理很好", sample_dictionary)
        assert isinstance(result, list)
        assert "".join(result) == "自然语言处理很好"


class TestEdgeCases:
    """边界情况测试"""

    def test_empty_text(self, sample_dictionary):
        """空文本处理"""
        assert forward_max_match("", sample_dictionary) == []
        assert backward_max_match("", sample_dictionary) == []
        assert bidirectional_max_match("", sample_dictionary) == []

    def test_unknown_chars(self, sample_dictionary):
        """未登录字处理"""
        text = "ABC测试XYZ"
        result = forward_max_match(text, sample_dictionary)
        assert isinstance(result, list)
        assert "".join(result) == text
        # 未登录字应逐字输出
        assert "A" in result
        assert "B" in result
        assert "C" in result
