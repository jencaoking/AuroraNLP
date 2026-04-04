import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from nlp_segment.segmentor import Segmentor
from nlp_segment.dictionary import Dictionary
from nlp_segment.tokenizer import (
    forward_max_match,
    backward_max_match,
    bidirectional_max_match
)

class TestDictionary:
    def test_add_word(self):
        d = Dictionary()
        d.add_word("测试")
        assert d.search_in_dict("测试") == True
        assert d.search_in_dict("不存在的词") == False

    def test_get_words(self):
        d = Dictionary()
        d.add_word("词1")
        d.add_word("词2")
        words = d.get_words()
        assert "词1" in words
        assert "词2" in words

class TestForwardMaxMatch:
    def test_simple_segmentation(self):
        d = Dictionary()
        d.add_word("今天")
        d.add_word("天气")
        result = forward_max_match("今天天气很好", d)
        assert "今天" in result
        assert "天气" in result

    def test_single_char(self):
        d = Dictionary()
        result = forward_max_match("你好世界", d)
        assert result == ["你", "好", "世", "界"]

class TestBackwardMaxMatch:
    def test_simple_segmentation(self):
        d = Dictionary()
        d.add_word("今天")
        d.add_word("天气")
        result = backward_max_match("今天天气很好", d)
        assert "今天" in result
        assert "天气" in result

class TestBidirectionalMaxMatch:
    def test_returns_list(self):
        d = Dictionary()
        d.add_word("研究")
        d.add_word("研究生命")
        result = bidirectional_max_match("研究生命起源", d)
        assert isinstance(result, list)

class TestSegmentor:
    def test_segment_default_mode(self):
        d = Dictionary()
        d.add_word("今天")
        d.add_word("天气")
        seg = Segmentor(d)
        result = seg.segment("今天天气很好")
        assert isinstance(result, list)

    def test_segment_forward_mode(self):
        d = Dictionary()
        d.add_word("今天")
        seg = Segmentor(d)
        result = seg.segment("今天天气很好", mode='forward')
        assert "今天" in result

    def test_segment_backward_mode(self):
        d = Dictionary()
        d.add_word("今天")
        seg = Segmentor(d)
        result = seg.segment("今天天气很好", mode='backward')
        assert "今天" in result

    def test_set_mode(self):
        seg = Segmentor()
        seg.set_mode('forward')
        assert seg.mode == 'forward'

    def test_load_dictionary(self, tmp_path):
        d = tmp_path / "dict.txt"
        d.write_text("今天\n天气\n很好\n", encoding='utf-8')
        seg = Segmentor()
        seg.load_dictionary(str(d))
        assert seg.dictionary.search_in_dict("今天") == True