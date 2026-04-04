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
        d = Dictionary(load_default=False)
        d.add_word("测试")
        assert d.search_in_dict("测试") == True
        assert d.search_in_dict("不存在的词") == False

    def test_get_words(self):
        d = Dictionary(load_default=False)
        d.add_word("词1")
        d.add_word("词2")
        words = d.get_words()
        assert "词1" in words
        assert "词2" in words

    def test_remove_word(self):
        d = Dictionary(load_default=False)
        d.add_word("测试词")
        assert d.search_in_dict("测试词") == True
        result = d.remove_word("测试词")
        assert result == True
        assert d.search_in_dict("测试词") == False

    def test_remove_nonexistent_word(self):
        d = Dictionary(load_default=False)
        result = d.remove_word("不存在的词")
        assert result == False

    def test_len(self):
        d = Dictionary(load_default=False)
        d.add_word("词1")
        d.add_word("词2")
        assert len(d) == 2

    def test_contains(self):
        d = Dictionary(load_default=False)
        d.add_word("测试")
        assert "测试" in d
        assert "不存在" not in d

    def test_load_default_dictionary(self):
        d = Dictionary(load_default=True)
        assert len(d) > 0
        assert "今天" in d

    def test_load_dictionary_file_not_found(self):
        d = Dictionary(load_default=False)
        with pytest.raises(FileNotFoundError):
            d.load_dictionary("nonexistent_dict.txt")


class TestForwardMaxMatch:
    def test_simple_segmentation(self):
        d = Dictionary(load_default=False)
        d.add_word("今天")
        d.add_word("天气")
        result = forward_max_match("今天天气很好", d)
        assert "今天" in result
        assert "天气" in result

    def test_single_char(self):
        d = Dictionary(load_default=False)
        result = forward_max_match("你好世界", d)
        assert result == ["你", "好", "世", "界"]


class TestBackwardMaxMatch:
    def test_simple_segmentation(self):
        d = Dictionary(load_default=False)
        d.add_word("今天")
        d.add_word("天气")
        result = backward_max_match("今天天气很好", d)
        assert "今天" in result
        assert "天气" in result


class TestBidirectionalMaxMatch:
    def test_returns_list(self):
        d = Dictionary(load_default=False)
        d.add_word("研究")
        d.add_word("研究生命")
        result = bidirectional_max_match("研究生命起源", d)
        assert isinstance(result, list)


class TestSegmentor:
    def test_segment_default_mode(self):
        d = Dictionary(load_default=False)
        d.add_word("今天")
        d.add_word("天气")
        seg = Segmentor(d)
        result = seg.segment("今天天气很好")
        assert isinstance(result, list)

    def test_segment_forward_mode(self):
        d = Dictionary(load_default=False)
        d.add_word("今天")
        seg = Segmentor(d)
        result = seg.segment("今天天气很好", mode='forward')
        assert "今天" in result

    def test_segment_backward_mode(self):
        d = Dictionary(load_default=False)
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
        seg = Segmentor(load_default_dict=False)
        seg.load_dictionary(str(d))
        assert seg.dictionary.search_in_dict("今天") == True

    def test_default_dictionary_loaded(self):
        seg = Segmentor()
        assert seg.get_dictionary_size() > 0
        assert "今天" in seg.dictionary

    def test_add_word(self):
        seg = Segmentor(load_default_dict=False)
        seg.add_word("新词")
        assert "新词" in seg.dictionary

    def test_remove_word(self):
        seg = Segmentor(load_default_dict=False)
        seg.add_word("测试词")
        result = seg.remove_word("测试词")
        assert result == True
        assert "测试词" not in seg.dictionary

    def test_segment_with_default_dict(self):
        seg = Segmentor()
        result = seg.segment("今天天气很好")
        assert isinstance(result, list)
        assert "今天" in result
        assert "天气" in result

    def test_invalid_mode(self):
        seg = Segmentor()
        with pytest.raises(ValueError):
            seg.segment("测试", mode='invalid')

    def test_set_invalid_mode(self):
        seg = Segmentor()
        with pytest.raises(ValueError):
            seg.set_mode('invalid')
