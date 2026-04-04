import pytest
import tempfile
import os
from nlp_segment.segmentor import Segmentor, POS_TAG_NAMES, NER_TAG_MAP
from nlp_segment.dictionary import Dictionary
from nlp_segment.trie import Trie
from nlp_segment.tokenizer import (
    forward_max_match,
    backward_max_match,
    bidirectional_max_match,
    forward_max_match_with_pos,
    backward_max_match_with_pos,
    bidirectional_max_match_with_pos
)


class TestTrie:
    def test_insert_and_search(self):
        trie = Trie()
        trie.insert("今天")
        assert trie.search("今天") == True
        assert trie.search("明") == False
        assert trie.search("今天天气") == False

    def test_starts_with(self):
        trie = Trie()
        trie.insert("今天")
        trie.insert("明天")
        assert trie.starts_with("今") == True
        assert trie.starts_with("明天") == True
        assert trie.starts_with("昨天") == False

    def test_get_max_match_length(self):
        trie = Trie()
        trie.insert("今天")
        trie.insert("今天天气")
        trie.insert("天气")

        text = "今天天气很好"
        assert trie.get_max_match_length(text, 0) == 4
        assert trie.get_max_match_length(text, 2) == 2

    def test_remove(self):
        trie = Trie()
        trie.insert("测试")
        assert trie.search("测试") == True
        result = trie.remove("测试")
        assert result == True
        assert trie.search("测试") == False

    def test_remove_nonexistent(self):
        trie = Trie()
        result = trie.remove("不存在")
        assert result == False

    def test_len(self):
        trie = Trie()
        trie.insert("词1")
        trie.insert("词2")
        assert len(trie) == 2

    def test_contains(self):
        trie = Trie()
        trie.insert("测试")
        assert "测试" in trie
        assert "不存在" not in trie

    def test_duplicate_insert(self):
        trie = Trie()
        trie.insert("测试")
        trie.insert("测试")
        assert len(trie) == 1

    def test_insert_with_pos(self):
        trie = Trie()
        trie.insert("今天", "t")
        found, pos = trie.search_with_pos("今天")
        assert found == True
        assert pos == "t"

    def test_search_with_pos_not_found(self):
        trie = Trie()
        found, pos = trie.search_with_pos("不存在")
        assert found == False
        assert pos is None

    def test_get_max_match_with_pos(self):
        trie = Trie()
        trie.insert("今天", "t")
        trie.insert("今天天气", "n")
        length, pos = trie.get_max_match_with_pos("今天天气很好", 0)
        assert length == 4
        assert pos == "n"


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

    def test_has_prefix(self):
        d = Dictionary(load_default=False)
        d.add_word("今天")
        d.add_word("明天")
        assert d.has_prefix("今") == True
        assert d.has_prefix("明天") == True
        assert d.has_prefix("昨天") == False

    def test_get_max_match_length(self):
        d = Dictionary(load_default=False)
        d.add_word("今天")
        d.add_word("今天天气")
        d.add_word("天气")

        text = "今天天气很好"
        assert d.get_max_match_length(text, 0) == 4
        assert d.get_max_match_length(text, 2) == 2

    def test_add_word_with_pos(self):
        d = Dictionary(load_default=False)
        d.add_word("今天", "t")
        pos = d.get_pos_tag("今天")
        assert pos == "t"

    def test_search_with_pos(self):
        d = Dictionary(load_default=False)
        d.add_word("天气", "n")
        found, pos = d.search_with_pos("天气")
        assert found == True
        assert pos == "n"

    def test_save_and_load_dictionary(self):
        d = Dictionary(load_default=False)
        d.add_word("测试词1", "n")
        d.add_word("测试词2", "v")

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            temp_path = f.name

        try:
            d.save_dictionary(temp_path)

            d2 = Dictionary(load_default=False)
            d2.load_dictionary(temp_path)

            assert "测试词1" in d2
            assert "测试词2" in d2
            assert d2.get_pos_tag("测试词1") == "n"
            assert d2.get_pos_tag("测试词2") == "v"
        finally:
            os.unlink(temp_path)


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

    def test_long_word_priority(self):
        d = Dictionary(load_default=False)
        d.add_word("今天")
        d.add_word("今天天气")
        result = forward_max_match("今天天气很好", d)
        assert result[0] == "今天天气"


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


class TestPosTagging:
    def test_forward_max_match_with_pos(self):
        d = Dictionary(load_default=False)
        d.add_word("今天", "t")
        d.add_word("天气", "n")
        d.add_word("很好", "a")
        result = forward_max_match_with_pos("今天天气很好", d)
        assert len(result) == 3
        assert result[0] == ("今天", "t")
        assert result[1] == ("天气", "n")
        assert result[2] == ("很好", "a")

    def test_backward_max_match_with_pos(self):
        d = Dictionary(load_default=False)
        d.add_word("今天", "t")
        d.add_word("天气", "n")
        result = backward_max_match_with_pos("今天天气", d)
        assert ("今天", "t") in result
        assert ("天气", "n") in result

    def test_bidirectional_max_match_with_pos(self):
        d = Dictionary(load_default=False)
        d.add_word("今天", "t")
        d.add_word("天气", "n")
        result = bidirectional_max_match_with_pos("今天天气", d)
        assert isinstance(result, list)
        assert all(isinstance(item, tuple) and len(item) == 2 for item in result)


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
        d.write_text("今天 t\n天气 n\n很好 a\n", encoding='utf-8')
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

    def test_segment_with_pos(self):
        seg = Segmentor()
        result = seg.segment_with_pos("今天天气很好")
        assert isinstance(result, list)
        assert all(isinstance(item, tuple) and len(item) == 2 for item in result)

    def test_segment_with_pos_has_correct_tags(self):
        seg = Segmentor()
        result = seg.segment_with_pos("今天天气很好")
        words = [w for w, _ in result]
        assert "今天" in words
        assert "天气" in words

    def test_recognize_entities(self):
        seg = Segmentor()
        entities = seg.recognize_entities("我在北京工作")
        entity_words = [w for w, _ in entities]
        assert "北京" in entity_words

    def test_segment_with_entities(self):
        seg = Segmentor()
        result = seg.segment_with_entities("我在北京工作")
        assert isinstance(result, list)
        assert all(isinstance(item, tuple) and len(item) == 3 for item in result)

    def test_save_dictionary(self):
        seg = Segmentor(load_default_dict=False)
        seg.add_word("测试词", "n")

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            temp_path = f.name

        try:
            seg.save_dictionary(temp_path)

            seg2 = Segmentor(load_default_dict=False)
            seg2.load_dictionary(temp_path)
            assert "测试词" in seg2.dictionary
        finally:
            os.unlink(temp_path)

    def test_add_word_with_pos(self):
        seg = Segmentor(load_default_dict=False)
        seg.add_word("新词", "n")
        pos = seg.dictionary.get_pos_tag("新词")
        assert pos == "n"

    def test_get_pos_tag_name(self):
        seg = Segmentor()
        assert seg.get_pos_tag_name("n") == "名词"
        assert seg.get_pos_tag_name("v") == "动词"
        assert seg.get_pos_tag_name("unknown") == "未知"

    def test_get_pos_tags(self):
        seg = Segmentor()
        tags = seg.get_pos_tags()
        assert isinstance(tags, dict)
        assert "n" in tags
        assert "v" in tags


class TestNER:
    def test_recognize_location(self):
        seg = Segmentor()
        entities = seg.recognize_entities("我在北京和上海工作")
        entity_words = [w for w, t in entities]
        assert "北京" in entity_words
        assert "上海" in entity_words

    def test_recognize_organization(self):
        seg = Segmentor()
        entities = seg.recognize_entities("他在清华大学学习")
        entity_words = [w for w, t in entities]
        assert "清华大学" in entity_words

    def test_entity_types(self):
        seg = Segmentor()
        entities = seg.recognize_entities("我在北京工作")
        for word, entity_type in entities:
            assert entity_type in ["PERSON", "LOCATION", "ORGANIZATION"]
