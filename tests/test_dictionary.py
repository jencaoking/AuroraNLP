"""Dictionary 和 UserDictionary 测试"""
import os
import pytest
from AuroraNLP.dictionary.dictionary import Dictionary, UserDictionary, DictionaryManager


class TestDictionary:
    """测试 Dictionary"""

    def test_add_and_search(self, empty_dictionary):
        """添加和搜索"""
        d = empty_dictionary
        d.add_word("测试", "n", 1.0)
        assert d.search_in_dict("测试") is True
        assert d.search_in_dict("不存在") is False

    def test_add_with_pos_weight(self, empty_dictionary):
        """带词性和权重添加"""
        d = empty_dictionary
        d.add_word("自然语言处理", "n", 1.5)
        found, pos, weight, priority = d.search_with_info("自然语言处理")
        assert found is True
        assert pos == "n"
        assert weight == 1.5

    def test_remove_word(self, empty_dictionary):
        """删除词"""
        d = empty_dictionary
        d.add_word("测试", "n", 1.0)
        assert d.remove_word("测试") is True
        assert d.search_in_dict("测试") is False
        assert d.remove_word("不存在") is False

    def test_get_words(self, sample_dictionary):
        """获取所有词"""
        words = sample_dictionary.get_words()
        assert "中国" in words
        assert "自然语言处理" in words
        assert isinstance(words, set)

    def test_has_prefix(self, sample_dictionary):
        """前缀检查"""
        assert sample_dictionary.has_prefix("中") is True
        assert sample_dictionary.has_prefix("中国") is True
        assert sample_dictionary.has_prefix("不存在前缀") is False

    def test_get_max_match_length(self, sample_dictionary):
        """最长匹配"""
        assert sample_dictionary.get_max_match_length("自然语言处理技术") == 6
        assert sample_dictionary.get_max_match_length("中国人") == 3

    def test_dictionary_len(self, sample_dictionary):
        """长度"""
        assert len(sample_dictionary) > 0
        assert len(sample_dictionary) == len(sample_dictionary.get_words())

    def test_dictionary_contains(self, sample_dictionary):
        """in 操作符"""
        assert "中国" in sample_dictionary
        assert "不存在" not in sample_dictionary

    def test_save_and_load_dictionary(self, empty_dictionary, tmp_path):
        """保存和加载词典"""
        d = empty_dictionary
        d.add_word("人工智能", "n", 1.5)
        d.add_word("深度学习", "n", 2.0)
        d.add_word("机器学习", "n", 1.8)

        # 保存
        dict_path = str(tmp_path / "test_dict.txt")
        d.save_dictionary(dict_path, include_weight_priority=True)

        # 加载到新词典
        d2 = Dictionary(load_default=False)
        d2.load_dictionary(dict_path)
        assert d2.search_in_dict("人工智能") is True
        assert d2.search_in_dict("深度学习") is True
        found, pos, weight, priority = d2.search_with_info("人工智能")
        assert pos == "n"
        assert weight == 1.5


class TestUserDictionary:
    """测试 UserDictionary"""

    def test_user_dictionary_priority(self, sample_user_dictionary):
        """用户词典优先级"""
        assert sample_user_dictionary.priority == 100

    def test_user_dictionary_default_weight(self, sample_user_dictionary):
        """用户词典默认权重"""
        ud = UserDictionary(name="test", priority=50)
        ud.add_word("测试词")
        found, pos, weight, priority = ud.search_with_info("测试词")
        assert weight == 10.0  # 默认权重
        assert priority == 50


class TestDictionaryManager:
    """测试 DictionaryManager"""

    def test_dictionary_manager_merge(self, sample_dictionary, sample_user_dictionary):
        """DictionaryManager 合并多个词典"""
        dm = DictionaryManager()
        dm.register_dictionary(sample_dictionary)
        dm.register_user_dictionary(sample_user_dictionary)

        # 系统词典中的词可以搜索到
        assert dm.search("中国") is True
        # 用户词典中的词可以搜索到
        assert dm.search("人工智能") is True

    def test_dictionary_manager_priority_order(self, sample_dictionary, sample_user_dictionary):
        """DictionaryManager 按优先级合并"""
        dm = DictionaryManager()
        dm.register_dictionary(sample_dictionary)
        dm.register_user_dictionary(sample_user_dictionary)

        # 用户词典优先级更高，搜索应返回用户词典的属性
        found, pos, weight, priority = dm.search_with_info("人工智能")
        assert found is True
        assert weight == 10.0  # 用户词典默认权重
        assert priority == 100  # 用户词典优先级
