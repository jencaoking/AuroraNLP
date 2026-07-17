"""Trie 树测试"""
import pytest
from AuroraNLP.dictionary.trie import Trie


class TestTrieInsertAndSearch:
    """测试插入和搜索"""

    def test_insert_and_search(self, empty_trie):
        """插入词并搜索"""
        trie = empty_trie
        trie.insert("中国", "ns", 1.0, 0)
        assert trie.search("中国") is True

    def test_search_not_found(self, empty_trie):
        """搜索不存在的词"""
        trie = empty_trie
        trie.insert("中国", "ns", 1.0, 0)
        assert trie.search("美国") is False

    def test_search_with_pos(self, empty_trie):
        """搜索带词性"""
        trie = empty_trie
        trie.insert("中国", "ns", 1.0, 0)
        found, pos = trie.search_with_pos("中国")
        assert found is True
        assert pos == "ns"

    def test_search_with_info(self, empty_trie):
        """搜索带完整信息(pos, weight, priority)"""
        trie = empty_trie
        trie.insert("自然语言处理", "n", 1.5, 10)
        found, pos, weight, priority = trie.search_with_info("自然语言处理")
        assert found is True
        assert pos == "n"
        assert weight == 1.5
        assert priority == 10

    def test_starts_with(self, sample_trie):
        """前缀匹配"""
        trie = sample_trie
        assert trie.starts_with("中") is True
        assert trie.starts_with("中国") is True
        assert trie.starts_with("中国人") is True
        assert trie.starts_with("中国大") is False

    def test_get_max_match_length(self, sample_trie):
        """最长匹配长度"""
        trie = sample_trie
        # "自然语言处理" 在词典中，长度为6
        assert trie.get_max_match_length("自然语言处理技术") == 6
        # "中国" 长度为2，"中国人" 长度为3
        assert trie.get_max_match_length("中国人") == 3
        # 不存在的字
        assert trie.get_max_match_length("美国") == 0

    def test_get_max_match_with_pos(self, sample_trie):
        """最长匹配带词性"""
        trie = sample_trie
        match_len, pos = trie.get_max_match_with_pos("自然语言处理技术", 0)
        assert match_len == 6
        assert pos == "n"

    def test_get_all_matches_with_info(self, sample_trie):
        """获取所有匹配"""
        trie = sample_trie
        matches = trie.get_all_matches_with_info("中国人", 0)
        # "中" 不在词典中，"中国" 在，"中国人" 在
        lengths = [m[0] for m in matches]
        assert 2 in lengths  # 中国
        assert 3 in lengths  # 中国人

    def test_remove(self, sample_trie):
        """删除词"""
        trie = sample_trie
        assert trie.search("中国") is True
        result = trie.remove("中国")
        assert result is True
        assert trie.search("中国") is False
        # 删除不存在的词
        result = trie.remove("不存在")
        assert result is False

    def test_len(self, sample_trie):
        """长度统计"""
        trie = sample_trie
        assert len(trie) > 0

    def test_contains(self, sample_trie):
        """in 操作符"""
        trie = sample_trie
        assert "中国" in trie
        assert "不存在" not in trie

    def test_insert_overwrite(self, empty_trie):
        """重复插入更新属性"""
        trie = empty_trie
        trie.insert("中国", "ns", 1.0, 0)
        trie.insert("中国", "nz", 2.0, 5)
        found, pos, weight, priority = trie.search_with_info("中国")
        assert found is True
        assert pos == "nz"
        assert weight == 2.0
        assert priority == 5

    def test_empty_trie_len(self, empty_trie):
        """空 trie 长度为0"""
        trie = empty_trie
        assert len(trie) == 0
