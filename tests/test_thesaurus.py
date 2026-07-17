"""测试同义词词典模块"""

import pytest

from AuroraNLP.dictionary.thesaurus import (
    Thesaurus,
    ThesaurusManager,
    ThesaurusEntry,
    SemanticCategory,
    WordRelation,
)


class TestThesaurus:

    def test_thesaurus_init(self):
        """测试初始化"""
        t = Thesaurus(load_default=False)
        assert t.is_loaded() is False

        t2 = Thesaurus(load_default=True)
        # 默认数据文件存在时应加载成功
        assert t2.is_loaded() is True

    def test_thesaurus_add_entry(self):
        """测试添加词条"""
        t = Thesaurus(load_default=False)
        t.add_entry('Aa01A01', ['人', '人们', '人员'], WordRelation.SYNONYM)
        assert t.has_word('人') is True
        assert t.has_word('人们') is True

    def test_thesaurus_search(self):
        """测试搜索"""
        t = Thesaurus(load_default=False)
        t.add_entry('Aa01A01', ['人', '人们', '人员'], WordRelation.SYNONYM)
        assert t.has_word('人') is True
        assert t.has_word('不存在') is False

    def test_thesaurus_get_synonyms(self):
        """测试获取同义词"""
        t = Thesaurus(load_default=False)
        t.add_entry('Aa01A01', ['人', '人们', '人员'], WordRelation.SYNONYM)
        synonyms = t.get_synonyms('人')
        assert '人们' in synonyms
        assert '人员' in synonyms
        # 不应包含自身
        assert '人' not in synonyms


class TestThesaurusManager:

    def test_thesaurus_manager_init(self):
        """测试管理器初始化"""
        manager = ThesaurusManager()
        assert manager.get_thesaurus() is None
        assert manager.is_loaded() is False


class TestSemanticCategory:

    def test_semantic_category(self):
        """测试语义分类"""
        cat = SemanticCategory(code='A', name='人物', level=1)
        assert cat.code == 'A'
        assert cat.name == '人物'
        assert cat.level == 1
        assert cat.parent_code is None


class TestWordRelation:

    def test_word_relation(self):
        """测试词关系枚举"""
        assert WordRelation.SYNONYM.value == '='
        assert WordRelation.RELATED.value == '#'
        assert WordRelation.INDEPENDENT.value == '@'
