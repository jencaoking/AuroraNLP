"""测试术语词典模块"""

import pytest

from AuroraNLP.terminology import (
    TerminologyDatabase,
    TerminologyManager,
    Term,
    TermDomain,
)


class TestTerminologyDatabase:

    def test_terminology_db_init(self):
        """测试初始化"""
        db = TerminologyDatabase(load_default=False, load_sogou=False)
        assert db.is_loaded() is False
        assert db.is_sogou_loaded() is False

    def test_terminology_add_term(self):
        """测试添加术语"""
        db = TerminologyDatabase(load_default=False, load_sogou=False)
        term = Term(
            term_id='test_001',
            name='人工智能',
            domain=TermDomain.IT,
            english='Artificial Intelligence',
            aliases=['AI'],
            definition='模拟人类智能的技术',
        )
        db.add_term(term)
        assert db.is_term('人工智能') is True
        assert db.get_by_id('test_001') is not None

    def test_terminology_search(self):
        """测试搜索术语"""
        db = TerminologyDatabase(load_default=False, load_sogou=False)
        term = Term(
            term_id='test_002',
            name='区块链',
            domain=TermDomain.IT,
            english='Blockchain',
        )
        db.add_term(term)

        results = db.search('区块链')
        assert len(results) > 0
        assert results[0].name == '区块链'


class TestTerminologyManager:

    def test_terminology_manager_init(self):
        """测试管理器初始化"""
        manager = TerminologyManager(load_default=False, load_sogou=False)
        assert manager.get_database() is None
        assert manager.is_loaded() is False


class TestTermDomain:

    def test_term_domain_enum(self):
        """测试领域枚举"""
        assert TermDomain.MEDICAL.value == 'medical'
        assert TermDomain.MEDICAL.get_name() == '医学'
        assert TermDomain.LEGAL.value == 'legal'
        assert TermDomain.LEGAL.get_name() == '法律'
        assert TermDomain.IT.value == 'it'
        assert TermDomain.IT.get_name() == 'IT'


class TestTerm:

    def test_term_creation(self):
        """测试 Term 对象创建"""
        term = Term(
            term_id='test_003',
            name='深度学习',
            domain=TermDomain.IT,
            sub_domain='机器学习',
            english='Deep Learning',
            aliases=['DL'],
            definition='一种机器学习方法',
            source='test',
        )
        assert term.term_id == 'test_003'
        assert term.name == '深度学习'
        assert term.domain == TermDomain.IT
        assert term.sub_domain == '机器学习'
        assert term.english == 'Deep Learning'
        assert term.aliases == ['DL']
        assert term.definition == '一种机器学习方法'
        assert term.source == 'test'

        # 测试 matches 方法
        assert term.matches('深度学习') is True
        assert term.matches('DL') is True
        assert term.matches('Deep Learning') is True
        assert term.matches('不存在的') is False

        # 测试 get_all_names 方法
        all_names = term.get_all_names()
        assert '深度学习' in all_names
        assert 'DL' in all_names
        assert 'Deep Learning' in all_names
