"""测试领域词典模块"""

import pytest

from AuroraNLP.domain_dictionary import DomainDictionary, DomainDictionaryManager


class TestDomainDictionary:

    def test_domain_dict_init(self):
        """测试领域词典初始化"""
        dd = DomainDictionary(domain='news', load_default=False)
        assert dd.domain == 'news'
        assert dd.domain_name == '新闻领域'
        assert dd.name == 'domain_news'

    def test_domain_dict_load(self):
        """测试加载领域词典"""
        dd = DomainDictionary(domain='news', load_default=True)
        assert dd.domain == 'news'
        assert len(dd) > 0

    def test_domain_dict_add_word(self):
        """测试添加词"""
        dd = DomainDictionary(domain='news', load_default=False)
        dd.add_word('人工智能', 'n', 5.0)
        assert dd.search_in_dict('人工智能') is True

    def test_domain_dict_search(self):
        """测试搜索词"""
        dd = DomainDictionary(domain='news', load_default=False)
        dd.add_word('深度学习', 'n', 5.0)
        assert dd.search_in_dict('深度学习') is True
        assert dd.search_in_dict('不存在的词') is False

    def test_domain_dict_priority(self):
        """测试优先级"""
        dd = DomainDictionary(domain='news', load_default=False, priority=80)
        assert dd.priority == 80

        dd2 = DomainDictionary(domain='news', load_default=False)
        assert dd2.priority == 50  # 默认优先级


class TestDomainDictionaryManager:

    def test_domain_manager_init(self):
        """测试管理器初始化"""
        manager = DomainDictionaryManager()
        assert len(manager) == 0

    def test_domain_manager_load_all(self):
        """测试加载所有领域"""
        manager = DomainDictionaryManager()
        manager.load_all_domains()
        assert len(manager) == 4
        assert manager.get_domain_dictionary('news') is not None
        assert manager.get_domain_dictionary('medical') is not None
        assert manager.get_domain_dictionary('legal') is not None
        assert manager.get_domain_dictionary('ecommerce') is not None

    def test_domain_manager_get_domain(self):
        """测试获取特定领域词典"""
        manager = DomainDictionaryManager()
        manager.load_all_domains()
        dd = manager.get_domain_dictionary('news')
        assert dd is not None
        assert dd.domain == 'news'

        # 获取不存在的领域
        dd_none = manager.get_domain_dictionary('nonexistent')
        assert dd_none is None
