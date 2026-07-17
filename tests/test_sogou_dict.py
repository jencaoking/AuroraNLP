"""测试搜狗词库模块"""

import pytest

from AuroraNLP.dictionary.sogou_dict import ScelConverter, SogouDictionary, SogouDictionaryManager


class TestSogouDictionary:

    def test_sogou_dict_init(self):
        """测试初始化"""
        sd = SogouDictionary(name='test_sogou', priority=80)
        assert sd.name == 'test_sogou'
        assert sd.priority == 80


class TestSogouDictionaryManager:

    def test_sogou_dict_manager_init(self):
        """测试管理器初始化"""
        manager = SogouDictionaryManager()
        assert manager is not None

    def test_sogou_dict_manager_get_available(self):
        """测试获取可用词库列表"""
        manager = SogouDictionaryManager()
        # 初始时没有创建词典，列表应为空
        info = manager.list_dictionaries()
        assert info == []

        # 创建一个词典后列表应包含该词典
        manager.create_dictionary('test', priority=80)
        info = manager.list_dictionaries()
        assert len(info) == 1
        assert info[0]['name'] == 'test'
