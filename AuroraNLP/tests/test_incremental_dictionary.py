"""测试增量词典模块"""

import pytest

from AuroraNLP.incremental_dictionary import (
    IncrementalDictionary,
    IncrementalUserDictionary,
    DictionaryUpdateEvent,
    DictionaryObserver,
    DictionaryUpdateManager,
)
from AuroraNLP.dictionary import DictionaryManager


class TestIncrementalDictionary:

    def test_incremental_dict_init(self):
        """测试初始化"""
        id_dict = IncrementalDictionary(load_default=False)
        assert id_dict.name == 'incremental'

    def test_incremental_dict_add_word(self):
        """测试添加词"""
        id_dict = IncrementalDictionary(load_default=False)
        id_dict.add_word('增量词', 'n', 5.0)
        assert id_dict.search_in_dict('增量词') is True

    def test_incremental_dict_remove_word(self):
        """测试删除词"""
        id_dict = IncrementalDictionary(load_default=False)
        id_dict.add_word('待删除词', 'n')
        assert id_dict.search_in_dict('待删除词') is True

        result = id_dict.remove_word('待删除词')
        assert result is True
        assert id_dict.search_in_dict('待删除词') is False


class TestDictionaryUpdateEvent:

    def test_update_event_creation(self):
        """测试创建更新事件"""
        event = DictionaryUpdateEvent('test_dict', 'add', ['词一', '词二'])
        assert event.dictionary_name == 'test_dict'
        assert event.update_type == 'add'
        assert event.words == ['词一', '词二']
        assert event.timestamp is not None
        assert event.timestamp_str is not None


class TestDictionaryObserver:

    def test_observer_interface(self):
        """测试观察者接口"""
        class MockObserver(DictionaryObserver):
            def __init__(self):
                self.received_events = []

            def on_dictionary_update(self, event):
                self.received_events.append(event)

        observer = MockObserver()
        id_dict = IncrementalDictionary(load_default=False)
        id_dict.add_observer(observer)

        id_dict.add_word('观察词', 'n')

        assert len(observer.received_events) == 1
        assert observer.received_events[0].update_type == 'add'
        assert observer.received_events[0].words == ['观察词']


class TestDictionaryUpdateManager:

    def test_update_manager_register(self):
        """测试注册词典"""
        dm = DictionaryManager()
        um = DictionaryUpdateManager(dm)
        assert um.dictionary_manager is dm


class TestIncrementalUserDictionary:

    def test_incremental_user_dict_init(self):
        """测试增量用户词典初始化"""
        iud = IncrementalUserDictionary(name='test_user', priority=150)
        assert iud.name == 'test_user'
        assert iud.priority == 150
