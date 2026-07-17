"""测试词典版本管理模块"""

import pytest
import tempfile
import shutil

from AuroraNLP.dictionary.dictionary_version import (
    DictionaryVersion,
    DictionaryVersionManager,
    VersionedDictionary,
)


class TestVersionedDictionary:

    def test_versioned_dict_init(self):
        """测试初始化"""
        vd = VersionedDictionary(load_default=False)
        assert vd.version_manager is not None

    def test_versioned_dict_add_word(self):
        """测试添加词"""
        vd = VersionedDictionary(load_default=False)
        vd.add_word('版本词', 'n', 5.0)
        assert vd.search_in_dict('版本词') is True

    def test_versioned_dict_commit(self):
        """测试提交版本"""
        with tempfile.TemporaryDirectory() as tmpdir:
            vm = DictionaryVersionManager(storage_dir=tmpdir)
            vd = VersionedDictionary(load_default=False, version_manager=vm)
            vd.add_word('提交词', 'n', 5.0)

            version_id = vd.commit('初始提交', 'tester')
            assert version_id is not None
            assert len(version_id) > 0

    def test_versioned_dict_get_history(self):
        """测试获取版本历史"""
        with tempfile.TemporaryDirectory() as tmpdir:
            vm = DictionaryVersionManager(storage_dir=tmpdir)
            vd = VersionedDictionary(load_default=False, version_manager=vm)
            vd.add_word('历史词', 'n')
            vd.commit('第一次提交', 'tester')

            vd.add_word('历史词二', 'v')
            vd.commit('第二次提交', 'tester')

            history = vd.get_version_history()
            assert len(history) >= 2

    def test_versioned_dict_checkout(self):
        """测试切换版本"""
        with tempfile.TemporaryDirectory() as tmpdir:
            vm = DictionaryVersionManager(storage_dir=tmpdir)
            vd = VersionedDictionary(load_default=False, version_manager=vm)
            vd.add_word('版本一', 'n')
            vid1 = vd.commit('版本一', 'tester')

            vd.add_word('版本二', 'v')
            vd.commit('版本二', 'tester')

            assert vd.search_in_dict('版本二') is True

            vd.checkout(vid1)
            assert vd.search_in_dict('版本一') is True
            assert vd.search_in_dict('版本二') is False

    def test_versioned_dict_rollback(self):
        """测试回滚版本"""
        with tempfile.TemporaryDirectory() as tmpdir:
            vm = DictionaryVersionManager(storage_dir=tmpdir)
            vd = VersionedDictionary(load_default=False, version_manager=vm)
            vd.add_word('回滚词', 'n')
            vd.commit('版本一', 'tester')

            vd.add_word('新词', 'v')
            vd.commit('版本二', 'tester')

            assert vd.search_in_dict('新词') is True

            vd.rollback(1)
            assert vd.search_in_dict('新词') is False


class TestDictionaryVersionManager:

    def test_version_manager_init(self):
        """测试版本管理器初始化"""
        with tempfile.TemporaryDirectory() as tmpdir:
            vm = DictionaryVersionManager(storage_dir=tmpdir)
            assert vm is not None
            assert vm.storage_dir == tmpdir


class TestDictionaryVersion:

    def test_dictionary_version_creation(self):
        """测试 DictionaryVersion 对象创建"""
        import time
        version = DictionaryVersion(
            version_id='abc123',
            message='测试版本',
            author='tester',
            timestamp=time.time()
        )
        assert version.version_id == 'abc123'
        assert version.message == '测试版本'
        assert version.author == 'tester'
        assert version.timestamp_str is not None
