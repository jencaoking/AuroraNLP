"""测试用户词典模块"""

import pytest

from AuroraNLP.dictionary.dictionary import UserDictionary


class TestUserDictionary:

    def test_user_dict_init(self):
        """测试初始化"""
        ud = UserDictionary()
        assert ud.name == 'user'
        assert ud.priority == 100

    def test_user_dict_add_word(self):
        """测试添加词"""
        ud = UserDictionary()
        ud.add_word('自然语言处理', 'n')
        assert ud.search_in_dict('自然语言处理') is True

    def test_user_dict_remove_word(self):
        """测试删除词"""
        ud = UserDictionary()
        ud.add_word('测试词', 'n')
        assert ud.search_in_dict('测试词') is True

        result = ud.remove_word('测试词')
        assert result is True
        assert ud.search_in_dict('测试词') is False

        # 删除不存在的词
        result2 = ud.remove_word('不存在的词')
        assert result2 is False

    def test_user_dict_search(self):
        """测试搜索"""
        ud = UserDictionary()
        ud.add_word('机器学习', 'n', 10.0)
        found, pos_tag = ud.search_with_pos('机器学习')
        assert found is True
        assert pos_tag == 'n'

        found2, _ = ud.search_with_pos('不存在的词')
        assert found2 is False

    def test_user_dict_priority(self):
        """测试优先级属性"""
        ud = UserDictionary(priority=200)
        assert ud.priority == 200

        ud.priority = 300
        assert ud.priority == 300

    def test_user_dict_default_weight(self):
        """测试默认权重"""
        ud = UserDictionary()
        assert ud.default_weight == 10.0

        ud.default_weight = 20.0
        assert ud.default_weight == 20.0

    def test_user_dict_load_from_file(self, tmp_path):
        """测试从文件加载"""
        dict_file = tmp_path / 'user_dict.txt'
        dict_file.write_text('自然语言处理\tn\t10.0\t100\n深度学习\tn\t8.0\t100\n', encoding='utf-8')

        ud = UserDictionary()
        ud.load_dictionary(str(dict_file))
        assert ud.search_in_dict('自然语言处理') is True
        assert ud.search_in_dict('深度学习') is True

    def test_user_dict_save_to_file(self, tmp_path):
        """测试保存到文件"""
        ud = UserDictionary()
        ud.add_word('人工智能', 'n', 10.0)

        save_file = tmp_path / 'saved_dict.txt'
        ud.save_dictionary(str(save_file))

        content = save_file.read_text(encoding='utf-8')
        assert '人工智能' in content

    def test_user_dict_get_words(self):
        """测试获取所有词"""
        ud = UserDictionary()
        ud.add_word('词一', 'n')
        ud.add_word('词二', 'v')
        ud.add_word('词三', 'a')

        words = ud.get_words()
        assert '词一' in words
        assert '词二' in words
        assert '词三' in words
        assert len(words) == 3

    def test_user_dict_len(self):
        """测试长度"""
        ud = UserDictionary()
        assert len(ud) == 0

        ud.add_word('词一', 'n')
        assert len(ud) == 1

        ud.add_word('词二', 'v')
        assert len(ud) == 2

        ud.remove_word('词一')
        assert len(ud) == 1
