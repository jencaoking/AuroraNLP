import unittest
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from AuroraNLP import (
    Segmentor, Dictionary, UserDictionary, DictionaryManager, Trie
)


class TestTrieWeightPriority(unittest.TestCase):
    def test_insert_with_weight_and_priority(self):
        trie = Trie()
        trie.insert("测试", "n", weight=5.0, priority=10)
        
        found, pos_tag, weight, priority = trie.search_with_info("测试")
        self.assertTrue(found)
        self.assertEqual(pos_tag, "n")
        self.assertEqual(weight, 5.0)
        self.assertEqual(priority, 10)

    def test_default_weight_and_priority(self):
        trie = Trie()
        trie.insert("默认词")
        
        found, pos_tag, weight, priority = trie.search_with_info("默认词")
        self.assertTrue(found)
        self.assertEqual(weight, 1.0)
        self.assertEqual(priority, 0)

    def test_set_weight(self):
        trie = Trie()
        trie.insert("权重测试")
        
        result = trie.set_weight("权重测试", 10.0)
        self.assertTrue(result)
        self.assertEqual(trie.get_weight("权重测试"), 10.0)

    def test_set_priority(self):
        trie = Trie()
        trie.insert("优先级测试")
        
        result = trie.set_priority("优先级测试", 100)
        self.assertTrue(result)
        self.assertEqual(trie.get_priority("优先级测试"), 100)

    def test_get_all_matches_with_info(self):
        trie = Trie()
        trie.insert("中", "f", weight=1.0, priority=0)
        trie.insert("中国", "ns", weight=5.0, priority=10)
        trie.insert("中国人", "n", weight=8.0, priority=20)
        
        matches = trie.get_all_matches_with_info("中国人", 0, 10)
        self.assertEqual(len(matches), 3)
        
        word_lens = [m[0] for m in matches]
        self.assertIn(1, word_lens)
        self.assertIn(2, word_lens)
        self.assertIn(3, word_lens)


class TestDictionaryWeightPriority(unittest.TestCase):
    def test_dictionary_priority(self):
        dict1 = Dictionary(load_default=False, priority=10)
        dict1.name = "dict1"
        self.assertEqual(dict1.priority, 10)
        
        dict1.priority = 20
        self.assertEqual(dict1.priority, 20)

    def test_add_word_with_weight_priority(self):
        dictionary = Dictionary(load_default=False)
        dictionary.add_word("测试词", "n", weight=5.0, priority=100)
        
        found, pos_tag, weight, priority = dictionary.search_with_info("测试词")
        self.assertTrue(found)
        self.assertEqual(weight, 5.0)
        self.assertEqual(priority, 100)

    def test_load_dictionary_with_weight_priority(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write("词语1 n 5.0 10\n")
            f.write("词语2 v 3.0 5\n")
            f.write("词语3 a\n")
            f.flush()
            
            dictionary = Dictionary(load_default=False)
            dictionary.load_dictionary(f.name)
            
            found1, _, weight1, priority1 = dictionary.search_with_info("词语1")
            self.assertTrue(found1)
            self.assertEqual(weight1, 5.0)
            self.assertEqual(priority1, 10)
            
            found2, _, weight2, priority2 = dictionary.search_with_info("词语2")
            self.assertTrue(found2)
            self.assertEqual(weight2, 3.0)
            self.assertEqual(priority2, 5)
            
            found3, _, weight3, priority3 = dictionary.search_with_info("词语3")
            self.assertTrue(found3)
            self.assertEqual(weight3, 1.0)
            self.assertEqual(priority3, 0)
        
        os.unlink(f.name)


class TestUserDictionary(unittest.TestCase):
    def test_create_user_dictionary(self):
        user_dict = UserDictionary(name="test_user", priority=100)
        self.assertEqual(user_dict.name, "test_user")
        self.assertEqual(user_dict.priority, 100)
        self.assertEqual(user_dict.default_weight, 10.0)

    def test_add_word_with_default_weight(self):
        user_dict = UserDictionary(name="test", priority=50)
        user_dict.default_weight = 20.0
        user_dict.add_word("用户词", "n")
        
        found, pos_tag, weight, priority = user_dict.search_with_info("用户词")
        self.assertTrue(found)
        self.assertEqual(weight, 20.0)
        self.assertEqual(priority, 50)

    def test_add_word_with_custom_weight_priority(self):
        user_dict = UserDictionary(name="test", priority=50)
        user_dict.add_word("自定义词", "v", weight=100.0, priority=200)
        
        found, pos_tag, weight, priority = user_dict.search_with_info("自定义词")
        self.assertTrue(found)
        self.assertEqual(weight, 100.0)
        self.assertEqual(priority, 200)

    def test_load_user_dictionary(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write("用户词1 n 15.0 100\n")
            f.write("用户词2 v\n")
            f.flush()
            
            user_dict = UserDictionary(name="test", priority=80)
            user_dict.default_weight = 12.0
            user_dict.load_dictionary(f.name)
            
            found1, _, weight1, priority1 = user_dict.search_with_info("用户词1")
            self.assertTrue(found1)
            self.assertEqual(weight1, 15.0)
            self.assertEqual(priority1, 100)
            
            found2, _, weight2, priority2 = user_dict.search_with_info("用户词2")
            self.assertTrue(found2)
            self.assertEqual(weight2, 12.0)
            self.assertEqual(priority2, 80)
        
        os.unlink(f.name)


class TestDictionaryManager(unittest.TestCase):
    def test_register_dictionaries(self):
        manager = DictionaryManager()
        
        dict1 = Dictionary(load_default=False, priority=10)
        dict1.name = "system"
        dict1.add_word("系统词", "n")
        
        user_dict = UserDictionary(name="user", priority=100)
        user_dict.add_word("用户词", "n")
        
        manager.register_dictionary(dict1)
        manager.register_user_dictionary(user_dict)
        
        info = manager.get_all_dictionaries_info()
        self.assertEqual(len(info), 2)

    def test_priority_based_search(self):
        manager = DictionaryManager()
        
        dict1 = Dictionary(load_default=False, priority=10)
        dict1.name = "low_priority"
        dict1.add_word("测试词", "n", weight=1.0, priority=10)
        
        user_dict = UserDictionary(name="high_priority", priority=100)
        user_dict.add_word("测试词", "v", weight=10.0, priority=100)
        
        manager.register_dictionary(dict1)
        manager.register_user_dictionary(user_dict)
        
        found, pos_tag, weight, priority = manager.search_with_info("测试词")
        self.assertTrue(found)
        self.assertEqual(pos_tag, "v")
        self.assertEqual(priority, 100)

    def test_unregister_dictionary(self):
        manager = DictionaryManager()
        
        user_dict = UserDictionary(name="test", priority=100)
        user_dict.add_word("测试词", "n")
        
        manager.register_user_dictionary(user_dict)
        self.assertTrue(manager.search("测试词"))
        
        result = manager.unregister_dictionary("test")
        self.assertTrue(result)
        
        manager.invalidate_cache()


class TestSegmentorUserDictionary(unittest.TestCase):
    def test_create_user_dictionary(self):
        seg = Segmentor(load_default_dict=False)
        user_dict = seg.create_user_dictionary("my_dict", priority=100, default_weight=15.0)
        
        self.assertIsNotNone(user_dict)
        self.assertEqual(user_dict.name, "my_dict")
        self.assertEqual(user_dict.priority, 100)

    def test_add_user_word(self):
        seg = Segmentor(load_default_dict=False)
        seg.add_user_word("新词汇", "n", weight=20.0, priority=150)
        
        info = seg.get_word_info("新词汇")
        self.assertTrue(info['found'])
        self.assertEqual(info['weight'], 20.0)
        self.assertEqual(info['priority'], 150)

    def test_weighted_segmentation(self):
        seg = Segmentor(load_default_dict=False)
        seg.set_use_weighted(True)
        
        seg.add_word("中", "f", weight=1.0)
        seg.add_word("中国", "ns", weight=5.0)
        seg.add_word("中国人", "n", weight=10.0)
        
        result = seg.segment("中国人")
        self.assertEqual(result, ["中国人"])

    def test_priority_based_segmentation(self):
        seg = Segmentor(load_default_dict=False)
        seg.set_use_weighted(True)
        
        seg.add_word("研究", "v", weight=1.0, priority=0)
        seg.add_word("研究生", "n", weight=5.0, priority=10)
        
        user_dict = seg.create_user_dictionary("user", priority=100)
        user_dict.add_word("研究生", "n", weight=20.0, priority=100)
        
        result = seg.segment("研究生")
        self.assertEqual(result, ["研究生"])

    def test_get_all_dictionaries_info(self):
        seg = Segmentor(load_default_dict=False)
        
        seg.create_user_dictionary("dict1", priority=50)
        seg.create_user_dictionary("dict2", priority=100)
        
        info = seg.get_all_dictionaries_info()
        self.assertGreaterEqual(len(info), 2)

    def test_remove_user_dictionary(self):
        seg = Segmentor(load_default_dict=False)
        seg.create_user_dictionary("test_dict", priority=100)
        
        result = seg.remove_user_dictionary("test_dict")
        self.assertTrue(result)
        
        user_dict = seg.get_user_dictionary("test_dict")
        self.assertIsNone(user_dict)

    def test_set_word_weight(self):
        seg = Segmentor(load_default_dict=False)
        seg.add_word("测试词", "n")
        
        result = seg.set_word_weight("测试词", 50.0)
        self.assertTrue(result)
        
        info = seg.get_word_info("测试词")
        self.assertEqual(info['weight'], 50.0)

    def test_set_word_priority(self):
        seg = Segmentor(load_default_dict=False)
        seg.add_word("测试词", "n")
        
        result = seg.set_word_priority("测试词", 200)
        self.assertTrue(result)
        
        info = seg.get_word_info("测试词")
        self.assertEqual(info['priority'], 200)


class TestWeightedTokenization(unittest.TestCase):
    def test_forward_max_match_weighted(self):
        from AuroraNLP.tokenizer import forward_max_match_weighted
        
        dictionary = Dictionary(load_default=False)
        dictionary.add_word("研究", "v", weight=1.0, priority=0)
        dictionary.add_word("研究生", "n", weight=10.0, priority=10)
        dictionary.add_word("生命", "n", weight=1.0, priority=0)
        dictionary.add_word("命", "n", weight=1.0, priority=0)
        
        result = forward_max_match_weighted("研究生", dictionary)
        self.assertEqual(result, ["研究生"])

    def test_bidirectional_max_match_weighted(self):
        from AuroraNLP.tokenizer import bidirectional_max_match_weighted
        
        dictionary = Dictionary(load_default=False)
        dictionary.add_word("中国", "ns", weight=5.0, priority=10)
        dictionary.add_word("中国人", "n", weight=10.0, priority=20)
        
        result = bidirectional_max_match_weighted("中国人", dictionary)
        self.assertEqual(result, ["中国人"])

    def test_weighted_with_pos(self):
        from AuroraNLP.tokenizer import forward_max_match_weighted_with_pos
        
        dictionary = Dictionary(load_default=False)
        dictionary.add_word("北京", "ns", weight=5.0, priority=10)
        dictionary.add_word("北京人", "n", weight=15.0, priority=20)
        
        result = forward_max_match_weighted_with_pos("北京人", dictionary)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], "北京人")
        self.assertEqual(result[0][1], "n")


if __name__ == '__main__':
    unittest.main()
