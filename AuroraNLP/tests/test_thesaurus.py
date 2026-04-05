import os
import tempfile
import unittest

from AuroraNLP.thesaurus import (
    WordRelation,
    CATEGORY_NAMES,
    SUBCATEGORY_NAMES,
    ThesaurusEntry,
    SemanticCategory,
    Thesaurus,
    ThesaurusManager,
)


class TestWordRelation(unittest.TestCase):
    def test_word_relation_values(self):
        self.assertEqual(WordRelation.SYNONYM.value, "=")
        self.assertEqual(WordRelation.RELATED.value, "#")
        self.assertEqual(WordRelation.INDEPENDENT.value, "@")


class TestCategoryNames(unittest.TestCase):
    def test_category_names_exist(self):
        self.assertIn("A", CATEGORY_NAMES)
        self.assertEqual(CATEGORY_NAMES["A"], "人物")
        self.assertIn("G", CATEGORY_NAMES)
        self.assertEqual(CATEGORY_NAMES["G"], "动作")

    def test_subcategory_names_exist(self):
        self.assertIn("A", SUBCATEGORY_NAMES)
        self.assertIn("a", SUBCATEGORY_NAMES["A"])
        self.assertEqual(SUBCATEGORY_NAMES["A"]["a"], "泛称")


class TestThesaurusEntry(unittest.TestCase):
    def test_entry_creation(self):
        entry = ThesaurusEntry(
            code="Aa01A01",
            words=["人", "人类", "人物"],
            relation=WordRelation.SYNONYM
        )
        self.assertEqual(entry.code, "Aa01A01")
        self.assertEqual(len(entry.words), 3)
        self.assertEqual(entry.relation, WordRelation.SYNONYM)
        self.assertEqual(entry.category, "A")
        self.assertEqual(entry.subcategory, "a")

    def test_entry_with_related_relation(self):
        entry = ThesaurusEntry(
            code="Aa01A02",
            words=["人口", "居民"],
            relation=WordRelation.RELATED
        )
        self.assertEqual(entry.relation, WordRelation.RELATED)


class TestSemanticCategory(unittest.TestCase):
    def test_category_creation(self):
        category = SemanticCategory(
            code="A",
            name="人物",
            level=1
        )
        self.assertEqual(category.code, "A")
        self.assertEqual(category.name, "人物")
        self.assertEqual(category.level, 1)
        self.assertIsNone(category.parent_code)
        self.assertEqual(len(category.children), 0)
        self.assertEqual(len(category.words), 0)

    def test_category_with_parent(self):
        category = SemanticCategory(
            code="Aa",
            name="泛称",
            level=2,
            parent_code="A"
        )
        self.assertEqual(category.parent_code, "A")


class TestThesaurus(unittest.TestCase):
    def setUp(self):
        self.thesaurus = Thesaurus(load_default=True)
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_default_thesaurus(self):
        self.assertTrue(self.thesaurus.is_loaded())
        self.assertGreater(self.thesaurus.get_word_count(), 0)

    def test_has_word(self):
        self.assertTrue(self.thesaurus.has_word("人"))
        self.assertTrue(self.thesaurus.has_word("美丽"))
        self.assertFalse(self.thesaurus.has_word("不存在的词xyz"))

    def test_get_synonyms(self):
        synonyms = self.thesaurus.get_synonyms("人")
        self.assertIn("人类", synonyms)
        self.assertIn("人物", synonyms)
        self.assertNotIn("人", synonyms)

    def test_get_synonyms_nonexistent_word(self):
        synonyms = self.thesaurus.get_synonyms("不存在的词xyz")
        self.assertEqual(len(synonyms), 0)

    def test_get_related_words(self):
        related = self.thesaurus.get_related_words("人")
        self.assertGreater(len(related), 0)

    def test_get_near_synonyms(self):
        near_synonyms = self.thesaurus.get_near_synonyms("美丽")
        self.assertIn("漂亮", near_synonyms)

    def test_get_category(self):
        category = self.thesaurus.get_category("人")
        self.assertIsNotNone(category)
        self.assertEqual(category.code, "A")
        self.assertEqual(category.name, "人物")

    def test_get_category_nonexistent_word(self):
        category = self.thesaurus.get_category("不存在的词xyz")
        self.assertIsNone(category)

    def test_get_subcategory(self):
        subcategory = self.thesaurus.get_subcategory("人")
        self.assertIsNotNone(subcategory)
        self.assertEqual(subcategory.code, "Aa")

    def test_get_category_name(self):
        name = self.thesaurus.get_category_name("人")
        self.assertEqual(name, "人物")

    def test_get_subcategory_name(self):
        name = self.thesaurus.get_subcategory_name("人")
        self.assertIsNotNone(name)

    def test_get_semantic_path(self):
        path = self.thesaurus.get_semantic_path("人")
        self.assertGreater(len(path), 0)
        self.assertEqual(path[0][0], "A")
        self.assertEqual(path[0][1], "人物")

    def test_get_semantic_path_nonexistent_word(self):
        path = self.thesaurus.get_semantic_path("不存在的词xyz")
        self.assertEqual(len(path), 0)

    def test_get_words_by_category(self):
        words = self.thesaurus.get_words_by_category("A")
        self.assertIn("人", words)
        self.assertIn("男人", words)

    def test_get_words_by_subcategory(self):
        words = self.thesaurus.get_words_by_subcategory("Aa")
        self.assertIn("人", words)

    def test_calculate_similarity_same_word(self):
        sim = self.thesaurus.calculate_similarity("人", "人")
        self.assertEqual(sim, 1.0)

    def test_calculate_similarity_synonyms(self):
        sim = self.thesaurus.calculate_similarity("人", "人类")
        self.assertGreater(sim, 0.5)

    def test_calculate_similarity_different_category(self):
        sim = self.thesaurus.calculate_similarity("人", "桌子")
        self.assertLess(sim, 0.5)

    def test_calculate_similarity_nonexistent_word(self):
        sim = self.thesaurus.calculate_similarity("不存在的词xyz", "人")
        self.assertEqual(sim, 0.0)

    def test_is_synonym(self):
        self.assertTrue(self.thesaurus.is_synonym("人", "人类"))
        self.assertTrue(self.thesaurus.is_synonym("人", "人"))
        self.assertFalse(self.thesaurus.is_synonym("人", "桌子"))

    def test_is_related(self):
        self.assertTrue(self.thesaurus.is_related("人", "人类"))
        self.assertTrue(self.thesaurus.is_related("人", "人"))

    def test_get_entry(self):
        entry = self.thesaurus.get_entry("Aa01A01")
        self.assertIsNotNone(entry)
        self.assertIn("人", entry.words)

    def test_get_word_codes(self):
        codes = self.thesaurus.get_word_codes("人")
        self.assertGreater(len(codes), 0)

    def test_get_all_categories(self):
        categories = self.thesaurus.get_all_categories()
        self.assertIn("A", categories)
        self.assertEqual(categories["A"].name, "人物")

    def test_get_all_words(self):
        words = self.thesaurus.get_all_words()
        self.assertIn("人", words)

    def test_add_entry(self):
        initial_count = self.thesaurus.get_entry_count()
        self.thesaurus.add_entry(
            code="Xx01A01=",
            words=["测试词1", "测试词2"],
            relation=WordRelation.SYNONYM
        )
        self.assertEqual(self.thesaurus.get_entry_count(), initial_count + 1)
        self.assertTrue(self.thesaurus.has_word("测试词1"))
        self.assertTrue(self.thesaurus.has_word("测试词2"))

    def test_save_and_load_thesaurus(self):
        test_file = os.path.join(self.temp_dir, "test_thesaurus.txt")
        
        self.thesaurus.add_entry(
            code="Xx01A01=",
            words=["测试词1", "测试词2"],
            relation=WordRelation.SYNONYM
        )
        
        self.thesaurus.save_thesaurus(test_file)
        self.assertTrue(os.path.exists(test_file))
        
        new_thesaurus = Thesaurus(load_default=False)
        new_thesaurus.load_thesaurus(test_file)
        
        self.assertTrue(new_thesaurus.has_word("测试词1"))
        self.assertTrue(new_thesaurus.has_word("测试词2"))

    def test_len(self):
        count = len(self.thesaurus)
        self.assertGreater(count, 0)

    def test_contains(self):
        self.assertIn("人", self.thesaurus)
        self.assertNotIn("不存在的词xyz", self.thesaurus)

    def test_repr(self):
        repr_str = repr(self.thesaurus)
        self.assertIn("Thesaurus", repr_str)
        self.assertIn("entries", repr_str)
        self.assertIn("words", repr_str)


class TestThesaurusManager(unittest.TestCase):
    def setUp(self):
        self.manager = ThesaurusManager()

    def test_load_default(self):
        self.manager.load()
        self.assertTrue(self.manager.is_loaded())

    def test_get_thesaurus(self):
        self.manager.load()
        thesaurus = self.manager.get_thesaurus()
        self.assertIsNotNone(thesaurus)

    def test_get_synonyms(self):
        self.manager.load()
        synonyms = self.manager.get_synonyms("人")
        self.assertIn("人类", synonyms)

    def test_get_related_words(self):
        self.manager.load()
        related = self.manager.get_related_words("人")
        self.assertGreater(len(related), 0)

    def test_get_near_synonyms(self):
        self.manager.load()
        near_synonyms = self.manager.get_near_synonyms("美丽")
        self.assertIn("漂亮", near_synonyms)

    def test_get_category_name(self):
        self.manager.load()
        name = self.manager.get_category_name("人")
        self.assertEqual(name, "人物")

    def test_get_semantic_path(self):
        self.manager.load()
        path = self.manager.get_semantic_path("人")
        self.assertGreater(len(path), 0)

    def test_calculate_similarity(self):
        self.manager.load()
        sim = self.manager.calculate_similarity("人", "人类")
        self.assertGreater(sim, 0.5)

    def test_is_synonym(self):
        self.manager.load()
        self.assertTrue(self.manager.is_synonym("人", "人类"))

    def test_is_related(self):
        self.manager.load()
        self.assertTrue(self.manager.is_related("人", "人类"))

    def test_not_loaded(self):
        manager = ThesaurusManager()
        self.assertFalse(manager.is_loaded())
        self.assertEqual(manager.get_synonyms("人"), [])
        self.assertEqual(manager.get_related_words("人"), [])
        self.assertEqual(manager.get_category_name("人"), None)


class TestThesaurusEdgeCases(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_nonexistent_file(self):
        thesaurus = Thesaurus(load_default=False)
        with self.assertRaises(FileNotFoundError):
            thesaurus.load_thesaurus("nonexistent_file.txt")

    def test_load_empty_file(self):
        empty_file = os.path.join(self.temp_dir, "empty.txt")
        with open(empty_file, 'w', encoding='utf-8') as f:
            f.write("")
        
        thesaurus = Thesaurus(load_default=False)
        thesaurus.load_thesaurus(empty_file)
        self.assertEqual(thesaurus.get_word_count(), 0)

    def test_load_file_with_comments(self):
        comment_file = os.path.join(self.temp_dir, "comments.txt")
        with open(comment_file, 'w', encoding='utf-8') as f:
            f.write("# This is a comment\n")
            f.write("# Another comment\n")
        
        thesaurus = Thesaurus(load_default=False)
        thesaurus.load_thesaurus(comment_file)
        self.assertEqual(thesaurus.get_word_count(), 0)

    def test_load_file_with_invalid_lines(self):
        invalid_file = os.path.join(self.temp_dir, "invalid.txt")
        with open(invalid_file, 'w', encoding='utf-8') as f:
            f.write("Aa01A01=\t人 人类\n")
            f.write("invalid line without tab\n")
            f.write("\n")
            f.write("Aa01A02=\t人口 居民\n")
        
        thesaurus = Thesaurus(load_default=False)
        thesaurus.load_thesaurus(invalid_file)
        self.assertTrue(thesaurus.has_word("人"))
        self.assertTrue(thesaurus.has_word("人口"))

    def test_parse_relation_types(self):
        thesaurus = Thesaurus(load_default=False)
        
        thesaurus.add_entry("Aa01A01=", ["词1", "词2"], WordRelation.SYNONYM)
        thesaurus.add_entry("Aa01A02#", ["词3", "词4"], WordRelation.RELATED)
        thesaurus.add_entry("Aa01A03@", ["词5"], WordRelation.INDEPENDENT)
        
        entry1 = thesaurus.get_entry("Aa01A01")
        self.assertEqual(entry1.relation, WordRelation.SYNONYM)
        
        entry2 = thesaurus.get_entry("Aa01A02")
        self.assertEqual(entry2.relation, WordRelation.RELATED)
        
        entry3 = thesaurus.get_entry("Aa01A03")
        self.assertEqual(entry3.relation, WordRelation.INDEPENDENT)


if __name__ == '__main__':
    unittest.main()
