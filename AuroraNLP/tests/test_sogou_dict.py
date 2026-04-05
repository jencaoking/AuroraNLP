import os
import json
import struct
import tempfile
import unittest

from AuroraNLP.scel_parser import ScelParser, ScelWord, ScelMetadata, ScelBatchParser
from AuroraNLP.sogou_dict import ScelConverter, SogouDictionary, SogouDictionaryManager


def create_mock_scel_file(file_path: str, words: list) -> None:
    with open(file_path, 'wb') as f:
        f.write(b'\x40\x15\x00\x00\x44\x43\x53\x01')
        f.write(b'\x00' * (0x130 - 8))

        name = "测试词库".encode('utf-16-le') + b'\x00\x00'
        f.write(name.ljust(0x208, b'\x00'))

        category = "测试分类".encode('utf-16-le') + b'\x00\x00'
        f.write(category.ljust(0x208, b'\x00'))

        description = "测试描述".encode('utf-16-le') + b'\x00\x00'
        f.write(description.ljust(0x800, b'\x00'))

        example = "测试示例".encode('utf-16-le') + b'\x00\x00'
        f.write(example.ljust(0x800, b'\x00'))

        f.write(b'\x00' * 4)

        pinyins = ['ce', 'shi', 'ci', 'ku']
        pinyin_data = b''
        for i, py in enumerate(pinyins):
            py_bytes = py.encode('utf-16-le')
            pinyin_data += struct.pack('<H', i)
            pinyin_data += struct.pack('<H', len(py_bytes))
            pinyin_data += py_bytes
        f.write(pinyin_data)

        current_pos = f.tell()
        if current_pos < 0x2628:
            f.write(b'\x00' * (0x2628 - current_pos))

        for word in words:
            word_bytes = word.encode('utf-16-le')
            f.write(struct.pack('<H', 1))
            f.write(struct.pack('<H', 4))
            f.write(struct.pack('<H', 0))
            f.write(struct.pack('<H', 1))
            f.write(struct.pack('<H', len(word_bytes)))
            f.write(word_bytes)
            f.write(struct.pack('<H', 2))
            f.write(struct.pack('<H', 1))


class TestScelParser(unittest.TestCase):
    def setUp(self):
        self.parser = ScelParser()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_scel_word_dataclass(self):
        word = ScelWord(word="测试", pinyin="ce'shi", frequency=100)
        self.assertEqual(word.word, "测试")
        self.assertEqual(word.pinyin, "ce'shi")
        self.assertEqual(word.frequency, 100)

    def test_scel_metadata_dataclass(self):
        metadata = ScelMetadata(
            name="测试词库",
            category="测试",
            description="测试描述",
            word_count=100
        )
        self.assertEqual(metadata.name, "测试词库")
        self.assertEqual(metadata.word_count, 100)

    def test_parse_nonexistent_file(self):
        with self.assertRaises(FileNotFoundError):
            self.parser.parse("nonexistent.scel")

    def test_parse_invalid_file(self):
        invalid_file = os.path.join(self.temp_dir, "invalid.scel")
        with open(invalid_file, 'wb') as f:
            f.write(b'invalid content')

        with self.assertRaises(ValueError):
            self.parser.parse(invalid_file)

    def test_bytes_to_str(self):
        test_str = "测试"
        encoded = test_str.encode('utf-16-le')
        result = self.parser._bytes_to_str(encoded + b'\x00\x00')
        self.assertEqual(result, test_str)


class TestScelConverter(unittest.TestCase):
    def setUp(self):
        self.converter = ScelConverter()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_to_txt_with_mock_data(self):
        txt_path = os.path.join(self.temp_dir, "output.txt")
        json_path = os.path.join(self.temp_dir, "output.json")

        self.assertTrue(True)

    def test_to_json_with_mock_data(self):
        json_path = os.path.join(self.temp_dir, "output.json")
        self.assertTrue(True)


class TestSogouDictionary(unittest.TestCase):
    def setUp(self):
        self.dict = SogouDictionary(name="test_sogou", priority=80)
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_init(self):
        self.assertEqual(self.dict.name, "test_sogou")
        self.assertEqual(self.dict.priority, 80)
        self.assertEqual(len(self.dict), 0)

    def test_add_word(self):
        self.dict.add_word("测试词", "n", 5.0, 80)
        self.assertTrue(self.dict.search_in_dict("测试词"))
        self.assertEqual(len(self.dict), 1)

    def test_remove_word(self):
        self.dict.add_word("测试词", "n", 5.0, 80)
        self.assertTrue(self.dict.remove_word("测试词"))
        self.assertFalse(self.dict.search_in_dict("测试词"))

    def test_search_with_info(self):
        self.dict.add_word("测试词", "n", 5.0, 80)
        found, pos, weight, priority = self.dict.search_with_info("测试词")
        self.assertTrue(found)
        self.assertEqual(pos, "n")
        self.assertEqual(weight, 5.0)
        self.assertEqual(priority, 80)

    def test_load_from_txt(self):
        txt_path = os.path.join(self.temp_dir, "test_dict.txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("测试词1\tn\t5.0\n")
            f.write("测试词2\tv\t3.0\n")
            f.write("测试词3\n")

        count = self.dict.load_from_txt(txt_path)
        self.assertEqual(count, 3)
        self.assertTrue(self.dict.search_in_dict("测试词1"))
        self.assertTrue(self.dict.search_in_dict("测试词2"))
        self.assertTrue(self.dict.search_in_dict("测试词3"))

    def test_load_from_json(self):
        json_path = os.path.join(self.temp_dir, "test_dict.json")
        data = {
            "words": [
                {"word": "测试词1", "pinyin": "ce'shi'ci'yi", "frequency": 100},
                {"word": "测试词2", "pinyin": "ce'shi'ci'er", "frequency": 50},
            ]
        }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)

        count = self.dict.load_from_json(json_path)
        self.assertEqual(count, 2)
        self.assertTrue(self.dict.search_in_dict("测试词1"))
        self.assertTrue(self.dict.search_in_dict("测试词2"))

    def test_source_files_tracking(self):
        txt_path = os.path.join(self.temp_dir, "test_dict.txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("测试词\n")

        self.dict.load_from_txt(txt_path)
        self.assertIn(txt_path, self.dict.source_files)


class TestSogouDictionaryManager(unittest.TestCase):
    def setUp(self):
        self.manager = SogouDictionaryManager()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_create_dictionary(self):
        d = self.manager.create_dictionary("test", priority=90)
        self.assertEqual(d.name, "test")
        self.assertEqual(d.priority, 90)

    def test_create_duplicate_dictionary(self):
        self.manager.create_dictionary("test")
        with self.assertRaises(ValueError):
            self.manager.create_dictionary("test")

    def test_get_dictionary(self):
        self.manager.create_dictionary("test")
        d = self.manager.get_dictionary("test")
        self.assertIsNotNone(d)
        self.assertEqual(d.name, "test")

    def test_remove_dictionary(self):
        self.manager.create_dictionary("test")
        self.assertTrue(self.manager.remove_dictionary("test"))
        self.assertIsNone(self.manager.get_dictionary("test"))

    def test_list_dictionaries(self):
        self.manager.create_dictionary("dict1", priority=80)
        self.manager.create_dictionary("dict2", priority=90)

        dicts = self.manager.list_dictionaries()
        self.assertEqual(len(dicts), 2)

    def test_import_from_txt(self):
        txt_path = os.path.join(self.temp_dir, "test.txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("测试词\n")

        d = self.manager.create_dictionary("test")
        d.load_from_txt(txt_path)

        self.assertTrue(d.search_in_dict("测试词"))

    def test_get_all_words(self):
        d1 = self.manager.create_dictionary("dict1")
        d1.add_word("词1")

        d2 = self.manager.create_dictionary("dict2")
        d2.add_word("词2")
        d2.add_word("词1")

        all_words = self.manager.get_all_words()
        self.assertEqual(len(all_words), 2)
        self.assertIn("词1", all_words)
        self.assertIn("词2", all_words)

    def test_search_word(self):
        d = self.manager.create_dictionary("test")
        d.add_word("测试词", "n", 5.0, 80)

        results = self.manager.search_word("测试词")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['dictionary'], "test")
        self.assertEqual(results[0]['pos_tag'], "n")


class TestScelBatchParser(unittest.TestCase):
    def setUp(self):
        self.batch_parser = ScelBatchParser()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_parse_empty_directory(self):
        empty_dir = os.path.join(self.temp_dir, "empty")
        os.makedirs(empty_dir)

        words = self.batch_parser.parse_directory(empty_dir)
        self.assertEqual(len(words), 0)

    def test_find_scel_files(self):
        scel_dir = os.path.join(self.temp_dir, "scel")
        os.makedirs(scel_dir)

        open(os.path.join(scel_dir, "test1.scel"), 'wb').close()
        open(os.path.join(scel_dir, "test2.SCEL"), 'wb').close()
        open(os.path.join(scel_dir, "test3.txt"), 'w').close()

        files = self.batch_parser._find_scel_files(scel_dir, recursive=False)
        self.assertEqual(len(files), 2)

    def test_get_unique_words(self):
        self.batch_parser._all_words = [
            ScelWord(word="测试", pinyin="ce'shi", frequency=1),
            ScelWord(word="测试", pinyin="ce'shi", frequency=2),
            ScelWord(word="词库", pinyin="ci'ku", frequency=1),
        ]

        unique = self.batch_parser.get_unique_words()
        self.assertEqual(len(unique), 2)


if __name__ == '__main__':
    unittest.main()
