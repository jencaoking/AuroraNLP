import os
import json
from typing import List, Dict, Optional, Set
from pathlib import Path

from .scel_parser import ScelParser, ScelWord, ScelBatchParser, ScelMetadata
from .dictionary import Dictionary, UserDictionary


class ScelConverter:
    def __init__(self):
        self._parser = ScelParser()

    def to_txt(
        self,
        input_path: str,
        output_path: str,
        include_pinyin: bool = False,
        include_frequency: bool = False,
        include_pos: bool = False,
        default_pos: str = 'n'
    ) -> int:
        words = self._parser.parse(input_path)

        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for w in words:
                parts = [w.word]

                if include_pos:
                    parts.append(w.pos_tag or default_pos)

                if include_frequency:
                    parts.append(str(w.frequency))

                if include_pinyin:
                    parts.append(w.pinyin)

                f.write('\t'.join(parts) + '\n')

        return len(words)

    def to_json(
        self,
        input_path: str,
        output_path: str,
        include_metadata: bool = True,
        pretty: bool = False
    ) -> int:
        words = self._parser.parse(input_path)

        result: Dict = {
            'words': [
                {
                    'word': w.word,
                    'pinyin': w.pinyin,
                    'frequency': w.frequency
                }
                for w in words
            ]
        }

        if include_metadata:
            result['metadata'] = {
                'name': self._parser.metadata.name,
                'category': self._parser.metadata.category,
                'description': self._parser.metadata.description,
                'example': self._parser.metadata.example,
                'word_count': self._parser.metadata.word_count
            }

        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            if pretty:
                json.dump(result, f, ensure_ascii=False, indent=2)
            else:
                json.dump(result, f, ensure_ascii=False)

        return len(words)

    def to_dictionary_format(
        self,
        input_path: str,
        output_path: str,
        default_weight: float = 1.0,
        default_priority: int = 50,
        default_pos: str = 'n'
    ) -> int:
        words = self._parser.parse(input_path)

        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for w in words:
                pos = w.pos_tag or default_pos
                weight = default_weight + (w.frequency / 10000.0)
                f.write(f"{w.word}\t{pos}\t{weight:.4f}\t{default_priority}\n")

        return len(words)

    def batch_convert(
        self,
        input_dir: str,
        output_dir: str,
        output_format: str = 'txt',
        recursive: bool = False
    ) -> Dict[str, int]:
        batch_parser = ScelBatchParser()
        batch_parser.parse_directory(input_dir, recursive)

        os.makedirs(output_dir, exist_ok=True)

        results: Dict[str, int] = {}

        if output_format == 'txt':
            output_path = os.path.join(output_dir, 'merged_dict.txt')
            with open(output_path, 'w', encoding='utf-8') as f:
                for w in batch_parser.get_unique_words():
                    f.write(f"{w.word}\n")
            results['merged'] = len(batch_parser.get_unique_words())
        elif output_format == 'json':
            output_path = os.path.join(output_dir, 'merged_dict.json')
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'words': [
                        {'word': w.word, 'pinyin': w.pinyin, 'frequency': w.frequency}
                        for w in batch_parser.get_unique_words()
                    ]
                }, f, ensure_ascii=False)
            results['merged'] = len(batch_parser.get_unique_words())
        elif output_format == 'dict':
            output_path = os.path.join(output_dir, 'merged_dict.txt')
            with open(output_path, 'w', encoding='utf-8') as f:
                for w in batch_parser.get_unique_words():
                    weight = 1.0 + (w.frequency / 10000.0)
                    f.write(f"{w.word}\tn\t{weight:.4f}\t50\n")
            results['merged'] = len(batch_parser.get_unique_words())

        results['total_files'] = batch_parser.parsed_files
        results['total_words'] = batch_parser.total_words

        return results


class SogouDictionary(UserDictionary):
    def __init__(self, name: str = "sogou", priority: int = 80):
        super().__init__(name=name, priority=priority)
        self._source_files: List[str] = []
        self._metadata_list: List[ScelMetadata] = []

    def load_scel(
        self,
        scel_path: str,
        default_weight: float = 5.0,
        default_pos: str = 'n'
    ) -> int:
        parser = ScelParser()
        words = parser.parse(scel_path)

        loaded_count = 0
        for w in words:
            weight = default_weight + (w.frequency / 10000.0)
            self.add_word(w.word, default_pos, weight, self._priority)
            loaded_count += 1

        self._source_files.append(scel_path)
        self._metadata_list.append(parser.metadata)

        return loaded_count

    def load_scel_directory(
        self,
        directory: str,
        recursive: bool = False,
        default_weight: float = 5.0,
        default_pos: str = 'n'
    ) -> Dict[str, int]:
        batch_parser = ScelBatchParser()
        batch_parser.parse_directory(directory, recursive)

        loaded_count = 0
        unique_words = batch_parser.get_unique_words()

        for w in unique_words:
            weight = default_weight + (w.frequency / 10000.0)
            self.add_word(w.word, default_pos, weight, self._priority)
            loaded_count += 1

        return {
            'loaded': loaded_count,
            'total_files': batch_parser.parsed_files,
            'total_words': batch_parser.total_words
        }

    def load_from_txt(
        self,
        txt_path: str,
        default_weight: float = 5.0,
        default_pos: str = 'n'
    ) -> int:
        loaded_count = 0
        with open(txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split('\t')
                word = parts[0]

                if len(parts) >= 2:
                    pos = parts[1]
                else:
                    pos = default_pos

                if len(parts) >= 3:
                    try:
                        weight = float(parts[2])
                    except ValueError:
                        weight = default_weight
                else:
                    weight = default_weight

                self.add_word(word, pos, weight, self._priority)
                loaded_count += 1

        self._source_files.append(txt_path)
        return loaded_count

    def load_from_json(
        self,
        json_path: str,
        default_weight: float = 5.0,
        default_pos: str = 'n'
    ) -> int:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        loaded_count = 0
        words = data.get('words', [])

        for item in words:
            word = item.get('word', '')
            if not word:
                continue

            frequency = item.get('frequency', 0)
            weight = default_weight + (frequency / 10000.0)

            self.add_word(word, default_pos, weight, self._priority)
            loaded_count += 1

        self._source_files.append(json_path)
        return loaded_count

    @property
    def source_files(self) -> List[str]:
        return self._source_files.copy()

    def get_metadata(self) -> List[Dict]:
        return [
            {
                'name': m.name,
                'category': m.category,
                'description': m.description,
                'word_count': m.word_count
            }
            for m in self._metadata_list
        ]


class SogouDictionaryManager:
    DEFAULT_SOGOU_DICT_PATH = os.path.join(os.path.dirname(__file__), 'data', 'sogou')

    def __init__(self):
        self._dictionaries: Dict[str, SogouDictionary] = {}
        self._converter = ScelConverter()

    def create_dictionary(
        self,
        name: str,
        priority: int = 80
    ) -> SogouDictionary:
        if name in self._dictionaries:
            raise ValueError(f"词典 '{name}' 已存在")

        dictionary = SogouDictionary(name=name, priority=priority)
        self._dictionaries[name] = dictionary
        return dictionary

    def get_dictionary(self, name: str) -> Optional[SogouDictionary]:
        return self._dictionaries.get(name)

    def remove_dictionary(self, name: str) -> bool:
        if name in self._dictionaries:
            del self._dictionaries[name]
            return True
        return False

    def list_dictionaries(self) -> List[Dict]:
        return [
            {
                'name': name,
                'priority': d.priority,
                'word_count': len(d),
                'source_files': d.source_files
            }
            for name, d in self._dictionaries.items()
        ]

    def import_scel(
        self,
        dict_name: str,
        scel_path: str,
        priority: int = 80
    ) -> int:
        if dict_name not in self._dictionaries:
            self.create_dictionary(dict_name, priority)

        return self._dictionaries[dict_name].load_scel(scel_path)

    def import_directory(
        self,
        dict_name: str,
        directory: str,
        recursive: bool = False,
        priority: int = 80
    ) -> Dict[str, int]:
        if dict_name not in self._dictionaries:
            self.create_dictionary(dict_name, priority)

        return self._dictionaries[dict_name].load_scel_directory(directory, recursive)

    def convert_scel_to_txt(
        self,
        input_path: str,
        output_path: str,
        include_pinyin: bool = False
    ) -> int:
        return self._converter.to_txt(input_path, output_path, include_pinyin)

    def convert_scel_to_json(
        self,
        input_path: str,
        output_path: str,
        pretty: bool = False
    ) -> int:
        return self._converter.to_json(input_path, output_path, pretty=pretty)

    def get_all_words(self) -> Set[str]:
        all_words: Set[str] = set()
        for d in self._dictionaries.values():
            all_words.update(d.get_words())
        return all_words

    def search_word(self, word: str) -> List[Dict]:
        results = []
        for name, d in self._dictionaries.items():
            found, pos_tag, weight, priority = d.search_with_info(word)
            if found:
                results.append({
                    'dictionary': name,
                    'word': word,
                    'pos_tag': pos_tag,
                    'weight': weight,
                    'priority': priority
                })
        return results
