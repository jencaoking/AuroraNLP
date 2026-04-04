import os
import warnings
from typing import Set, Optional, List, Tuple

from .trie import Trie


class Dictionary:
    DEFAULT_DICT_PATH = os.path.join(os.path.dirname(__file__), 'data', 'dictionary.txt')

    def __init__(self, load_default: bool = True):
        self._trie = Trie()
        self._words_cache: Optional[Set[str]] = None
        if load_default:
            self._load_default_dictionary()

    def _load_default_dictionary(self) -> None:
        if os.path.exists(self.DEFAULT_DICT_PATH):
            self.load_dictionary(self.DEFAULT_DICT_PATH)

    def load_dictionary(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"词典文件不存在: {path}")
        
        loaded_count = 0
        error_count = 0
        
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) >= 2:
                    word, pos_tag = parts[0], parts[1]
                    if not word:
                        warnings.warn(
                            f"第 {line_num} 行: 空词汇将被跳过",
                            UserWarning,
                            stacklevel=2
                        )
                        error_count += 1
                        continue
                    if not pos_tag or not pos_tag.replace('/', '').isalpha():
                        warnings.warn(
                            f"第 {line_num} 行: 无效的词性标签 '{pos_tag}'，将使用默认值",
                            UserWarning,
                            stacklevel=2
                        )
                        pos_tag = 'x'
                    self._trie.insert(word, pos_tag)
                    loaded_count += 1
                    if len(parts) > 2:
                        warnings.warn(
                            f"词典行格式多余字段将被忽略: '{line}'",
                            UserWarning,
                            stacklevel=2
                        )
                elif len(parts) == 1:
                    word = parts[0]
                    if word:
                        self._trie.insert(word)
                        loaded_count += 1
                    else:
                        warnings.warn(
                            f"第 {line_num} 行: 空词汇将被跳过",
                            UserWarning,
                            stacklevel=2
                        )
                        error_count += 1
        
        self._words_cache = None
        
        if loaded_count == 0:
            warnings.warn(
                f"词典文件 '{path}' 未加载任何有效词汇",
                UserWarning,
                stacklevel=2
            )
        
        if error_count > 0:
            warnings.warn(
                f"词典加载完成，共加载 {loaded_count} 个词汇，跳过 {error_count} 个无效条目",
                UserWarning,
                stacklevel=2
            )

    def save_dictionary(self, path: str) -> None:
        with open(path, 'w', encoding='utf-8') as f:
            for word in self.get_words():
                _, pos_tag = self._trie.search_with_pos(word)
                if pos_tag:
                    f.write(f"{word} {pos_tag}\n")
                else:
                    f.write(f"{word}\n")

    def get_words(self) -> Set[str]:
        if self._words_cache is None:
            self._words_cache = set()
            self._collect_words(self._trie.root, "", self._words_cache)
        return self._words_cache.copy()

    def _collect_words(self, node, prefix: str, result: Set[str]) -> None:
        if node.is_word:
            result.add(prefix)
        for char, child in node.children.items():
            self._collect_words(child, prefix + char, result)

    def add_word(self, word: str, pos_tag: Optional[str] = None) -> None:
        self._trie.insert(word, pos_tag)
        self._words_cache = None

    def remove_word(self, word: str) -> bool:
        result = self._trie.remove(word)
        if result:
            self._words_cache = None
        return result

    def search_in_dict(self, word: str) -> bool:
        return self._trie.search(word)

    def search_with_pos(self, word: str) -> Tuple[bool, Optional[str]]:
        return self._trie.search_with_pos(word)

    def get_pos_tag(self, word: str) -> Optional[str]:
        _, pos_tag = self._trie.search_with_pos(word)
        return pos_tag

    def has_prefix(self, prefix: str) -> bool:
        return self._trie.starts_with(prefix)

    def get_max_match_length(self, text: str, start: int = 0, max_len: int = 15) -> int:
        return self._trie.get_max_match_length(text, start, max_len)

    def get_max_match_with_pos(self, text: str, start: int = 0, max_len: int = 15) -> Tuple[int, Optional[str]]:
        return self._trie.get_max_match_with_pos(text, start, max_len)

    def __len__(self) -> int:
        return len(self._trie)

    def __contains__(self, word: str) -> bool:
        return word in self._trie
