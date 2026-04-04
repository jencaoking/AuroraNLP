import os
from typing import Set, Optional

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
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip()
                if word:
                    self._trie.insert(word)
        self._words_cache = None

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

    def add_word(self, word: str) -> None:
        self._trie.insert(word)
        self._words_cache = None

    def remove_word(self, word: str) -> bool:
        result = self._trie.remove(word)
        if result:
            self._words_cache = None
        return result

    def search_in_dict(self, word: str) -> bool:
        return self._trie.search(word)

    def has_prefix(self, prefix: str) -> bool:
        return self._trie.starts_with(prefix)

    def get_max_match_length(self, text: str, start: int = 0, max_len: int = 15) -> int:
        return self._trie.get_max_match_length(text, start, max_len)

    def __len__(self) -> int:
        return len(self._trie)

    def __contains__(self, word: str) -> bool:
        return word in self._trie
