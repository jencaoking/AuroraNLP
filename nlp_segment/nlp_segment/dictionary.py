import os
from typing import Set, Optional


class Dictionary:
    DEFAULT_DICT_PATH = os.path.join(os.path.dirname(__file__), 'data', 'dictionary.txt')

    def __init__(self, load_default: bool = True):
        self.words: Set[str] = set()
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
                    self.words.add(word)

    def get_words(self) -> Set[str]:
        return self.words

    def add_word(self, word: str) -> None:
        self.words.add(word)

    def remove_word(self, word: str) -> bool:
        if word in self.words:
            self.words.remove(word)
            return True
        return False

    def search_in_dict(self, word: str) -> bool:
        return word in self.words

    def __len__(self) -> int:
        return len(self.words)

    def __contains__(self, word: str) -> bool:
        return word in self.words
