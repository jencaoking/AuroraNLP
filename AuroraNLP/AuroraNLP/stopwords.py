import os
from typing import Set, List, Optional


class StopWords:
    DEFAULT_STOPWORDS_PATH = os.path.join(os.path.dirname(__file__), 'data', 'stopwords.txt')

    def __init__(self, load_default: bool = True):
        self._stopwords: Set[str] = set()
        if load_default:
            self._load_default_stopwords()

    def _load_default_stopwords(self) -> None:
        if os.path.exists(self.DEFAULT_STOPWORDS_PATH):
            self.load_stopwords(self.DEFAULT_STOPWORDS_PATH)

    def load_stopwords(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"停用词文件不存在: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip()
                if word:
                    self._stopwords.add(word)

    def add_stopword(self, word: str) -> None:
        self._stopwords.add(word)

    def remove_stopword(self, word: str) -> bool:
        if word in self._stopwords:
            self._stopwords.remove(word)
            return True
        return False

    def is_stopword(self, word: str) -> bool:
        return word in self._stopwords

    def filter(self, words: List[str]) -> List[str]:
        return [word for word in words if word not in self._stopwords]

    def filter_with_pos(self, words_with_pos: List[tuple]) -> List[tuple]:
        return [(word, pos) for word, pos in words_with_pos if word not in self._stopwords]

    def get_stopwords(self) -> Set[str]:
        return self._stopwords.copy()

    def __len__(self) -> int:
        return len(self._stopwords)

    def __contains__(self, word: str) -> bool:
        return word in self._stopwords

    def save_stopwords(self, path: str) -> None:
        with open(path, 'w', encoding='utf-8') as f:
            for word in sorted(self._stopwords):
                f.write(f"{word}\n")


__all__ = ['StopWords']
