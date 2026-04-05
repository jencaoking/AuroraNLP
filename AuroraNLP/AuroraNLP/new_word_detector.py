from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
import math
import re


class NewWordDetector:
    def __init__(
        self,
        min_freq: int = 5,
        min_pmi: float = 1.0,
        min_entropy: float = 0.5,
        min_word_len: int = 2,
        max_word_len: int = 6
    ):
        self.min_freq = min_freq
        self.min_pmi = min_pmi
        self.min_entropy = min_entropy
        self.min_word_len = min_word_len
        self.max_word_len = max_word_len

        self._char_freq: Dict[str, int] = defaultdict(int)
        self._word_freq: Dict[str, int] = defaultdict(int)
        self._left_context: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._right_context: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._total_chars = 0
        self._total_candidates = 0

        self._pmi_cache: Dict[str, float] = {}
        self._left_entropy_cache: Dict[str, float] = {}
        self._right_entropy_cache: Dict[str, float] = {}

        self._trained = False
        self._stop_chars: Set[str] = set('，。！？、；：""''（）【】《》\n\r\t ')
        self._stop_patterns = None

    def _is_valid_word(self, word: str) -> bool:
        if not word:
            return False
        if len(word) < self.min_word_len or len(word) > self.max_word_len:
            return False
        for char in word:
            if char in self._stop_chars:
                return False
            if char.isdigit():
                return False
        return True

    def _extract_candidates(self, text: str) -> List[str]:
        candidates = []
        text_len = len(text)

        for i in range(text_len):
            for length in range(self.min_word_len, min(self.max_word_len + 1, text_len - i + 1)):
                candidate = text[i:i + length]
                if self._is_valid_word(candidate):
                    candidates.append(candidate)

        return candidates

    def train(self, corpus: List[str]) -> None:
        self._char_freq.clear()
        self._word_freq.clear()
        self._left_context.clear()
        self._right_context.clear()
        self._total_chars = 0
        self._total_candidates = 0
        self._pmi_cache.clear()
        self._left_entropy_cache.clear()
        self._right_entropy_cache.clear()

        for text in corpus:
            if not text:
                continue

            for char in text:
                if char not in self._stop_chars:
                    self._char_freq[char] += 1
                    self._total_chars += 1

            candidates = self._extract_candidates(text)
            for candidate in candidates:
                self._word_freq[candidate] += 1
                self._total_candidates += 1

            for i, char in enumerate(text):
                if i > 0 and char not in self._stop_chars:
                    left_char = text[i - 1]
                    if left_char not in self._stop_chars:
                        self._left_context[char][left_char] += 1

                if i < len(text) - 1 and char not in self._stop_chars:
                    right_char = text[i + 1]
                    if right_char not in self._stop_chars:
                        self._right_context[char][right_char] += 1

        self._trained = True

    def train_from_file(self, filepath: str, encoding: str = 'utf-8') -> None:
        corpus = []
        with open(filepath, 'r', encoding=encoding) as f:
            for line in f:
                line = line.strip()
                if line:
                    corpus.append(line)
        self.train(corpus)

    def _calculate_pmi(self, word: str) -> float:
        if word in self._pmi_cache:
            return self._pmi_cache[word]

        if not word or len(word) < 2:
            return 0.0

        word_freq = self._word_freq.get(word, 0)
        if word_freq == 0:
            return float('-inf')

        if self._total_candidates == 0:
            return float('-inf')

        p_word = word_freq / self._total_candidates
        if p_word == 0:
            return float('-inf')

        if self._total_chars == 0:
            return float('inf')

        p_chars = 1.0
        for char in word:
            char_freq = self._char_freq.get(char, 0)
            if char_freq == 0:
                return float('inf')
            p_char = char_freq / self._total_chars
            p_chars *= p_char

        if p_chars == 0:
            return float('inf')

        pmi = math.log(p_word / p_chars)

        self._pmi_cache[word] = pmi
        return pmi

    def _calculate_entropy(self, context_dict: Dict[str, int]) -> float:
        if not context_dict:
            return 0.0

        total = sum(context_dict.values())
        if total == 0:
            return 0.0

        entropy = 0.0
        for count in context_dict.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log(p)

        return entropy

    def _calculate_left_entropy(self, word: str) -> float:
        if word in self._left_entropy_cache:
            return self._left_entropy_cache[word]

        if not word:
            return 0.0

        left_context = self._left_context.get(word, {})
        entropy = self._calculate_entropy(left_context)

        self._left_entropy_cache[word] = entropy
        return entropy

    def _calculate_right_entropy(self, word: str) -> float:
        if word in self._right_entropy_cache:
            return self._right_entropy_cache[word]

        if not word:
            return 0.0

        right_context = self._right_context.get(word, {})
        entropy = self._calculate_entropy(right_context)

        self._right_entropy_cache[word] = entropy
        return entropy

    def get_pmi(self, word: str) -> float:
        if not self._trained:
            raise RuntimeError("Model has not been trained. Call train() first.")
        return self._calculate_pmi(word)

    def get_left_entropy(self, word: str) -> float:
        if not self._trained:
            raise RuntimeError("Model has not been trained. Call train() first.")
        return self._calculate_left_entropy(word)

    def get_right_entropy(self, word: str) -> float:
        if not self._trained:
            raise RuntimeError("Model has not been trained. Call train() first.")
        return self._calculate_right_entropy(word)

    def get_word_score(self, word: str) -> Dict[str, float]:
        if not self._trained:
            raise RuntimeError("Model has not been trained. Call train() first.")

        freq = self._word_freq.get(word, 0)
        pmi = self._calculate_pmi(word)
        left_entropy = self._calculate_left_entropy(word)
        right_entropy = self._calculate_right_entropy(word)

        avg_entropy = (left_entropy + right_entropy) / 2

        return {
            'word': word,
            'frequency': freq,
            'pmi': pmi,
            'left_entropy': left_entropy,
            'right_entropy': right_entropy,
            'avg_entropy': avg_entropy
        }

    def detect(
        self,
        top_k: int = 100,
        min_freq: Optional[int] = None,
        min_pmi: Optional[float] = None,
        min_entropy: Optional[float] = None
    ) -> List[Tuple[str, Dict[str, float]]]:
        if not self._trained:
            raise RuntimeError("Model has not been trained. Call train() first.")

        min_freq = min_freq if min_freq is not None else self.min_freq
        min_pmi = min_pmi if min_pmi is not None else self.min_pmi
        min_entropy = min_entropy if min_entropy is not None else self.min_entropy

        candidates = []

        for word, freq in self._word_freq.items():
            if freq < min_freq:
                continue

            if not self._is_valid_word(word):
                continue

            pmi = self._calculate_pmi(word)
            if pmi < min_pmi:
                continue

            left_entropy = self._calculate_left_entropy(word)
            right_entropy = self._calculate_right_entropy(word)
            avg_entropy = (left_entropy + right_entropy) / 2

            if avg_entropy < min_entropy:
                continue

            score_info = {
                'frequency': freq,
                'pmi': pmi,
                'left_entropy': left_entropy,
                'right_entropy': right_entropy,
                'avg_entropy': avg_entropy
            }

            candidates.append((word, score_info))

        candidates.sort(key=lambda x: (x[1]['pmi'], x[1]['avg_entropy'], x[1]['frequency']), reverse=True)

        return candidates[:top_k]

    def detect_from_text(
        self,
        text: str,
        top_k: int = 20,
        min_freq: Optional[int] = None,
        min_pmi: Optional[float] = None,
        min_entropy: Optional[float] = None
    ) -> List[Tuple[str, Dict[str, float]]]:
        if not self._trained:
            raise RuntimeError("Model has not been trained. Call train() first.")

        candidates = self._extract_candidates(text)
        unique_candidates = set(candidates)

        results = []
        for word in unique_candidates:
            freq = self._word_freq.get(word, 0)
            if freq == 0:
                continue

            min_freq_val = min_freq if min_freq is not None else self.min_freq
            min_pmi_val = min_pmi if min_pmi is not None else self.min_pmi
            min_entropy_val = min_entropy if min_entropy is not None else self.min_entropy

            if freq < min_freq_val:
                continue

            pmi = self._calculate_pmi(word)
            if pmi < min_pmi_val:
                continue

            left_entropy = self._calculate_left_entropy(word)
            right_entropy = self._calculate_right_entropy(word)
            avg_entropy = (left_entropy + right_entropy) / 2

            if avg_entropy < min_entropy_val:
                continue

            score_info = {
                'frequency': freq,
                'pmi': pmi,
                'left_entropy': left_entropy,
                'right_entropy': right_entropy,
                'avg_entropy': avg_entropy
            }

            results.append((word, score_info))

        results.sort(key=lambda x: (x[1]['pmi'], x[1]['avg_entropy'], x[1]['frequency']), reverse=True)

        return results[:top_k]

    def get_new_words(
        self,
        existing_words: Set[str],
        top_k: int = 100,
        min_freq: Optional[int] = None,
        min_pmi: Optional[float] = None,
        min_entropy: Optional[float] = None
    ) -> List[Tuple[str, Dict[str, float]]]:
        all_candidates = self.detect(top_k * 2, min_freq, min_pmi, min_entropy)

        new_words = []
        for word, info in all_candidates:
            if word not in existing_words:
                new_words.append((word, info))
                if len(new_words) >= top_k:
                    break

        return new_words

    def auto_extend_dictionary(
        self,
        dictionary,
        top_k: int = 50,
        min_freq: Optional[int] = None,
        min_pmi: Optional[float] = None,
        min_entropy: Optional[float] = None,
        pos_tag: Optional[str] = None,
        weight: float = 1.0
    ) -> List[Tuple[str, Dict[str, float]]]:
        if not self._trained:
            raise RuntimeError("Model has not been trained. Call train() first.")

        if not (hasattr(dictionary, 'get_words') and hasattr(dictionary, 'add_word')):
            raise TypeError("dictionary must have both 'get_words' and 'add_word' methods")

        existing_words = dictionary.get_words()

        new_words = self.get_new_words(
            existing_words,
            top_k=top_k,
            min_freq=min_freq,
            min_pmi=min_pmi,
            min_entropy=min_entropy
        )

        added_words = []
        for word, info in new_words:
            dictionary.add_word(word, pos_tag, weight)
            added_words.append((word, info))

        return added_words

    def get_statistics(self) -> Dict:
        if not self._trained:
            return {'trained': False}

        return {
            'trained': True,
            'total_chars': self._total_chars,
            'total_candidates': self._total_candidates,
            'unique_chars': len(self._char_freq),
            'unique_candidates': len(self._word_freq),
            'min_freq': self.min_freq,
            'min_pmi': self.min_pmi,
            'min_entropy': self.min_entropy,
            'min_word_len': self.min_word_len,
            'max_word_len': self.max_word_len
        }

    def set_thresholds(
        self,
        min_freq: Optional[int] = None,
        min_pmi: Optional[float] = None,
        min_entropy: Optional[float] = None
    ) -> None:
        if min_freq is not None:
            self.min_freq = min_freq
        if min_pmi is not None:
            self.min_pmi = min_pmi
        if min_entropy is not None:
            self.min_entropy = min_entropy

    def set_word_length_range(self, min_len: int, max_len: int) -> None:
        if min_len < 1:
            raise ValueError("min_len must be at least 1")
        if max_len < min_len:
            raise ValueError("max_len must be >= min_len")

        self.min_word_len = min_len
        self.max_word_len = max_len

    def add_stop_chars(self, chars: str) -> None:
        for char in chars:
            self._stop_chars.add(char)

    def remove_stop_chars(self, chars: str) -> None:
        for char in chars:
            self._stop_chars.discard(char)

    def get_stop_chars(self) -> Set[str]:
        return self._stop_chars.copy()

    def is_trained(self) -> bool:
        return self._trained

    def get_word_frequency(self, word: str) -> int:
        return self._word_freq.get(word, 0)

    def get_char_frequency(self, char: str) -> int:
        return self._char_freq.get(char, 0)

    def get_all_candidates(self) -> Dict[str, int]:
        return dict(self._word_freq)

    def clear_cache(self) -> None:
        self._pmi_cache.clear()
        self._left_entropy_cache.clear()
        self._right_entropy_cache.clear()


class MutualInformation:
    def __init__(self):
        self._bigram_freq: Dict[Tuple[str, str], int] = defaultdict(int)
        self._unigram_freq: Dict[str, int] = defaultdict(int)
        self._total_bigrams = 0
        self._total_unigrams = 0
        self._trained = False

    def train(self, corpus: List[str], window_size: int = 1) -> None:
        self._bigram_freq.clear()
        self._unigram_freq.clear()
        self._total_bigrams = 0
        self._total_unigrams = 0

        for text in corpus:
            if not text:
                continue

            chars = [c for c in text if c.strip()]

            for i, char in enumerate(chars):
                self._unigram_freq[char] += 1
                self._total_unigrams += 1

                for j in range(1, window_size + 1):
                    if i + j < len(chars):
                        bigram = (char, chars[i + j])
                        self._bigram_freq[bigram] += 1
                        self._total_bigrams += 1

        self._trained = True

    def calculate_pmi(self, char1: str, char2: str) -> float:
        if not self._trained:
            raise RuntimeError("Model has not been trained. Call train() first.")

        bigram = (char1, char2)
        bigram_count = self._bigram_freq.get(bigram, 0)

        if bigram_count == 0:
            return float('-inf')

        p_xy = bigram_count / self._total_bigrams if self._total_bigrams > 0 else 0

        count_x = self._unigram_freq.get(char1, 0)
        count_y = self._unigram_freq.get(char2, 0)

        if count_x == 0 or count_y == 0:
            return float('-inf')

        p_x = count_x / self._total_unigrams
        p_y = count_y / self._total_unigrams

        if p_xy == 0 or p_x == 0 or p_y == 0:
            return float('-inf')

        return math.log(p_xy / (p_x * p_y))

    def calculate_word_pmi(self, word: str) -> float:
        if not self._trained:
            raise RuntimeError("Model has not been trained. Call train() first.")

        if len(word) < 2:
            return 0.0

        total_pmi = 0.0
        count = 0

        for i in range(len(word) - 1):
            pmi = self.calculate_pmi(word[i], word[i + 1])
            if pmi != float('-inf'):
                total_pmi += pmi
                count += 1

        if count == 0:
            return float('-inf')

        return total_pmi / count

    def get_top_collocations(
        self,
        min_freq: int = 5,
        min_pmi: float = 0.0,
        top_k: int = 20
    ) -> List[Tuple[str, str, int, float]]:
        if not self._trained:
            raise RuntimeError("Model has not been trained. Call train() first.")

        results = []

        for (char1, char2), freq in self._bigram_freq.items():
            if freq < min_freq:
                continue

            pmi = self.calculate_pmi(char1, char2)
            if pmi >= min_pmi:
                results.append((char1, char2, freq, pmi))

        results.sort(key=lambda x: x[3], reverse=True)
        return results[:top_k]

    def is_trained(self) -> bool:
        return self._trained


class EntropyCalculator:
    def __init__(self):
        self._left_context: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._right_context: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._trained = False

    def train(self, corpus: List[str]) -> None:
        self._left_context.clear()
        self._right_context.clear()

        for text in corpus:
            if not text:
                continue

            chars = [c for c in text if c.strip()]

            for i, char in enumerate(chars):
                if i > 0:
                    left_char = chars[i - 1]
                    self._left_context[char][left_char] += 1

                if i < len(chars) - 1:
                    right_char = chars[i + 1]
                    self._right_context[char][right_char] += 1

        self._trained = True

    def calculate_entropy(self, context_dict: Dict[str, int]) -> float:
        if not context_dict:
            return 0.0

        total = sum(context_dict.values())
        if total == 0:
            return 0.0

        entropy = 0.0
        for count in context_dict.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log(p)

        return entropy

    def get_left_entropy(self, word: str) -> float:
        if not self._trained:
            raise RuntimeError("Model has not been trained. Call train() first.")

        context = self._left_context.get(word, {})
        return self.calculate_entropy(context)

    def get_right_entropy(self, word: str) -> float:
        if not self._trained:
            raise RuntimeError("Model has not been trained. Call train() first.")

        context = self._right_context.get(word, {})
        return self.calculate_entropy(context)

    def get_avg_entropy(self, word: str) -> float:
        left_entropy = self.get_left_entropy(word)
        right_entropy = self.get_right_entropy(word)
        return (left_entropy + right_entropy) / 2

    def is_trained(self) -> bool:
        return self._trained


__all__ = ['NewWordDetector', 'MutualInformation', 'EntropyCalculator']
