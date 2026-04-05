import math
from typing import List, Dict, Set, Optional, Tuple
from collections import Counter


class Similarity:
    def __init__(self):
        self._idf_cache: Dict[str, float] = {}
        self._document_count: int = 0
        self._document_freq: Dict[str, int] = {}

    def _tokenize(self, text: str, segmentor) -> List[str]:
        return segmentor.segment(text)

    def _build_vocab(self, words_list: List[List[str]]) -> Set[str]:
        vocab = set()
        for words in words_list:
            vocab.update(words)
        return vocab

    def _compute_tf(self, words: List[str]) -> Dict[str, float]:
        word_count = Counter(words)
        total = len(words)
        if total == 0:
            return {}
        return {word: count / total for word, count in word_count.items()}

    def _compute_idf(self, word: str) -> float:
        if word in self._idf_cache:
            return self._idf_cache[word]

        doc_freq = self._document_freq.get(word, 0)
        if doc_freq == 0:
            idf = 0.0
        else:
            idf = math.log((self._document_count + 1) / (doc_freq + 1)) + 1

        self._idf_cache[word] = idf
        return idf

    def build_idf_corpus(self, documents: List[str], segmentor) -> None:
        self._document_count = len(documents)
        self._document_freq = {}
        self._idf_cache = {}

        for doc in documents:
            words = set(self._tokenize(doc, segmentor))
            for word in words:
                self._document_freq[word] = self._document_freq.get(word, 0) + 1

    def _get_tfidf_vector(
        self,
        words: List[str],
        vocab: Set[str]
    ) -> Dict[str, float]:
        tf = self._compute_tf(words)
        vector = {}
        for word in vocab:
            tf_score = tf.get(word, 0.0)
            idf_score = self._compute_idf(word)
            vector[word] = tf_score * idf_score
        return vector

    def cosine_similarity(
        self,
        text1: str,
        text2: str,
        segmentor,
        stopwords: Optional[Set[str]] = None
    ) -> float:
        words1 = self._tokenize(text1, segmentor)
        words2 = self._tokenize(text2, segmentor)

        if stopwords:
            words1 = [w for w in words1 if w not in stopwords]
            words2 = [w for w in words2 if w not in stopwords]

        if not words1 or not words2:
            return 0.0

        vocab = self._build_vocab([words1, words2])

        vec1 = self._get_tfidf_vector(words1, vocab)
        vec2 = self._get_tfidf_vector(words2, vocab)

        dot_product = sum(vec1.get(w, 0.0) * vec2.get(w, 0.0) for w in vocab)

        norm1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
        norm2 = math.sqrt(sum(v ** 2 for v in vec2.values()))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def jaccard_similarity(
        self,
        text1: str,
        text2: str,
        segmentor,
        stopwords: Optional[Set[str]] = None
    ) -> float:
        words1 = self._tokenize(text1, segmentor)
        words2 = self._tokenize(text2, segmentor)

        if stopwords:
            words1 = [w for w in words1 if w not in stopwords]
            words2 = [w for w in words2 if w not in stopwords]

        if not words1 or not words2:
            return 0.0

        set1 = set(words1)
        set2 = set(words2)

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        if union == 0:
            return 0.0

        return intersection / union

    def dice_similarity(
        self,
        text1: str,
        text2: str,
        segmentor,
        stopwords: Optional[Set[str]] = None
    ) -> float:
        words1 = self._tokenize(text1, segmentor)
        words2 = self._tokenize(text2, segmentor)

        if stopwords:
            words1 = [w for w in words1 if w not in stopwords]
            words2 = [w for w in words2 if w not in stopwords]

        if not words1 or not words2:
            return 0.0

        set1 = set(words1)
        set2 = set(words2)

        intersection = len(set1 & set2)
        total = len(set1) + len(set2)

        if total == 0:
            return 0.0

        return 2 * intersection / total

    def overlap_similarity(
        self,
        text1: str,
        text2: str,
        segmentor,
        stopwords: Optional[Set[str]] = None
    ) -> float:
        words1 = self._tokenize(text1, segmentor)
        words2 = self._tokenize(text2, segmentor)

        if stopwords:
            words1 = [w for w in words1 if w not in stopwords]
            words2 = [w for w in words2 if w not in stopwords]

        if not words1 or not words2:
            return 0.0

        set1 = set(words1)
        set2 = set(words2)

        intersection = len(set1 & set2)
        min_size = min(len(set1), len(set2))

        if min_size == 0:
            return 0.0

        return intersection / min_size

    def edit_distance(
        self,
        text1: str,
        text2: str
    ) -> int:
        m, n = len(text1), len(text2)

        if m == 0:
            return n
        if n == 0:
            return m

        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i - 1] == text2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min(
                        dp[i - 1][j] + 1,
                        dp[i][j - 1] + 1,
                        dp[i - 1][j - 1] + 1
                    )

        return dp[m][n]

    def edit_similarity(
        self,
        text1: str,
        text2: str,
        segmentor=None,
        stopwords: Optional[Set[str]] = None
    ) -> float:
        distance = self.edit_distance(text1, text2)
        max_len = max(len(text1), len(text2))

        if max_len == 0:
            return 1.0

        return 1.0 - distance / max_len

    def batch_similarity(
        self,
        query: str,
        documents: List[str],
        segmentor,
        method: str = 'cosine',
        stopwords: Optional[Set[str]] = None
    ) -> List[Tuple[str, float]]:
        valid_methods = ['cosine', 'jaccard', 'dice', 'overlap', 'edit']
        if method not in valid_methods:
            raise ValueError(f"Unknown method: {method}. Use one of {valid_methods}.")
        
        method_func = {
            'cosine': self.cosine_similarity,
            'jaccard': self.jaccard_similarity,
            'dice': self.dice_similarity,
            'overlap': self.overlap_similarity,
            'edit': self.edit_similarity
        }[method]

        results = []
        for doc in documents:
            score = method_func(query, doc, segmentor, stopwords)
            results.append((doc, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results


__all__ = ['Similarity']
