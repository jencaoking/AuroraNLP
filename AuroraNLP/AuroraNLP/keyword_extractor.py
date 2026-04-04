import math
from typing import List, Tuple, Dict, Optional, Set
from collections import Counter


class KeywordExtractor:
    def __init__(self):
        self._idf_cache: Dict[str, float] = {}
        self._document_count: int = 0
        self._document_freq: Dict[str, int] = {}

    def _tokenize(self, text: str, segmentor) -> List[str]:
        return segmentor.segment(text)

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

    def extract_keywords_tfidf(
        self,
        text: str,
        segmentor,
        top_k: int = 10,
        stopwords: Optional[Set[str]] = None
    ) -> List[Tuple[str, float]]:
        words = self._tokenize(text, segmentor)

        if stopwords:
            words = [w for w in words if w not in stopwords]

        if not words:
            return []

        tf = self._compute_tf(words)

        tfidf_scores = {}
        for word, tf_score in tf.items():
            idf_score = self._compute_idf(word)
            tfidf_scores[word] = tf_score * idf_score

        sorted_keywords = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)

        return sorted_keywords[:top_k]

    def extract_keywords_freq(
        self,
        text: str,
        segmentor,
        top_k: int = 10,
        stopwords: Optional[Set[str]] = None,
        min_length: int = 1
    ) -> List[Tuple[str, int]]:
        words = self._tokenize(text, segmentor)

        if stopwords:
            words = [w for w in words if w not in stopwords]

        if min_length > 1:
            words = [w for w in words if len(w) >= min_length]

        word_count = Counter(words)

        return word_count.most_common(top_k)

    def extract_keywords_textrank(
        self,
        text: str,
        segmentor,
        top_k: int = 10,
        window_size: int = 4,
        damping: float = 0.85,
        max_iter: int = 100,
        stopwords: Optional[Set[str]] = None
    ) -> List[Tuple[str, float]]:
        words = self._tokenize(text, segmentor)

        if stopwords:
            words = [w for w in words if w not in stopwords]

        if len(words) < 2:
            return [(w, 1.0) for w in words[:top_k]]

        word_set = list(set(words))
        word_to_idx = {word: idx for idx, word in enumerate(word_set)}
        n = len(word_set)

        graph = [[0.0] * n for _ in range(n)]

        for i in range(len(words)):
            for j in range(i + 1, min(i + window_size, len(words))):
                if words[i] != words[j]:
                    idx1 = word_to_idx[words[i]]
                    idx2 = word_to_idx[words[j]]
                    graph[idx1][idx2] += 1.0
                    graph[idx2][idx1] += 1.0

        out_weights = [sum(graph[i]) for i in range(n)]

        scores = [1.0] * n

        for _ in range(max_iter):
            new_scores = []
            for i in range(n):
                score = 0.0
                for j in range(n):
                    if graph[j][i] > 0 and out_weights[j] > 0:
                        score += graph[j][i] / out_weights[j] * scores[j]
                new_scores.append((1 - damping) + damping * score)

            scores = new_scores

        word_scores = [(word_set[i], scores[i]) for i in range(n)]
        word_scores.sort(key=lambda x: x[1], reverse=True)

        return word_scores[:top_k]

    def get_document_count(self) -> int:
        return self._document_count

    def get_vocabulary_size(self) -> int:
        return len(self._document_freq)


__all__ = ['KeywordExtractor']
