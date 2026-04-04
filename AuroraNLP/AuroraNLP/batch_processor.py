from typing import List, Tuple, Optional, Iterator
from .segmentor import Segmentor


class BatchProcessor:
    def __init__(self, segmentor: Segmentor, batch_size: int = 100):
        self.segmentor = segmentor
        self.batch_size = batch_size

    def segment_batch(
        self,
        texts: List[str],
        mode: Optional[str] = None
    ) -> List[List[str]]:
        return [self.segmentor.segment(text, mode) for text in texts]

    def segment_batch_iter(
        self,
        texts: List[str],
        mode: Optional[str] = None
    ) -> Iterator[List[str]]:
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            for text in batch:
                yield self.segmentor.segment(text, mode)

    def segment_with_pos_batch(
        self,
        texts: List[str],
        mode: Optional[str] = None
    ) -> List[List[Tuple[str, str]]]:
        return [self.segmentor.segment_with_pos(text, mode) for text in texts]

    def segment_without_stopwords_batch(
        self,
        texts: List[str],
        mode: Optional[str] = None
    ) -> List[List[str]]:
        return [self.segmentor.segment_without_stopwords(text, mode) for text in texts]

    def extract_keywords_batch(
        self,
        texts: List[str],
        top_k: int = 10,
        method: str = 'tfidf',
        use_stopwords: bool = True
    ) -> List[List[Tuple[str, float]]]:
        return [
            self.segmentor.extract_keywords(text, top_k, method, use_stopwords)
            for text in texts
        ]

    def compute_similarity_matrix(
        self,
        texts: List[str],
        method: str = 'cosine',
        use_stopwords: bool = True
    ) -> List[List[float]]:
        n = len(texts)
        matrix = [[0.0] * n for _ in range(n)]

        for i in range(n):
            for j in range(i, n):
                if i == j:
                    matrix[i][j] = 1.0
                else:
                    sim = self.segmentor.compute_similarity(
                        texts[i], texts[j], method, use_stopwords
                    )
                    matrix[i][j] = sim
                    matrix[j][i] = sim

        return matrix

    def find_similar_texts(
        self,
        query: str,
        texts: List[str],
        threshold: float = 0.5,
        method: str = 'cosine',
        use_stopwords: bool = True
    ) -> List[Tuple[int, str, float]]:
        results = []
        similarities = self.segmentor.batch_similarity(
            query, texts, method, use_stopwords
        )

        for i, (text, score) in enumerate(similarities):
            if score >= threshold:
                results.append((i, text, score))

        return results

    def segment_large_text(
        self,
        text: str,
        chunk_size: int = 10000,
        mode: Optional[str] = None
    ) -> List[str]:
        if len(text) <= chunk_size:
            return self.segmentor.segment(text, mode)

        results = []
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i + chunk_size]
            results.extend(self.segmentor.segment(chunk, mode))

        return results

    def segment_file(
        self,
        file_path: str,
        encoding: str = 'utf-8',
        mode: Optional[str] = None,
        line_by_line: bool = True
    ) -> List[List[str]]:
        results = []

        with open(file_path, 'r', encoding=encoding) as f:
            if line_by_line:
                for line in f:
                    line = line.strip()
                    if line:
                        results.append(self.segmentor.segment(line, mode))
            else:
                content = f.read()
                results.append(self.segmentor.segment(content, mode))

        return results


__all__ = ['BatchProcessor']
