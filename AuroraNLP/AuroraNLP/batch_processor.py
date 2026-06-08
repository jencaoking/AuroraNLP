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
        mode: Optional[str] = None,
        overlap: int = 50
    ) -> List[str]:
        """对长文本分词，避免跨越 chunk 边界的词被错误切分。

        切分策略：
        1. 在 ``[chunk_end - overlap, chunk_end]`` 内从后向前搜索标点/空白等
           安全切分点；找到则在该点切分（不产生重叠）。
        2. 窗口内未找到安全切分点时，回退到真实 overlap 机制：在 ``chunk_end``
           处切分，下一 chunk 从 ``chunk_end - overlap`` 开始使边界处被切断的
           token 在下一 chunk 中被完整重新分词，同时丢弃当前 chunk 末尾位于
           overlap 区域的 token 以避免重复。
        """
        if len(text) <= chunk_size:
            return self.segmentor.segment(text, mode)

        BREAK_CHARS = frozenset('，。！？；：、\n\t\r ')
        results: List[str] = []
        i = 0
        text_len = len(text)

        while i < text_len:
            remaining = text_len - i
            if remaining <= chunk_size:
                results.extend(self.segmentor.segment(text[i:], mode))
                break

            chunk_end = i + chunk_size
            # 在 [max(i + 1, chunk_end - overlap), chunk_end] 内从后向前搜索安全切分点
            break_point = -1
            search_start = max(i + 1, chunk_end - overlap)
            for j in range(chunk_end - 1, search_start - 1, -1):
                if text[j] in BREAK_CHARS:
                    break_point = j + 1
                    break

            if break_point != -1:
                # 找到安全切分点：直接切分，无重叠无重复
                results.extend(self.segmentor.segment(text[i:break_point], mode))
                i = break_point
            else:
                # 未找到安全切分点：使用真实 overlap 机制
                chunk_tokens = self.segmentor.segment(text[i:chunk_end], mode)
                if overlap > 0 and chunk_tokens:
                    # 丢弃 chunk 末尾位于 overlap 区域的 token（将在下一 chunk 重分）
                    overlap_offset_in_chunk = chunk_size - overlap
                    consumed = 0
                    cutoff = len(chunk_tokens)
                    for idx, tok in enumerate(chunk_tokens):
                        if consumed >= overlap_offset_in_chunk:
                            cutoff = idx
                            break
                        consumed += len(tok)
                    results.extend(chunk_tokens[:cutoff])
                    # 下一 chunk 从 chunk_end - overlap 开始，让边界 token 被完整重分
                    i = chunk_end - overlap
                else:
                    results.extend(chunk_tokens)
                    i = chunk_end

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
