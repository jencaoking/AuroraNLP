from typing import List, Tuple, Optional, Set
from .tokenizer import (
    forward_max_match,
    backward_max_match,
    bidirectional_max_match,
    forward_max_match_with_pos,
    backward_max_match_with_pos,
    bidirectional_max_match_with_pos
)
from .dictionary import Dictionary
from .stopwords import StopWords
from .keyword_extractor import KeywordExtractor
from .similarity import Similarity
from .hmm import HMMSegmentor


POS_TAG_NAMES = {
    'n': '名词',
    'v': '动词',
    'a': '形容词',
    'd': '副词',
    't': '时间词',
    'r': '代词',
    'p': '介词',
    'c': '连词',
    'm': '数词',
    'q': '量词',
    'u': '助词',
    'w': '标点',
    'nr': '人名',
    'ns': '地名',
    'nt': '机构名',
    'nz': '其他专名',
    'vn': '动名词',
    'x': '未知',
}

NER_TAG_MAP = {
    'nr': 'PERSON',
    'ns': 'LOCATION',
    'nt': 'ORGANIZATION',
}


class Segmentor:
    def __init__(
        self,
        dictionary: Optional[Dictionary] = None,
        load_default_dict: bool = True,
        load_default_stopwords: bool = True,
        use_hmm: bool = False
    ):
        if dictionary is not None:
            self.dictionary = dictionary
        else:
            self.dictionary = Dictionary(load_default=load_default_dict)

        self.stopwords = StopWords(load_default=load_default_stopwords)
        self.keyword_extractor = KeywordExtractor()
        self.similarity = Similarity()
        self.mode: str = 'bidirectional'
        
        self.use_hmm = use_hmm
        self.hmm_segmentor = HMMSegmentor()

    def segment(self, text: str, mode: Optional[str] = None) -> List[str]:
        if mode is None:
            mode = self.mode

        if mode == 'hmm' or self.use_hmm:
            if not self.hmm_segmentor.is_trained():
                raise RuntimeError("HMM model has not been trained. Call train_hmm() or load_hmm_model() first.")
            return self.hmm_segmentor.segment(text)
        elif mode == 'forward':
            return forward_max_match(text, self.dictionary)
        elif mode == 'backward':
            return backward_max_match(text, self.dictionary)
        elif mode == 'bidirectional':
            return bidirectional_max_match(text, self.dictionary)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def segment_without_stopwords(self, text: str, mode: Optional[str] = None) -> List[str]:
        words = self.segment(text, mode)
        return self.stopwords.filter(words)

    def segment_with_pos(self, text: str, mode: Optional[str] = None) -> List[Tuple[str, str]]:
        if mode is None:
            mode = self.mode

        if mode == 'forward':
            return forward_max_match_with_pos(text, self.dictionary)
        elif mode == 'backward':
            return backward_max_match_with_pos(text, self.dictionary)
        elif mode == 'bidirectional':
            return bidirectional_max_match_with_pos(text, self.dictionary)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def segment_with_pos_without_stopwords(self, text: str, mode: Optional[str] = None) -> List[Tuple[str, str]]:
        words_with_pos = self.segment_with_pos(text, mode)
        return self.stopwords.filter_with_pos(words_with_pos)

    def recognize_entities(self, text: str, mode: Optional[str] = None) -> List[Tuple[str, str]]:
        pos_result = self.segment_with_pos(text, mode)
        entities = []

        for word, pos_tag in pos_result:
            if pos_tag in NER_TAG_MAP:
                entities.append((word, NER_TAG_MAP[pos_tag]))

        return entities

    def segment_with_entities(self, text: str, mode: Optional[str] = None) -> List[Tuple[str, str, str]]:
        pos_result = self.segment_with_pos(text, mode)
        result = []

        for word, pos_tag in pos_result:
            entity_type = NER_TAG_MAP.get(pos_tag, 'O')
            result.append((word, pos_tag, entity_type))

        return result

    def extract_keywords(
        self,
        text: str,
        top_k: int = 10,
        method: str = 'tfidf',
        use_stopwords: bool = True,
        min_length: int = 1
    ) -> List[Tuple[str, float]]:
        stopwords = self.stopwords.get_stopwords() if use_stopwords else None

        if method == 'tfidf':
            return self.keyword_extractor.extract_keywords_tfidf(
                text, self, top_k, stopwords
            )
        elif method == 'freq':
            result = self.keyword_extractor.extract_keywords_freq(
                text, self, top_k, stopwords, min_length
            )
            return [(word, float(count)) for word, count in result]
        elif method == 'textrank':
            return self.keyword_extractor.extract_keywords_textrank(
                text, self, top_k, stopwords=stopwords
            )
        else:
            raise ValueError(f"Unknown method: {method}. Use 'tfidf', 'freq', or 'textrank'.")

    def build_keyword_corpus(self, documents: List[str]) -> None:
        self.keyword_extractor.build_idf_corpus(documents, self)

    def compute_similarity(
        self,
        text1: str,
        text2: str,
        method: str = 'cosine',
        use_stopwords: bool = True
    ) -> float:
        stopwords = self.stopwords.get_stopwords() if use_stopwords else None

        method_func = {
            'cosine': self.similarity.cosine_similarity,
            'jaccard': self.similarity.jaccard_similarity,
            'dice': self.similarity.dice_similarity,
            'overlap': self.similarity.overlap_similarity,
            'edit': self.similarity.edit_similarity
        }

        if method not in method_func:
            raise ValueError(f"Unknown method: {method}. Use 'cosine', 'jaccard', 'dice', 'overlap', or 'edit'.")

        if method == 'edit':
            return self.similarity.edit_similarity(text1, text2)

        return method_func[method](text1, text2, self, stopwords)

    def build_similarity_corpus(self, documents: List[str]) -> None:
        self.similarity.build_idf_corpus(documents, self)

    def batch_similarity(
        self,
        query: str,
        documents: List[str],
        method: str = 'cosine',
        use_stopwords: bool = True
    ) -> List[Tuple[str, float]]:
        stopwords = self.stopwords.get_stopwords() if use_stopwords else None
        return self.similarity.batch_similarity(query, documents, self, method, stopwords)

    def set_mode(self, mode: str) -> None:
        if mode not in ('forward', 'backward', 'bidirectional'):
            raise ValueError(f"Unknown mode: {mode}")
        self.mode = mode

    def load_dictionary(self, path: str) -> None:
        self.dictionary.load_dictionary(path)

    def save_dictionary(self, path: str) -> None:
        self.dictionary.save_dictionary(path)

    def add_word(self, word: str, pos_tag: Optional[str] = None) -> None:
        self.dictionary.add_word(word, pos_tag)

    def remove_word(self, word: str) -> bool:
        return self.dictionary.remove_word(word)

    def get_dictionary_size(self) -> int:
        return len(self.dictionary)

    def add_stopword(self, word: str) -> None:
        self.stopwords.add_stopword(word)

    def remove_stopword(self, word: str) -> bool:
        return self.stopwords.remove_stopword(word)

    def is_stopword(self, word: str) -> bool:
        return word in self.stopwords

    def get_stopwords(self) -> Set[str]:
        return self.stopwords.get_stopwords()

    def load_stopwords(self, path: str) -> None:
        self.stopwords.load_stopwords(path)

    def save_stopwords(self, path: str) -> None:
        self.stopwords.save_stopwords(path)

    def get_pos_tag_name(self, pos_tag: str) -> str:
        return POS_TAG_NAMES.get(pos_tag, '未知')

    def get_pos_tags(self) -> dict:
        return POS_TAG_NAMES.copy()
    
    def train_hmm(self, corpus: List[List[str]], smooth: float = 1.0) -> None:
        self.hmm_segmentor.train(corpus, smooth)
    
    def train_hmm_from_file(self, filepath: str, encoding: str = 'utf-8') -> None:
        from .hmm import train_from_file
        train_from_file(self.hmm_segmentor, filepath, encoding)
    
    def load_hmm_model(self, filepath: str) -> None:
        self.hmm_segmentor.load_model(filepath)
    
    def save_hmm_model(self, filepath: str) -> None:
        self.hmm_segmentor.save_model(filepath)
    
    def segment_with_hmm_states(self, text: str) -> List[Tuple[str, str]]:
        if not self.hmm_segmentor.is_trained():
            raise RuntimeError("HMM model has not been trained. Call train_hmm() or load_hmm_model() first.")
        return self.hmm_segmentor.segment_with_states(text)
    
    def get_hmm_model_info(self) -> dict:
        return self.hmm_segmentor.get_model_info()
    
    def is_hmm_trained(self) -> bool:
        return self.hmm_segmentor.is_trained()
    
    def set_use_hmm(self, use_hmm: bool) -> None:
        self.use_hmm = use_hmm
