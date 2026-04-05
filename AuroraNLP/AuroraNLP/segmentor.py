from typing import List, Tuple, Optional, Set, Dict, Any
from .tokenizer import (
    forward_max_match,
    backward_max_match,
    bidirectional_max_match,
    forward_max_match_with_pos,
    backward_max_match_with_pos,
    bidirectional_max_match_with_pos,
    forward_max_match_weighted,
    backward_max_match_weighted,
    bidirectional_max_match_weighted,
    forward_max_match_weighted_with_pos,
    backward_max_match_weighted_with_pos,
    bidirectional_max_match_weighted_with_pos
)
from .dictionary import Dictionary, UserDictionary, DictionaryManager
from .stopwords import StopWords
from .keyword_extractor import KeywordExtractor
from .similarity import Similarity
from .hmm import HMMSegmentor
from .crf import CRFSegmentor
from .perceptron import PerceptronSegmentor
from .lattice import LatticeSegmentor, Lattice
from .ambiguity import AmbiguityDetector, AmbiguityResult, AmbiguityType, AmbiguityRegion
from .new_word_detector import NewWordDetector
from .hybrid import (
    HybridSegmentor,
    HybridConfig,
    HybridStrategy,
    SegmenterResult,
    SegmenterType
)


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
        use_hmm: bool = False,
        use_crf: bool = False,
        use_perceptron: bool = False,
        use_lattice: bool = False,
        use_weighted: bool = False,
        use_hybrid: bool = False,
        hybrid_config: Optional[HybridConfig] = None
    ):
        self._dict_manager = DictionaryManager()

        if dictionary is not None:
            self.dictionary = dictionary
            self._dict_manager.register_dictionary(dictionary)
        else:
            self.dictionary = Dictionary(load_default=load_default_dict)
            self._dict_manager.register_dictionary(self.dictionary)

        self._user_dictionaries: Dict[str, UserDictionary] = {}

        self.stopwords = StopWords(load_default=load_default_stopwords)
        self.keyword_extractor = KeywordExtractor()
        self.similarity = Similarity()
        self.mode: str = 'bidirectional'
        self.use_weighted = use_weighted

        self.use_hmm = use_hmm
        self.hmm_segmentor = HMMSegmentor()

        self.use_crf = use_crf
        self.crf_segmentor = CRFSegmentor()

        self.use_perceptron = use_perceptron
        self.perceptron_segmentor = PerceptronSegmentor()

        self.use_lattice = use_lattice
        self.lattice_segmentor = LatticeSegmentor(self.dictionary)

        self.ambiguity_detector = AmbiguityDetector(self.dictionary)

        self.new_word_detector = NewWordDetector()
        
        self.use_hybrid = use_hybrid
        self._hybrid_config = hybrid_config
        self._hybrid_segmentor: Optional[HybridSegmentor] = None
        
        if use_hybrid:
            self._init_hybrid_segmentor()

    @property
    def dict_manager(self) -> DictionaryManager:
        return self._dict_manager

    def _get_active_dictionary(self):
        if self._user_dictionaries:
            return self._dict_manager
        return self.dictionary

    def create_user_dictionary(
        self,
        name: str,
        priority: int = 100,
        default_weight: float = 10.0
    ) -> UserDictionary:
        user_dict = UserDictionary(name=name, priority=priority)
        user_dict.default_weight = default_weight
        self._user_dictionaries[name] = user_dict
        self._dict_manager.register_user_dictionary(user_dict)
        return user_dict

    def get_user_dictionary(self, name: str) -> Optional[UserDictionary]:
        return self._user_dictionaries.get(name)

    def remove_user_dictionary(self, name: str) -> bool:
        if name in self._user_dictionaries:
            del self._user_dictionaries[name]
            self._dict_manager.unregister_dictionary(name)
            return True
        return False

    def load_user_dictionary(
        self,
        path: str,
        name: Optional[str] = None,
        priority: int = 100,
        default_weight: float = 10.0
    ) -> UserDictionary:
        if name is None:
            name = f"user_{len(self._user_dictionaries)}"

        user_dict = self.create_user_dictionary(name, priority, default_weight)
        user_dict.load_dictionary(path, priority=priority, default_weight=default_weight)
        return user_dict

    def add_user_word(
        self,
        word: str,
        pos_tag: Optional[str] = None,
        weight: Optional[float] = None,
        priority: Optional[int] = None,
        dict_name: Optional[str] = None
    ) -> None:
        if dict_name:
            user_dict = self._user_dictionaries.get(dict_name)
            if user_dict:
                user_dict.add_word(word, pos_tag, weight, priority)
                self._dict_manager.invalidate_cache()
                return

        if self._user_dictionaries:
            default_dict = list(self._user_dictionaries.values())[0]
            default_dict.add_word(word, pos_tag, weight, priority)
            self._dict_manager.invalidate_cache()
        else:
            user_dict = self.create_user_dictionary("default_user")
            user_dict.add_word(word, pos_tag, weight, priority)
            self._dict_manager.invalidate_cache()

    def set_word_weight(
        self,
        word: str,
        weight: float,
        dict_name: Optional[str] = None
    ) -> bool:
        if dict_name:
            user_dict = self._user_dictionaries.get(dict_name)
            if user_dict:
                return user_dict.set_weight(word, weight)

        for user_dict in self._user_dictionaries.values():
            if user_dict.set_weight(word, weight):
                self._dict_manager.invalidate_cache()
                return True

        return self.dictionary.set_weight(word, weight)

    def set_word_priority(
        self,
        word: str,
        priority: int,
        dict_name: Optional[str] = None
    ) -> bool:
        if dict_name:
            user_dict = self._user_dictionaries.get(dict_name)
            if user_dict:
                return user_dict.set_priority(word, priority)

        for user_dict in self._user_dictionaries.values():
            if user_dict.set_priority(word, priority):
                self._dict_manager.invalidate_cache()
                return True

        return self.dictionary.set_priority(word, priority)

    def get_word_info(self, word: str) -> Dict[str, Any]:
        result = {'word': word, 'found': False}

        for name, user_dict in self._user_dictionaries.items():
            found, pos_tag, weight, priority = user_dict.search_with_info(word)
            if found:
                result.update({
                    'found': True,
                    'pos_tag': pos_tag,
                    'weight': weight,
                    'priority': priority,
                    'dictionary': name,
                    'dictionary_type': 'user'
                })
                return result

        found, pos_tag, weight, priority = self.dictionary.search_with_info(word)
        if found:
            result.update({
                'found': True,
                'pos_tag': pos_tag,
                'weight': weight,
                'priority': priority,
                'dictionary': self.dictionary.name,
                'dictionary_type': 'system'
            })

        return result

    def get_all_dictionaries_info(self) -> List[Dict[str, Any]]:
        return self._dict_manager.get_all_dictionaries_info()

    def set_use_weighted(self, use_weighted: bool) -> None:
        self.use_weighted = use_weighted

    def segment(self, text: str, mode: Optional[str] = None) -> List[str]:
        if mode is None:
            mode = self.mode
        
        if mode == 'hybrid':
            return self.segment_hybrid(text)

        active_dict = self._get_active_dictionary()

        if mode == 'hmm':
            if not self.hmm_segmentor.is_trained():
                raise RuntimeError("HMM model has not been trained. Call train_hmm() or load_hmm_model() first.")
            return self.hmm_segmentor.segment(text)
        elif mode == 'crf':
            if not self.crf_segmentor.is_trained():
                raise RuntimeError("CRF model has not been trained. Call train_crf() or load_crf_model() first.")
            return self.crf_segmentor.segment(text)
        elif mode == 'perceptron':
            if not self.perceptron_segmentor.is_trained():
                raise RuntimeError("Perceptron model has not been trained. Call train_perceptron() or load_perceptron_model() first.")
            return self.perceptron_segmentor.segment(text)
        elif mode == 'lattice':
            return self.lattice_segmentor.segment(text)
        elif mode == 'forward':
            if self.use_weighted:
                return forward_max_match_weighted(text, active_dict)
            return forward_max_match(text, active_dict)
        elif mode == 'backward':
            if self.use_weighted:
                return backward_max_match_weighted(text, active_dict)
            return backward_max_match(text, active_dict)
        elif mode == 'bidirectional':
            if self.use_weighted:
                return bidirectional_max_match_weighted(text, active_dict)
            return bidirectional_max_match(text, active_dict)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'forward', 'backward', 'bidirectional', 'hmm', 'crf', 'perceptron', 'lattice', or 'hybrid'.")

    def segment_without_stopwords(self, text: str, mode: Optional[str] = None) -> List[str]:
        words = self.segment(text, mode)
        return self.stopwords.filter(words)

    def segment_with_pos(self, text: str, mode: Optional[str] = None) -> List[Tuple[str, str]]:
        if mode is None:
            mode = self.mode

        active_dict = self._get_active_dictionary()

        if mode == 'hmm':
            raise ValueError("HMM mode does not support POS tagging. Use 'forward', 'backward', or 'bidirectional' mode for POS tagging.")
        elif mode == 'crf':
            raise ValueError("CRF mode does not support POS tagging. Use 'forward', 'backward', or 'bidirectional' mode for POS tagging.")
        elif mode == 'perceptron':
            raise ValueError("Perceptron mode does not support POS tagging. Use 'forward', 'backward', or 'bidirectional' mode for POS tagging.")
        elif mode == 'forward':
            if self.use_weighted:
                return forward_max_match_weighted_with_pos(text, active_dict)
            return forward_max_match_with_pos(text, active_dict)
        elif mode == 'backward':
            if self.use_weighted:
                return backward_max_match_weighted_with_pos(text, active_dict)
            return backward_max_match_with_pos(text, active_dict)
        elif mode == 'bidirectional':
            if self.use_weighted:
                return bidirectional_max_match_weighted_with_pos(text, active_dict)
            return bidirectional_max_match_with_pos(text, active_dict)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'forward', 'backward', or 'bidirectional'.")

    def segment_without_stopwords_with_pos(self, text: str, mode: Optional[str] = None) -> List[Tuple[str, str]]:
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
        if mode not in ('forward', 'backward', 'bidirectional', 'hmm', 'crf', 'perceptron', 'lattice', 'hybrid'):
            raise ValueError(f"Unknown mode: {mode}. Use 'forward', 'backward', 'bidirectional', 'hmm', 'crf', 'perceptron', 'lattice', or 'hybrid'.")
        self.mode = mode

    def load_dictionary(self, path: str) -> None:
        self.dictionary.load_dictionary(path)

    def save_dictionary(self, path: str) -> None:
        self.dictionary.save_dictionary(path)

    def add_word(
        self,
        word: str,
        pos_tag: Optional[str] = None,
        weight: float = 1.0,
        priority: Optional[int] = None
    ) -> None:
        self.dictionary.add_word(word, pos_tag, weight, priority)

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

    def load_hmm_model(self, filepath: str, key: Optional[str] = None, verify: bool = True) -> None:
        self.hmm_segmentor.load_model(filepath, key, verify)

    def save_hmm_model(self, filepath: str, key: Optional[str] = None) -> None:
        self.hmm_segmentor.save_model(filepath, key)

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

    def train_crf(
        self,
        corpus: List[List[str]],
        learning_rate: float = 0.1,
        l2_reg: float = 0.01,
        max_iter: int = 100,
        epsilon: float = 1e-6,
        verbose: bool = True
    ) -> None:
        self.crf_segmentor.train(
            corpus,
            learning_rate=learning_rate,
            l2_reg=l2_reg,
            max_iter=max_iter,
            epsilon=epsilon,
            verbose=verbose
        )

    def train_crf_from_file(self, filepath: str, encoding: str = 'utf-8') -> None:
        corpus = []

        with open(filepath, 'r', encoding=encoding) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                words = line.split()
                if words:
                    corpus.append(words)

        self.crf_segmentor.train(corpus)

    def load_crf_model(self, filepath: str) -> None:
        self.crf_segmentor.load_model(filepath)

    def save_crf_model(self, filepath: str) -> None:
        self.crf_segmentor.save_model(filepath)

    def segment_with_crf_states(self, text: str) -> List[Tuple[str, str]]:
        if not self.crf_segmentor.is_trained():
            raise RuntimeError("CRF model has not been trained. Call train_crf() or load_crf_model() first.")
        return self.crf_segmentor.segment_with_states(text)

    def get_crf_model_info(self) -> dict:
        return self.crf_segmentor.get_model_info()

    def is_crf_trained(self) -> bool:
        return self.crf_segmentor.is_trained()

    def set_use_crf(self, use_crf: bool) -> None:
        self.use_crf = use_crf

    def train_perceptron(
        self,
        corpus: List[List[str]],
        learning_rate: float = 1.0,
        max_iter: int = 10,
        averaged: bool = True,
        verbose: bool = True
    ) -> None:
        self.perceptron_segmentor.train(
            corpus,
            learning_rate=learning_rate,
            max_iter=max_iter,
            averaged=averaged,
            verbose=verbose
        )

    def train_perceptron_online(self, tokens: List[str], update_weights: bool = True) -> Tuple[bool, float]:
        return self.perceptron_segmentor.train_online(tokens, update_weights)

    def partial_fit_perceptron(
        self,
        corpus: List[List[str]],
        learning_rate: Optional[float] = None,
        verbose: bool = False
    ) -> None:
        self.perceptron_segmentor.partial_fit(corpus, learning_rate, verbose)

    def train_perceptron_from_file(self, filepath: str, encoding: str = 'utf-8') -> None:
        from .perceptron import train_from_file as perceptron_train_from_file
        perceptron_train_from_file(self.perceptron_segmentor, filepath, encoding)

    def load_perceptron_model(self, filepath: str) -> None:
        self.perceptron_segmentor.load_model(filepath)

    def save_perceptron_model(self, filepath: str) -> None:
        self.perceptron_segmentor.save_model(filepath)

    def segment_with_perceptron_states(self, text: str) -> List[Tuple[str, str]]:
        if not self.perceptron_segmentor.is_trained():
            raise RuntimeError("Perceptron model has not been trained. Call train_perceptron() or load_perceptron_model() first.")
        return self.perceptron_segmentor.segment_with_states(text)

    def get_perceptron_model_info(self) -> dict:
        return self.perceptron_segmentor.get_model_info()

    def is_perceptron_trained(self) -> bool:
        return self.perceptron_segmentor.is_trained()

    def set_use_perceptron(self, use_perceptron: bool) -> None:
        self.use_perceptron = use_perceptron

    def set_lattice_scoring_method(self, method: str) -> None:
        self.lattice_segmentor.set_scoring_method(method)

    def set_lattice_ngram_model(self, ngram_model) -> None:
        self.lattice_segmentor.set_ngram_model(ngram_model)

    def set_lattice_word_frequency(self, freq_dict: Dict[str, int]) -> None:
        self.lattice_segmentor.set_word_frequency(freq_dict)

    def segment_with_lattice(self, text: str) -> Tuple[List[str], Lattice]:
        return self.lattice_segmentor.segment_with_lattice(text)

    def segment_with_lattice_pos(self, text: str) -> List[Tuple[str, Optional[str]]]:
        return self.lattice_segmentor.segment_with_pos(text)

    def get_all_lattice_segmentations(self, text: str, max_results: int = 10) -> List[List[str]]:
        return self.lattice_segmentor.get_all_segmentations(text, max_results)

    def detect_lattice_ambiguity(self, text: str) -> List[Dict]:
        return self.lattice_segmentor.detect_ambiguity(text)

    def build_lattice(self, text: str) -> Lattice:
        return self.lattice_segmentor.build_lattice(text)

    def find_k_best_paths(self, text: str, k: int = 5) -> List[List[str]]:
        lattice = self.build_lattice(text)
        paths = self.lattice_segmentor.find_k_best_paths(lattice, k)
        return [lattice.get_path_words(path) for path in paths]

    def set_use_lattice(self, use_lattice: bool) -> None:
        self.use_lattice = use_lattice

    def detect_ambiguity(self, text: str) -> AmbiguityResult:
        return self.ambiguity_detector.detect(text)

    def detect_ambiguity_from_lattice(self, text: str) -> AmbiguityResult:
        lattice = self.build_lattice(text)
        return self.ambiguity_detector.detect_from_lattice(lattice)

    def get_ambiguity_statistics(self, text: str) -> Dict:
        return self.ambiguity_detector.get_ambiguity_statistics(text)

    def has_ambiguity(self, text: str) -> bool:
        result = self.detect_ambiguity(text)
        return result.has_ambiguity()

    def get_cross_ambiguities(self, text: str) -> List[AmbiguityRegion]:
        result = self.detect_ambiguity(text)
        return result.get_regions_by_type(AmbiguityType.CROSS)

    def get_combination_ambiguities(self, text: str) -> List[AmbiguityRegion]:
        result = self.detect_ambiguity(text)
        return result.get_regions_by_type(AmbiguityType.COMBINATION)

    def get_overlap_ambiguities(self, text: str) -> List[AmbiguityRegion]:
        result = self.detect_ambiguity(text)
        return result.get_regions_by_type(AmbiguityType.OVERLAP)

    def resolve_ambiguity(
        self, 
        text: str, 
        method: str = 'shortest'
    ) -> List[str]:
        if method == 'shortest':
            return self.segment(text, mode='lattice')
        elif method == 'ngram':
            lattice = self.build_lattice(text)
            self.lattice_segmentor.set_scoring_method('ngram')
            return self.lattice_segmentor.segment(text)
        elif method == 'frequency':
            self.lattice_segmentor.set_scoring_method('frequency')
            return self.lattice_segmentor.segment(text)
        else:
            return self.segment(text, mode='lattice')

    def train_new_word_detector(
        self,
        corpus: List[str],
        min_freq: int = 5,
        min_pmi: float = 1.0,
        min_entropy: float = 0.5
    ) -> None:
        self.new_word_detector.min_freq = min_freq
        self.new_word_detector.min_pmi = min_pmi
        self.new_word_detector.min_entropy = min_entropy
        self.new_word_detector.train(corpus)

    def train_new_word_detector_from_file(
        self,
        filepath: str,
        encoding: str = 'utf-8',
        min_freq: int = 5,
        min_pmi: float = 1.0,
        min_entropy: float = 0.5
    ) -> None:
        self.new_word_detector.min_freq = min_freq
        self.new_word_detector.min_pmi = min_pmi
        self.new_word_detector.min_entropy = min_entropy
        self.new_word_detector.train_from_file(filepath, encoding)

    def detect_new_words(
        self,
        top_k: int = 100,
        min_freq: Optional[int] = None,
        min_pmi: Optional[float] = None,
        min_entropy: Optional[float] = None
    ) -> List[Tuple[str, Dict[str, float]]]:
        return self.new_word_detector.detect(top_k, min_freq, min_pmi, min_entropy)

    def detect_new_words_from_text(
        self,
        text: str,
        top_k: int = 20,
        min_freq: Optional[int] = None,
        min_pmi: Optional[float] = None,
        min_entropy: Optional[float] = None
    ) -> List[Tuple[str, Dict[str, float]]]:
        return self.new_word_detector.detect_from_text(text, top_k, min_freq, min_pmi, min_entropy)

    def get_new_word_score(self, word: str) -> Dict[str, float]:
        return self.new_word_detector.get_word_score(word)

    def get_new_word_pmi(self, word: str) -> float:
        return self.new_word_detector.get_pmi(word)

    def get_new_word_entropy(self, word: str) -> Tuple[float, float]:
        left_entropy = self.new_word_detector.get_left_entropy(word)
        right_entropy = self.new_word_detector.get_right_entropy(word)
        return (left_entropy, right_entropy)

    def auto_extend_dictionary_with_new_words(
        self,
        top_k: int = 50,
        min_freq: Optional[int] = None,
        min_pmi: Optional[float] = None,
        min_entropy: Optional[float] = None,
        pos_tag: Optional[str] = None,
        weight: float = 1.0
    ) -> List[Tuple[str, Dict[str, float]]]:
        return self.new_word_detector.auto_extend_dictionary(
            self.dictionary,
            top_k=top_k,
            min_freq=min_freq,
            min_pmi=min_pmi,
            min_entropy=min_entropy,
            pos_tag=pos_tag,
            weight=weight
        )

    def is_new_word_detector_trained(self) -> bool:
        return self.new_word_detector.is_trained()

    def get_new_word_detector_statistics(self) -> Dict:
        return self.new_word_detector.get_statistics()

    def set_new_word_detector_thresholds(
        self,
        min_freq: Optional[int] = None,
        min_pmi: Optional[float] = None,
        min_entropy: Optional[float] = None
    ) -> None:
        self.new_word_detector.set_thresholds(min_freq, min_pmi, min_entropy)

    def set_new_word_length_range(self, min_len: int, max_len: int) -> None:
        self.new_word_detector.set_word_length_range(min_len, max_len)
    
    def _init_hybrid_segmentor(self) -> None:
        config = self._hybrid_config or HybridConfig()
        self._hybrid_segmentor = HybridSegmentor(
            dictionary=self.dictionary,
            config=config
        )
        
        if self.hmm_segmentor.is_trained():
            self._hybrid_segmentor.set_hmm_segmentor(self.hmm_segmentor)
        if self.crf_segmentor.is_trained():
            self._hybrid_segmentor.set_crf_segmentor(self.crf_segmentor)
        if self.perceptron_segmentor.is_trained():
            self._hybrid_segmentor.set_perceptron_segmentor(self.perceptron_segmentor)
        self._hybrid_segmentor.set_lattice_segmentor(self.lattice_segmentor)
    
    def enable_hybrid(self, config: Optional[HybridConfig] = None) -> None:
        self.use_hybrid = True
        self._hybrid_config = config
        self._init_hybrid_segmentor()
    
    def disable_hybrid(self) -> None:
        self.use_hybrid = False
    
    def set_hybrid_strategy(self, strategy: HybridStrategy) -> None:
        if self._hybrid_segmentor is None:
            self._init_hybrid_segmentor()
        
        if self._hybrid_config is None:
            self._hybrid_config = HybridConfig()
        
        self._hybrid_config.strategy = strategy
        self._hybrid_segmentor.set_config(self._hybrid_config)
    
    def set_hybrid_weights(self, weights: Dict[str, float]) -> None:
        if self._hybrid_segmentor is None:
            self._init_hybrid_segmentor()
        
        if self._hybrid_config is None:
            self._hybrid_config = HybridConfig()
        
        self._hybrid_config.weights = weights
        self._hybrid_segmentor.set_config(self._hybrid_config)
    
    def set_hybrid_cascade_order(self, order: List[str]) -> None:
        if self._hybrid_segmentor is None:
            self._init_hybrid_segmentor()
        
        if self._hybrid_config is None:
            self._hybrid_config = HybridConfig(strategy=HybridStrategy.CASCADE)
        
        self._hybrid_config.cascade_order = order
        self._hybrid_segmentor.set_config(self._hybrid_config)
    
    def segment_hybrid(self, text: str) -> List[str]:
        if not self.use_hybrid or self._hybrid_segmentor is None:
            self.enable_hybrid()
        
        return self._hybrid_segmentor.segment(text)
    
    def segment_hybrid_with_details(self, text: str) -> Tuple[List[str], Dict[str, Any]]:
        if not self.use_hybrid or self._hybrid_segmentor is None:
            self.enable_hybrid()
        
        return self._hybrid_segmentor.segment_with_details(text)
    
    def get_hybrid_available_segmenters(self) -> List[str]:
        if self._hybrid_segmentor is None:
            return []
        return self._hybrid_segmentor.get_available_segmenters()
    
    def get_hybrid_config(self) -> Optional[HybridConfig]:
        return self._hybrid_config
    
    def load_hybrid_dl_model(self, model_path: str, model_type: str = 'bert') -> bool:
        if self._hybrid_segmentor is None:
            self._init_hybrid_segmentor()
        
        return self._hybrid_segmentor.load_dl_model(model_path, model_type)
    
    def sync_hybrid_segmentors(self) -> None:
        if self._hybrid_segmentor is None:
            return
        
        if self.hmm_segmentor.is_trained():
            self._hybrid_segmentor.set_hmm_segmentor(self.hmm_segmentor)
        if self.crf_segmentor.is_trained():
            self._hybrid_segmentor.set_crf_segmentor(self.crf_segmentor)
        if self.perceptron_segmentor.is_trained():
            self._hybrid_segmentor.set_perceptron_segmentor(self.perceptron_segmentor)
    
    def is_hybrid_enabled(self) -> bool:
        return self.use_hybrid and self._hybrid_segmentor is not None
