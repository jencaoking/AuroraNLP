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
from .traditional_chinese import TraditionalChineseConverter, TraditionalChineseDictionary
from .managers import (
    DictionaryService,
    StopWordsManager,
    KeywordExtractorManager,
    SimilarityManager,
    MLSegmentorManager,
    LatticeSegmentorManager,
    AmbiguityDetectorManager,
    NewWordDetectorManager,
    HybridSegmentorManager
)
from .lattice import Lattice
from .ambiguity import AmbiguityResult, AmbiguityType, AmbiguityRegion
from .hybrid import HybridConfig, HybridStrategy


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
        self.dict_manager = DictionaryService(dictionary, load_default_dict)
        
        self.stopwords_manager = StopWordsManager(load_default_stopwords)
        self.keyword_extractor_manager = KeywordExtractorManager()
        self.similarity_manager = SimilarityManager()
        
        self.ml_segmentor_manager = MLSegmentorManager(use_hmm, use_crf, use_perceptron)
        self.lattice_manager = LatticeSegmentorManager(self.dict_manager.dictionary, use_lattice)
        self.ambiguity_manager = AmbiguityDetectorManager(self.dict_manager.dictionary)
        self.new_word_manager = NewWordDetectorManager()
        
        # 繁体中文支持
        self.traditional_converter = TraditionalChineseConverter()
        self.traditional_dictionary = TraditionalChineseDictionary(self.traditional_converter)
        
        self._init_hybrid_manager(hybrid_config)
        
        self.mode: str = 'bidirectional'
        self.use_weighted = use_weighted
        self.use_hybrid = use_hybrid
        
        if use_hybrid:
            self.hybrid_manager.enable(hybrid_config)
    
    def _init_hybrid_manager(self, hybrid_config: Optional[HybridConfig]) -> None:
        self.hybrid_manager = HybridSegmentorManager(
            self.dict_manager.dictionary,
            hybrid_config,
            self.ml_segmentor_manager.hmm_segmentor,
            self.ml_segmentor_manager.crf_segmentor,
            self.ml_segmentor_manager.perceptron_segmentor,
            lattice_segmentor_provider=lambda: self.lattice_manager.lattice_segmentor
        )
    
    @property
    def dictionary(self) -> Dictionary:
        return self.dict_manager.dictionary
    
    @property
    def stopwords(self) -> StopWordsManager:
        return self.stopwords_manager
    
    @property
    def keyword_extractor(self) -> KeywordExtractorManager:
        return self.keyword_extractor_manager
    
    @property
    def similarity(self) -> SimilarityManager:
        return self.similarity_manager
    
    @property
    def hmm_segmentor(self):
        return self.ml_segmentor_manager.hmm_segmentor
    
    @property
    def crf_segmentor(self):
        return self.ml_segmentor_manager.crf_segmentor
    
    @property
    def perceptron_segmentor(self):
        return self.ml_segmentor_manager.perceptron_segmentor
    
    @property
    def lattice_segmentor(self):
        return self.lattice_manager.lattice_segmentor
    
    @property
    def ambiguity_detector(self):
        return self.ambiguity_manager.ambiguity_detector
    
    @property
    def new_word_detector(self):
        return self.new_word_manager.new_word_detector
    
    def create_user_dictionary(
        self,
        name: str,
        priority: int = 100,
        default_weight: float = 10.0
    ) -> UserDictionary:
        return self.dict_manager.create_user_dictionary(name, priority, default_weight)
    
    def get_user_dictionary(self, name: str) -> Optional[UserDictionary]:
        return self.dict_manager.get_user_dictionary(name)
    
    def remove_user_dictionary(self, name: str) -> bool:
        return self.dict_manager.remove_user_dictionary(name)
    
    def load_user_dictionary(
        self,
        path: str,
        name: Optional[str] = None,
        priority: int = 100,
        default_weight: float = 10.0
    ) -> UserDictionary:
        return self.dict_manager.load_user_dictionary(path, name, priority, default_weight)
    
    def add_user_word(
        self,
        word: str,
        pos_tag: Optional[str] = None,
        weight: Optional[float] = None,
        priority: Optional[int] = None,
        dict_name: Optional[str] = None
    ) -> None:
        self.dict_manager.add_user_word(word, pos_tag, weight, priority, dict_name)
    
    def set_word_weight(
        self,
        word: str,
        weight: float,
        dict_name: Optional[str] = None
    ) -> bool:
        return self.dict_manager.set_word_weight(word, weight, dict_name)
    
    def set_word_priority(
        self,
        word: str,
        priority: int,
        dict_name: Optional[str] = None
    ) -> bool:
        return self.dict_manager.set_word_priority(word, priority, dict_name)
    
    def get_word_info(self, word: str) -> Dict[str, Any]:
        return self.dict_manager.get_word_info(word)
    
    def get_all_dictionaries_info(self) -> List[Dict[str, Any]]:
        return self.dict_manager.get_all_dictionaries_info()
    
    def set_use_weighted(self, use_weighted: bool) -> None:
        self.use_weighted = use_weighted
    
    def segment(self, text: str, mode: Optional[str] = None) -> List[str]:
        if mode is None:
            mode = self.mode
        
        if mode == 'hybrid':
            return self.hybrid_manager.segment(text)
        
        active_dict = self.dict_manager._get_active_dictionary()
        
        if mode == 'hmm':
            return self.ml_segmentor_manager.segment_hmm(text)
        elif mode == 'crf':
            return self.ml_segmentor_manager.segment_crf(text)
        elif mode == 'perceptron':
            return self.ml_segmentor_manager.segment_perceptron(text)
        elif mode == 'lattice':
            return self.lattice_manager.segment(text)
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
        return self.stopwords_manager.filter(words)
    
    def segment_with_pos(self, text: str, mode: Optional[str] = None) -> List[Tuple[str, str]]:
        if mode is None:
            mode = self.mode
        
        active_dict = self.dict_manager._get_active_dictionary()
        
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
        return self.stopwords_manager.filter_with_pos(words_with_pos)
    
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
        stopwords = self.stopwords_manager.get_stopwords() if use_stopwords else None
        return self.keyword_extractor_manager.extract_keywords(text, self, top_k, method, stopwords, min_length)
    
    def build_keyword_corpus(self, documents: List[str]) -> None:
        self.keyword_extractor_manager.build_idf_corpus(documents, self)
    
    def compute_similarity(
        self,
        text1: str,
        text2: str,
        method: str = 'cosine',
        use_stopwords: bool = True
    ) -> float:
        stopwords = self.stopwords_manager.get_stopwords() if use_stopwords else None
        return self.similarity_manager.compute_similarity(text1, text2, self, method, stopwords)
    
    def build_similarity_corpus(self, documents: List[str]) -> None:
        self.similarity_manager.build_idf_corpus(documents, self)
    
    def batch_similarity(
        self,
        query: str,
        documents: List[str],
        method: str = 'cosine',
        use_stopwords: bool = True
    ) -> List[Tuple[str, float]]:
        stopwords = self.stopwords_manager.get_stopwords() if use_stopwords else None
        return self.similarity_manager.batch_similarity(query, documents, self, method, stopwords)
    
    def set_mode(self, mode: str) -> None:
        if mode not in ('forward', 'backward', 'bidirectional', 'hmm', 'crf', 'perceptron', 'lattice', 'hybrid'):
            raise ValueError(f"Unknown mode: {mode}. Use 'forward', 'backward', 'bidirectional', 'hmm', 'crf', 'perceptron', 'lattice', or 'hybrid'.")
        self.mode = mode
    
    def load_dictionary(self, path: str) -> None:
        self.dict_manager.load_dictionary(path)
    
    def save_dictionary(self, path: str) -> None:
        self.dict_manager.save_dictionary(path)
    
    def add_word(
        self,
        word: str,
        pos_tag: Optional[str] = None,
        weight: float = 1.0,
        priority: Optional[int] = None
    ) -> None:
        self.dict_manager.add_word(word, pos_tag, weight, priority)
    
    def remove_word(self, word: str) -> bool:
        return self.dict_manager.remove_word(word)
    
    def get_dictionary_size(self) -> int:
        return self.dict_manager.get_dictionary_size()
    
    def add_stopword(self, word: str) -> None:
        self.stopwords_manager.add_stopword(word)
    
    def remove_stopword(self, word: str) -> bool:
        return self.stopwords_manager.remove_stopword(word)
    
    def is_stopword(self, word: str) -> bool:
        return self.stopwords_manager.is_stopword(word)
    
    def get_stopwords(self) -> Set[str]:
        return self.stopwords_manager.get_stopwords()
    
    def load_stopwords(self, path: str) -> None:
        self.stopwords_manager.load_stopwords(path)
    
    def save_stopwords(self, path: str) -> None:
        self.stopwords_manager.save_stopwords(path)
    
    def get_pos_tag_name(self, pos_tag: str) -> str:
        return POS_TAG_NAMES.get(pos_tag, '未知')
    
    def get_pos_tags(self) -> dict:
        return POS_TAG_NAMES.copy()
    
    def train_hmm(self, corpus: List[List[str]], smooth: float = 1.0) -> None:
        self.ml_segmentor_manager.train_hmm(corpus, smooth)
    
    def train_hmm_from_file(self, filepath: str, encoding: str = 'utf-8') -> None:
        self.ml_segmentor_manager.train_hmm_from_file(filepath, encoding)
    
    def load_hmm_model(self, filepath: str, key: Optional[str] = None, verify: bool = True) -> None:
        self.ml_segmentor_manager.load_hmm_model(filepath, key, verify)
    
    def save_hmm_model(self, filepath: str, key: Optional[str] = None) -> None:
        self.ml_segmentor_manager.save_hmm_model(filepath, key)
    
    def segment_with_hmm_states(self, text: str) -> List[Tuple[str, str]]:
        return self.ml_segmentor_manager.segment_with_hmm_states(text)
    
    def get_hmm_model_info(self) -> dict:
        return self.ml_segmentor_manager.get_hmm_model_info()
    
    def is_hmm_trained(self) -> bool:
        return self.ml_segmentor_manager.is_hmm_trained()
    
    def set_use_hmm(self, use_hmm: bool) -> None:
        self.ml_segmentor_manager.set_use_hmm(use_hmm)
    
    def train_crf(
        self,
        corpus: List[List[str]],
        learning_rate: float = 0.1,
        l2_reg: float = 0.01,
        max_iter: int = 100,
        epsilon: float = 1e-6,
        verbose: bool = True
    ) -> None:
        self.ml_segmentor_manager.train_crf(corpus, learning_rate, l2_reg, max_iter, epsilon, verbose)
    
    def train_crf_from_file(self, filepath: str, encoding: str = 'utf-8') -> None:
        self.ml_segmentor_manager.train_crf_from_file(filepath, encoding)
    
    def load_crf_model(self, filepath: str) -> None:
        self.ml_segmentor_manager.load_crf_model(filepath)
    
    def save_crf_model(self, filepath: str) -> None:
        self.ml_segmentor_manager.save_crf_model(filepath)
    
    def segment_with_crf_states(self, text: str) -> List[Tuple[str, str]]:
        return self.ml_segmentor_manager.segment_with_crf_states(text)
    
    def get_crf_model_info(self) -> dict:
        return self.ml_segmentor_manager.get_crf_model_info()
    
    def is_crf_trained(self) -> bool:
        return self.ml_segmentor_manager.is_crf_trained()
    
    def set_use_crf(self, use_crf: bool) -> None:
        self.ml_segmentor_manager.set_use_crf(use_crf)
    
    def train_perceptron(
        self,
        corpus: List[List[str]],
        learning_rate: float = 1.0,
        max_iter: int = 10,
        averaged: bool = True,
        verbose: bool = True
    ) -> None:
        self.ml_segmentor_manager.train_perceptron(corpus, learning_rate, max_iter, averaged, verbose)
    
    def train_perceptron_online(self, tokens: List[str], update_weights: bool = True) -> Tuple[bool, float]:
        return self.ml_segmentor_manager.train_perceptron_online(tokens, update_weights)
    
    def partial_fit_perceptron(
        self,
        corpus: List[List[str]],
        learning_rate: Optional[float] = None,
        verbose: bool = False
    ) -> None:
        self.ml_segmentor_manager.partial_fit_perceptron(corpus, learning_rate, verbose)
    
    def train_perceptron_from_file(self, filepath: str, encoding: str = 'utf-8') -> None:
        self.ml_segmentor_manager.train_perceptron_from_file(filepath, encoding)
    
    def load_perceptron_model(self, filepath: str) -> None:
        self.ml_segmentor_manager.load_perceptron_model(filepath)
    
    def save_perceptron_model(self, filepath: str) -> None:
        self.ml_segmentor_manager.save_perceptron_model(filepath)
    
    def segment_with_perceptron_states(self, text: str) -> List[Tuple[str, str]]:
        return self.ml_segmentor_manager.segment_with_perceptron_states(text)
    
    def get_perceptron_model_info(self) -> dict:
        return self.ml_segmentor_manager.get_perceptron_model_info()
    
    def is_perceptron_trained(self) -> bool:
        return self.ml_segmentor_manager.is_perceptron_trained()
    
    def set_use_perceptron(self, use_perceptron: bool) -> None:
        self.ml_segmentor_manager.set_use_perceptron(use_perceptron)
    
    def set_lattice_scoring_method(self, method: str) -> None:
        self.lattice_manager.set_scoring_method(method)
    
    def set_lattice_ngram_model(self, ngram_model) -> None:
        self.lattice_manager.set_ngram_model(ngram_model)
    
    def set_lattice_word_frequency(self, freq_dict: Dict[str, int]) -> None:
        self.lattice_manager.set_word_frequency(freq_dict)
    
    def segment_with_lattice(self, text: str) -> Tuple[List[str], Lattice]:
        return self.lattice_manager.segment_with_lattice(text)
    
    def segment_with_lattice_pos(self, text: str) -> List[Tuple[str, Optional[str]]]:
        return self.lattice_manager.segment_with_pos(text)
    
    def get_all_lattice_segmentations(self, text: str, max_results: int = 10) -> List[List[str]]:
        return self.lattice_manager.get_all_segmentations(text, max_results)
    
    def detect_lattice_ambiguity(self, text: str) -> List[Dict]:
        return self.lattice_manager.detect_ambiguity(text)
    
    def build_lattice(self, text: str) -> Lattice:
        return self.lattice_manager.build_lattice(text)
    
    def find_k_best_paths(self, text: str, k: int = 5) -> List[List[str]]:
        return self.lattice_manager.find_k_best_paths(text, k)
    
    def set_use_lattice(self, use_lattice: bool) -> None:
        self.lattice_manager.set_use_lattice(use_lattice)
    
    def detect_ambiguity(self, text: str) -> AmbiguityResult:
        return self.ambiguity_manager.detect(text)
    
    def detect_ambiguity_from_lattice(self, text: str) -> AmbiguityResult:
        lattice = self.build_lattice(text)
        return self.ambiguity_manager.detect_from_lattice(lattice)
    
    def get_ambiguity_statistics(self, text: str) -> Dict:
        return self.ambiguity_manager.get_ambiguity_statistics(text)
    
    def has_ambiguity(self, text: str) -> bool:
        return self.ambiguity_manager.has_ambiguity(text)
    
    def get_cross_ambiguities(self, text: str) -> List[AmbiguityRegion]:
        return self.ambiguity_manager.get_cross_ambiguities(text)
    
    def get_combination_ambiguities(self, text: str) -> List[AmbiguityRegion]:
        return self.ambiguity_manager.get_combination_ambiguities(text)
    
    def get_overlap_ambiguities(self, text: str) -> List[AmbiguityRegion]:
        return self.ambiguity_manager.get_overlap_ambiguities(text)
    
    def resolve_ambiguity(
        self, 
        text: str, 
        method: str = 'shortest'
    ) -> List[str]:
        if method == 'shortest':
            return self.segment(text, mode='lattice')
        elif method == 'ngram':
            lattice = self.build_lattice(text)
            self.lattice_manager.set_scoring_method('ngram')
            return self.lattice_manager.segment(text)
        elif method == 'frequency':
            self.lattice_manager.set_scoring_method('frequency')
            return self.lattice_manager.segment(text)
        else:
            return self.segment(text, mode='lattice')
    
    def train_new_word_detector(
        self,
        corpus: List[str],
        min_freq: int = 5,
        min_pmi: float = 1.0,
        min_entropy: float = 0.5
    ) -> None:
        self.new_word_manager.train(corpus, min_freq, min_pmi, min_entropy)
    
    def train_new_word_detector_from_file(
        self,
        filepath: str,
        encoding: str = 'utf-8',
        min_freq: int = 5,
        min_pmi: float = 1.0,
        min_entropy: float = 0.5
    ) -> None:
        self.new_word_manager.train_from_file(filepath, encoding, min_freq, min_pmi, min_entropy)
    
    def detect_new_words(
        self,
        top_k: int = 100,
        min_freq: Optional[int] = None,
        min_pmi: Optional[float] = None,
        min_entropy: Optional[float] = None
    ) -> List[Tuple[str, Dict[str, float]]]:
        return self.new_word_manager.detect(top_k, min_freq, min_pmi, min_entropy)
    
    def detect_new_words_from_text(
        self,
        text: str,
        top_k: int = 20,
        min_freq: Optional[int] = None,
        min_pmi: Optional[float] = None,
        min_entropy: Optional[float] = None
    ) -> List[Tuple[str, Dict[str, float]]]:
        return self.new_word_manager.detect_from_text(text, top_k, min_freq, min_pmi, min_entropy)
    
    def get_new_word_score(self, word: str) -> Dict[str, float]:
        return self.new_word_manager.get_word_score(word)
    
    def get_new_word_pmi(self, word: str) -> float:
        return self.new_word_manager.get_pmi(word)
    
    def get_new_word_entropy(self, word: str) -> Tuple[float, float]:
        return self.new_word_manager.get_entropy(word)
    
    def auto_extend_dictionary_with_new_words(
        self,
        top_k: int = 50,
        min_freq: Optional[int] = None,
        min_pmi: Optional[float] = None,
        min_entropy: Optional[float] = None,
        pos_tag: Optional[str] = None,
        weight: float = 1.0
    ) -> List[Tuple[str, Dict[str, float]]]:
        return self.new_word_manager.auto_extend_dictionary(
            self.dict_manager.dictionary,
            top_k=top_k,
            min_freq=min_freq,
            min_pmi=min_pmi,
            min_entropy=min_entropy,
            pos_tag=pos_tag,
            weight=weight
        )
    
    def is_new_word_detector_trained(self) -> bool:
        return self.new_word_manager.is_trained()
    
    def get_new_word_detector_statistics(self) -> Dict:
        return self.new_word_manager.get_statistics()
    
    def set_new_word_detector_thresholds(
        self,
        min_freq: Optional[int] = None,
        min_pmi: Optional[float] = None,
        min_entropy: Optional[float] = None
    ) -> None:
        self.new_word_manager.set_thresholds(min_freq, min_pmi, min_entropy)
    
    def set_new_word_length_range(self, min_len: int, max_len: int) -> None:
        self.new_word_manager.set_word_length_range(min_len, max_len)
    
    def enable_hybrid(self, config: Optional[HybridConfig] = None) -> None:
        self.use_hybrid = True
        self.hybrid_manager.enable(config)
    
    def disable_hybrid(self) -> None:
        self.use_hybrid = False
        self.hybrid_manager.disable()
    
    def set_hybrid_strategy(self, strategy: HybridStrategy) -> None:
        self.hybrid_manager.set_strategy(strategy)
    
    def set_hybrid_weights(self, weights: Dict[str, float]) -> None:
        self.hybrid_manager.set_weights(weights)
    
    def set_hybrid_cascade_order(self, order: List[str]) -> None:
        self.hybrid_manager.set_cascade_order(order)
    
    def segment_hybrid(self, text: str) -> List[str]:
        return self.hybrid_manager.segment(text)
    
    def segment_hybrid_with_details(self, text: str) -> Tuple[List[str], Dict[str, Any]]:
        return self.hybrid_manager.segment_with_details(text)
    
    def get_hybrid_available_segmenters(self) -> List[str]:
        return self.hybrid_manager.get_available_segmenters()
    
    def get_hybrid_config(self) -> Optional[HybridConfig]:
        return self.hybrid_manager.get_config()
    
    def load_hybrid_dl_model(self, model_path: str, model_type: str = 'bert') -> bool:
        return self.hybrid_manager.load_dl_model(model_path, model_type)
    
    def sync_hybrid_segmentors(self) -> None:
        self.hybrid_manager.sync_segmentors(
            self.ml_segmentor_manager.hmm_segmentor,
            self.ml_segmentor_manager.crf_segmentor,
            self.ml_segmentor_manager.perceptron_segmentor,
            self.lattice_manager.lattice_segmentor
        )
    
    def is_hybrid_enabled(self) -> bool:
        return self.hybrid_manager.is_enabled()
    
    def add_traditional_word(self, word: str, traditional: Optional[str] = None) -> None:
        """
        添加繁体中文词汇
        
        Args:
            word: 词汇（可以是简体或繁体）
            traditional: 对应的繁体形式（可选）
        """
        self.traditional_dictionary.add_word(word, traditional)
    
    def add_traditional_words(self, words: List[str]) -> None:
        """
        批量添加繁体中文词汇
        
        Args:
            words: 词汇列表
        """
        self.traditional_dictionary.add_words(words)
    
    def load_traditional_dictionary(self, file_path: str) -> int:
        """
        从文件加载繁体中文词典
        
        Args:
            file_path: 文件路径
            
        Returns:
            加载的词汇数量
        """
        return self.traditional_dictionary.load_from_file(file_path)
    
    def save_traditional_dictionary(self, file_path: str) -> None:
        """
        保存繁体中文词典到文件
        
        Args:
            file_path: 文件路径
        """
        self.traditional_dictionary.save_to_file(file_path)
    
    def segment_traditional(self, text: str, mode: Optional[str] = None, region: Optional[str] = None) -> List[str]:
        """
        分词繁体中文文本
        
        Args:
            text: 繁体中文字符串
            mode: 分词模式
            region: 地区代码 ('tw', 'hk', 'mo')，可选
            
        Returns:
            分词结果
        """
        # 转换为简体中文后分词
        simplified_text = self.traditional_converter.traditional_to_simplified(text, region)
        return self.segment(simplified_text, mode)
    
    def segment_with_traditional(self, text: str, mode: Optional[str] = None, region: Optional[str] = None) -> List[str]:
        """
        处理包含繁体中文的文本
        
        Args:
            text: 中文字符串（可能包含繁体）
            mode: 分词模式
            region: 地区代码 ('tw', 'hk', 'mo')，可选
            
        Returns:
            分词结果
        """
        # 检测是否包含繁体中文
        has_traditional = any(char in self.traditional_converter.get_traditional_chars() for char in text)
        
        if has_traditional:
            # 转换为简体中文后分词
            simplified_text = self.traditional_converter.traditional_to_simplified(text)
            return self.segment(simplified_text, mode)
        else:
            # 直接分词
            return self.segment(text, mode)
    
    def simplified_to_traditional(self, text: str, region: Optional[str] = None) -> str:
        """
        将简体中文转换为繁体中文
        
        Args:
            text: 简体中文字符串
            region: 地区代码 ('tw', 'hk', 'mo')，可选
            
        Returns:
            繁体中文字符串
        """
        return self.traditional_converter.simplified_to_traditional(text, region)
    
    def traditional_to_simplified(self, text: str) -> str:
        """
        将繁体中文转换为简体中文
        
        Args:
            text: 繁体中文字符串
            
        Returns:
            简体中文字符串
        """
        return self.traditional_converter.traditional_to_simplified(text)
    
    def detect_language_variant(self, text: str) -> Optional[str]:
        """
        检测文本的繁体中文变体类型
        
        Args:
            text: 中文文本
            
        Returns:
            地区代码 ('tw', 'hk', 'mo') 或 None
        """
        return self.traditional_converter.detect_language_variant(text)
