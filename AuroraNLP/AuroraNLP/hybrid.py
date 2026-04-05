from typing import List, Tuple, Dict, Optional, Callable, Any, Union, TYPE_CHECKING
from dataclasses import dataclass, field
from enum import Enum
from collections import Counter, defaultdict
import math
import pickle
import os

if TYPE_CHECKING:
    from .dictionary import Dictionary, DictionaryManager
    from .hmm import HMMSegmentor
    from .crf import CRFSegmentor
    from .perceptron import PerceptronSegmentor
    from .lattice import LatticeSegmentor, Lattice


class HybridStrategy(Enum):
    VOTE = "vote"
    WEIGHTED = "weighted"
    CASCADE = "cascade"
    ADAPTIVE = "adaptive"
    CONFIDENCE = "confidence"


class SegmenterType(Enum):
    RULE_BASED = "rule_based"
    STATISTICAL = "statistical"
    DEEP_LEARNING = "deep_learning"
    HYBRID = "hybrid"


@dataclass
class SegmenterResult:
    words: List[str]
    confidence: float = 1.0
    segmenter_type: SegmenterType = SegmenterType.RULE_BASED
    segmenter_name: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.segmenter_name:
            self.segmenter_name = self.segmenter_type.value
    
    @property
    def word_count(self) -> int:
        return len(self.words)
    
    @property
    def avg_word_length(self) -> float:
        if not self.words:
            return 0.0
        return sum(len(w) for w in self.words) / len(self.words)
    
    @property
    def single_char_ratio(self) -> float:
        if not self.words:
            return 0.0
        return sum(1 for w in self.words if len(w) == 1) / len(self.words)


@dataclass
class HybridConfig:
    strategy: HybridStrategy = HybridStrategy.WEIGHTED
    weights: Dict[str, float] = field(default_factory=lambda: {
        'dict': 0.3,
        'hmm': 0.25,
        'crf': 0.25,
        'perceptron': 0.2
    })
    cascade_order: List[str] = field(default_factory=lambda: ['dict', 'hmm', 'crf'])
    confidence_threshold: float = 0.7
    min_confidence: float = 0.3
    use_dict_fallback: bool = True
    enable_cache: bool = True
    cache_size: int = 1000
    
    def validate(self) -> bool:
        if self.strategy == HybridStrategy.WEIGHTED:
            if not self.weights:
                return False
            total = sum(self.weights.values())
            if total <= 0:
                return False
        elif self.strategy == HybridStrategy.CASCADE:
            if not self.cascade_order:
                return False
        return True


class ConfidenceEstimator:
    def __init__(self):
        self._word_freq: Dict[str, int] = {}
        self._total_words: int = 0
    
    def set_word_frequency(self, freq_dict: Dict[str, int]) -> None:
        self._word_freq = freq_dict
        self._total_words = sum(freq_dict.values())
    
    def estimate_from_result(self, result: SegmenterResult, dictionary: Optional['Dictionary'] = None) -> float:
        if not result.words:
            return 0.0
        
        scores = []
        
        scores.append(self._estimate_by_coverage(result, dictionary))
        scores.append(self._estimate_by_word_quality(result))
        scores.append(self._estimate_by_segmentation_quality(result))
        
        return sum(scores) / len(scores)
    
    def _estimate_by_coverage(self, result: SegmenterResult, dictionary: Optional['Dictionary']) -> float:
        if dictionary is None:
            return 0.5
        
        known_words = 0
        total_chars = 0
        
        for word in result.words:
            total_chars += len(word)
            if dictionary.search_in_dict(word):
                known_words += len(word)
        
        if total_chars == 0:
            return 0.0
        
        return known_words / total_chars
    
    def _estimate_by_word_quality(self, result: SegmenterResult) -> float:
        if not result.words:
            return 0.0
        
        quality_scores = []
        
        for word in result.words:
            word_score = 1.0
            
            if len(word) == 1:
                word_score *= 0.7
            
            if self._word_freq:
                freq = self._word_freq.get(word, 0)
                if freq > 0:
                    word_score *= min(1.0, math.log(freq + 1) / 10)
                else:
                    word_score *= 0.5
            
            quality_scores.append(word_score)
        
        return sum(quality_scores) / len(quality_scores)
    
    def _estimate_by_segmentation_quality(self, result: SegmenterResult) -> float:
        if not result.words:
            return 0.0
        
        score = 1.0
        
        single_ratio = result.single_char_ratio
        if single_ratio > 0.5:
            score *= (1.0 - single_ratio * 0.5)
        
        avg_len = result.avg_word_length
        if avg_len < 1.5:
            score *= 0.7
        elif avg_len > 4:
            score *= 0.9
        
        return max(0.1, score)


class VoteFusion:
    @staticmethod
    def fuse(results: List[SegmenterResult]) -> List[str]:
        if not results:
            return []
        
        if len(results) == 1:
            return results[0].words
        
        text_length = sum(len(w) for w in results[0].words)
        position_votes: Dict[int, Counter] = defaultdict(Counter)
        
        for result in results:
            pos = 0
            for word in result.words:
                boundary = pos + len(word)
                position_votes[pos][boundary] += 1
                pos = boundary
        
        final_words = []
        pos = 0
        
        while pos < text_length:
            if pos in position_votes:
                best_boundary = position_votes[pos].most_common(1)[0][0]
                final_words.append(results[0].words[0][:best_boundary - pos] if best_boundary > pos else "")
                pos = best_boundary
            else:
                pos += 1
        
        return VoteFusion._reconstruct_words(final_words, results)
    
    @staticmethod
    def _reconstruct_words(boundaries: List[int], results: List[SegmenterResult]) -> List[str]:
        if not results:
            return []
        
        text = ''.join(results[0].words)
        
        vote_matrix: Dict[int, Counter] = defaultdict(Counter)
        
        for result in results:
            pos = 0
            for word in result.words:
                boundary = pos + len(word)
                vote_matrix[pos][boundary] += 1
                pos = boundary
        
        final_words = []
        pos = 0
        
        while pos < len(text):
            if pos in vote_matrix:
                best_boundary = vote_matrix[pos].most_common(1)[0][0]
                final_words.append(text[pos:best_boundary])
                pos = best_boundary
            else:
                final_words.append(text[pos])
                pos += 1
        
        return final_words


class WeightedFusion:
    def __init__(self):
        self._position_scores: Dict[int, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
    
    def fuse(self, results: List[SegmenterResult], weights: Dict[str, float]) -> List[str]:
        if not results:
            return []
        
        if len(results) == 1:
            return results[0].words
        
        normalized_weights = self._normalize_weights(weights)
        
        text_length = sum(len(w) for w in results[0].words)
        self._position_scores.clear()
        
        for result in results:
            weight = normalized_weights.get(result.segmenter_name, 1.0 / len(results))
            confidence_factor = result.confidence
            
            pos = 0
            for word in result.words:
                boundary = pos + len(word)
                self._position_scores[pos][boundary] += weight * confidence_factor
                pos = boundary
        
        final_words = []
        pos = 0
        text = ''.join(results[0].words)
        
        while pos < text_length:
            if pos in self._position_scores:
                best_boundary = max(
                    self._position_scores[pos].items(),
                    key=lambda x: x[1]
                )[0]
                final_words.append(text[pos:best_boundary])
                pos = best_boundary
            else:
                final_words.append(text[pos])
                pos += 1
        
        return final_words
    
    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        total = sum(weights.values())
        if total <= 0:
            return {k: 1.0 / len(weights) for k in weights}
        return {k: v / total for k, v in weights.items()}


class CascadeFusion:
    def __init__(self, confidence_threshold: float = 0.7):
        self.confidence_threshold = confidence_threshold
    
    def fuse(self, results: List[SegmenterResult], order: List[str]) -> List[str]:
        if not results:
            return []
        
        results_by_name = {r.segmenter_name: r for r in results}
        
        for segmenter_name in order:
            if segmenter_name in results_by_name:
                result = results_by_name[segmenter_name]
                if result.confidence >= self.confidence_threshold:
                    return result.words
        
        if results:
            best_result = max(results, key=lambda r: r.confidence)
            return best_result.words
        
        return []


class AdaptiveFusion:
    def __init__(self):
        self._text_classifier = TextClassifier()
        self._strategy_selector = StrategySelector()
    
    def fuse(
        self, 
        results: List[SegmenterResult], 
        text: str,
        dictionary: Optional['Dictionary'] = None
    ) -> List[str]:
        if not results:
            return []
        
        if len(results) == 1:
            return results[0].words
        
        text_features = self._text_classifier.extract_features(text)
        best_strategy = self._strategy_selector.select(text_features, results)
        
        if best_strategy == 'dict':
            dict_results = [r for r in results if r.segmenter_type == SegmenterType.RULE_BASED]
            if dict_results:
                return dict_results[0].words
        elif best_strategy == 'statistical':
            stat_results = [r for r in results if r.segmenter_type == SegmenterType.STATISTICAL]
            if stat_results:
                best = max(stat_results, key=lambda r: r.confidence)
                return best.words
        
        best_result = max(results, key=lambda r: r.confidence)
        return best_result.words


class TextClassifier:
    def extract_features(self, text: str) -> Dict[str, float]:
        features = {}
        
        features['length'] = len(text)
        features['avg_char_per_word_estimate'] = self._estimate_avg_word_length(text)
        features['digit_ratio'] = self._calc_digit_ratio(text)
        features['punct_ratio'] = self._calc_punct_ratio(text)
        features['english_ratio'] = self._calc_english_ratio(text)
        features['chinese_ratio'] = self._calc_chinese_ratio(text)
        
        return features
    
    def _estimate_avg_word_length(self, text: str) -> float:
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        if chinese_chars == 0:
            return 1.0
        return min(4.0, max(1.5, chinese_chars / max(1, text.count(' ') + 1)))
    
    def _calc_digit_ratio(self, text: str) -> float:
        if not text:
            return 0.0
        return sum(1 for c in text if c.isdigit()) / len(text)
    
    def _calc_punct_ratio(self, text: str) -> float:
        if not text:
            return 0.0
        puncts = set('，。！？、；：""''（）【】《》—…·')
        return sum(1 for c in text if c in puncts or c in '.,!?;:\"\'()[]<>') / len(text)
    
    def _calc_english_ratio(self, text: str) -> float:
        if not text:
            return 0.0
        return sum(1 for c in text if c.isalpha() and ord(c) < 128) / len(text)
    
    def _calc_chinese_ratio(self, text: str) -> float:
        if not text:
            return 0.0
        return sum(1 for c in text if '\u4e00' <= c <= '\u9fff') / len(text)


class StrategySelector:
    def select(self, features: Dict[str, float], results: List[SegmenterResult]) -> str:
        chinese_ratio = features.get('chinese_ratio', 0)
        english_ratio = features.get('english_ratio', 0)
        digit_ratio = features.get('digit_ratio', 0)
        
        if english_ratio > 0.5:
            return 'dict'
        
        if digit_ratio > 0.3:
            return 'dict'
        
        if chinese_ratio > 0.7:
            stat_results = [r for r in results if r.segmenter_type == SegmenterType.STATISTICAL]
            if stat_results:
                return 'statistical'
        
        return 'best_confidence'


class DeepLearningInterface:
    def __init__(self):
        self._model = None
        self._model_loaded = False
        self._model_path: Optional[str] = None
    
    def load_model(self, model_path: str, model_type: str = 'bert') -> bool:
        try:
            self._model_path = model_path
            self._model_loaded = True
            return True
        except Exception:
            return False
    
    def is_loaded(self) -> bool:
        return self._model_loaded
    
    def segment(self, text: str) -> SegmenterResult:
        if not self._model_loaded:
            return SegmenterResult(
                words=list(text),
                confidence=0.0,
                segmenter_type=SegmenterType.DEEP_LEARNING,
                segmenter_name='dl_unloaded'
            )
        
        words = self._predict(text)
        return SegmenterResult(
            words=words,
            confidence=0.9,
            segmenter_type=SegmenterType.DEEP_LEARNING,
            segmenter_name='bert'
        )
    
    def _predict(self, text: str) -> List[str]:
        return list(text)
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            'loaded': self._model_loaded,
            'model_path': self._model_path,
            'model_type': 'bert' if self._model_loaded else None
        }


class HybridSegmentor:
    def __init__(
        self,
        dictionary: Optional['Dictionary'] = None,
        config: Optional[HybridConfig] = None
    ):
        self._dictionary = dictionary
        self._config = config or HybridConfig()
        
        self._hmm_segmentor: Optional['HMMSegmentor'] = None
        self._crf_segmentor: Optional['CRFSegmentor'] = None
        self._perceptron_segmentor: Optional['PerceptronSegmentor'] = None
        self._lattice_segmentor: Optional['LatticeSegmentor'] = None
        self._dl_interface = DeepLearningInterface()
        
        self._confidence_estimator = ConfidenceEstimator()
        self._vote_fusion = VoteFusion()
        self._weighted_fusion = WeightedFusion()
        self._cascade_fusion = CascadeFusion(self._config.confidence_threshold)
        self._adaptive_fusion = AdaptiveFusion()
        
        self._result_cache: Dict[str, List[str]] = {}
    
    def set_dictionary(self, dictionary: 'Dictionary') -> None:
        self._dictionary = dictionary
    
    def set_hmm_segmentor(self, segmentor: 'HMMSegmentor') -> None:
        self._hmm_segmentor = segmentor
    
    def set_crf_segmentor(self, segmentor: 'CRFSegmentor') -> None:
        self._crf_segmentor = segmentor
    
    def set_perceptron_segmentor(self, segmentor: 'PerceptronSegmentor') -> None:
        self._perceptron_segmentor = segmentor
    
    def set_lattice_segmentor(self, segmentor: 'LatticeSegmentor') -> None:
        self._lattice_segmentor = segmentor
    
    def set_word_frequency(self, freq_dict: Dict[str, int]) -> None:
        self._confidence_estimator.set_word_frequency(freq_dict)
    
    def set_config(self, config: HybridConfig) -> None:
        self._config = config
        self._cascade_fusion = CascadeFusion(config.confidence_threshold)
    
    def load_dl_model(self, model_path: str, model_type: str = 'bert') -> bool:
        return self._dl_interface.load_model(model_path, model_type)
    
    def segment(self, text: str) -> List[str]:
        if not text:
            return []
        
        if self._config.enable_cache and text in self._result_cache:
            return self._result_cache[text]
        
        results = self._get_all_segmenter_results(text)
        
        if not results:
            return list(text)
        
        final_words = self._fuse_results(text, results)
        
        if self._config.enable_cache:
            if len(self._result_cache) >= self._config.cache_size:
                self._result_cache.clear()
            self._result_cache[text] = final_words
        
        return final_words
    
    def segment_with_details(self, text: str) -> Tuple[List[str], Dict[str, Any]]:
        if not text:
            return [], {'results': [], 'strategy': None}
        
        results = self._get_all_segmenter_results(text)
        
        if not results:
            return list(text), {'results': [], 'strategy': self._config.strategy.value}
        
        final_words = self._fuse_results(text, results)
        
        details = {
            'strategy': self._config.strategy.value,
            'results': [
                {
                    'segmenter': r.segmenter_name,
                    'words': r.words,
                    'confidence': r.confidence,
                    'type': r.segmenter_type.value
                }
                for r in results
            ],
            'final_confidence': self._calculate_final_confidence(results, final_words)
        }
        
        return final_words, details
    
    def _get_all_segmenter_results(self, text: str) -> List[SegmenterResult]:
        results = []
        
        dict_result = self._segment_with_dict(text)
        if dict_result:
            results.append(dict_result)
        
        hmm_result = self._segment_with_hmm(text)
        if hmm_result:
            results.append(hmm_result)
        
        crf_result = self._segment_with_crf(text)
        if crf_result:
            results.append(crf_result)
        
        perceptron_result = self._segment_with_perceptron(text)
        if perceptron_result:
            results.append(perceptron_result)
        
        lattice_result = self._segment_with_lattice(text)
        if lattice_result:
            results.append(lattice_result)
        
        dl_result = self._segment_with_dl(text)
        if dl_result and dl_result.confidence > 0:
            results.append(dl_result)
        
        return results
    
    def _segment_with_dict(self, text: str) -> Optional[SegmenterResult]:
        if self._dictionary is None:
            return None
        
        from .tokenizer import bidirectional_max_match
        
        words = bidirectional_max_match(text, self._dictionary)
        
        result = SegmenterResult(
            words=words,
            confidence=0.0,
            segmenter_type=SegmenterType.RULE_BASED,
            segmenter_name='dict'
        )
        
        result.confidence = self._confidence_estimator.estimate_from_result(result, self._dictionary)
        
        return result
    
    def _segment_with_hmm(self, text: str) -> Optional[SegmenterResult]:
        if self._hmm_segmentor is None or not self._hmm_segmentor.is_trained():
            return None
        
        words = self._hmm_segmentor.segment(text)
        
        result = SegmenterResult(
            words=words,
            confidence=0.0,
            segmenter_type=SegmenterType.STATISTICAL,
            segmenter_name='hmm'
        )
        
        result.confidence = self._confidence_estimator.estimate_from_result(result, self._dictionary)
        
        return result
    
    def _segment_with_crf(self, text: str) -> Optional[SegmenterResult]:
        if self._crf_segmentor is None or not self._crf_segmentor.is_trained():
            return None
        
        words = self._crf_segmentor.segment(text)
        
        result = SegmenterResult(
            words=words,
            confidence=0.0,
            segmenter_type=SegmenterType.STATISTICAL,
            segmenter_name='crf'
        )
        
        result.confidence = self._confidence_estimator.estimate_from_result(result, self._dictionary)
        
        return result
    
    def _segment_with_perceptron(self, text: str) -> Optional[SegmenterResult]:
        if self._perceptron_segmentor is None or not self._perceptron_segmentor.is_trained():
            return None
        
        words = self._perceptron_segmentor.segment(text)
        
        result = SegmenterResult(
            words=words,
            confidence=0.0,
            segmenter_type=SegmenterType.STATISTICAL,
            segmenter_name='perceptron'
        )
        
        result.confidence = self._confidence_estimator.estimate_from_result(result, self._dictionary)
        
        return result
    
    def _segment_with_lattice(self, text: str) -> Optional[SegmenterResult]:
        if self._lattice_segmentor is None:
            return None
        
        words = self._lattice_segmentor.segment(text)
        
        result = SegmenterResult(
            words=words,
            confidence=0.0,
            segmenter_type=SegmenterType.HYBRID,
            segmenter_name='lattice'
        )
        
        result.confidence = self._confidence_estimator.estimate_from_result(result, self._dictionary)
        
        return result
    
    def _segment_with_dl(self, text: str) -> Optional[SegmenterResult]:
        if not self._dl_interface.is_loaded():
            return None
        
        return self._dl_interface.segment(text)
    
    def _fuse_results(self, text: str, results: List[SegmenterResult]) -> List[str]:
        if len(results) == 1:
            return results[0].words
        
        strategy = self._config.strategy
        
        if strategy == HybridStrategy.VOTE:
            return self._vote_fusion.fuse(results)
        elif strategy == HybridStrategy.WEIGHTED:
            return self._weighted_fusion.fuse(results, self._config.weights)
        elif strategy == HybridStrategy.CASCADE:
            return self._cascade_fusion.fuse(results, self._config.cascade_order)
        elif strategy == HybridStrategy.ADAPTIVE:
            return self._adaptive_fusion.fuse(results, text, self._dictionary)
        elif strategy == HybridStrategy.CONFIDENCE:
            best_result = max(results, key=lambda r: r.confidence)
            return best_result.words
        else:
            return self._weighted_fusion.fuse(results, self._config.weights)
    
    def _calculate_final_confidence(self, results: List[SegmenterResult], final_words: List[str]) -> float:
        if not results:
            return 0.0
        
        final_result = SegmenterResult(
            words=final_words,
            segmenter_type=SegmenterType.HYBRID,
            segmenter_name='hybrid'
        )
        
        return self._confidence_estimator.estimate_from_result(final_result, self._dictionary)
    
    def get_available_segmenters(self) -> List[str]:
        available = []
        
        if self._dictionary is not None:
            available.append('dict')
        if self._hmm_segmentor is not None and self._hmm_segmentor.is_trained():
            available.append('hmm')
        if self._crf_segmentor is not None and self._crf_segmentor.is_trained():
            available.append('crf')
        if self._perceptron_segmentor is not None and self._perceptron_segmentor.is_trained():
            available.append('perceptron')
        if self._lattice_segmentor is not None:
            available.append('lattice')
        if self._dl_interface.is_loaded():
            available.append('dl')
        
        return available
    
    def get_config(self) -> HybridConfig:
        return self._config
    
    def clear_cache(self) -> None:
        self._result_cache.clear()


__all__ = [
    'HybridStrategy',
    'SegmenterType',
    'SegmenterResult',
    'HybridConfig',
    'ConfidenceEstimator',
    'VoteFusion',
    'WeightedFusion',
    'CascadeFusion',
    'AdaptiveFusion',
    'TextClassifier',
    'StrategySelector',
    'DeepLearningInterface',
    'HybridSegmentor',
]
