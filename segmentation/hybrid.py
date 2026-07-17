from typing import List, Tuple, Dict, Optional, Callable, Any, Union, TYPE_CHECKING
from dataclasses import dataclass, field
from enum import Enum
from collections import Counter, defaultdict
from abc import ABC, abstractmethod
import math
import pickle
import os

if TYPE_CHECKING:
    from AuroraNLP.dictionary.dictionary import Dictionary, DictionaryManager
    from AuroraNLP.segmentation.hmm import HMMSegmentor
    from AuroraNLP.segmentation.crf import CRFSegmentor
    from AuroraNLP.segmentation.perceptron import PerceptronSegmentor
    from AuroraNLP.segmentation.lattice import LatticeSegmentor, Lattice


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


@dataclass
class FusionContext:
    """融合上下文，包含融合所需的所有信息"""
    text: str
    results: List[SegmenterResult]
    dictionary: Optional['Dictionary'] = None
    config: Optional[HybridConfig] = None
    word_frequency: Optional[Dict[str, int]] = None


class FusionStrategy(ABC):
    """融合策略抽象基类"""
    
    @abstractmethod
    def fuse(self, context: FusionContext) -> List[str]:
        """执行融合策略，返回最终分词结果"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """获取策略名称"""
        pass
    
    def calculate_confidence(self, context: FusionContext, result: List[str]) -> float:
        """计算融合结果的置信度，默认实现"""
        return 0.8


class VoteFusionStrategy(FusionStrategy):
    """投票融合策略"""
    
    def fuse(self, context: FusionContext) -> List[str]:
        results = context.results
        if not results:
            return []
        
        if len(results) == 1:
            return results[0].words
        
        text = context.text
        text_length = len(text)
        
        position_votes: Dict[int, Counter] = defaultdict(Counter)
        
        for result in results:
            pos = 0
            for word in result.words:
                boundary = pos + len(word)
                position_votes[pos][boundary] += 1
                pos = boundary
        
        boundaries = self._collect_boundaries(position_votes, text_length)
        return self._reconstruct_words(boundaries, text)
    
    def _collect_boundaries(self, position_votes: Dict[int, Counter], text_length: int) -> List[int]:
        boundaries = []
        pos = 0
        
        while pos < text_length:
            if pos in position_votes:
                best_boundary = position_votes[pos].most_common(1)[0][0]
                boundaries.append(best_boundary)
                pos = best_boundary
            else:
                boundaries.append(pos + 1)
                pos += 1
        
        return boundaries
    
    def _reconstruct_words(self, boundaries: List[int], text: str) -> List[str]:
        if not boundaries or not text:
            return []
        
        words = []
        start = 0
        
        for boundary in boundaries:
            if boundary > start:
                words.append(text[start:boundary])
            start = boundary
        
        return words
    
    def get_name(self) -> str:
        return "vote"


class WeightedFusionStrategy(FusionStrategy):
    """加权融合策略"""
    
    def __init__(self):
        self._position_scores: Dict[int, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
    
    def fuse(self, context: FusionContext) -> List[str]:
        results = context.results
        if not results:
            return []
        
        if len(results) == 1:
            return results[0].words
        
        config = context.config
        weights = config.weights if config else {}
        normalized_weights = self._normalize_weights(weights)
        
        # 使用原始文本长度计算
        text_length = len(context.text)
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
        text = context.text
        
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
    
    def get_name(self) -> str:
        return "weighted"


class CascadeFusionStrategy(FusionStrategy):
    """级联融合策略"""
    
    def __init__(self, confidence_threshold: float = 0.7):
        self.confidence_threshold = confidence_threshold
    
    def fuse(self, context: FusionContext) -> List[str]:
        results = context.results
        if not results:
            return []
        
        config = context.config
        order = config.cascade_order if config else []
        
        # 使用配置中的置信度阈值
        threshold = config.confidence_threshold if config else self.confidence_threshold
        
        results_by_name = {r.segmenter_name: r for r in results}
        
        for segmenter_name in order:
            if segmenter_name in results_by_name:
                result = results_by_name[segmenter_name]
                if result.confidence >= threshold:
                    return result.words
        
        if results:
            best_result = max(results, key=lambda r: r.confidence)
            return best_result.words
        
        return []
    
    def get_name(self) -> str:
        return "cascade"


class AdaptiveFusionStrategy(FusionStrategy):
    """自适应融合策略"""
    
    def __init__(self):
        self._text_classifier = TextClassifier()
        self._strategy_selector = StrategySelector()
    
    def fuse(self, context: FusionContext) -> List[str]:
        results = context.results
        if not results:
            return []
        
        if len(results) == 1:
            return results[0].words
        
        text = context.text
        features = self._text_classifier.extract_features(text)
        
        selected_strategy = self._strategy_selector.select(features, results)
        
        if selected_strategy == 'dict':
            dict_results = [r for r in results if r.segmenter_name == 'dict']
            if dict_results:
                return dict_results[0].words
        elif selected_strategy == 'statistical':
            stat_results = [r for r in results if r.segmenter_type == SegmenterType.STATISTICAL]
            if stat_results:
                best_stat = max(stat_results, key=lambda r: r.confidence)
                return best_stat.words
        
        # 默认使用置信度最高的结果
        best_result = max(results, key=lambda r: r.confidence)
        return best_result.words
    
    def get_name(self) -> str:
        return "adaptive"





class ConfidenceFusionStrategy(FusionStrategy):
    """置信度融合策略"""
    
    def fuse(self, context: FusionContext) -> List[str]:
        results = context.results
        if not results:
            return []
        
        best_result = max(results, key=lambda r: r.confidence)
        return best_result.words
    
    def get_name(self) -> str:
        return "confidence"


class FusionStrategyFactory:
    """融合策略工厂"""
    
    def __init__(self):
        self._strategies: Dict[str, FusionStrategy] = {}
        self._register_default_strategies()
    
    def _register_default_strategies(self):
        """注册默认的融合策略"""
        self.register_strategy('vote', VoteFusionStrategy())
        self.register_strategy('weighted', WeightedFusionStrategy())
        self.register_strategy('cascade', CascadeFusionStrategy())
        self.register_strategy('adaptive', AdaptiveFusionStrategy())
        self.register_strategy('confidence', ConfidenceFusionStrategy())
    
    def register_strategy(self, name: str, strategy: FusionStrategy):
        """注册新的融合策略"""
        self._strategies[name] = strategy
    
    def get_strategy(self, name: str) -> Optional[FusionStrategy]:
        """获取指定名称的融合策略"""
        return self._strategies.get(name)
    
    def get_available_strategies(self) -> List[str]:
        """获取所有可用的融合策略名称"""
        return list(self._strategies.keys())
    
    def create_strategy(self, name: str, config: Optional[HybridConfig] = None) -> Optional[FusionStrategy]:
        """创建融合策略实例（支持配置）"""
        strategy = self.get_strategy(name)
        if strategy and config:
            # 如果策略需要配置，可以在这里进行配置
            if isinstance(strategy, CascadeFusionStrategy):
                strategy.confidence_threshold = config.confidence_threshold
        return strategy


class ConfidenceEstimator(ABC):
    """置信度估计器抽象基类"""
    
    @abstractmethod
    def estimate(self, result: SegmenterResult, context: Optional[Dict[str, Any]] = None) -> float:
        """估计分词结果的置信度"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """获取估计器名称"""
        pass


class CoverageConfidenceEstimator(ConfidenceEstimator):
    """基于覆盖率的置信度估计器"""
    
    def estimate(self, result: SegmenterResult, context: Optional[Dict[str, Any]] = None) -> float:
        if not result.words:
            return 0.0
        
        dictionary = context.get('dictionary') if context else None
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
    
    def get_name(self) -> str:
        return "coverage"


class WordQualityConfidenceEstimator(ConfidenceEstimator):
    """基于词质量的置信度估计器"""
    
    def __init__(self):
        self._word_freq: Dict[str, int] = {}
        self._total_words: int = 0
    
    def set_word_frequency(self, freq_dict: Dict[str, int]) -> None:
        self._word_freq = freq_dict
        self._total_words = sum(freq_dict.values())
    
    def estimate(self, result: SegmenterResult, context: Optional[Dict[str, Any]] = None) -> float:
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
    
    def get_name(self) -> str:
        return "word_quality"


class SegmentationQualityConfidenceEstimator(ConfidenceEstimator):
    """基于分词质量的置信度估计器"""
    
    def estimate(self, result: SegmenterResult, context: Optional[Dict[str, Any]] = None) -> float:
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
    
    def get_name(self) -> str:
        return "segmentation_quality"


class CompositeConfidenceEstimator(ConfidenceEstimator):
    """组合置信度估计器"""
    
    def __init__(self, estimators: Optional[List[ConfidenceEstimator]] = None):
        self.estimators = estimators or [
            CoverageConfidenceEstimator(),
            WordQualityConfidenceEstimator(),
            SegmentationQualityConfidenceEstimator()
        ]
    
    def add_estimator(self, estimator: ConfidenceEstimator):
        """添加置信度估计器"""
        self.estimators.append(estimator)
    
    def remove_estimator(self, name: str):
        """移除指定名称的置信度估计器"""
        self.estimators = [e for e in self.estimators if e.get_name() != name]
    
    def estimate(self, result: SegmenterResult, context: Optional[Dict[str, Any]] = None) -> float:
        if not result.words:
            return 0.0
        
        scores = []
        for estimator in self.estimators:
            score = estimator.estimate(result, context)
            scores.append(score)
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def get_name(self) -> str:
        return "composite"


class LegacyConfidenceEstimator:
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
    """混合分词器（重构后）"""
    
    def __init__(
        self,
        dictionary: Optional['Dictionary'] = None,
        config: Optional[HybridConfig] = None
    ):
        self._dictionary = dictionary
        self._config = config or HybridConfig()
        
        # 分词器组件
        self._hmm_segmentor: Optional['HMMSegmentor'] = None
        self._crf_segmentor: Optional['CRFSegmentor'] = None
        self._perceptron_segmentor: Optional['PerceptronSegmentor'] = None
        self._lattice_segmentor: Optional['LatticeSegmentor'] = None
        self._dl_interface = DeepLearningInterface()
        
        # 融合策略组件
        self._strategy_factory = FusionStrategyFactory()
        self._confidence_estimator = CompositeConfidenceEstimator()
        
        # 缓存组件
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
        """设置词频，用于置信度估计"""
        for estimator in self._confidence_estimator.estimators:
            if isinstance(estimator, WordQualityConfidenceEstimator):
                estimator.set_word_frequency(freq_dict)
    
    def set_config(self, config: HybridConfig) -> None:
        self._config = config
    
    def load_dl_model(self, model_path: str, model_type: str = 'bert') -> bool:
        return self._dl_interface.load_model(model_path, model_type)
    
    def register_fusion_strategy(self, name: str, strategy: FusionStrategy):
        """注册新的融合策略"""
        self._strategy_factory.register_strategy(name, strategy)
    
    def segment(self, text: str) -> List[str]:
        """分词主方法"""
        if not text:
            return []
        
        if self._config.enable_cache and text in self._result_cache:
            return self._result_cache[text]
        
        # 获取所有分词器的结果
        results = self._get_all_segmenter_results(text)
        
        if not results:
            return list(text)
        
        # 构建融合上下文
        context = self._build_fusion_context(text, results)
        
        # 获取融合策略并执行融合
        strategy_name = self._config.strategy.value
        strategy = self._strategy_factory.create_strategy(strategy_name, self._config)
        
        if strategy is None:
            strategy = self._strategy_factory.get_strategy('weighted')
        
        final_words = strategy.fuse(context)
        
        # 缓存结果
        if self._config.enable_cache:
            if len(self._result_cache) >= self._config.cache_size:
                self._result_cache.clear()
            self._result_cache[text] = final_words
        
        return final_words
    
    def segment_with_details(self, text: str) -> Tuple[List[str], Dict[str, Any]]:
        """带详情的分词方法"""
        if not text:
            return [], {'results': [], 'strategy': None}
        
        results = self._get_all_segmenter_results(text)
        
        if not results:
            return list(text), {'results': [], 'strategy': self._config.strategy.value}
        
        context = self._build_fusion_context(text, results)
        
        strategy_name = self._config.strategy.value
        strategy = self._strategy_factory.create_strategy(strategy_name, self._config)
        
        if strategy is None:
            strategy = self._strategy_factory.get_strategy('weighted')
        
        final_words = strategy.fuse(context)
        
        details = {
            'strategy': strategy_name,
            'results': [
                {
                    'segmenter': r.segmenter_name,
                    'words': r.words,
                    'confidence': r.confidence,
                    'type': r.segmenter_type.value
                }
                for r in results
            ],
            'final_confidence': self._calculate_final_confidence(final_words)
        }
        
        return final_words, details
    
    def _build_fusion_context(self, text: str, results: List[SegmenterResult]) -> FusionContext:
        """构建融合上下文"""
        return FusionContext(
            text=text,
            results=results,
            dictionary=self._dictionary,
            config=self._config
        )
    
    def _get_all_segmenter_results(self, text: str) -> List[SegmenterResult]:
        """获取所有分词器的结果"""
        results = []
        
        # 字典分词
        dict_result = self._segment_with_dict(text)
        if dict_result:
            results.append(dict_result)
        
        # HMM分词
        hmm_result = self._segment_with_hmm(text)
        if hmm_result:
            results.append(hmm_result)
        
        # CRF分词
        crf_result = self._segment_with_crf(text)
        if crf_result:
            results.append(crf_result)
        
        # 感知机分词
        perceptron_result = self._segment_with_perceptron(text)
        if perceptron_result:
            results.append(perceptron_result)
        
        # 格网分词
        lattice_result = self._segment_with_lattice(text)
        if lattice_result:
            results.append(lattice_result)
        
        # 深度学习分词
        dl_result = self._segment_with_dl(text)
        if dl_result and dl_result.confidence > 0:
            results.append(dl_result)
        
        return results
    
    def _segment_with_dict(self, text: str) -> Optional[SegmenterResult]:
        """字典分词"""
        if self._dictionary is None:
            return None
        
        from AuroraNLP.core.tokenizer import bidirectional_max_match
        
        words = bidirectional_max_match(text, self._dictionary)
        
        result = SegmenterResult(
            words=words,
            confidence=0.0,
            segmenter_type=SegmenterType.RULE_BASED,
            segmenter_name='dict'
        )
        
        # 使用置信度估计器
        context = {'dictionary': self._dictionary}
        result.confidence = self._confidence_estimator.estimate(result, context)
        
        return result
    
    def _segment_with_hmm(self, text: str) -> Optional[SegmenterResult]:
        """HMM分词"""
        if self._hmm_segmentor is None or not self._hmm_segmentor.is_trained():
            return None
        
        words = self._hmm_segmentor.segment(text)
        
        result = SegmenterResult(
            words=words,
            confidence=0.0,
            segmenter_type=SegmenterType.STATISTICAL,
            segmenter_name='hmm'
        )
        
        context = {'dictionary': self._dictionary}
        result.confidence = self._confidence_estimator.estimate(result, context)
        
        return result
    
    def _segment_with_crf(self, text: str) -> Optional[SegmenterResult]:
        """CRF分词"""
        if self._crf_segmentor is None or not self._crf_segmentor.is_trained():
            return None
        
        words = self._crf_segmentor.segment(text)
        
        result = SegmenterResult(
            words=words,
            confidence=0.0,
            segmenter_type=SegmenterType.STATISTICAL,
            segmenter_name='crf'
        )
        
        context = {'dictionary': self._dictionary}
        result.confidence = self._confidence_estimator.estimate(result, context)
        
        return result
    
    def _segment_with_perceptron(self, text: str) -> Optional[SegmenterResult]:
        """感知机分词"""
        if self._perceptron_segmentor is None or not self._perceptron_segmentor.is_trained():
            return None
        
        words = self._perceptron_segmentor.segment(text)
        
        result = SegmenterResult(
            words=words,
            confidence=0.0,
            segmenter_type=SegmenterType.STATISTICAL,
            segmenter_name='perceptron'
        )
        
        context = {'dictionary': self._dictionary}
        result.confidence = self._confidence_estimator.estimate(result, context)
        
        return result
    
    def _segment_with_lattice(self, text: str) -> Optional[SegmenterResult]:
        """格网分词"""
        if self._lattice_segmentor is None:
            return None
        
        words = self._lattice_segmentor.segment(text)
        
        result = SegmenterResult(
            words=words,
            confidence=0.0,
            segmenter_type=SegmenterType.HYBRID,
            segmenter_name='lattice'
        )
        
        context = {'dictionary': self._dictionary}
        result.confidence = self._confidence_estimator.estimate(result, context)
        
        return result
    
    def _segment_with_dl(self, text: str) -> Optional[SegmenterResult]:
        """深度学习分词"""
        if not self._dl_interface.is_loaded():
            return None
        
        return self._dl_interface.segment(text)
    
    def _calculate_final_confidence(self, final_words: List[str]) -> float:
        """计算最终结果的置信度"""
        if not final_words:
            return 0.0
        
        result = SegmenterResult(
            words=final_words,
            segmenter_type=SegmenterType.HYBRID,
            segmenter_name='hybrid'
        )
        
        context = {'dictionary': self._dictionary}
        return self._confidence_estimator.estimate(result, context)
    
    def get_available_segmenters(self) -> List[str]:
        """获取可用的分词器列表"""
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
    
    def get_available_strategies(self) -> List[str]:
        """获取可用的融合策略列表"""
        return self._strategy_factory.get_available_strategies()
    
    def get_config(self) -> HybridConfig:
        """获取当前配置"""
        return self._config
    
    def clear_cache(self) -> None:
        """清空缓存"""
        self._result_cache.clear()


__all__ = [
    'HybridStrategy',
    'SegmenterType',
    'SegmenterResult',
    'HybridConfig',
    'FusionContext',
    'FusionStrategy',
    'VoteFusionStrategy',
    'WeightedFusionStrategy',
    'CascadeFusionStrategy',
    'AdaptiveFusionStrategy',
    'ConfidenceFusionStrategy',
    'FusionStrategyFactory',
    'ConfidenceEstimator',
    'CoverageConfidenceEstimator',
    'WordQualityConfidenceEstimator',
    'SegmentationQualityConfidenceEstimator',
    'CompositeConfidenceEstimator',
    'TextClassifier',
    'StrategySelector',
    'DeepLearningInterface',
    'HybridSegmentor',
]
