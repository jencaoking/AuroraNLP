__version__ = "0.2.0-beta"
__codename__ = "coca"
__version_info__ = (0, 2, 0, "beta")

from .segmentor import Segmentor
from .dictionary import Dictionary, UserDictionary, DictionaryManager
from .stopwords import StopWords
from .keyword_extractor import KeywordExtractor
from .similarity import Similarity
from .trie import Trie
from .benchmark import PerformanceBenchmark, BenchmarkResult, measure_time
from .batch_processor import BatchProcessor
from .hmm import HMMSegmentor, train_from_file
from .ngram import NGramModel, BigramModel, TrigramModel
from .crf import CRFModel, CRFSegmentor, CRFFeatureTemplate
from .perceptron import StructuredPerceptron, PerceptronSegmentor, PerceptronFeatureTemplate
from .lattice import Lattice, LatticeEdge, LatticeNode, LatticeBuilder, LatticeSegmentor, PathScorer
from .ambiguity import AmbiguityType, AmbiguityRegion, AmbiguityResult, AmbiguityDetector
from .new_word_detector import NewWordDetector, MutualInformation, EntropyCalculator
from .pos_tagger import POS_TAGS, DEFAULT_TAGS, HMMPOSTagger, CRFPOSTagger, train_pos_from_file
from .ner import (
    NER_ENTITY_TYPES,
    NER_TAGS,
    DEFAULT_NER_TAGS,
    Entity,
    NERFeatureTemplate,
    CRFNERModel,
    NERRecognizer,
    create_sample_ner_corpus,
    train_ner_from_file,
)

__all__ = [
    "Segmentor",
    "Dictionary",
    "UserDictionary",
    "DictionaryManager",
    "StopWords",
    "KeywordExtractor",
    "Similarity",
    "Trie",
    "PerformanceBenchmark",
    "BenchmarkResult",
    "measure_time",
    "BatchProcessor",
    "HMMSegmentor",
    "train_from_file",
    "NGramModel",
    "BigramModel",
    "TrigramModel",
    "CRFModel",
    "CRFSegmentor",
    "CRFFeatureTemplate",
    "StructuredPerceptron",
    "PerceptronSegmentor",
    "PerceptronFeatureTemplate",
    "Lattice",
    "LatticeEdge",
    "LatticeNode",
    "LatticeBuilder",
    "LatticeSegmentor",
    "PathScorer",
    "AmbiguityType",
    "AmbiguityRegion",
    "AmbiguityResult",
    "AmbiguityDetector",
    "NewWordDetector",
    "MutualInformation",
    "EntropyCalculator",
    "POS_TAGS",
    "DEFAULT_TAGS",
    "HMMPOSTagger",
    "CRFPOSTagger",
    "train_pos_from_file",
    "NER_ENTITY_TYPES",
    "NER_TAGS",
    "DEFAULT_NER_TAGS",
    "Entity",
    "NERFeatureTemplate",
    "CRFNERModel",
    "NERRecognizer",
    "create_sample_ner_corpus",
    "train_ner_from_file",
]
