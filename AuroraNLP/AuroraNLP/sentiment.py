"""
情感词典模块 - 中文情感分析基础

提供中文情感词典功能，包括：
- 正面/负面情感词库
- 情感强度标注（1-5级）
- 否定词处理
- 程度副词处理
- 情感极性计算
"""

import os
import re
from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum


class SentimentPolarity(Enum):
    """情感极性"""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class SentimentIntensity(Enum):
    """情感强度等级"""
    VERY_WEAK = 1
    WEAK = 2
    MEDIUM = 3
    STRONG = 4
    VERY_STRONG = 5


INTENSITY_SCORES = {
    SentimentIntensity.VERY_WEAK: 0.2,
    SentimentIntensity.WEAK: 0.4,
    SentimentIntensity.MEDIUM: 0.6,
    SentimentIntensity.STRONG: 0.8,
    SentimentIntensity.VERY_STRONG: 1.0,
}

INTENSITY_MAP = {
    1: SentimentIntensity.VERY_WEAK,
    2: SentimentIntensity.WEAK,
    3: SentimentIntensity.MEDIUM,
    4: SentimentIntensity.STRONG,
    5: SentimentIntensity.VERY_STRONG,
}


@dataclass
class SentimentWord:
    """情感词条目"""
    word: str
    polarity: SentimentPolarity
    intensity: SentimentIntensity
    category: str = ""
    examples: List[str] = field(default_factory=list)
    
    @property
    def score(self) -> float:
        """获取情感分数"""
        base_score = INTENSITY_SCORES.get(self.intensity, 0.5)
        if self.polarity == SentimentPolarity.NEGATIVE:
            return -base_score
        return base_score


@dataclass
class DegreeWord:
    """程度副词条目"""
    word: str
    degree: float
    category: str = ""


@dataclass
class NegationWord:
    """否定词条目"""
    word: str
    strength: float = 1.0


@dataclass
class SentimentResult:
    """情感分析结果"""
    text: str
    score: float
    polarity: SentimentPolarity
    positive_words: List[Tuple[str, float]] = field(default_factory=list)
    negative_words: List[Tuple[str, float]] = field(default_factory=list)
    intensity: float = 0.0
    confidence: float = 0.0


class SentimentDictionary:
    """情感词典类"""
    
    DEFAULT_DICT_PATH = os.path.join(os.path.dirname(__file__), 'data', 'sentiment.txt')
    DEFAULT_NEGATION_PATH = os.path.join(os.path.dirname(__file__), 'data', 'negation_words.txt')
    DEFAULT_DEGREE_PATH = os.path.join(os.path.dirname(__file__), 'data', 'degree_words.txt')
    
    POSITIVE_CATEGORIES = {
        "joy": "喜悦",
        "love": "喜爱",
        "praise": "赞扬",
        "trust": "信任",
        "anticipation": "期待",
        "surprise_good": "惊喜",
    }
    
    NEGATIVE_CATEGORIES = {
        "anger": "愤怒",
        "sadness": "悲伤",
        "fear": "恐惧",
        "disgust": "厌恶",
        "surprise_bad": "震惊",
    }
    
    def __init__(self, load_default: bool = True):
        self._positive_words: Dict[str, SentimentWord] = {}
        self._negative_words: Dict[str, SentimentWord] = {}
        self._negation_words: Dict[str, NegationWord] = {}
        self._degree_words: Dict[str, DegreeWord] = {}
        self._word_count: int = 0
        self._loaded: bool = False
        
        if load_default:
            self._load_default_dictionaries()
    
    def _load_default_dictionaries(self) -> None:
        """加载默认词典"""
        if os.path.exists(self.DEFAULT_DICT_PATH):
            self.load_sentiment_words(self.DEFAULT_DICT_PATH)
        else:
            self._load_builtin_sentiment_words()
        
        if os.path.exists(self.DEFAULT_NEGATION_PATH):
            self.load_negation_words(self.DEFAULT_NEGATION_PATH)
        else:
            self._load_builtin_negation_words()
        
        if os.path.exists(self.DEFAULT_DEGREE_PATH):
            self.load_degree_words(self.DEFAULT_DEGREE_PATH)
        else:
            self._load_builtin_degree_words()
    
    def _load_builtin_sentiment_words(self) -> None:
        """加载内置情感词"""
        positive_words = [
            ("好", 3, "joy"), ("很好", 4, "joy"), ("非常好", 5, "joy"),
            ("开心", 3, "joy"), ("高兴", 3, "joy"), ("快乐", 3, "joy"),
            ("愉快", 3, "joy"), ("欢乐", 4, "joy"), ("幸福", 4, "joy"),
            ("满意", 3, "joy"), ("欣喜", 4, "joy"), ("狂喜", 5, "joy"),
            ("喜欢", 3, "love"), ("喜爱", 4, "love"), ("热爱", 5, "love"),
            ("棒", 4, "praise"), ("优秀", 5, "praise"), ("完美", 5, "praise"),
            ("精彩", 4, "praise"), ("出色", 4, "praise"), ("杰出", 5, "praise"),
            ("赞", 4, "praise"), ("信任", 4, "trust"), ("期待", 3, "anticipation"),
            ("惊喜", 4, "surprise_good"),
        ]
        
        negative_words = [
            ("愤怒", 4, "anger"), ("生气", 3, "anger"), ("气愤", 4, "anger"),
            ("悲伤", 4, "sadness"), ("难过", 3, "sadness"), ("伤心", 3, "sadness"),
            ("痛苦", 4, "sadness"), ("失望", 3, "sadness"), ("绝望", 5, "sadness"),
            ("害怕", 3, "fear"), ("恐惧", 4, "fear"), ("担心", 3, "fear"),
            ("焦虑", 4, "fear"), ("紧张", 3, "fear"), ("恐慌", 5, "fear"),
            ("恶心", 4, "disgust"), ("反感", 3, "disgust"), ("厌恶", 4, "disgust"),
            ("讨厌", 3, "disgust"), ("憎恨", 5, "disgust"),
            ("震惊", 4, "surprise_bad"), ("糟糕", 3, "negative"),
            ("差", 2, "negative"), ("坏", 3, "negative"), ("失败", 4, "negative"),
        ]
        
        for word, intensity, category in positive_words:
            self.add_positive_word(word, INTENSITY_MAP.get(intensity, SentimentIntensity.MEDIUM), category)
        
        for word, intensity, category in negative_words:
            self.add_negative_word(word, INTENSITY_MAP.get(intensity, SentimentIntensity.MEDIUM), category)
    
    def _load_builtin_negation_words(self) -> None:
        """加载内置否定词"""
        negations = [
            ("不", 1.0), ("没", 1.0), ("没有", 1.0), ("无", 1.0),
            ("非", 1.0), ("勿", 1.0), ("别", 1.0), ("未", 1.0),
            ("不曾", 1.0), ("未必", 0.8), ("难以", 0.8),
        ]
        for word, strength in negations:
            self.add_negation_word(word, strength)
    
    def _load_builtin_degree_words(self) -> None:
        """加载内置程度副词"""
        degrees = [
            ("很", 1.2, "很"), ("非常", 1.4, "很"), ("十分", 1.4, "很"),
            ("特别", 1.5, "很"), ("相当", 1.4, "很"), ("挺", 1.2, "很"),
            ("极", 1.8, "极其"), ("极其", 1.8, "极其"), ("最", 2.0, "极其"),
            ("太", 1.6, "极其"), ("过于", 1.6, "极其"), ("超级", 1.8, "极其"),
            ("稍微", 0.3, "轻微"), ("有点", 0.4, "轻微"), ("有些", 0.4, "轻微"),
            ("比较", 0.6, "比较"), ("相对", 0.6, "比较"), ("更", 1.3, "更"),
            ("更加", 1.4, "更"), ("越", 1.3, "更"), ("越来越", 1.5, "更"),
        ]
        for word, degree, category in degrees:
            self.add_degree_word(word, degree, category)
    
    def load_sentiment_words(self, path: str) -> None:
        """从文件加载情感词"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"情感词典文件不存在: {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split('\t')
                if len(parts) < 3:
                    continue
                
                word = parts[0].strip()
                polarity_str = parts[1].strip().lower()
                intensity = int(parts[2].strip())
                category = parts[3].strip() if len(parts) > 3 else ""
                
                if polarity_str == 'positive':
                    polarity = SentimentPolarity.POSITIVE
                elif polarity_str == 'negative':
                    polarity = SentimentPolarity.NEGATIVE
                else:
                    continue
                
                intensity_enum = INTENSITY_MAP.get(intensity, SentimentIntensity.MEDIUM)
                
                if polarity == SentimentPolarity.POSITIVE:
                    self.add_positive_word(word, intensity_enum, category)
                else:
                    self.add_negative_word(word, intensity_enum, category)
        
        self._loaded = True
    
    def load_negation_words(self, path: str) -> None:
        """从文件加载否定词"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"否定词文件不存在: {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split('\t')
                word = parts[0].strip()
                strength = float(parts[1]) if len(parts) > 1 else 1.0
                self.add_negation_word(word, strength)
    
    def load_degree_words(self, path: str) -> None:
        """从文件加载程度副词"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"程度副词文件不存在: {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split('\t')
                if len(parts) < 2:
                    continue
                
                word = parts[0].strip()
                degree = float(parts[1])
                category = parts[2].strip() if len(parts) > 2 else ""
                self.add_degree_word(word, degree, category)
    
    def add_positive_word(self, word: str, intensity: SentimentIntensity = SentimentIntensity.MEDIUM, category: str = "") -> None:
        """添加正面情感词"""
        sentiment_word = SentimentWord(
            word=word,
            polarity=SentimentPolarity.POSITIVE,
            intensity=intensity,
            category=category
        )
        self._positive_words[word] = sentiment_word
        self._word_count += 1
    
    def add_negative_word(self, word: str, intensity: SentimentIntensity = SentimentIntensity.MEDIUM, category: str = "") -> None:
        """添加负面情感词"""
        sentiment_word = SentimentWord(
            word=word,
            polarity=SentimentPolarity.NEGATIVE,
            intensity=intensity,
            category=category
        )
        self._negative_words[word] = sentiment_word
        self._word_count += 1
    
    def add_negation_word(self, word: str, strength: float = 1.0) -> None:
        """添加否定词"""
        self._negation_words[word] = NegationWord(word=word, strength=strength)
    
    def add_degree_word(self, word: str, degree: float, category: str = "") -> None:
        """添加程度副词"""
        self._degree_words[word] = DegreeWord(word=word, degree=degree, category=category)
    
    def get_sentiment_word(self, word: str) -> Optional[SentimentWord]:
        """获取情感词信息"""
        if word in self._positive_words:
            return self._positive_words[word]
        if word in self._negative_words:
            return self._negative_words[word]
        return None
    
    def is_positive(self, word: str) -> bool:
        """判断是否为正面情感词"""
        return word in self._positive_words
    
    def is_negative(self, word: str) -> bool:
        """判断是否为负面情感词"""
        return word in self._negative_words
    
    def is_sentiment_word(self, word: str) -> bool:
        """判断是否为情感词"""
        return self.is_positive(word) or self.is_negative(word)
    
    def is_negation_word(self, word: str) -> bool:
        """判断是否为否定词"""
        return word in self._negation_words
    
    def is_degree_word(self, word: str) -> bool:
        """判断是否为程度副词"""
        return word in self._degree_words
    
    def get_word_score(self, word: str) -> float:
        """获取词语情感分数"""
        sentiment_word = self.get_sentiment_word(word)
        if sentiment_word:
            return sentiment_word.score
        return 0.0
    
    def get_word_intensity(self, word: str) -> Optional[SentimentIntensity]:
        """获取词语情感强度"""
        sentiment_word = self.get_sentiment_word(word)
        if sentiment_word:
            return sentiment_word.intensity
        return None
    
    def get_word_category(self, word: str) -> Optional[str]:
        """获取词语情感类别"""
        sentiment_word = self.get_sentiment_word(word)
        if sentiment_word:
            return sentiment_word.category
        return None
    
    def get_positive_words(self) -> List[str]:
        """获取所有正面情感词"""
        return list(self._positive_words.keys())
    
    def get_negative_words(self) -> List[str]:
        """获取所有负面情感词"""
        return list(self._negative_words.keys())
    
    def get_words_by_category(self, category: str) -> List[str]:
        """按类别获取情感词"""
        words = []
        for word, sentiment in self._positive_words.items():
            if sentiment.category == category:
                words.append(word)
        for word, sentiment in self._negative_words.items():
            if sentiment.category == category:
                words.append(word)
        return words
    
    def get_words_by_intensity(self, intensity: SentimentIntensity) -> List[str]:
        """按强度获取情感词"""
        words = []
        for word, sentiment in self._positive_words.items():
            if sentiment.intensity == intensity:
                words.append(word)
        for word, sentiment in self._negative_words.items():
            if sentiment.intensity == intensity:
                words.append(word)
        return words
    
    def get_degree(self, word: str) -> float:
        """获取程度副词的程度值"""
        degree_word = self._degree_words.get(word)
        if degree_word:
            return degree_word.degree
        return 1.0
    
    def get_negation_strength(self, word: str) -> float:
        """获取否定词的否定强度"""
        negation_word = self._negation_words.get(word)
        if negation_word:
            return negation_word.strength
        return 0.0
    
    def analyze(self, text: str, words: List[str] = None) -> SentimentResult:
        """分析文本情感"""
        if words is None:
            words = list(text)
        
        positive_words = []
        negative_words = []
        total_score = 0.0
        
        i = 0
        while i < len(words):
            word = words[i]
            
            # 检查否定词
            negation = 1.0
            if i > 0 and self.is_negation_word(words[i - 1]):
                negation = -1.0
            
            # 检查程度副词
            degree = 1.0
            if i > 0 and self.is_degree_word(words[i - 1]):
                degree = self.get_degree(words[i - 1])
            
            sentiment_word = self.get_sentiment_word(word)
            if sentiment_word:
                score = sentiment_word.score * negation * degree
                total_score += score
                
                if score > 0:
                    positive_words.append((word, score))
                elif score < 0:
                    negative_words.append((word, score))
            
            i += 1
        
        # 确定极性
        if total_score > 0:
            polarity = SentimentPolarity.POSITIVE
        elif total_score < 0:
            polarity = SentimentPolarity.NEGATIVE
        else:
            polarity = SentimentPolarity.NEUTRAL
        
        # 计算置信度
        word_count = len(positive_words) + len(negative_words)
        confidence = min(1.0, word_count * 0.2 + 0.3) if word_count > 0 else 0.0
        
        # 计算强度
        intensity = abs(total_score)
        
        return SentimentResult(
            text=text,
            score=max(-1.0, min(1.0, total_score)),
            polarity=polarity,
            positive_words=positive_words,
            negative_words=negative_words,
            intensity=intensity,
            confidence=confidence
        )
    
    def get_positive_word_count(self) -> int:
        """获取正面情感词数量"""
        return len(self._positive_words)
    
    def get_negative_word_count(self) -> int:
        """获取负面情感词数量"""
        return len(self._negative_words)
    
    def get_total_word_count(self) -> int:
        """获取情感词总数"""
        return len(self._positive_words) + len(self._negative_words)
    
    def get_negation_word_count(self) -> int:
        """获取否定词数量"""
        return len(self._negation_words)
    
    def get_degree_word_count(self) -> int:
        """获取程度副词数量"""
        return len(self._degree_words)
    
    def is_loaded(self) -> bool:
        """检查词典是否已加载"""
        return self._loaded or self.get_total_word_count() > 0
    
    def save_sentiment_words(self, path: str) -> None:
        """保存情感词到文件"""
        with open(path, 'w', encoding='utf-8') as f:
            f.write("# AuroraNLP 情感词典\\n")
            f.write("# 格式: 词语\\t极性\\t强度\\t类别\\n\\n")
            
            f.write("# 正面情感词\\n")
            for word, sentiment in sorted(self._positive_words.items()):
                f.write(f"{word}\\t{sentiment.polarity.value}\\t{sentiment.intensity.value}\\t{sentiment.category}\\n")
            
            f.write("\\n# 负面情感词\\n")
            for word, sentiment in sorted(self._negative_words.items()):
                f.write(f"{word}\\t{sentiment.polarity.value}\\t{sentiment.intensity.value}\\t{sentiment.category}\\n")
    
    def __len__(self) -> int:
        return self.get_total_word_count()
    
    def __contains__(self, word: str) -> bool:
        return self.is_sentiment_word(word)


class SentimentAnalyzer:
    """情感分析器"""
    
    def __init__(self, dictionary: SentimentDictionary = None):
        self._dictionary = dictionary or SentimentDictionary()
    
    def analyze(self, text: str, words: List[str] = None) -> SentimentResult:
        """分析文本情感"""
        return self._dictionary.analyze(text, words)
    
    def get_dictionary(self) -> SentimentDictionary:
        """获取情感词典"""
        return self._dictionary


__all__ = [
    'SentimentPolarity',
    'SentimentIntensity',
    'SentimentWord',
    'DegreeWord',
    'NegationWord',
    'SentimentResult',
    'SentimentDictionary',
    'SentimentAnalyzer',
]