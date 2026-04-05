"""
同义词词林模块 - 开放词林整合

基于哈工大同义词词林扩展版格式，提供同义词、近义词和语义分类功能。

编码格式说明：
- 编码长度为7-8位，如 Aa01A01= 或 Aa01A02#
- 第1位：大类（A-Z），共12个大类
- 第2位：中类（a-z），每个大类下最多26个中类
- 第3-4位：小类编号（01-99）
- 第5位：词群编号（A-Z）
- 第6-7位：词群内编号（01-99）
- 第8位：词语关系标记
  - '=' 表示同义词（语义完全相同）
  - '#' 表示相关词（语义相关但不完全相同）
  - '@' 表示独立词（无同义词）

大类说明：
A: 人物
B: 物品
C: 时间
D: 空间
E: 抽象事物
F: 特征
G: 动作
H: 心理活动
I: 活动
J: 现象状态
K: 关联
L: 助语
"""

import os
from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum


class WordRelation(Enum):
    SYNONYM = "="
    RELATED = "#"
    INDEPENDENT = "@"


CATEGORY_NAMES: Dict[str, str] = {
    "A": "人物",
    "B": "物品",
    "C": "时间",
    "D": "空间",
    "E": "抽象事物",
    "F": "特征",
    "G": "动作",
    "H": "心理活动",
    "I": "活动",
    "J": "现象状态",
    "K": "关联",
    "L": "助语",
}

SUBCATEGORY_NAMES: Dict[str, Dict[str, str]] = {
    "A": {
        "a": "泛称",
        "b": "男女老少",
        "c": "职业",
        "d": "身份",
        "e": "亲属",
        "f": "民族",
        "g": "信仰",
        "h": "身体状况",
        "i": "人才",
        "j": "坏人",
    },
    "B": {
        "a": "统称",
        "b": "自然物",
        "c": "人工物",
        "d": "植物",
        "e": "动物",
        "f": "微生物",
    },
    "C": {
        "a": "时令",
        "b": "时期",
        "c": "时间",
        "d": "时段",
        "e": "节假日",
    },
    "D": {
        "a": "地理",
        "b": "行政区",
        "c": "场所",
        "d": "方位",
        "e": "距离",
    },
    "E": {
        "a": "事理",
        "b": "知识",
        "c": "信息",
        "d": "道理",
        "e": "规律",
    },
    "F": {
        "a": "外形",
        "b": "性质",
        "c": "状态",
        "d": "颜色",
        "e": "声音",
        "f": "味道",
        "g": "气味",
    },
    "G": {
        "a": "行为",
        "b": "动作",
        "c": "生产",
        "d": "经营",
        "e": "社交",
        "f": "生活",
    },
    "H": {
        "a": "情感",
        "b": "认知",
        "c": "意愿",
        "d": "思维",
    },
    "I": {
        "a": "政治",
        "b": "军事",
        "c": "经济",
        "d": "文化",
        "e": "教育",
        "f": "体育",
        "g": "娱乐",
    },
    "J": {
        "a": "自然现象",
        "b": "社会现象",
        "c": "生理现象",
        "d": "状态",
    },
    "K": {
        "a": "关系",
        "b": "区别",
        "c": "因果",
        "d": "条件",
        "e": "目的",
    },
    "L": {
        "a": "虚词",
        "b": "标点",
    },
}


@dataclass
class ThesaurusEntry:
    code: str
    words: List[str]
    relation: WordRelation
    category: str = ""
    subcategory: str = ""
    
    def __post_init__(self):
        if self.code:
            self.category = self.code[0] if len(self.code) > 0 else ""
            self.subcategory = self.code[1] if len(self.code) > 1 else ""


@dataclass
class SemanticCategory:
    code: str
    name: str
    level: int
    parent_code: Optional[str] = None
    children: List[str] = field(default_factory=list)
    words: List[str] = field(default_factory=list)


class Thesaurus:
    DEFAULT_THESAURUS_PATH = os.path.join(
        os.path.dirname(__file__), 'data', 'thesaurus.txt'
    )
    
    def __init__(self, load_default: bool = True):
        self._entries: Dict[str, ThesaurusEntry] = {}
        self._word_to_codes: Dict[str, Set[str]] = {}
        self._categories: Dict[str, SemanticCategory] = {}
        self._synonym_groups: Dict[str, Set[str]] = {}
        self._loaded: bool = False
        self._word_count: int = 0
        self._entry_count: int = 0
        
        if load_default:
            self._load_default_thesaurus()
    
    def _load_default_thesaurus(self) -> None:
        if os.path.exists(self.DEFAULT_THESAURUS_PATH):
            self.load_thesaurus(self.DEFAULT_THESAURUS_PATH)
    
    def load_thesaurus(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"同义词词林文件不存在: {path}")
        
        self._entries.clear()
        self._word_to_codes.clear()
        self._categories.clear()
        self._synonym_groups.clear()
        self._word_count = 0
        self._entry_count = 0
        
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split('\t')
                if len(parts) < 2:
                    continue
                
                code = parts[0].strip()
                words_str = parts[1].strip()
                
                if not code or not words_str:
                    continue
                
                relation = self._parse_relation(code)
                clean_code = self._clean_code(code)
                
                words = [w.strip() for w in words_str.split() if w.strip()]
                
                if not words:
                    continue
                
                entry = ThesaurusEntry(
                    code=clean_code,
                    words=words,
                    relation=relation
                )
                
                self._entries[clean_code] = entry
                self._entry_count += 1
                
                for word in words:
                    if word not in self._word_to_codes:
                        self._word_to_codes[word] = set()
                    self._word_to_codes[word].add(clean_code)
                    self._word_count += 1
                
                if relation == WordRelation.SYNONYM:
                    group_key = clean_code[:5]
                    if group_key not in self._synonym_groups:
                        self._synonym_groups[group_key] = set()
                    self._synonym_groups[group_key].update(words)
        
        self._build_category_tree()
        self._loaded = True
    
    def _parse_relation(self, code: str) -> WordRelation:
        if code.endswith('='):
            return WordRelation.SYNONYM
        elif code.endswith('#'):
            return WordRelation.RELATED
        elif code.endswith('@'):
            return WordRelation.INDEPENDENT
        return WordRelation.RELATED
    
    def _clean_code(self, code: str) -> str:
        if code and code[-1] in '=#@':
            return code[:-1]
        return code
    
    def _build_category_tree(self) -> None:
        for cat_code, cat_name in CATEGORY_NAMES.items():
            self._categories[cat_code] = SemanticCategory(
                code=cat_code,
                name=cat_name,
                level=1,
                parent_code=None
            )
        
        for entry in self._entries.values():
            code = entry.code
            if len(code) >= 2:
                cat_code = code[0]
                sub_code = code[:2]
                
                if cat_code in CATEGORY_NAMES:
                    sub_names = SUBCATEGORY_NAMES.get(cat_code, {})
                    sub_name = sub_names.get(code[1], f"子类{code[1]}")
                    
                    if sub_code not in self._categories:
                        self._categories[sub_code] = SemanticCategory(
                            code=sub_code,
                            name=sub_name,
                            level=2,
                            parent_code=cat_code
                        )
                        if cat_code in self._categories:
                            self._categories[cat_code].children.append(sub_code)
    
    def get_synonyms(self, word: str) -> List[str]:
        if word not in self._word_to_codes:
            return []
        
        synonyms: Set[str] = set()
        for code in self._word_to_codes[word]:
            entry = self._entries.get(code)
            if entry and entry.relation == WordRelation.SYNONYM:
                synonyms.update(entry.words)
        
        synonyms.discard(word)
        return list(synonyms)
    
    def get_related_words(self, word: str) -> List[str]:
        if word not in self._word_to_codes:
            return []
        
        related: Set[str] = set()
        for code in self._word_to_codes[word]:
            entry = self._entries.get(code)
            if entry:
                related.update(entry.words)
        
        related.discard(word)
        return list(related)
    
    def get_near_synonyms(self, word: str, include_related: bool = True) -> List[str]:
        if word not in self._word_to_codes:
            return []
        
        near_synonyms: Set[str] = set()
        
        for code in self._word_to_codes[word]:
            entry = self._entries.get(code)
            if entry:
                near_synonyms.update(entry.words)
            
            if include_related:
                group_key = code[:5]
                if group_key in self._synonym_groups:
                    near_synonyms.update(self._synonym_groups[group_key])
        
        near_synonyms.discard(word)
        return list(near_synonyms)
    
    def get_category(self, word: str) -> Optional[SemanticCategory]:
        if word not in self._word_to_codes:
            return None
        
        codes = self._word_to_codes[word]
        if not codes:
            return None
        
        code = next(iter(codes))
        cat_code = code[0]
        return self._categories.get(cat_code)
    
    def get_subcategory(self, word: str) -> Optional[SemanticCategory]:
        if word not in self._word_to_codes:
            return None
        
        codes = self._word_to_codes[word]
        if not codes:
            return None
        
        code = next(iter(codes))
        if len(code) >= 2:
            sub_code = code[:2]
            return self._categories.get(sub_code)
        return None
    
    def get_category_name(self, word: str) -> Optional[str]:
        category = self.get_category(word)
        return category.name if category else None
    
    def get_subcategory_name(self, word: str) -> Optional[str]:
        subcategory = self.get_subcategory(word)
        return subcategory.name if subcategory else None
    
    def get_semantic_path(self, word: str) -> List[Tuple[str, str]]:
        if word not in self._word_to_codes:
            return []
        
        codes = self._word_to_codes[word]
        if not codes:
            return []
        
        code = next(iter(codes))
        path: List[Tuple[str, str]] = []
        
        if len(code) >= 1:
            cat_code = code[0]
            if cat_code in self._categories:
                path.append((cat_code, self._categories[cat_code].name))
        
        if len(code) >= 2:
            sub_code = code[:2]
            if sub_code in self._categories:
                path.append((sub_code, self._categories[sub_code].name))
        
        if len(code) >= 4:
            small_cat_code = code[:4]
            path.append((small_cat_code, f"小类{small_cat_code[2:4]}"))
        
        if len(code) >= 5:
            group_code = code[:5]
            path.append((group_code, f"词群{group_code[4]}"))
        
        return path
    
    def get_words_by_category(self, category_code: str) -> List[str]:
        words: Set[str] = set()
        
        for code, entry in self._entries.items():
            if code.startswith(category_code):
                words.update(entry.words)
        
        return list(words)
    
    def get_words_by_subcategory(self, subcategory_code: str) -> List[str]:
        if len(subcategory_code) < 2:
            return []
        return self.get_words_by_category(subcategory_code)
    
    def calculate_similarity(self, word1: str, word2: str) -> float:
        if word1 == word2:
            return 1.0
        
        if word1 not in self._word_to_codes or word2 not in self._word_to_codes:
            return 0.0
        
        codes1 = self._word_to_codes[word1]
        codes2 = self._word_to_codes[word2]
        
        max_sim = 0.0
        
        for code1 in codes1:
            for code2 in codes2:
                sim = self._code_similarity(code1, code2)
                max_sim = max(max_sim, sim)
        
        return max_sim
    
    def _code_similarity(self, code1: str, code2: str) -> float:
        if code1 == code2:
            return 1.0
        
        min_len = min(len(code1), len(code2))
        common_prefix = 0
        
        for i in range(min_len):
            if code1[i] == code2[i]:
                common_prefix += 1
            else:
                break
        
        if common_prefix == 0:
            return 0.0
        
        weights = [0.1, 0.2, 0.3, 0.5, 0.6, 0.8, 1.0]
        
        if common_prefix < len(weights):
            return weights[common_prefix - 1]
        
        return 1.0
    
    def is_synonym(self, word1: str, word2: str) -> bool:
        if word1 == word2:
            return True
        
        if word1 not in self._word_to_codes:
            return False
        
        for code in self._word_to_codes[word1]:
            entry = self._entries.get(code)
            if entry and entry.relation == WordRelation.SYNONYM:
                if word2 in entry.words:
                    return True
        
        return False
    
    def is_related(self, word1: str, word2: str) -> bool:
        if word1 == word2:
            return True
        
        if word1 not in self._word_to_codes:
            return False
        
        for code in self._word_to_codes[word1]:
            entry = self._entries.get(code)
            if entry and word2 in entry.words:
                return True
        
        return False
    
    def has_word(self, word: str) -> bool:
        return word in self._word_to_codes
    
    def get_entry(self, code: str) -> Optional[ThesaurusEntry]:
        return self._entries.get(code)
    
    def get_word_codes(self, word: str) -> Set[str]:
        return self._word_to_codes.get(word, set()).copy()
    
    def get_all_categories(self) -> Dict[str, SemanticCategory]:
        return self._categories.copy()
    
    def get_all_words(self) -> Set[str]:
        return set(self._word_to_codes.keys())
    
    def get_entry_count(self) -> int:
        return self._entry_count
    
    def get_word_count(self) -> int:
        return len(self._word_to_codes)
    
    def is_loaded(self) -> bool:
        return self._loaded
    
    def add_entry(
        self,
        code: str,
        words: List[str],
        relation: WordRelation = WordRelation.RELATED
    ) -> None:
        clean_code = self._clean_code(code)
        
        entry = ThesaurusEntry(
            code=clean_code,
            words=words,
            relation=relation
        )
        
        self._entries[clean_code] = entry
        self._entry_count += 1
        
        for word in words:
            if word not in self._word_to_codes:
                self._word_to_codes[word] = set()
            self._word_to_codes[word].add(clean_code)
        
        if relation == WordRelation.SYNONYM:
            group_key = clean_code[:5]
            if group_key not in self._synonym_groups:
                self._synonym_groups[group_key] = set()
            self._synonym_groups[group_key].update(words)
    
    def save_thesaurus(self, path: str) -> None:
        with open(path, 'w', encoding='utf-8') as f:
            f.write("# 同义词词林数据文件\n")
            f.write("# 编码格式: 大类(1位)中类(1位)小类(2位)词群(1位)编号(2位)标记(1位)\n")
            f.write("# 标记: = 同义词, # 相关词, @ 独立词\n")
            f.write("#\n")
            
            for code, entry in sorted(self._entries.items()):
                relation_char = entry.relation.value
                full_code = code + relation_char
                words_str = ' '.join(entry.words)
                f.write(f"{full_code}\t{words_str}\n")
    
    def __len__(self) -> int:
        return self.get_word_count()
    
    def __contains__(self, word: str) -> bool:
        return self.has_word(word)
    
    def __repr__(self) -> str:
        return (
            f"Thesaurus(entries={self._entry_count}, "
            f"words={self.get_word_count()}, loaded={self._loaded})"
        )


class ThesaurusManager:
    def __init__(self):
        self._thesaurus: Optional[Thesaurus] = None
    
    def load(self, path: Optional[str] = None) -> None:
        if path:
            self._thesaurus = Thesaurus(load_default=False)
            self._thesaurus.load_thesaurus(path)
        else:
            self._thesaurus = Thesaurus(load_default=True)
    
    def get_thesaurus(self) -> Optional[Thesaurus]:
        return self._thesaurus
    
    def get_synonyms(self, word: str) -> List[str]:
        if self._thesaurus is None:
            return []
        return self._thesaurus.get_synonyms(word)
    
    def get_related_words(self, word: str) -> List[str]:
        if self._thesaurus is None:
            return []
        return self._thesaurus.get_related_words(word)
    
    def get_near_synonyms(self, word: str) -> List[str]:
        if self._thesaurus is None:
            return []
        return self._thesaurus.get_near_synonyms(word)
    
    def get_category_name(self, word: str) -> Optional[str]:
        if self._thesaurus is None:
            return None
        return self._thesaurus.get_category_name(word)
    
    def get_semantic_path(self, word: str) -> List[Tuple[str, str]]:
        if self._thesaurus is None:
            return []
        return self._thesaurus.get_semantic_path(word)
    
    def calculate_similarity(self, word1: str, word2: str) -> float:
        if self._thesaurus is None:
            return 0.0
        return self._thesaurus.calculate_similarity(word1, word2)
    
    def is_synonym(self, word1: str, word2: str) -> bool:
        if self._thesaurus is None:
            return False
        return self._thesaurus.is_synonym(word1, word2)
    
    def is_related(self, word1: str, word2: str) -> bool:
        if self._thesaurus is None:
            return False
        return self._thesaurus.is_related(word1, word2)
    
    def is_loaded(self) -> bool:
        return self._thesaurus is not None and self._thesaurus.is_loaded()
