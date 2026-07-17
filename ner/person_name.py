"""
人名词库模块 - 人名词库构建

提供中文人名识别、姓氏库、名字用字库和性别推断功能。

功能：
- 常见姓氏库：包含中国常见姓氏及其频率统计
- 名字用字库：包含名字中常用的字及其性别倾向
- 性别推断：根据名字推断性别
- 人名识别：识别文本中的人名
- 人名生成：生成随机人名

数据格式说明：
- 姓氏格式：姓氏\t频率\t类型(单姓/复姓)
- 名字用字格式：用字\t男性频率\t女性频率\t中性频率\t字义分类
"""

import os
import random
from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum


class Gender(Enum):
    MALE = "male"
    FEMALE = "female"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"


class SurnameType(Enum):
    SINGLE = "single"
    COMPOUND = "compound"


@dataclass
class Surname:
    name: str
    frequency: float
    surname_type: SurnameType
    pinyin: Optional[str] = None
    origin: Optional[str] = None
    
    def is_compound(self) -> bool:
        return self.surname_type == SurnameType.COMPOUND
    
    def is_single(self) -> bool:
        return self.surname_type == SurnameType.SINGLE


@dataclass
class NameChar:
    char: str
    male_freq: float
    female_freq: float
    neutral_freq: float
    category: str = ""
    meaning: str = ""
    pinyin: Optional[str] = None
    
    @property
    def total_freq(self) -> float:
        return self.male_freq + self.female_freq + self.neutral_freq
    
    def get_gender_tendency(self) -> Gender:
        if self.male_freq > self.female_freq and self.male_freq > self.neutral_freq:
            return Gender.MALE
        elif self.female_freq > self.male_freq and self.female_freq > self.neutral_freq:
            return Gender.FEMALE
        elif self.neutral_freq > self.male_freq and self.neutral_freq > self.female_freq:
            return Gender.NEUTRAL
        else:
            return Gender.NEUTRAL


@dataclass
class PersonName:
    full_name: str
    surname: str
    given_name: str
    gender: Gender = Gender.UNKNOWN
    confidence: float = 0.0
    
    def __str__(self) -> str:
        return self.full_name
    
    def __repr__(self) -> str:
        return (
            f"PersonName(full_name='{self.full_name}', surname='{self.surname}', "
            f"given_name='{self.given_name}', gender={self.gender.value}, "
            f"confidence={self.confidence:.2f})"
        )


NAME_CHAR_CATEGORIES: Dict[str, str] = {
    "virtue": "德行",
    "nature": "自然",
    "wisdom": "智慧",
    "beauty": "美好",
    "strength": "力量",
    "literary": "文雅",
    "auspicious": "吉祥",
    "season": "季节",
    "color": "颜色",
    "gem": "珍宝",
    "plant": "植物",
    "animal": "动物",
    "water": "水",
    "mountain": "山",
    "sky": "天空",
    "other": "其他",
}


class PersonNameDictionary:
    DEFAULT_DATA_PATH = os.path.join(
        os.path.dirname(__file__), 'data', 'person_names.txt'
    )
    
    def __init__(self, load_default: bool = True):
        self._surnames: Dict[str, Surname] = {}
        self._compound_surnames: Dict[str, Surname] = {}
        self._name_chars: Dict[str, NameChar] = {}
        self._surname_set: Set[str] = set()
        self._compound_surname_set: Set[str] = set()
        self._loaded: bool = False
        self._surname_count: int = 0
        self._name_char_count: int = 0
        
        if load_default:
            self._load_default_data()
    
    def _load_default_data(self) -> None:
        if os.path.exists(self.DEFAULT_DATA_PATH):
            self.load_data(self.DEFAULT_DATA_PATH)
    
    def load_data(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"人名数据文件不存在: {path}")
        
        self._surnames.clear()
        self._compound_surnames.clear()
        self._name_chars.clear()
        self._surname_set.clear()
        self._compound_surname_set.clear()
        self._surname_count = 0
        self._name_char_count = 0
        
        current_section = None
        
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    if line.startswith('# @'):
                        section = line[3:].strip()
                        if section == 'surnames':
                            current_section = 'surnames'
                        elif section == 'compound_surnames':
                            current_section = 'compound_surnames'
                        elif section == 'name_chars':
                            current_section = 'name_chars'
                    continue
                
                if current_section == 'surnames':
                    self._parse_surname(line, SurnameType.SINGLE)
                elif current_section == 'compound_surnames':
                    self._parse_surname(line, SurnameType.COMPOUND)
                elif current_section == 'name_chars':
                    self._parse_name_char(line)
        
        self._surname_set = set(self._surnames.keys())
        self._compound_surname_set = set(self._compound_surnames.keys())
        self._loaded = True
    
    def _parse_surname(self, line: str, surname_type: SurnameType) -> None:
        parts = line.split('\t')
        if len(parts) < 2:
            return
        
        name = parts[0].strip()
        try:
            frequency = float(parts[1].strip())
        except ValueError:
            frequency = 0.0
        
        pinyin = parts[2].strip() if len(parts) > 2 else None
        origin = parts[3].strip() if len(parts) > 3 else None
        
        surname = Surname(
            name=name,
            frequency=frequency,
            surname_type=surname_type,
            pinyin=pinyin,
            origin=origin
        )
        
        if surname_type == SurnameType.SINGLE:
            self._surnames[name] = surname
            self._surname_count += 1
        else:
            self._compound_surnames[name] = surname
    
    def _parse_name_char(self, line: str) -> None:
        parts = line.split('\t')
        if len(parts) < 4:
            return
        
        char = parts[0].strip()
        try:
            male_freq = float(parts[1].strip())
            female_freq = float(parts[2].strip())
            neutral_freq = float(parts[3].strip())
        except ValueError:
            return
        
        category = parts[4].strip() if len(parts) > 4 else ""
        meaning = parts[5].strip() if len(parts) > 5 else ""
        pinyin = parts[6].strip() if len(parts) > 6 else None
        
        name_char = NameChar(
            char=char,
            male_freq=male_freq,
            female_freq=female_freq,
            neutral_freq=neutral_freq,
            category=category,
            meaning=meaning,
            pinyin=pinyin
        )
        
        self._name_chars[char] = name_char
        self._name_char_count += 1
    
    def is_surname(self, text: str) -> bool:
        if len(text) == 1:
            return text in self._surname_set
        elif len(text) == 2:
            return text in self._compound_surname_set
        return False
    
    def get_surname(self, name: str) -> Optional[Surname]:
        if name in self._compound_surnames:
            return self._compound_surnames[name]
        if name in self._surnames:
            return self._surnames[name]
        return None
    
    def get_all_surnames(self) -> List[Surname]:
        return list(self._surnames.values()) + list(self._compound_surnames.values())
    
    def get_single_surnames(self) -> List[Surname]:
        return list(self._surnames.values())
    
    def get_compound_surnames(self) -> List[Surname]:
        return list(self._compound_surnames.values())
    
    def get_top_surnames(self, n: int = 100) -> List[Surname]:
        all_surnames = self.get_all_surnames()
        sorted_surnames = sorted(all_surnames, key=lambda s: s.frequency, reverse=True)
        return sorted_surnames[:n]
    
    def is_name_char(self, char: str) -> bool:
        return char in self._name_chars
    
    def get_name_char(self, char: str) -> Optional[NameChar]:
        return self._name_chars.get(char)
    
    def get_all_name_chars(self) -> List[NameChar]:
        return list(self._name_chars.values())
    
    def get_name_chars_by_category(self, category: str) -> List[NameChar]:
        return [nc for nc in self._name_chars.values() if nc.category == category]
    
    def get_name_chars_by_gender(self, gender: Gender, min_freq: float = 0.0, top_n: int = 100) -> List[NameChar]:
        result = []
        for nc in self._name_chars.values():
            if gender == Gender.MALE:
                if nc.male_freq > min_freq and nc.male_freq >= nc.female_freq:
                    result.append(nc)
            elif gender == Gender.FEMALE:
                if nc.female_freq > min_freq and nc.female_freq >= nc.male_freq:
                    result.append(nc)
            elif gender == Gender.NEUTRAL:
                if nc.neutral_freq > min_freq:
                    result.append(nc)
        
        if gender == Gender.MALE:
            result.sort(key=lambda nc: nc.male_freq, reverse=True)
        elif gender == Gender.FEMALE:
            result.sort(key=lambda nc: nc.female_freq, reverse=True)
        else:
            result.sort(key=lambda nc: nc.neutral_freq, reverse=True)
        
        return result[:top_n]
    
    def extract_surname(self, full_name: str) -> Tuple[Optional[str], str]:
        if not full_name or len(full_name) < 2:
            return None, full_name
        
        if len(full_name) >= 3:
            two_char = full_name[:2]
            if two_char in self._compound_surname_set:
                return two_char, full_name[2:]
        
        one_char = full_name[0]
        if one_char in self._surname_set:
            return one_char, full_name[1:]
        
        return None, full_name
    
    def infer_gender(self, name: str) -> Tuple[Gender, float]:
        surname, given_name = self.extract_surname(name)
        
        if not given_name:
            return Gender.UNKNOWN, 0.0
        
        male_score = 0.0
        female_score = 0.0
        neutral_score = 0.0
        char_count = 0
        
        for char in given_name:
            name_char = self._name_chars.get(char)
            if name_char:
                male_score += name_char.male_freq
                female_score += name_char.female_freq
                neutral_score += name_char.neutral_freq
                char_count += 1
        
        if char_count == 0:
            return Gender.UNKNOWN, 0.0
        
        male_score /= char_count
        female_score /= char_count
        neutral_score /= char_count
        
        total = male_score + female_score + neutral_score
        if total == 0:
            return Gender.UNKNOWN, 0.0
        
        if male_score > female_score and male_score > neutral_score:
            confidence = male_score / total
            return Gender.MALE, confidence
        elif female_score > male_score and female_score > neutral_score:
            confidence = female_score / total
            return Gender.FEMALE, confidence
        else:
            confidence = neutral_score / total if neutral_score > 0 else 0.5
            return Gender.NEUTRAL, confidence
    
    def parse_name(self, full_name: str) -> Optional[PersonName]:
        if not full_name or len(full_name) < 2:
            return None
        
        surname, given_name = self.extract_surname(full_name)
        
        if not surname:
            return None
        
        gender, confidence = self.infer_gender(full_name)
        
        return PersonName(
            full_name=full_name,
            surname=surname,
            given_name=given_name,
            gender=gender,
            confidence=confidence
        )
    
    def is_person_name(self, text: str, min_given_name_len: int = 1) -> bool:
        if not text or len(text) < 2:
            return False
        
        surname, given_name = self.extract_surname(text)
        
        if not surname:
            return False
        
        if len(given_name) < min_given_name_len:
            return False
        
        name_char_count = sum(1 for c in given_name if c in self._name_chars)
        
        return name_char_count >= len(given_name) * 0.5
    
    def recognize_names(self, text: str) -> List[PersonName]:
        names: List[PersonName] = []
        
        i = 0
        while i < len(text):
            found = False
            
            if i + 3 <= len(text):
                two_char = text[i:i+2]
                if two_char in self._compound_surname_set:
                    given_name_len = 0
                    for j in range(i + 2, min(i + 5, len(text))):
                        char = text[j]
                        if char in self._name_chars:
                            given_name_len += 1
                        else:
                            break
                    
                    if given_name_len >= 1:
                        candidate = text[i:i+2+given_name_len]
                        name = self.parse_name(candidate)
                        if name:
                            names.append(name)
                            i = i + 2 + given_name_len
                            found = True
            
            if not found and i + 2 <= len(text):
                one_char = text[i]
                if one_char in self._surname_set:
                    given_name_len = 0
                    for j in range(i + 1, min(i + 4, len(text))):
                        char = text[j]
                        if char in self._name_chars:
                            given_name_len += 1
                        else:
                            break
                    
                    if given_name_len >= 1:
                        candidate = text[i:i+1+given_name_len]
                        name = self.parse_name(candidate)
                        if name:
                            names.append(name)
                            i = i + 1 + given_name_len
                            found = True
            
            if not found:
                i += 1
        
        return names
    
    def generate_name(
        self,
        surname: Optional[str] = None,
        gender: Optional[Gender] = None,
        given_name_len: int = 2,
        category: Optional[str] = None
    ) -> str:
        if surname is None:
            surnames = self.get_single_surnames()
            if surnames:
                surname = random.choice(surnames).name
            else:
                surname = "张"
        
        if gender is None:
            gender = random.choice([Gender.MALE, Gender.FEMALE])
        
        if category:
            chars = self.get_name_chars_by_category(category)
        else:
            chars = self.get_name_chars_by_gender(gender)
        
        if not chars:
            chars = list(self._name_chars.values())
        
        if not chars:
            return surname + "明"
        
        given_name = ""
        for _ in range(given_name_len):
            name_char = random.choice(chars)
            given_name += name_char.char
        
        return surname + given_name
    
    def generate_names(
        self,
        count: int = 10,
        gender: Optional[Gender] = None,
        given_name_len: int = 2,
        category: Optional[str] = None
    ) -> List[str]:
        names = []
        for _ in range(count):
            name = self.generate_name(
                gender=gender,
                given_name_len=given_name_len,
                category=category
            )
            names.append(name)
        return names
    
    def get_surname_count(self) -> int:
        return len(self._surnames) + len(self._compound_surnames)
    
    def get_name_char_count(self) -> int:
        return len(self._name_chars)
    
    def is_loaded(self) -> bool:
        return self._loaded
    
    def get_statistics(self) -> Dict[str, Any]:
        return {
            "loaded": self._loaded,
            "single_surname_count": len(self._surnames),
            "compound_surname_count": len(self._compound_surnames),
            "total_surname_count": self.get_surname_count(),
            "name_char_count": self.get_name_char_count(),
            "categories": list(set(nc.category for nc in self._name_chars.values() if nc.category)),
        }
    
    def add_surname(
        self,
        name: str,
        frequency: float = 0.0,
        surname_type: SurnameType = SurnameType.SINGLE,
        pinyin: Optional[str] = None,
        origin: Optional[str] = None
    ) -> None:
        surname = Surname(
            name=name,
            frequency=frequency,
            surname_type=surname_type,
            pinyin=pinyin,
            origin=origin
        )
        
        if surname_type == SurnameType.SINGLE:
            self._surnames[name] = surname
            self._surname_set.add(name)
        else:
            self._compound_surnames[name] = surname
            self._compound_surname_set.add(name)
    
    def add_name_char(
        self,
        char: str,
        male_freq: float = 0.0,
        female_freq: float = 0.0,
        neutral_freq: float = 0.0,
        category: str = "",
        meaning: str = "",
        pinyin: Optional[str] = None
    ) -> None:
        name_char = NameChar(
            char=char,
            male_freq=male_freq,
            female_freq=female_freq,
            neutral_freq=neutral_freq,
            category=category,
            meaning=meaning,
            pinyin=pinyin
        )
        self._name_chars[char] = name_char
    
    def save_data(self, path: str) -> None:
        with open(path, 'w', encoding='utf-8') as f:
            f.write("# 人名词库数据文件\n")
            f.write("# 格式说明:\n")
            f.write("# 姓氏: 姓氏\\t频率\\t拼音\\t起源\n")
            f.write("# 名字用字: 用字\\t男性频率\\t女性频率\\t中性频率\\t分类\\t含义\\t拼音\n")
            f.write("#\n")
            
            f.write("# @surnames\n")
            for surname in sorted(self._surnames.values(), key=lambda s: s.frequency, reverse=True):
                parts = [surname.name, str(surname.frequency)]
                if surname.pinyin:
                    parts.append(surname.pinyin)
                if surname.origin:
                    parts.append(surname.origin)
                f.write('\t'.join(parts) + '\n')
            
            f.write("\n# @compound_surnames\n")
            for surname in sorted(self._compound_surnames.values(), key=lambda s: s.frequency, reverse=True):
                parts = [surname.name, str(surname.frequency)]
                if surname.pinyin:
                    parts.append(surname.pinyin)
                if surname.origin:
                    parts.append(surname.origin)
                f.write('\t'.join(parts) + '\n')
            
            f.write("\n# @name_chars\n")
            for name_char in sorted(self._name_chars.values(), key=lambda nc: nc.total_freq, reverse=True):
                parts = [
                    name_char.char,
                    str(name_char.male_freq),
                    str(name_char.female_freq),
                    str(name_char.neutral_freq),
                    name_char.category,
                    name_char.meaning
                ]
                if name_char.pinyin:
                    parts.append(name_char.pinyin)
                f.write('\t'.join(parts) + '\n')
    
    def __len__(self) -> int:
        return self.get_surname_count() + self.get_name_char_count()
    
    def __contains__(self, text: str) -> bool:
        return self.is_surname(text) or self.is_name_char(text)
    
    def __repr__(self) -> str:
        return (
            f"PersonNameDictionary(surnames={self.get_surname_count()}, "
            f"name_chars={self.get_name_char_count()}, loaded={self._loaded})"
        )


class PersonNameManager:
    def __init__(self, load_default: bool = True):
        self._dictionary: Optional[PersonNameDictionary] = None
        if load_default:
            self._dictionary = PersonNameDictionary(load_default=True)
    
    def load(self, path: Optional[str] = None) -> None:
        if path:
            self._dictionary = PersonNameDictionary(load_default=False)
            self._dictionary.load_data(path)
        else:
            self._dictionary = PersonNameDictionary(load_default=True)
    
    def get_dictionary(self) -> Optional[PersonNameDictionary]:
        return self._dictionary
    
    def is_surname(self, text: str) -> bool:
        if self._dictionary is None:
            return False
        return self._dictionary.is_surname(text)
    
    def get_surname(self, name: str) -> Optional[Surname]:
        if self._dictionary is None:
            return None
        return self._dictionary.get_surname(name)
    
    def get_all_surnames(self) -> List[Surname]:
        if self._dictionary is None:
            return []
        return self._dictionary.get_all_surnames()
    
    def get_top_surnames(self, n: int = 100) -> List[Surname]:
        if self._dictionary is None:
            return []
        return self._dictionary.get_top_surnames(n)
    
    def is_name_char(self, char: str) -> bool:
        if self._dictionary is None:
            return False
        return self._dictionary.is_name_char(char)
    
    def get_name_char(self, char: str) -> Optional[NameChar]:
        if self._dictionary is None:
            return None
        return self._dictionary.get_name_char(char)
    
    def get_name_chars_by_gender(self, gender: Gender, min_freq: float = 0.0) -> List[NameChar]:
        if self._dictionary is None:
            return []
        return self._dictionary.get_name_chars_by_gender(gender, min_freq)
    
    def infer_gender(self, name: str) -> Tuple[Gender, float]:
        if self._dictionary is None:
            return Gender.UNKNOWN, 0.0
        return self._dictionary.infer_gender(name)
    
    def parse_name(self, full_name: str) -> Optional[PersonName]:
        if self._dictionary is None:
            return None
        return self._dictionary.parse_name(full_name)
    
    def is_person_name(self, text: str) -> bool:
        if self._dictionary is None:
            return False
        return self._dictionary.is_person_name(text)
    
    def recognize_names(self, text: str) -> List[PersonName]:
        if self._dictionary is None:
            return []
        return self._dictionary.recognize_names(text)
    
    def generate_name(
        self,
        surname: Optional[str] = None,
        gender: Optional[Gender] = None,
        given_name_len: int = 2,
        category: Optional[str] = None
    ) -> str:
        if self._dictionary is None:
            return ""
        return self._dictionary.generate_name(surname, gender, given_name_len, category)
    
    def generate_names(
        self,
        count: int = 10,
        gender: Optional[Gender] = None,
        given_name_len: int = 2,
        category: Optional[str] = None
    ) -> List[str]:
        if self._dictionary is None:
            return []
        return self._dictionary.generate_names(count, gender, given_name_len, category)
    
    def get_statistics(self) -> Dict[str, Any]:
        if self._dictionary is None:
            return {"loaded": False}
        return self._dictionary.get_statistics()
    
    def is_loaded(self) -> bool:
        return self._dictionary is not None and self._dictionary.is_loaded()
