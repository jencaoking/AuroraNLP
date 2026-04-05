"""
专业术语库模块 - 多领域专业术语管理

提供医学、法律、金融、IT等多个领域的专业术语识别、查询和管理功能。

功能：
- 多领域术语支持：医学、法律、金融、IT等
- 术语查询：按领域、名称、别名查询
- 术语识别：从文本中识别专业术语
- 术语分类：自动判断术语所属领域
- 搜狗词库整合：支持从.scel文件导入术语

数据格式说明：
- 术语格式：术语ID\t名称\t领域\t子领域\t英文\t别名(逗号分隔)\t解释\t来源
"""

import os
import re
from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

from .scel_parser import ScelParser, ScelWord


class TermDomain(Enum):
    MEDICAL = "medical"
    LEGAL = "legal"
    FINANCE = "finance"
    IT = "it"
    AUTOMOTIVE = "automotive"
    FOOD = "food"
    ENTERTAINMENT = "entertainment"
    GOVERNMENT = "government"
    ECOMMERCE = "ecommerce"
    OTHER = "other"
    
    def get_name(self) -> str:
        names = {
            self.MEDICAL: "医学",
            self.LEGAL: "法律",
            self.FINANCE: "金融",
            self.IT: "IT",
            self.AUTOMOTIVE: "汽车",
            self.FOOD: "饮食",
            self.ENTERTAINMENT: "娱乐",
            self.GOVERNMENT: "政府",
            self.ECOMMERCE: "电商",
            self.OTHER: "其他",
        }
        return names.get(self, "未知")


DOMAIN_SCEL_MAPPING: Dict[str, TermDomain] = {
    "法律词汇大全": TermDomain.LEGAL,
    "股票基金词库": TermDomain.FINANCE,
    "网络安全": TermDomain.IT,
    "网络安全及黑客": TermDomain.IT,
    "网络工程": TermDomain.IT,
    "前端工程师": TermDomain.IT,
    "开发大神": TermDomain.IT,
    "手机词汇": TermDomain.IT,
    "汽车词汇": TermDomain.AUTOMOTIVE,
    "饮食大全": TermDomain.FOOD,
    "明星": TermDomain.ENTERTAINMENT,
    "歌手人名": TermDomain.ENTERTAINMENT,
    "政府机关": TermDomain.GOVERNMENT,
    "电子商务": TermDomain.ECOMMERCE,
}


@dataclass
class Term:
    term_id: str
    name: str
    domain: TermDomain
    sub_domain: Optional[str] = None
    english: Optional[str] = None
    aliases: List[str] = field(default_factory=list)
    definition: Optional[str] = None
    source: Optional[str] = None
    frequency: int = 0
    
    def __str__(self) -> str:
        return self.name
    
    def __repr__(self) -> str:
        return (
            f"Term(id='{self.term_id}', name='{self.name}', "
            f"domain={self.domain.get_name()})"
        )
    
    def get_all_names(self) -> List[str]:
        names = [self.name]
        names.extend(self.aliases)
        if self.english:
            names.append(self.english)
        return names
    
    def matches(self, text: str) -> bool:
        if text == self.name:
            return True
        if text == self.term_id:
            return True
        if text in self.aliases:
            return True
        if self.english and text.lower() == self.english.lower():
            return True
        return False


class TerminologyDatabase:
    DEFAULT_DATA_PATH = os.path.join(
        os.path.dirname(__file__), 'data', 'terminology.txt'
    )
    DEFAULT_SOGOU_PATH = os.path.join(
        os.path.dirname(__file__), 'data', 'sogou'
    )
    
    def __init__(self, load_default: bool = True, load_sogou: bool = True):
        self._terms: Dict[str, Term] = {}
        self._name_index: Dict[str, Set[str]] = {}
        self._alias_index: Dict[str, str] = {}
        self._english_index: Dict[str, str] = {}
        self._domain_index: Dict[TermDomain, Set[str]] = {
            domain: set() for domain in TermDomain
        }
        self._sub_domain_index: Dict[str, Set[str]] = {}
        self._loaded: bool = False
        self._term_count: int = 0
        self._sogou_loaded: bool = False
        
        if load_default:
            self._load_default_data()
        if load_sogou:
            self._load_sogou_data()
    
    def _load_default_data(self) -> None:
        if os.path.exists(self.DEFAULT_DATA_PATH):
            self.load_data(self.DEFAULT_DATA_PATH)
    
    def _load_sogou_data(self) -> None:
        if not os.path.exists(self.DEFAULT_SOGOU_PATH):
            return
        
        for filename in os.listdir(self.DEFAULT_SOGOU_PATH):
            if not filename.endswith('.scel'):
                continue
            
            domain = self._guess_domain_from_filename(filename)
            if domain is None:
                continue
            
            filepath = os.path.join(self.DEFAULT_SOGOU_PATH, filename)
            try:
                self.load_scel(filepath, domain)
            except Exception:
                pass
        
        self._sogou_loaded = True
    
    def _guess_domain_from_filename(self, filename: str) -> Optional[TermDomain]:
        for pattern, domain in DOMAIN_SCEL_MAPPING.items():
            if pattern in filename:
                return domain
        return None
    
    def load_data(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"术语数据文件不存在: {path}")
        
        current_section = None
        
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    if line.startswith('# @'):
                        section = line[3:].strip()
                        if section in ['medical', 'legal', 'finance', 'it', 'other']:
                            current_section = section
                    continue
                
                self._parse_term(line, current_section)
        
        self._loaded = True
    
    def _parse_term(self, line: str, section: Optional[str]) -> None:
        parts = line.split('\t')
        if len(parts) < 3:
            return
        
        term_id = parts[0].strip()
        name = parts[1].strip()
        
        domain_str = parts[2].strip().lower()
        domain = self._parse_domain(domain_str)
        
        sub_domain = parts[3].strip() if len(parts) > 3 and parts[3].strip() else None
        english = parts[4].strip() if len(parts) > 4 and parts[4].strip() else None
        
        aliases = []
        if len(parts) > 5 and parts[5].strip():
            aliases = [a.strip() for a in parts[5].split(',') if a.strip()]
        
        definition = parts[6].strip() if len(parts) > 6 and parts[6].strip() else None
        source = parts[7].strip() if len(parts) > 7 and parts[7].strip() else None
        
        term = Term(
            term_id=term_id,
            name=name,
            domain=domain,
            sub_domain=sub_domain,
            english=english,
            aliases=aliases,
            definition=definition,
            source=source,
        )
        
        self._add_term(term)
    
    def _parse_domain(self, domain_str: str) -> TermDomain:
        mapping = {
            'medical': TermDomain.MEDICAL,
            '医学': TermDomain.MEDICAL,
            'legal': TermDomain.LEGAL,
            '法律': TermDomain.LEGAL,
            'finance': TermDomain.FINANCE,
            '金融': TermDomain.FINANCE,
            'it': TermDomain.IT,
            '信息技术': TermDomain.IT,
            'automotive': TermDomain.AUTOMOTIVE,
            '汽车': TermDomain.AUTOMOTIVE,
            'food': TermDomain.FOOD,
            '饮食': TermDomain.FOOD,
            'entertainment': TermDomain.ENTERTAINMENT,
            '娱乐': TermDomain.ENTERTAINMENT,
            'government': TermDomain.GOVERNMENT,
            '政府': TermDomain.GOVERNMENT,
            'ecommerce': TermDomain.ECOMMERCE,
            '电商': TermDomain.ECOMMERCE,
        }
        return mapping.get(domain_str.lower(), TermDomain.OTHER)
    
    def _add_term(self, term: Term) -> None:
        self._terms[term.term_id] = term
        self._term_count += 1
        
        if term.name not in self._name_index:
            self._name_index[term.name] = set()
        self._name_index[term.name].add(term.term_id)
        
        for alias in term.aliases:
            self._alias_index[alias] = term.term_id
        
        if term.english:
            self._english_index[term.english.lower()] = term.term_id
        
        self._domain_index[term.domain].add(term.term_id)
        
        if term.sub_domain:
            if term.sub_domain not in self._sub_domain_index:
                self._sub_domain_index[term.sub_domain] = set()
            self._sub_domain_index[term.sub_domain].add(term.term_id)
    
    def load_scel(
        self,
        scel_path: str,
        domain: TermDomain,
        sub_domain: Optional[str] = None,
        source: Optional[str] = None
    ) -> int:
        parser = ScelParser()
        words = parser.parse(scel_path)
        
        loaded_count = 0
        base_id = f"{domain.value}_{os.path.basename(scel_path)[:10]}"
        
        for i, w in enumerate(words):
            term_id = f"{base_id}_{i:06d}"
            
            if w.word in self._name_index:
                continue
            
            term = Term(
                term_id=term_id,
                name=w.word,
                domain=domain,
                sub_domain=sub_domain,
                aliases=[],
                definition=None,
                source=source or parser.metadata.name,
                frequency=w.frequency,
            )
            
            self._add_term(term)
            loaded_count += 1
        
        return loaded_count
    
    def load_from_txt(
        self,
        txt_path: str,
        domain: TermDomain,
        sub_domain: Optional[str] = None,
        source: Optional[str] = None
    ) -> int:
        loaded_count = 0
        base_id = f"{domain.value}_{os.path.basename(txt_path)[:10]}"
        
        with open(txt_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                term_id = f"{base_id}_{i:06d}"
                
                if line in self._name_index:
                    continue
                
                term = Term(
                    term_id=term_id,
                    name=line,
                    domain=domain,
                    sub_domain=sub_domain,
                    source=source,
                )
                
                self._add_term(term)
                loaded_count += 1
        
        return loaded_count
    
    def get_by_id(self, term_id: str) -> Optional[Term]:
        return self._terms.get(term_id)
    
    def get_by_name(self, name: str) -> List[Term]:
        ids = self._name_index.get(name, set())
        return [self._terms[tid] for tid in ids if tid in self._terms]
    
    def get_by_alias(self, alias: str) -> Optional[Term]:
        tid = self._alias_index.get(alias)
        if tid:
            return self._terms.get(tid)
        return None
    
    def get_by_english(self, english: str) -> Optional[Term]:
        tid = self._english_index.get(english.lower())
        if tid:
            return self._terms.get(tid)
        return None
    
    def search(self, query: str) -> List[Term]:
        results: List[Term] = []
        
        if query in self._terms:
            results.append(self._terms[query])
        
        results.extend(self.get_by_name(query))
        
        term = self.get_by_alias(query)
        if term and term not in results:
            results.append(term)
        
        term = self.get_by_english(query)
        if term and term not in results:
            results.append(term)
        
        for name, ids in self._name_index.items():
            if query in name and name != query:
                for tid in ids:
                    t = self._terms.get(tid)
                    if t and t not in results:
                        results.append(t)
        
        return results
    
    def get_by_domain(self, domain: TermDomain) -> List[Term]:
        ids = self._domain_index.get(domain, set())
        return [self._terms[tid] for tid in ids if tid in self._terms]
    
    def get_medical_terms(self) -> List[Term]:
        return self.get_by_domain(TermDomain.MEDICAL)
    
    def get_legal_terms(self) -> List[Term]:
        return self.get_by_domain(TermDomain.LEGAL)
    
    def get_finance_terms(self) -> List[Term]:
        return self.get_by_domain(TermDomain.FINANCE)
    
    def get_it_terms(self) -> List[Term]:
        return self.get_by_domain(TermDomain.IT)
    
    def get_by_sub_domain(self, sub_domain: str) -> List[Term]:
        ids = self._sub_domain_index.get(sub_domain, set())
        return [self._terms[tid] for tid in ids if tid in self._terms]
    
    def is_term(self, text: str) -> bool:
        if text in self._terms:
            return True
        if text in self._name_index:
            return True
        if text in self._alias_index:
            return True
        if text.lower() in self._english_index:
            return True
        return False
    
    def get_term_domain(self, text: str) -> Optional[TermDomain]:
        terms = self.get_by_name(text)
        if terms:
            return terms[0].domain
        
        term = self.get_by_alias(text)
        if term:
            return term.domain
        
        term = self.get_by_english(text)
        if term:
            return term.domain
        
        return None
    
    def recognize_terms(self, text: str) -> List[Tuple[Term, int, int]]:
        results: List[Tuple[Term, int, int]] = []
        used_positions: Set[int] = set()
        
        all_names = set(self._name_index.keys()) | set(self._alias_index.keys())
        sorted_names = sorted(all_names, key=len, reverse=True)
        
        for name in sorted_names:
            start = 0
            while True:
                pos = text.find(name, start)
                if pos == -1:
                    break
                
                end = pos + len(name)
                overlap = False
                for i in range(pos, end):
                    if i in used_positions:
                        overlap = True
                        break
                
                if not overlap:
                    if name in self._alias_index:
                        tid = self._alias_index[name]
                        term = self._terms.get(tid)
                    else:
                        ids = self._name_index.get(name, set())
                        term = self._terms.get(next(iter(ids), ''))
                    
                    if term:
                        results.append((term, pos, end))
                        for i in range(pos, end):
                            used_positions.add(i)
                
                start = pos + 1
        
        results.sort(key=lambda x: x[1])
        return results
    
    def recognize_terms_by_domain(
        self,
        text: str,
        domain: TermDomain
    ) -> List[Tuple[Term, int, int]]:
        all_results = self.recognize_terms(text)
        return [(t, s, e) for t, s, e in all_results if t.domain == domain]
    
    def get_all_terms(self) -> List[Term]:
        return list(self._terms.values())
    
    def get_term_count(self) -> int:
        return self._term_count
    
    def is_loaded(self) -> bool:
        return self._loaded
    
    def is_sogou_loaded(self) -> bool:
        return self._sogou_loaded
    
    def get_statistics(self) -> Dict[str, Any]:
        return {
            "loaded": self._loaded,
            "sogou_loaded": self._sogou_loaded,
            "total_count": self._term_count,
            "medical_count": len(self._domain_index[TermDomain.MEDICAL]),
            "legal_count": len(self._domain_index[TermDomain.LEGAL]),
            "finance_count": len(self._domain_index[TermDomain.FINANCE]),
            "it_count": len(self._domain_index[TermDomain.IT]),
            "automotive_count": len(self._domain_index[TermDomain.AUTOMOTIVE]),
            "food_count": len(self._domain_index[TermDomain.FOOD]),
            "entertainment_count": len(self._domain_index[TermDomain.ENTERTAINMENT]),
            "government_count": len(self._domain_index[TermDomain.GOVERNMENT]),
            "ecommerce_count": len(self._domain_index[TermDomain.ECOMMERCE]),
            "other_count": len(self._domain_index[TermDomain.OTHER]),
            "alias_count": len(self._alias_index),
            "english_count": len(self._english_index),
            "sub_domain_count": len(self._sub_domain_index),
        }
    
    def add_term(self, term: Term) -> None:
        self._add_term(term)
    
    def add_term_simple(
        self,
        name: str,
        domain: TermDomain,
        sub_domain: Optional[str] = None,
        english: Optional[str] = None,
        aliases: Optional[List[str]] = None,
        definition: Optional[str] = None,
        source: Optional[str] = None,
    ) -> None:
        term_id = f"{domain.value}_{self._term_count:06d}"
        term = Term(
            term_id=term_id,
            name=name,
            domain=domain,
            sub_domain=sub_domain,
            english=english,
            aliases=aliases or [],
            definition=definition,
            source=source,
        )
        self._add_term(term)
    
    def save_data(self, path: str) -> None:
        with open(path, 'w', encoding='utf-8') as f:
            f.write("# 专业术语库数据文件\n")
            f.write("# 格式说明:\n")
            f.write("# 术语ID\\t名称\\t领域\\t子领域\\t英文\\t别名\\t解释\\t来源\n")
            f.write("#\n")
            
            for domain in TermDomain:
                if domain == TermDomain.OTHER:
                    continue
                
                f.write(f"\n# @{domain.value}\n")
                ids = sorted(self._domain_index[domain])
                for tid in ids:
                    term = self._terms.get(tid)
                    if term:
                        parts = [
                            term.term_id,
                            term.name,
                            term.domain.value,
                            term.sub_domain or "",
                            term.english or "",
                            ','.join(term.aliases) if term.aliases else "",
                            term.definition or "",
                            term.source or "",
                        ]
                        f.write('\t'.join(parts) + '\n')
    
    def __len__(self) -> int:
        return self._term_count
    
    def __contains__(self, text: str) -> bool:
        return self.is_term(text)
    
    def __getitem__(self, term_id: str) -> Optional[Term]:
        return self.get_by_id(term_id)
    
    def __repr__(self) -> str:
        return (
            f"TerminologyDatabase(terms={self._term_count}, "
            f"loaded={self._loaded}, sogou_loaded={self._sogou_loaded})"
        )


class TerminologyManager:
    def __init__(self, load_default: bool = True, load_sogou: bool = True):
        self._database: Optional[TerminologyDatabase] = None
        if load_default or load_sogou:
            self._database = TerminologyDatabase(
                load_default=load_default,
                load_sogou=load_sogou
            )
    
    def load(self, path: Optional[str] = None, load_sogou: bool = True) -> None:
        self._database = TerminologyDatabase(
            load_default=False,
            load_sogou=False
        )
        if path:
            self._database.load_data(path)
        if load_sogou:
            self._database._load_sogou_data()
            self._database._sogou_loaded = True
    
    def get_database(self) -> Optional[TerminologyDatabase]:
        return self._database
    
    def get_by_id(self, term_id: str) -> Optional[Term]:
        if self._database is None:
            return None
        return self._database.get_by_id(term_id)
    
    def get_by_name(self, name: str) -> List[Term]:
        if self._database is None:
            return []
        return self._database.get_by_name(name)
    
    def get_by_alias(self, alias: str) -> Optional[Term]:
        if self._database is None:
            return None
        return self._database.get_by_alias(alias)
    
    def get_by_english(self, english: str) -> Optional[Term]:
        if self._database is None:
            return None
        return self._database.get_by_english(english)
    
    def search(self, query: str) -> List[Term]:
        if self._database is None:
            return []
        return self._database.search(query)
    
    def get_by_domain(self, domain: TermDomain) -> List[Term]:
        if self._database is None:
            return []
        return self._database.get_by_domain(domain)
    
    def get_medical_terms(self) -> List[Term]:
        if self._database is None:
            return []
        return self._database.get_medical_terms()
    
    def get_legal_terms(self) -> List[Term]:
        if self._database is None:
            return []
        return self._database.get_legal_terms()
    
    def get_finance_terms(self) -> List[Term]:
        if self._database is None:
            return []
        return self._database.get_finance_terms()
    
    def get_it_terms(self) -> List[Term]:
        if self._database is None:
            return []
        return self._database.get_it_terms()
    
    def is_term(self, text: str) -> bool:
        if self._database is None:
            return False
        return self._database.is_term(text)
    
    def get_term_domain(self, text: str) -> Optional[TermDomain]:
        if self._database is None:
            return None
        return self._database.get_term_domain(text)
    
    def recognize_terms(self, text: str) -> List[Tuple[Term, int, int]]:
        if self._database is None:
            return []
        return self._database.recognize_terms(text)
    
    def recognize_terms_by_domain(
        self,
        text: str,
        domain: TermDomain
    ) -> List[Tuple[Term, int, int]]:
        if self._database is None:
            return []
        return self._database.recognize_terms_by_domain(text, domain)
    
    def get_statistics(self) -> Dict[str, Any]:
        if self._database is None:
            return {"loaded": False}
        return self._database.get_statistics()
    
    def is_loaded(self) -> bool:
        return self._database is not None and self._database.is_loaded()
