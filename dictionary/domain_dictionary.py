import os
from typing import Optional, Dict, List, Any
from AuroraNLP.dictionary.dictionary import Dictionary


class DomainDictionary(Dictionary):
    DOMAIN_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data', 'domain_dictionaries')
    
    # 预定义领域及其默认优先级
    DOMAINS = {
        'news': {'name': '新闻领域', 'priority': 50},
        'medical': {'name': '医疗领域', 'priority': 50},
        'legal': {'name': '法律领域', 'priority': 50},
        'ecommerce': {'name': '电商领域', 'priority': 50}
    }
    
    def __init__(self, domain: str, load_default: bool = True, priority: Optional[int] = None):
        if domain not in self.DOMAINS:
            raise ValueError(f"不支持的领域: {domain}")
        
        self._domain = domain
        self._domain_name = self.DOMAINS[domain]['name']
        
        if priority is None:
            priority = self.DOMAINS[domain]['priority']
        
        super().__init__(load_default=False, priority=priority)
        self.name = f"domain_{domain}"
        
        if load_default:
            self._load_domain_dictionary()
    
    @property
    def domain(self) -> str:
        return self._domain
    
    @property
    def domain_name(self) -> str:
        return self._domain_name
    
    def _load_domain_dictionary(self) -> None:
        domain_file = os.path.join(self.DOMAIN_DATA_DIR, f"{self._domain}.txt")
        if os.path.exists(domain_file):
            self.load_dictionary(domain_file)
    
    def get_domain_info(self) -> Dict[str, Any]:
        return {
            'domain': self._domain,
            'domain_name': self._domain_name,
            'name': self.name,
            'priority': self.priority,
            'word_count': len(self)
        }
    
    @classmethod
    def get_supported_domains(cls) -> List[str]:
        return list(cls.DOMAINS.keys())
    
    @classmethod
    def create_domain_dictionary(cls, domain: str, priority: Optional[int] = None) -> 'DomainDictionary':
        return cls(domain, load_default=True, priority=priority)


class DomainDictionaryManager:
    def __init__(self):
        self._domain_dictionaries: Dict[str, DomainDictionary] = {}
    
    def register_domain_dictionary(self, domain_dict: DomainDictionary) -> None:
        self._domain_dictionaries[domain_dict.domain] = domain_dict
    
    def unregister_domain_dictionary(self, domain: str) -> bool:
        if domain in self._domain_dictionaries:
            del self._domain_dictionaries[domain]
            return True
        return False
    
    def get_domain_dictionary(self, domain: str) -> Optional[DomainDictionary]:
        return self._domain_dictionaries.get(domain)
    
    def get_all_domain_dictionaries(self) -> Dict[str, DomainDictionary]:
        return self._domain_dictionaries.copy()
    
    def load_all_domains(self, priority: Optional[int] = None) -> None:
        for domain in DomainDictionary.get_supported_domains():
            domain_dict = DomainDictionary.create_domain_dictionary(domain, priority)
            self.register_domain_dictionary(domain_dict)
    
    def get_all_domains_info(self) -> List[Dict[str, Any]]:
        return [
            dict({'type': 'domain'}, **dd.get_domain_info())
            for dd in self._domain_dictionaries.values()
        ]
    
    def __len__(self) -> int:
        return len(self._domain_dictionaries)