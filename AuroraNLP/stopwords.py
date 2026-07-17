import os
import json
from typing import Set, List, Optional, Dict, Any


class StopWords:
    DEFAULT_STOPWORDS_PATH = os.path.join(os.path.dirname(__file__), 'data', 'stopwords.txt')
    STOPWORDS_DIR = os.path.join(os.path.dirname(__file__), 'data', 'stopwords')
    SCENARIOS_PATH = os.path.join(STOPWORDS_DIR, 'scenario', 'scenarios.json')

    def __init__(self, load_default: bool = True, scenario: Optional[str] = None):
        self._stopwords: Set[str] = set()
        self._scenario: Optional[str] = scenario
        self._scenarios: Dict[str, Any] = {}
        self._load_scenarios()
        if load_default:
            if scenario:
                self.load_scenario(scenario)
            else:
                self._load_default_stopwords()

    def _load_scenarios(self) -> None:
        if os.path.exists(self.SCENARIOS_PATH):
            with open(self.SCENARIOS_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self._scenarios = data.get('scenarios', {})

    def _load_default_stopwords(self) -> None:
        if os.path.exists(self.DEFAULT_STOPWORDS_PATH):
            self.load_stopwords(self.DEFAULT_STOPWORDS_PATH)
        else:
            common_path = os.path.join(self.STOPWORDS_DIR, 'common', 'common.txt')
            if os.path.exists(common_path):
                self.load_stopwords(common_path)

    def load_stopwords(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"停用词文件不存在: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip()
                if word:
                    self._stopwords.add(word)

    def load_common_stopwords(self, filename: str = 'common.txt') -> None:
        common_path = os.path.join(self.STOPWORDS_DIR, 'common', filename)
        self.load_stopwords(common_path)

    def load_domain_stopwords(self, domain: str, filename: str = None) -> None:
        domain_path = os.path.join(self.STOPWORDS_DIR, 'domain', domain)
        if filename:
            file_path = os.path.join(domain_path, filename)
            self.load_stopwords(file_path)
        else:
            if os.path.exists(domain_path):
                for file in os.listdir(domain_path):
                    if file.endswith('.txt'):
                        file_path = os.path.join(domain_path, file)
                        self.load_stopwords(file_path)

    def load_scenario(self, scenario_name: str) -> None:
        if scenario_name not in self._scenarios:
            raise ValueError(f"场景不存在: {scenario_name}")
        scenario = self._scenarios[scenario_name]
        stopwords_config = scenario.get('stopwords', {})
        
        common_files = stopwords_config.get('common', [])
        for file in common_files:
            file_path = os.path.join(self.STOPWORDS_DIR, 'common', file)
            if os.path.exists(file_path):
                self.load_stopwords(file_path)
        
        domain_files = stopwords_config.get('domain', [])
        for file in domain_files:
            file_path = os.path.join(self.STOPWORDS_DIR, 'domain', file)
            if os.path.exists(file_path):
                self.load_stopwords(file_path)
        
        custom_files = stopwords_config.get('custom', [])
        for file in custom_files:
            if os.path.exists(file):
                self.load_stopwords(file)

    def add_stopword(self, word: str) -> None:
        self._stopwords.add(word)

    def remove_stopword(self, word: str) -> bool:
        if word in self._stopwords:
            self._stopwords.remove(word)
            return True
        return False

    def is_stopword(self, word: str) -> bool:
        return word in self._stopwords

    def filter(self, words: List[str]) -> List[str]:
        return [word for word in words if word not in self._stopwords]

    def filter_with_pos(self, words_with_pos: List[tuple]) -> List[tuple]:
        return [(word, pos) for word, pos in words_with_pos if word not in self._stopwords]

    def get_stopwords(self) -> Set[str]:
        return self._stopwords.copy()

    def get_scenarios(self) -> Dict[str, Any]:
        return self._scenarios.copy()

    def get_current_scenario(self) -> Optional[str]:
        return self._scenario

    def __len__(self) -> int:
        return len(self._stopwords)

    def __contains__(self, word: str) -> bool:
        return word in self._stopwords

    def save_stopwords(self, path: str) -> None:
        with open(path, 'w', encoding='utf-8') as f:
            for word in sorted(self._stopwords):
                f.write(f"{word}\n")


__all__ = ['StopWords']
