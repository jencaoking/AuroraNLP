import os
import threading
import warnings
from typing import Set, Optional, List, Tuple, Dict, Any

from AuroraNLP.dictionary.trie import Trie


class Dictionary:
    DEFAULT_DICT_PATH = os.path.join(os.path.dirname(__file__), 'data', 'dictionary.txt')

    def __init__(self, load_default: bool = True, priority: int = 0):
        self._trie = Trie()
        self._words_cache: Optional[Set[str]] = None
        self._priority: int = priority
        self._name: str = "default"
        if load_default:
            self._load_default_dictionary()

    @property
    def priority(self) -> int:
        return self._priority

    @priority.setter
    def priority(self, value: int) -> None:
        self._priority = value

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    def _load_default_dictionary(self) -> None:
        if os.path.exists(self.DEFAULT_DICT_PATH):
            self.load_dictionary(self.DEFAULT_DICT_PATH)

    def load_dictionary(
        self,
        path: str,
        priority: Optional[int] = None,
        default_weight: float = 1.0
    ) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"词典文件不存在: {path}")

        if priority is None:
            priority = self._priority

        loaded_count = 0
        error_count = 0

        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) >= 4:
                    word, pos_tag, weight_str, priority_str = parts[0], parts[1], parts[2], parts[3]
                    try:
                        weight = float(weight_str)
                        word_priority = int(priority_str)
                    except ValueError as e:
                        warnings.warn(
                            f"第 {line_num} 行: 权重或优先级格式错误，使用默认值: {e}",
                            UserWarning,
                            stacklevel=2
                        )
                        weight = default_weight
                        word_priority = priority
                    self._trie.insert(word, pos_tag, weight, word_priority)
                    loaded_count += 1
                elif len(parts) >= 3:
                    word, pos_tag, weight_str = parts[0], parts[1], parts[2]
                    try:
                        weight = float(weight_str)
                    except ValueError:
                        weight = default_weight
                    self._trie.insert(word, pos_tag, weight, priority)
                    loaded_count += 1
                elif len(parts) >= 2:
                    word, pos_tag = parts[0], parts[1]
                    if not word:
                        warnings.warn(
                            f"第 {line_num} 行: 空词汇将被跳过",
                            UserWarning,
                            stacklevel=2
                        )
                        error_count += 1
                        continue
                    if not pos_tag or not pos_tag.replace('/', '').isalpha():
                        warnings.warn(
                            f"第 {line_num} 行: 无效的词性标签 '{pos_tag}'，将使用默认值",
                            UserWarning,
                            stacklevel=2
                        )
                        pos_tag = 'x'
                    self._trie.insert(word, pos_tag, default_weight, priority)
                    loaded_count += 1
                    if len(parts) > 2:
                        warnings.warn(
                            f"词典行格式多余字段将被忽略: '{line}'",
                            UserWarning,
                            stacklevel=2
                        )
                elif len(parts) == 1:
                    word = parts[0]
                    if word:
                        self._trie.insert(word, None, default_weight, priority)
                        loaded_count += 1
                    else:
                        warnings.warn(
                            f"第 {line_num} 行: 空词汇将被跳过",
                            UserWarning,
                            stacklevel=2
                        )
                        error_count += 1

        self._words_cache = None

        if loaded_count == 0:
            warnings.warn(
                f"词典文件 '{path}' 未加载任何有效词汇",
                UserWarning,
                stacklevel=2
            )

        if error_count > 0:
            warnings.warn(
                f"词典加载完成，共加载 {loaded_count} 个词汇，跳过 {error_count} 个无效条目",
                UserWarning,
                stacklevel=2
            )

    def save_dictionary(self, path: str, include_weight_priority: bool = False) -> None:
        with open(path, 'w', encoding='utf-8') as f:
            for word in self.get_words():
                found, pos_tag, weight, priority = self._trie.search_with_info(word)
                if include_weight_priority:
                    if pos_tag:
                        f.write(f"{word} {pos_tag} {weight} {priority}\n")
                    else:
                        f.write(f"{word} x {weight} {priority}\n")
                else:
                    if pos_tag:
                        f.write(f"{word} {pos_tag}\n")
                    else:
                        f.write(f"{word}\n")

    def get_words(self) -> Set[str]:
        if self._words_cache is None:
            self._words_cache = set()
            self._collect_words(self._trie.root, "", self._words_cache)
        return self._words_cache.copy()

    def _collect_words(self, node, prefix: str, result: Set[str]) -> None:
        if node.is_word:
            result.add(prefix)
        for char, child in node.children.items():
            self._collect_words(child, prefix + char, result)

    def add_word(
        self,
        word: str,
        pos_tag: Optional[str] = None,
        weight: float = 1.0,
        priority: Optional[int] = None
    ) -> None:
        if priority is None:
            priority = self._priority
        self._trie.insert(word, pos_tag, weight, priority)
        self._words_cache = None

    def remove_word(self, word: str) -> bool:
        result = self._trie.remove(word)
        if result:
            self._words_cache = None
        return result

    def search_in_dict(self, word: str) -> bool:
        return self._trie.search(word)

    def search_with_pos(self, word: str) -> Tuple[bool, Optional[str]]:
        return self._trie.search_with_pos(word)

    def search_with_info(self, word: str) -> Tuple[bool, Optional[str], float, int]:
        return self._trie.search_with_info(word)

    def get_pos_tag(self, word: str) -> Optional[str]:
        _, pos_tag = self._trie.search_with_pos(word)
        return pos_tag

    def get_weight(self, word: str) -> float:
        return self._trie.get_weight(word)

    def get_priority(self, word: str) -> int:
        return self._trie.get_priority(word)

    def set_weight(self, word: str, weight: float) -> bool:
        return self._trie.set_weight(word, weight)

    def set_priority(self, word: str, priority: int) -> bool:
        return self._trie.set_priority(word, priority)

    def has_prefix(self, prefix: str) -> bool:
        return self._trie.starts_with(prefix)

    def get_max_match_length(self, text: str, start: int = 0, max_len: int = 15) -> int:
        return self._trie.get_max_match_length(text, start, max_len)

    def get_max_match_with_pos(self, text: str, start: int = 0, max_len: int = 15) -> Tuple[int, Optional[str]]:
        return self._trie.get_max_match_with_pos(text, start, max_len)

    def get_max_match_with_info(
        self,
        text: str,
        start: int = 0,
        max_len: int = 15
    ) -> Tuple[int, Optional[str], float, int]:
        return self._trie.get_max_match_with_info(text, start, max_len)

    def get_all_matches_with_info(
        self,
        text: str,
        start: int = 0,
        max_len: int = 15
    ) -> List[Tuple[int, str, Optional[str], float, int]]:
        return self._trie.get_all_matches_with_info(text, start, max_len)

    def __len__(self) -> int:
        return len(self._trie)

    def __contains__(self, word: str) -> bool:
        return word in self._trie


class UserDictionary:
    def __init__(self, name: str = "user", priority: int = 100):
        self._trie = Trie()
        self._words_cache: Optional[Set[str]] = None
        self._priority: int = priority
        self._name: str = name
        self._default_weight: float = 10.0

    @property
    def priority(self) -> int:
        return self._priority

    @priority.setter
    def priority(self, value: int) -> None:
        self._priority = value

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    @property
    def default_weight(self) -> float:
        return self._default_weight

    @default_weight.setter
    def default_weight(self, value: float) -> None:
        self._default_weight = value

    def load_dictionary(
        self,
        path: str,
        priority: Optional[int] = None,
        default_weight: Optional[float] = None
    ) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"词典文件不存在: {path}")

        if priority is None:
            priority = self._priority
        if default_weight is None:
            default_weight = self._default_weight

        loaded_count = 0
        error_count = 0

        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) >= 4:
                    word, pos_tag, weight_str, priority_str = parts[0], parts[1], parts[2], parts[3]
                    try:
                        weight = float(weight_str)
                        word_priority = int(priority_str)
                    except ValueError:
                        weight = default_weight
                        word_priority = priority
                    self._trie.insert(word, pos_tag, weight, word_priority)
                    loaded_count += 1
                elif len(parts) >= 3:
                    word, pos_tag, weight_str = parts[0], parts[1], parts[2]
                    try:
                        weight = float(weight_str)
                    except ValueError:
                        weight = default_weight
                    self._trie.insert(word, pos_tag, weight, priority)
                    loaded_count += 1
                elif len(parts) >= 2:
                    word, pos_tag = parts[0], parts[1]
                    if word:
                        self._trie.insert(word, pos_tag, default_weight, priority)
                        loaded_count += 1
                elif len(parts) == 1:
                    word = parts[0]
                    if word:
                        self._trie.insert(word, None, default_weight, priority)
                        loaded_count += 1

        self._words_cache = None

        if loaded_count > 0:
            warnings.warn(f"用户词典 '{self._name}' 加载完成，共加载 {loaded_count} 个词汇", UserWarning)

    def save_dictionary(self, path: str) -> None:
        with open(path, 'w', encoding='utf-8') as f:
            for word in self.get_words():
                found, pos_tag, weight, priority = self._trie.search_with_info(word)
                if pos_tag:
                    f.write(f"{word} {pos_tag} {weight} {priority}\n")
                else:
                    f.write(f"{word} x {weight} {priority}\n")

    def get_words(self) -> Set[str]:
        if self._words_cache is None:
            self._words_cache = set()
            self._collect_words(self._trie.root, "", self._words_cache)
        return self._words_cache.copy()

    def _collect_words(self, node, prefix: str, result: Set[str]) -> None:
        if node.is_word:
            result.add(prefix)
        for char, child in node.children.items():
            self._collect_words(child, prefix + char, result)

    def add_word(
        self,
        word: str,
        pos_tag: Optional[str] = None,
        weight: Optional[float] = None,
        priority: Optional[int] = None
    ) -> None:
        if weight is None:
            weight = self._default_weight
        if priority is None:
            priority = self._priority
        self._trie.insert(word, pos_tag, weight, priority)
        self._words_cache = None

    def remove_word(self, word: str) -> bool:
        result = self._trie.remove(word)
        if result:
            self._words_cache = None
        return result

    def search_in_dict(self, word: str) -> bool:
        return self._trie.search(word)

    def search_with_pos(self, word: str) -> Tuple[bool, Optional[str]]:
        return self._trie.search_with_pos(word)

    def search_with_info(self, word: str) -> Tuple[bool, Optional[str], float, int]:
        return self._trie.search_with_info(word)

    def get_pos_tag(self, word: str) -> Optional[str]:
        _, pos_tag = self._trie.search_with_pos(word)
        return pos_tag

    def get_weight(self, word: str) -> float:
        return self._trie.get_weight(word)

    def get_priority(self, word: str) -> int:
        return self._trie.get_priority(word)

    def set_weight(self, word: str, weight: float) -> bool:
        return self._trie.set_weight(word, weight)

    def set_priority(self, word: str, priority: int) -> bool:
        return self._trie.set_priority(word, priority)

    def has_prefix(self, prefix: str) -> bool:
        return self._trie.starts_with(prefix)

    def get_max_match_length(self, text: str, start: int = 0, max_len: int = 15) -> int:
        return self._trie.get_max_match_length(text, start, max_len)

    def get_max_match_with_pos(self, text: str, start: int = 0, max_len: int = 15) -> Tuple[int, Optional[str]]:
        return self._trie.get_max_match_with_pos(text, start, max_len)

    def get_max_match_with_info(
        self,
        text: str,
        start: int = 0,
        max_len: int = 15
    ) -> Tuple[int, Optional[str], float, int]:
        return self._trie.get_max_match_with_info(text, start, max_len)

    def get_all_matches_with_info(
        self,
        text: str,
        start: int = 0,
        max_len: int = 15
    ) -> List[Tuple[int, str, Optional[str], float, int]]:
        return self._trie.get_all_matches_with_info(text, start, max_len)

    def __len__(self) -> int:
        return len(self._trie)

    def __contains__(self, word: str) -> bool:
        return word in self._trie


class DictionaryManager:
    def __init__(self):
        self._dictionaries: Dict[str, Dictionary] = {}
        self._user_dictionaries: Dict[str, UserDictionary] = {}
        self._domain_dictionaries: Dict[str, 'DomainDictionary'] = {}
        self._merged_trie: Optional[Trie] = None
        self._cache_valid: bool = False
        self._lock = threading.RLock()

    def register_dictionary(self, dictionary: Dictionary) -> None:
        with self._lock:
            self._dictionaries[dictionary.name] = dictionary
            self._cache_valid = False

    def register_user_dictionary(self, user_dict: UserDictionary) -> None:
        with self._lock:
            self._user_dictionaries[user_dict.name] = user_dict
            self._cache_valid = False

    def register_domain_dictionary(self, domain_dict: 'DomainDictionary') -> None:
        with self._lock:
            self._domain_dictionaries[domain_dict.domain] = domain_dict
            self._cache_valid = False

    def unregister_dictionary(self, name: str) -> bool:
        with self._lock:
            if name in self._dictionaries:
                del self._dictionaries[name]
                self._cache_valid = False
                return True
            if name in self._user_dictionaries:
                del self._user_dictionaries[name]
                self._cache_valid = False
                return True
            if name in self._domain_dictionaries:
                del self._domain_dictionaries[name]
                self._cache_valid = False
                return True
        return False

    def get_dictionary(self, name: str) -> Optional[Dictionary]:
        return self._dictionaries.get(name)

    def get_user_dictionary(self, name: str) -> Optional[UserDictionary]:
        return self._user_dictionaries.get(name)

    def get_domain_dictionary(self, domain: str) -> Optional['DomainDictionary']:
        return self._domain_dictionaries.get(domain)

    def _rebuild_cache(self) -> None:
        if self._cache_valid and self._merged_trie is not None:
            return

        self._merged_trie = Trie()

        all_dicts: List[Tuple[int, Any]] = []
        for d in self._dictionaries.values():
            all_dicts.append((d.priority, d))
        for d in self._user_dictionaries.values():
            all_dicts.append((d.priority, d))
        for d in self._domain_dictionaries.values():
            all_dicts.append((d.priority, d))

        all_dicts.sort(key=lambda x: x[0])

        for _, d in all_dicts:
            self._merge_dictionary(d)

        self._cache_valid = True

    def _merge_dictionary(self, dictionary) -> None:
        words = dictionary.get_words()
        for word in words:
            found, pos_tag, weight, priority = dictionary.search_with_info(word)
            existing = self._merged_trie.search_with_info(word)
            if not existing[0] or priority > existing[3]:
                self._merged_trie.insert(word, pos_tag, weight, priority)
            elif priority == existing[3] and weight > existing[2]:
                self._merged_trie.insert(word, pos_tag, weight, priority)

    def search(self, word: str) -> bool:
        with self._lock:
            self._rebuild_cache()
            return self._merged_trie.search(word)

    def search_with_pos(self, word: str) -> Tuple[bool, Optional[str]]:
        with self._lock:
            self._rebuild_cache()
            return self._merged_trie.search_with_pos(word)

    def search_with_info(self, word: str) -> Tuple[bool, Optional[str], float, int]:
        with self._lock:
            self._rebuild_cache()
            return self._merged_trie.search_with_info(word)

    def get_max_match_length(self, text: str, start: int = 0, max_len: int = 15) -> int:
        with self._lock:
            self._rebuild_cache()
            return self._merged_trie.get_max_match_length(text, start, max_len)

    def get_max_match_with_pos(self, text: str, start: int = 0, max_len: int = 15) -> Tuple[int, Optional[str]]:
        with self._lock:
            self._rebuild_cache()
            return self._merged_trie.get_max_match_with_pos(text, start, max_len)

    def get_max_match_with_info(
        self,
        text: str,
        start: int = 0,
        max_len: int = 15
    ) -> Tuple[int, Optional[str], float, int]:
        with self._lock:
            self._rebuild_cache()
            return self._merged_trie.get_max_match_with_info(text, start, max_len)

    def get_all_matches_with_info(
        self,
        text: str,
        start: int = 0,
        max_len: int = 15
    ) -> List[Tuple[int, str, Optional[str], float, int]]:
        with self._lock:
            self._rebuild_cache()
            return self._merged_trie.get_all_matches_with_info(text, start, max_len)

    def get_all_dictionaries_info(self) -> List[Dict[str, Any]]:
        result = []
        for name, d in self._dictionaries.items():
            result.append({
                'name': name,
                'type': 'system',
                'priority': d.priority,
                'word_count': len(d)
            })
        for name, d in self._user_dictionaries.items():
            result.append({
                'name': name,
                'type': 'user',
                'priority': d.priority,
                'word_count': len(d)
            })
        for domain, d in self._domain_dictionaries.items():
            result.append({
                'name': d.name,
                'type': 'domain',
                'domain': domain,
                'domain_name': d.domain_name,
                'priority': d.priority,
                'word_count': len(d)
            })
        return sorted(result, key=lambda x: x['priority'], reverse=True)

    def invalidate_cache(self) -> None:
        self._cache_valid = False

    def __len__(self) -> int:
        self._rebuild_cache()
        return len(self._merged_trie)
