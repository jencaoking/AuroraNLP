import os
import json
import time
import requests
import threading
from datetime import datetime, timedelta
from typing import Set, Optional, List, Dict, Any

from .dictionary import Dictionary


class NetworkDictionary(Dictionary):
    DEFAULT_NETWORK_DICT_PATH = os.path.join(os.path.dirname(__file__), 'data', 'network_words.json')
    DEFAULT_UPDATE_INTERVAL = 24 * 60 * 60  # 24小时
    DEFAULT_EXPIRY_DAYS = 30  # 30天过期

    def __init__(self, load_default: bool = True, priority: int = 50):
        super().__init__(load_default=False, priority=priority)
        self._name = "network"
        self._update_interval = self.DEFAULT_UPDATE_INTERVAL
        self._expiry_days = self.DEFAULT_EXPIRY_DAYS
        self._last_update = 0
        self._words_with_timestamp: Dict[str, Dict[str, Any]] = {}
        
        if load_default:
            self._load_network_dictionary()
            self._start_auto_update()

    @property
    def update_interval(self) -> int:
        return self._update_interval

    @update_interval.setter
    def update_interval(self, value: int) -> None:
        self._update_interval = value

    @property
    def expiry_days(self) -> int:
        return self._expiry_days

    @expiry_days.setter
    def expiry_days(self, value: int) -> None:
        self._expiry_days = value

    @property
    def last_update(self) -> int:
        return self._last_update

    def _load_network_dictionary(self) -> None:
        if os.path.exists(self.DEFAULT_NETWORK_DICT_PATH):
            try:
                with open(self.DEFAULT_NETWORK_DICT_PATH, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self._words_with_timestamp = data.get('words', {})
                    self._last_update = data.get('last_update', 0)
                    
                # 加载到Trie树
                for word, info in self._words_with_timestamp.items():
                    pos_tag = info.get('pos_tag', 'x')
                    weight = info.get('weight', 1.0)
                    priority = info.get('priority', self._priority)
                    self._trie.insert(word, pos_tag, weight, priority)
                    
            except Exception as e:
                print(f"加载网络词典失败: {e}")

    def save_network_dictionary(self) -> None:
        data = {
            'words': self._words_with_timestamp,
            'last_update': self._last_update
        }
        
        try:
            os.makedirs(os.path.dirname(self.DEFAULT_NETWORK_DICT_PATH), exist_ok=True)
            with open(self.DEFAULT_NETWORK_DICT_PATH, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存网络词典失败: {e}")

    def add_word(
        self,
        word: str,
        pos_tag: Optional[str] = None,
        weight: float = 1.0,
        priority: Optional[int] = None,
        timestamp: Optional[int] = None
    ) -> None:
        if priority is None:
            priority = self._priority
        if timestamp is None:
            timestamp = int(time.time())
        
        super().add_word(word, pos_tag, weight, priority)
        
        self._words_with_timestamp[word] = {
            'pos_tag': pos_tag or 'x',
            'weight': weight,
            'priority': priority,
            'timestamp': timestamp
        }

    def remove_word(self, word: str) -> bool:
        result = super().remove_word(word)
        if result and word in self._words_with_timestamp:
            del self._words_with_timestamp[word]
        return result

    def _crawl_network_hotwords(self) -> List[str]:
        hotwords = []
        
        # 爬取微博热词
        try:
            response = requests.get('https://api.weibo.com/2/trends/hot.json', timeout=10)
            if response.status_code == 200:
                data = response.json()
                if 'trends' in data:
                    hotwords.extend([item['name'] for item in data['trends'][:20]])
        except Exception as e:
            print(f"爬取微博热词失败: {e}")
        
        # 爬取百度热词
        try:
            response = requests.get('https://top.baidu.com/api/board', timeout=10)
            if response.status_code == 200:
                data = response.json()
                if 'data' in data and 'cards' in data['data']:
                    for card in data['data']['cards']:
                        if 'content' in card:
                            hotwords.extend([item['word'] for item in card['content'][:20]])
        except Exception as e:
            print(f"爬取百度热词失败: {e}")
        
        # 爬取知乎热词
        try:
            response = requests.get('https://www.zhihu.com/api/v3/feed/topstory/hot-lists/total', timeout=10)
            if response.status_code == 200:
                data = response.json()
                if 'data' in data:
                    hotwords.extend([item['target']['title'] for item in data['data'][:20]])
        except Exception as e:
            print(f"爬取知乎热词失败: {e}")
        
        return list(set(hotwords))

    def update_hotwords(self) -> int:
        hotwords = self._crawl_network_hotwords()
        added_count = 0
        
        for word in hotwords:
            if word and word not in self._words_with_timestamp:
                self.add_word(word, pos_tag='x', weight=5.0, priority=self._priority)
                added_count += 1
        
        self._last_update = int(time.time())
        self.save_network_dictionary()
        return added_count

    def cleanup_expired_words(self) -> int:
        expired_count = 0
        current_time = int(time.time())
        expiry_timestamp = current_time - (self._expiry_days * 24 * 60 * 60)
        
        expired_words = []
        for word, info in self._words_with_timestamp.items():
            if info.get('timestamp', 0) < expiry_timestamp:
                expired_words.append(word)
        
        for word in expired_words:
            self.remove_word(word)
            expired_count += 1
        
        if expired_count > 0:
            self.save_network_dictionary()
        
        return expired_count

    def _auto_update_task(self):
        while True:
            try:
                current_time = int(time.time())
                if current_time - self._last_update >= self._update_interval:
                    added = self.update_hotwords()
                    expired = self.cleanup_expired_words()
                    print(f"网络新词库自动更新: 添加{added}个新词, 清理{expired}个过期词")
            except Exception as e:
                print(f"自动更新任务失败: {e}")
            finally:
                time.sleep(self._update_interval)

    def _start_auto_update(self):
        thread = threading.Thread(target=self._auto_update_task, daemon=True)
        thread.start()

    def get_words_with_timestamps(self) -> Dict[str, Dict[str, Any]]:
        return self._words_with_timestamp.copy()

    def get_recent_words(self, days: int = 7) -> List[str]:
        recent_timestamp = int(time.time()) - (days * 24 * 60 * 60)
        recent_words = []
        
        for word, info in self._words_with_timestamp.items():
            if info.get('timestamp', 0) >= recent_timestamp:
                recent_words.append(word)
        
        return recent_words

    def get_expired_words(self) -> List[str]:
        current_time = int(time.time())
        expiry_timestamp = current_time - (self._expiry_days * 24 * 60 * 60)
        expired_words = []
        
        for word, info in self._words_with_timestamp.items():
            if info.get('timestamp', 0) < expiry_timestamp:
                expired_words.append(word)
        
        return expired_words

    def get_statistics(self) -> Dict[str, Any]:
        current_time = int(time.time())
        expiry_timestamp = current_time - (self._expiry_days * 24 * 60 * 60)
        
        total_words = len(self._words_with_timestamp)
        recent_words = 0
        expired_words = 0
        
        for info in self._words_with_timestamp.values():
            if info.get('timestamp', 0) >= current_time - (7 * 24 * 60 * 60):
                recent_words += 1
            if info.get('timestamp', 0) < expiry_timestamp:
                expired_words += 1
        
        return {
            'total_words': total_words,
            'recent_words': recent_words,
            'expired_words': expired_words,
            'last_update': self._last_update,
            'update_interval': self._update_interval,
            'expiry_days': self._expiry_days
        }
