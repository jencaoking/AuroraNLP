import os
import time
import threading
from typing import Set, Optional, List, Tuple, Dict, Any, Callable
from datetime import datetime

from AuroraNLP.dictionary.dictionary import Dictionary, UserDictionary, DictionaryManager
from AuroraNLP.dictionary.trie import Trie


class DictionaryUpdateEvent:
    """词典更新事件"""
    def __init__(self, dictionary_name: str, update_type: str, words: List[str]):
        self.dictionary_name = dictionary_name
        self.update_type = update_type  # 'add', 'remove', 'update'
        self.words = words
        self.timestamp = datetime.now()
        self.timestamp_str = self.timestamp.strftime('%Y-%m-%d %H:%M:%S')


class DictionaryObserver:
    """词典观察者接口"""
    def on_dictionary_update(self, event: DictionaryUpdateEvent) -> None:
        """词典更新回调"""
        pass


class IncrementalDictionary(Dictionary):
    """支持增量更新的词典类"""
    def __init__(self, load_default: bool = True, priority: int = 0, name: str = "incremental"):
        super().__init__(load_default, priority)
        self._name = name
        self._observers: List[DictionaryObserver] = []
        self._lock = threading.RLock()
    
    def add_observer(self, observer: DictionaryObserver) -> None:
        """添加观察者"""
        if observer not in self._observers:
            self._observers.append(observer)
    
    def remove_observer(self, observer: DictionaryObserver) -> None:
        """移除观察者"""
        if observer in self._observers:
            self._observers.remove(observer)
    
    def _notify_observers(self, update_type: str, words: List[str]) -> None:
        """通知观察者"""
        event = DictionaryUpdateEvent(self._name, update_type, words)
        for observer in self._observers:
            try:
                observer.on_dictionary_update(event)
            except Exception as e:
                print(f"通知观察者时出错: {e}")
    
    def add_word(self, word: str, pos_tag: Optional[str] = None, weight: float = 1.0, priority: Optional[int] = None) -> None:
        """添加词语并通知观察者"""
        with self._lock:
            super().add_word(word, pos_tag, weight, priority)
            self._notify_observers('add', [word])
    
    def remove_word(self, word: str) -> bool:
        """删除词语并通知观察者"""
        with self._lock:
            result = super().remove_word(word)
            if result:
                self._notify_observers('remove', [word])
            return result
    
    def update_word(self, word: str, pos_tag: Optional[str] = None, weight: Optional[float] = None, priority: Optional[int] = None) -> bool:
        """更新词语并通知观察者"""
        with self._lock:
            if not self.search_in_dict(word):
                return False
            
            # 保存旧值
            old_found, old_pos, old_weight, old_priority = self.search_with_info(word)
            
            # 更新词语
            if pos_tag is not None:
                # 先删除再添加
                self.remove_word(word)
                self.add_word(word, pos_tag, weight or old_weight, priority or old_priority)
            else:
                # 只更新权重和优先级
                if weight is not None:
                    self._trie.set_weight(word, weight)
                if priority is not None:
                    self._trie.set_priority(word, priority)
                self._words_cache = None
            
            self._notify_observers('update', [word])
            return True
    
    def load_incremental(self, path: str) -> int:
        """增量加载词典文件"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"词典文件不存在: {path}")
        
        added_words = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) >= 2:
                    word = parts[0]
                    pos_tag = parts[1]
                    weight = float(parts[2]) if len(parts) > 2 else 1.0
                    priority = int(parts[3]) if len(parts) > 3 else self._priority
                    self.add_word(word, pos_tag, weight, priority)
                    added_words.append(word)
                elif len(parts) == 1:
                    word = parts[0]
                    self.add_word(word)
                    added_words.append(word)
        
        return len(added_words)


class IncrementalUserDictionary(UserDictionary):
    """支持增量更新的用户词典类"""
    def __init__(self, name: str = "user", priority: int = 100):
        super().__init__(name, priority)
        self._observers: List[DictionaryObserver] = []
        self._lock = threading.RLock()
    
    def add_observer(self, observer: DictionaryObserver) -> None:
        """添加观察者"""
        if observer not in self._observers:
            self._observers.append(observer)
    
    def remove_observer(self, observer: DictionaryObserver) -> None:
        """移除观察者"""
        if observer in self._observers:
            self._observers.remove(observer)
    
    def _notify_observers(self, update_type: str, words: List[str]) -> None:
        """通知观察者"""
        event = DictionaryUpdateEvent(self._name, update_type, words)
        for observer in self._observers:
            try:
                observer.on_dictionary_update(event)
            except Exception as e:
                print(f"通知观察者时出错: {e}")
    
    def add_word(self, word: str, pos_tag: Optional[str] = None, weight: Optional[float] = None, priority: Optional[int] = None) -> None:
        """添加词语并通知观察者"""
        with self._lock:
            super().add_word(word, pos_tag, weight, priority)
            self._notify_observers('add', [word])
    
    def remove_word(self, word: str) -> bool:
        """删除词语并通知观察者"""
        with self._lock:
            result = super().remove_word(word)
            if result:
                self._notify_observers('remove', [word])
            return result
    
    def update_word(self, word: str, pos_tag: Optional[str] = None, weight: Optional[float] = None, priority: Optional[int] = None) -> bool:
        """更新词语并通知观察者"""
        with self._lock:
            if not self.search_in_dict(word):
                return False
            
            # 保存旧值
            old_found, old_pos, old_weight, old_priority = self.search_with_info(word)
            
            # 更新词语
            if pos_tag is not None:
                # 先删除再添加
                self.remove_word(word)
                self.add_word(word, pos_tag, weight or old_weight, priority or old_priority)
            else:
                # 只更新权重和优先级
                if weight is not None:
                    self._trie.set_weight(word, weight)
                if priority is not None:
                    self._trie.set_priority(word, priority)
                self._words_cache = None
            
            self._notify_observers('update', [word])
            return True


class DictionaryUpdateManager(DictionaryObserver):
    """词典更新管理器"""
    def __init__(self, dictionary_manager: DictionaryManager):
        self.dictionary_manager = dictionary_manager
        self._monitored_files: Dict[str, float] = {}
        self._monitor_thread: Optional[threading.Thread] = None
        self._running = False
        self._update_interval = 5  # 5秒检查一次
    
    def add_monitored_file(self, file_path: str, dictionary: IncrementalDictionary) -> None:
        """添加要监控的词典文件"""
        if os.path.exists(file_path):
            self._monitored_files[file_path] = {
                'mtime': os.path.getmtime(file_path),
                'dictionary': dictionary
            }
    
    def remove_monitored_file(self, file_path: str) -> None:
        """移除监控的词典文件"""
        if file_path in self._monitored_files:
            del self._monitored_files[file_path]
    
    def start_monitoring(self) -> None:
        """开始监控文件变化"""
        if not self._running:
            self._running = True
            self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._monitor_thread.start()
    
    def stop_monitoring(self) -> None:
        """停止监控"""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2)
    
    def _monitor_loop(self) -> None:
        """监控循环"""
        while self._running:
            self._check_files()
            time.sleep(self._update_interval)
    
    def _check_files(self) -> None:
        """检查文件变化"""
        for file_path, info in list(self._monitored_files.items()):
            if not os.path.exists(file_path):
                continue
            
            current_mtime = os.path.getmtime(file_path)
            if current_mtime > info['mtime']:
                # 文件已修改
                try:
                    dictionary = info['dictionary']
                    added_count = dictionary.load_incremental(file_path)
                    if added_count > 0:
                        print(f"词典文件 {file_path} 已更新，新增 {added_count} 个词语")
                        # 使词典管理器的缓存失效
                        self.dictionary_manager.invalidate_cache()
                except Exception as e:
                    print(f"更新词典文件 {file_path} 时出错: {e}")
                finally:
                    # 更新文件修改时间
                    info['mtime'] = current_mtime
    
    def on_dictionary_update(self, event: DictionaryUpdateEvent) -> None:
        """词典更新回调"""
        # 使词典管理器的缓存失效
        self.dictionary_manager.invalidate_cache()
        print(f"词典 {event.dictionary_name} 已更新: {event.update_type} {len(event.words)} 个词语")
    
    def trigger_manual_update(self, dictionary_name: str, words: List[Dict[str, Any]]) -> int:
        """手动触发词典更新"""
        # 查找对应的词典
        for name, dictionary in self.dictionary_manager._dictionaries.items():
            if name == dictionary_name and isinstance(dictionary, IncrementalDictionary):
                count = 0
                for word_info in words:
                    word = word_info.get('word')
                    if not word:
                        continue
                    
                    pos_tag = word_info.get('pos_tag')
                    weight = word_info.get('weight', 1.0)
                    priority = word_info.get('priority', dictionary.priority)
                    
                    action = word_info.get('action', 'add')
                    if action == 'add':
                        dictionary.add_word(word, pos_tag, weight, priority)
                        count += 1
                    elif action == 'remove':
                        if dictionary.remove_word(word):
                            count += 1
                    elif action == 'update':
                        if dictionary.update_word(word, pos_tag, weight, priority):
                            count += 1
                return count
        
        # 查找用户词典
        for name, dictionary in self.dictionary_manager._user_dictionaries.items():
            if name == dictionary_name and isinstance(dictionary, IncrementalUserDictionary):
                count = 0
                for word_info in words:
                    word = word_info.get('word')
                    if not word:
                        continue
                    
                    pos_tag = word_info.get('pos_tag')
                    weight = word_info.get('weight')
                    priority = word_info.get('priority')
                    
                    action = word_info.get('action', 'add')
                    if action == 'add':
                        dictionary.add_word(word, pos_tag, weight, priority)
                        count += 1
                    elif action == 'remove':
                        if dictionary.remove_word(word):
                            count += 1
                    elif action == 'update':
                        if dictionary.update_word(word, pos_tag, weight, priority):
                            count += 1
                return count
        
        return 0


class HotUpdateDictionaryManager(DictionaryManager):
    """支持热更新的词典管理器"""
    def __init__(self):
        super().__init__()
        self.update_manager = DictionaryUpdateManager(self)
    
    def register_dictionary(self, dictionary: Dictionary) -> None:
        """注册词典并添加观察者"""
        super().register_dictionary(dictionary)
        if isinstance(dictionary, (IncrementalDictionary, IncrementalUserDictionary)):
            dictionary.add_observer(self.update_manager)
    
    def register_user_dictionary(self, user_dict: UserDictionary) -> None:
        """注册用户词典并添加观察者"""
        super().register_user_dictionary(user_dict)
        if isinstance(user_dict, (IncrementalDictionary, IncrementalUserDictionary)):
            user_dict.add_observer(self.update_manager)
    
    def start_file_monitoring(self) -> None:
        """开始文件监控"""
        self.update_manager.start_monitoring()
    
    def stop_file_monitoring(self) -> None:
        """停止文件监控"""
        self.update_manager.stop_monitoring()
    
    def add_monitored_file(self, file_path: str, dictionary: IncrementalDictionary) -> None:
        """添加监控文件"""
        self.update_manager.add_monitored_file(file_path, dictionary)
    
    def update_dictionary(self, dictionary_name: str, words: List[Dict[str, Any]]) -> int:
        """更新词典"""
        return self.update_manager.trigger_manual_update(dictionary_name, words)
