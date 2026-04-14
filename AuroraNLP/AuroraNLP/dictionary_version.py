import os
import json
import time
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from .dictionary import Dictionary, UserDictionary
from .trie import Trie


class DictionaryVersion:
    """词典版本类"""
    def __init__(self, version_id: str, message: str, author: str, timestamp: float):
        self.version_id = version_id
        self.message = message
        self.author = author
        self.timestamp = timestamp
        self.timestamp_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
        self.changes: Dict[str, List[Tuple[str, Any, Any]]] = {}
        self.dictionary_state: Dict[str, Dict[str, Any]] = {}
    
    def add_change(self, word: str, old_value: Any, new_value: Any):
        """添加变更记录"""
        if word not in self.changes:
            self.changes[word] = []
        self.changes[word].append(('modify', old_value, new_value))
    
    def add_addition(self, word: str, value: Any):
        """添加新增记录"""
        if word not in self.changes:
            self.changes[word] = []
        self.changes[word].append(('add', None, value))
    
    def add_deletion(self, word: str, value: Any):
        """添加删除记录"""
        if word not in self.changes:
            self.changes[word] = []
        self.changes[word].append(('delete', value, None))
    
    def save_dictionary_state(self, dictionary):
        """保存词典状态"""
        for word in dictionary.get_words():
            found, pos_tag, weight, priority = dictionary.search_with_info(word)
            if found:
                self.dictionary_state[word] = {
                    'pos_tag': pos_tag,
                    'weight': weight,
                    'priority': priority
                }
    
    def restore_dictionary_state(self, dictionary):
        """从版本恢复词典状态"""
        # 先清空词典
        for word in list(dictionary.get_words()):
            dictionary.remove_word(word)
        # 恢复词典状态
        for word, info in self.dictionary_state.items():
            dictionary.add_word(
                word,
                info['pos_tag'],
                info['weight'],
                info['priority']
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'version_id': self.version_id,
            'message': self.message,
            'author': self.author,
            'timestamp': self.timestamp,
            'timestamp_str': self.timestamp_str,
            'changes': self.changes,
            'dictionary_state': self.dictionary_state
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DictionaryVersion':
        """从字典创建实例"""
        version = cls(
            data['version_id'],
            data['message'],
            data['author'],
            data['timestamp']
        )
        version.changes = data.get('changes', {})
        version.dictionary_state = data.get('dictionary_state', {})
        return version


class DictionaryVersionManager:
    """词典版本管理器"""
    def __init__(self, storage_dir: str = None):
        self.storage_dir = storage_dir or os.path.join(os.path.dirname(__file__), 'data', 'versions')
        os.makedirs(self.storage_dir, exist_ok=True)
        self.versions: Dict[str, DictionaryVersion] = {}
        self.current_version: Optional[str] = None
        self._load_versions()
    
    def _load_versions(self):
        """加载版本记录"""
        version_files = [f for f in os.listdir(self.storage_dir) if f.endswith('.json')]
        for file_name in version_files:
            file_path = os.path.join(self.storage_dir, file_name)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    version = DictionaryVersion.from_dict(data)
                    self.versions[version.version_id] = version
            except Exception as e:
                print(f"加载版本文件 {file_name} 失败: {e}")
        
        # 按时间戳排序，最新的版本作为当前版本
        if self.versions:
            sorted_versions = sorted(self.versions.values(), key=lambda v: v.timestamp, reverse=True)
            self.current_version = sorted_versions[0].version_id
    
    def _save_version(self, version: DictionaryVersion):
        """保存版本记录"""
        file_path = os.path.join(self.storage_dir, f"{version.version_id}.json")
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(version.to_dict(), f, ensure_ascii=False, indent=2)
    
    def _generate_version_id(self, dictionary) -> str:
        """生成版本ID"""
        words = sorted(dictionary.get_words())
        content = ''.join(words)
        timestamp = str(time.time())
        combined = content + timestamp
        return hashlib.sha256(combined.encode('utf-8')).hexdigest()[:10]
    
    def commit(self, dictionary, message: str, author: str = 'anonymous') -> str:
        """提交词典变更"""
        # 生成版本ID
        version_id = self._generate_version_id(dictionary)
        
        # 创建版本记录
        version = DictionaryVersion(version_id, message, author, time.time())
        
        # 与上一个版本比较，记录变更
        if self.current_version:
            prev_version = self.versions[self.current_version]
            
            # 获取当前词典和上一版本词典的词语集合
            current_words = set(dictionary.get_words())
            prev_words = set(prev_version.dictionary_state.keys())
            
            # 找出新增的词语
            added_words = current_words - prev_words
            for word in added_words:
                found, pos_tag, weight, priority = dictionary.search_with_info(word)
                if found:
                    value = {'pos_tag': pos_tag, 'weight': weight, 'priority': priority}
                    version.add_addition(word, value)
            
            # 找出删除的词语
            deleted_words = prev_words - current_words
            for word in deleted_words:
                if word in prev_version.dictionary_state:
                    value = prev_version.dictionary_state[word]
                    version.add_deletion(word, value)
            
            # 找出修改的词语
            common_words = current_words & prev_words
            for word in common_words:
                found_current, pos_tag_current, weight_current, priority_current = dictionary.search_with_info(word)
                if found_current:
                    current_value = {'pos_tag': pos_tag_current, 'weight': weight_current, 'priority': priority_current}
                    prev_value = prev_version.dictionary_state.get(word, {})
                    if current_value != prev_value:
                        version.add_change(word, prev_value, current_value)
        
        # 保存当前词典状态
        version.save_dictionary_state(dictionary)
        
        # 保存版本
        self.versions[version_id] = version
        self._save_version(version)
        self.current_version = version_id
        
        return version_id
    
    def checkout(self, version_id: str, dictionary):
        """切换到指定版本"""
        if version_id not in self.versions:
            raise ValueError(f"版本 {version_id} 不存在")
        
        # 从版本恢复词典状态
        version = self.versions[version_id]
        version.restore_dictionary_state(dictionary)
        
        self.current_version = version_id
        return version
    
    def rollback(self, steps: int, dictionary) -> str:
        """回滚到之前的版本"""
        if not self.versions or not self.current_version:
            raise ValueError("没有版本记录")
        
        # 按时间戳排序版本
        sorted_versions = sorted(self.versions.values(), key=lambda v: v.timestamp, reverse=True)
        current_index = next((i for i, v in enumerate(sorted_versions) if v.version_id == self.current_version), -1)
        
        if current_index == -1:
            raise ValueError("当前版本未找到")
        
        target_index = current_index + steps
        if target_index >= len(sorted_versions):
            raise ValueError("回滚步数过多")
        
        target_version = sorted_versions[target_index]
        self.checkout(target_version.version_id, dictionary)
        return target_version.version_id
    
    def get_version_history(self) -> List[DictionaryVersion]:
        """获取版本历史"""
        return sorted(self.versions.values(), key=lambda v: v.timestamp, reverse=True)
    
    def get_version_info(self, version_id: str) -> Optional[DictionaryVersion]:
        """获取版本信息"""
        return self.versions.get(version_id)
    
    def diff(self, version_id1: str, version_id2: str) -> Dict[str, Any]:
        """比较两个版本的差异"""
        if version_id1 not in self.versions or version_id2 not in self.versions:
            raise ValueError("指定的版本不存在")
        
        version1 = self.versions[version_id1]
        version2 = self.versions[version_id2]
        
        # 比较两个版本的词典状态
        state1 = version1.dictionary_state
        state2 = version2.dictionary_state
        
        # 找出新增的词语
        added_words = set(state2.keys()) - set(state1.keys())
        
        # 找出删除的词语
        deleted_words = set(state1.keys()) - set(state2.keys())
        
        # 找出修改的词语
        common_words = set(state1.keys()) & set(state2.keys())
        modified_words = {word for word in common_words if state1[word] != state2[word]}
        
        # 构建差异结果
        changes = {
            'added': {word: state2[word] for word in added_words},
            'deleted': {word: state1[word] for word in deleted_words},
            'modified': {word: {'old': state1[word], 'new': state2[word]} for word in modified_words}
        }
        
        return {
            'version1': version1.to_dict(),
            'version2': version2.to_dict(),
            'changes': changes
        }


class VersionedDictionary(Dictionary):
    """支持版本控制的词典类"""
    def __init__(self, load_default: bool = True, priority: int = 0, version_manager: Optional[DictionaryVersionManager] = None):
        super().__init__(load_default, priority)
        self.version_manager = version_manager or DictionaryVersionManager()
    
    def add_word(self, word: str, pos_tag: Optional[str] = None, weight: float = 1.0, priority: Optional[int] = None) -> None:
        """添加词语并记录变更"""
        old_info = None
        if self.search_in_dict(word):
            old_info = self.search_with_info(word)
        super().add_word(word, pos_tag, weight, priority)
    
    def remove_word(self, word: str) -> bool:
        """删除词语并记录变更"""
        old_info = None
        if self.search_in_dict(word):
            old_info = self.search_with_info(word)
        result = super().remove_word(word)
        return result
    
    def commit(self, message: str, author: str = 'anonymous') -> str:
        """提交变更"""
        return self.version_manager.commit(self, message, author)
    
    def checkout(self, version_id: str):
        """切换到指定版本"""
        return self.version_manager.checkout(version_id, self)
    
    def rollback(self, steps: int) -> str:
        """回滚到之前的版本"""
        return self.version_manager.rollback(steps, self)
    
    def get_version_history(self) -> List[DictionaryVersion]:
        """获取版本历史"""
        return self.version_manager.get_version_history()
    
    def get_current_version(self) -> Optional[DictionaryVersion]:
        """获取当前版本"""
        if self.version_manager.current_version:
            return self.version_manager.get_version_info(self.version_manager.current_version)
        return None


class VersionedUserDictionary(UserDictionary):
    """支持版本控制的用户词典类"""
    def __init__(self, name: str = "user", priority: int = 100, version_manager: Optional[DictionaryVersionManager] = None):
        super().__init__(name, priority)
        self.version_manager = version_manager or DictionaryVersionManager()
    
    def add_word(self, word: str, pos_tag: Optional[str] = None, weight: Optional[float] = None, priority: Optional[int] = None) -> None:
        """添加词语并记录变更"""
        old_info = None
        if self.search_in_dict(word):
            old_info = self.search_with_info(word)
        super().add_word(word, pos_tag, weight, priority)
    
    def remove_word(self, word: str) -> bool:
        """删除词语并记录变更"""
        old_info = None
        if self.search_in_dict(word):
            old_info = self.search_with_info(word)
        result = super().remove_word(word)
        return result
    
    def commit(self, message: str, author: str = 'anonymous') -> str:
        """提交变更"""
        return self.version_manager.commit(self, message, author)
    
    def checkout(self, version_id: str):
        """切换到指定版本"""
        return self.version_manager.checkout(version_id, self)
    
    def rollback(self, steps: int) -> str:
        """回滚到之前的版本"""
        return self.version_manager.rollback(steps, self)
    
    def get_version_history(self) -> List[DictionaryVersion]:
        """获取版本历史"""
        return self.version_manager.get_version_history()
    
    def get_current_version(self) -> Optional[DictionaryVersion]:
        """获取当前版本"""
        if self.version_manager.current_version:
            return self.version_manager.get_version_info(self.version_manager.current_version)
        return None