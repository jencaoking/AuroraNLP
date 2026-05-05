"""
AuroraNLP 架构重构模块 - Pipeline系统

本模块实现了NLP处理流水线的完整架构，包括：
- 步骤51: Pipeline架构设计 - 模块化组件流水线，组件顺序配置，条件分支支持
- 步骤52: 组件注册机制 - 插件化扩展，装饰器注册，自动发现机制
- 步骤53: 组件配置系统 - JSON配置支持，配置验证
- 步骤54: 组件冻结机制 - 部分组件不更新，训练时冻结，参数锁定
- 步骤55: Doc对象设计 - 统一数据结构，文本容器，属性存储
- 步骤56: Span对象实现 - 文本片段抽象，切片操作，属性继承
- 步骤57: Token对象实现 - 词元级别操作，属性访问，关系链接
- 步骤58: 词汇表共享 - StringStore实现，内存优化，ID映射
- 步骤59: 模型版本管理 - 模型生命周期，版本号规范，兼容性检查
- 步骤60: 模型缓存机制 - 减少加载时间，LRU缓存策略，内存管理
- 步骤61: RESTful API设计 - HTTP服务接口，请求验证
- 步骤62: gRPC接口实现 - 高性能RPC调用框架
- 步骤63: 异步处理支持 - asyncio集成，并发处理，协程调度
- 步骤64: 流式处理实现 - 大文件流式分词，内存友好，进度回调
- 步骤65: 插件系统设计 - 第三方扩展机制，插件生命周期，依赖管理

约束：零外部依赖，纯Python标准库实现
"""

import json
import hashlib
import time
import threading
import asyncio
import functools
import os
import sys
import struct
import socket
import traceback
import importlib
import inspect
import weakref
import copy
import re
from typing import (
    List, Dict, Any, Optional, Tuple, Set, Callable, Union,
    Iterator, Iterable, Type, TypeVar, Generic, NamedTuple
)
from collections import OrderedDict
from enum import Enum, IntEnum
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from io import StringIO, BytesIO
from concurrent.futures import ThreadPoolExecutor
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


# ============================================================
# 步骤58: 词汇表共享 - StringStore实现
# ============================================================

class StringStore:
    """
    字符串存储池 - 实现字符串的内存优化和ID映射

    通过字符串驻留机制，确保相同的字符串只存储一份，
    所有引用共享同一个对象，从而减少内存使用。

    特性：
    - 字符串到整型ID的双向映射
    - 内存优化（相同字符串共享存储）
    - 线程安全操作
    - 支持序列化和反序列化
    """

    def __init__(self):
        """初始化字符串存储池"""
        self._str_to_id: Dict[str, int] = {}
        self._id_to_str: Dict[int, str] = {}
        self._next_id: int = 0
        self._lock = threading.RLock()

    def __len__(self) -> int:
        """返回存储池中字符串的数量"""
        return len(self._str_to_id)

    def __contains__(self, key: Union[str, int]) -> bool:
        """检查字符串或ID是否存在于存储池中"""
        if isinstance(key, str):
            return key in self._str_to_id
        return key in self._id_to_str

    def __getitem__(self, key: Union[str, int]) -> Union[str, int]:
        """通过字符串获取ID，或通过ID获取字符串"""
        if isinstance(key, str):
            if key not in self._str_to_id:
                raise KeyError(f"字符串 '{key}' 不在存储池中")
            return self._str_to_id[key]
        else:
            if key not in self._id_to_str:
                raise KeyError(f"ID {key} 不在存储池中")
            return self._id_to_str[key]

    def __iter__(self) -> Iterator[str]:
        """迭代存储池中的所有字符串"""
        return iter(self._str_to_id)

    def __repr__(self) -> str:
        return f"StringStore(size={len(self)})"

    def add(self, s: str) -> int:
        """
        添加字符串到存储池，返回对应的ID

        如果字符串已存在，返回已有的ID。

        Args:
            s: 要添加的字符串

        Returns:
            字符串对应的整型ID
        """
        with self._lock:
            if s in self._str_to_id:
                return self._str_to_id[s]
            str_id = self._next_id
            self._str_to_id[s] = str_id
            self._id_to_str[str_id] = s
            self._next_id += 1
            return str_id

    def get_id(self, s: str) -> Optional[int]:
        """
        获取字符串对应的ID，不存在则返回None

        Args:
            s: 要查询的字符串

        Returns:
            对应的ID，或None
        """
        return self._str_to_id.get(s)

    def get_str(self, str_id: int) -> Optional[str]:
        """
        获取ID对应的字符串，不存在则返回None

        Args:
            str_id: 要查询的ID

        Returns:
            对应的字符串，或None
        """
        return self._id_to_str.get(str_id)

    def lookup(self, s: str) -> Optional[str]:
        """
        查找字符串的驻留版本

        如果字符串存在于存储池中，返回池中的引用；
        否则返回None。

        Args:
            s: 要查找的字符串

        Returns:
            池中的字符串引用，或None
        """
        if s in self._str_to_id:
            return self._id_to_str[self._str_to_id[s]]
        return None

    def to_json(self) -> str:
        """将存储池序列化为JSON字符串"""
        return json.dumps(self._str_to_id, ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> 'StringStore':
        """
        从JSON字符串反序列化存储池

        Args:
            json_str: JSON格式的字符串

        Returns:
            反序列化后的StringStore实例
        """
        store = cls()
        data = json.loads(json_str)
        for s, str_id in data.items():
            store._str_to_id[s] = str_id
            store._id_to_str[str_id] = s
            if str_id >= store._next_id:
                store._next_id = str_id + 1
        return store

    def clear(self) -> None:
        """清空存储池"""
        with self._lock:
            self._str_to_id.clear()
            self._id_to_str.clear()
            self._next_id = 0

    def batch_add(self, strings: Iterable[str]) -> List[int]:
        """
        批量添加字符串

        Args:
            strings: 可迭代的字符串集合

        Returns:
            对应的ID列表
        """
        with self._lock:
            return [self.add(s) for s in strings]


# 全局共享词汇表实例
_global_string_store = StringStore()


def get_string_store() -> StringStore:
    """获取全局共享的字符串存储池"""
    return _global_string_store


# ============================================================
# 步骤55: Doc对象设计 - 统一数据结构
# ============================================================

class Doc:
    """
    文档对象 - NLP处理流水线的统一数据容器

    Doc是流水线中所有组件共享的核心数据结构，
    承载文本及其所有标注信息（分词、词性、命名实体等）。

    特性：
    - 文本容器：存储原始文本和处理结果
    - 属性存储：支持任意扩展属性
    - 集合视图：提供tokens、spans、entities等便捷访问
    - 延迟计算：支持按需计算属性
    """

    def __init__(
        self,
        text: str,
        string_store: Optional[StringStore] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        初始化文档对象

        Args:
            text: 原始文本内容
            string_store: 共享的字符串存储池，默认使用全局实例
            metadata: 文档元数据
        """
        self._text = text
        self._string_store = string_store or get_string_store()
        self._tokens: List['Token'] = []
        self._spans: List['Span'] = []
        self._entities: List['Span'] = []
        self._attrs: Dict[str, Any] = {}
        self._metadata: Dict[str, Any] = metadata or {}
        self._user_data: Dict[str, Any] = {}
        self._callbacks: Dict[str, List[Callable]] = {}

    def __repr__(self) -> str:
        preview = self._text[:50] + "..." if len(self._text) > 50 else self._text
        return f"Doc(text=\"{preview}\", tokens={len(self._tokens)})"

    def __len__(self) -> int:
        """返回文档的字符长度"""
        return len(self._text)

    def __getitem__(self, key: str) -> Any:
        """获取文档属性"""
        if key == 'text':
            return self._text
        if key == 'tokens':
            return self._tokens
        if key == 'spans':
            return self._spans
        if key == 'entities':
            return self._entities
        return self._attrs.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        """设置文档属性"""
        self._attrs[key] = value

    def __contains__(self, key: str) -> bool:
        """检查属性是否存在"""
        if key in ('text', 'tokens', 'spans', 'entities'):
            return True
        return key in self._attrs

    @property
    def text(self) -> str:
        """获取原始文本"""
        return self._text

    @property
    def tokens(self) -> List['Token']:
        """获取词元列表"""
        return self._tokens

    @property
    def spans(self) -> List['Span']:
        """获取文本片段列表"""
        return self._spans

    @property
    def entities(self) -> List['Span']:
        """获取命名实体列表"""
        return self._entities

    @property
    def string_store(self) -> StringStore:
        """获取共享字符串存储池"""
        return self._string_store

    @property
    def metadata(self) -> Dict[str, Any]:
        """获取文档元数据"""
        return self._metadata

    @property
    def user_data(self) -> Dict[str, Any]:
        """获取用户自定义数据"""
        return self._user_data

    def set_token(self, index: int, token: 'Token') -> None:
        """
        设置指定位置的词元

        Args:
            index: 词元位置索引
            token: Token对象
        """
        if index < len(self._tokens):
            self._tokens[index] = token
        else:
            while len(self._tokens) <= index:
                self._tokens.append(None)  # type: ignore
            self._tokens[index] = token

    def add_token(self, token: 'Token') -> int:
        """
        添加词元到文档

        Args:
            token: Token对象

        Returns:
            词元的索引位置
        """
        idx = len(self._tokens)
        self._tokens.append(token)
        return idx

    def add_span(self, span: 'Span') -> None:
        """
        添加文本片段到文档

        Args:
            span: Span对象
        """
        self._spans.append(span)

    def add_entity(self, entity: 'Span') -> None:
        """
        添加命名实体到文档

        Args:
            entity: 实体Span对象
        """
        self._entities.append(entity)

    def get_attr(self, key: str, default: Any = None) -> Any:
        """
        获取文档属性

        Args:
            key: 属性名
            default: 默认值

        Returns:
            属性值
        """
        return self._attrs.get(key, default)

    def set_attr(self, key: str, value: Any) -> None:
        """
        设置文档属性

        Args:
            key: 属性名
            value: 属性值
        """
        self._attrs[key] = value

    def has_attr(self, key: str) -> bool:
        """检查属性是否存在"""
        return key in self._attrs

    def char_span(self, start: int, end: int, label: str = "", **kwargs) -> 'Span':
        """
        创建基于字符偏移的文本片段

        Args:
            start: 起始字符偏移
            end: 结束字符偏移
            label: 标签
            **kwargs: 额外属性

        Returns:
            Span对象
        """
        span = Span(
            doc=self,
            start=start,
            end=end,
            label=label,
            string_store=self._string_store,
            **kwargs
        )
        return span

    def register_callback(self, event: str, callback: Callable) -> None:
        """
        注册事件回调

        Args:
            event: 事件名称
            callback: 回调函数
        """
        if event not in self._callbacks:
            self._callbacks[event] = []
        self._callbacks[event].append(callback)

    def trigger_event(self, event: str, **kwargs) -> None:
        """
        触发事件

        Args:
            event: 事件名称
            **kwargs: 事件参数
        """
        for callback in self._callbacks.get(event, []):
            try:
                callback(self, **kwargs)
            except Exception:
                pass  # 回调异常不影响主流程

    def copy(self) -> 'Doc':
        """
        创建文档的浅拷贝

        Returns:
            新的Doc对象
        """
        new_doc = Doc(
            text=self._text,
            string_store=self._string_store,
            metadata=dict(self._metadata)
        )
        new_doc._attrs = dict(self._attrs)
        new_doc._user_data = dict(self._user_data)
        new_doc._tokens = list(self._tokens)
        new_doc._spans = list(self._spans)
        new_doc._entities = list(self._entities)
        return new_doc


# ============================================================
# 步骤56: Span对象实现 - 文本片段抽象
# ============================================================

class Span:
    """
    文本片段对象 - 表示文档中的一个连续文本区间

    Span是对文档文本的一个切片引用，支持属性继承和
    各种文本操作。

    特性：
    - 字符偏移定位
    - 属性继承（从Token继承属性）
    - 切片操作
    - 关系链接
    """

    def __init__(
        self,
        doc: Doc,
        start: int,
        end: int,
        label: str = "",
        string_store: Optional[StringStore] = None,
        **kwargs
    ):
        """
        初始化文本片段

        Args:
            doc: 所属的Doc对象
            start: 起始字符偏移（包含）
            end: 结束字符偏移（不包含）
            label: 标签（如实体类型）
            string_store: 共享字符串存储池
            **kwargs: 额外属性
        """
        self._doc = weakref.ref(doc)
        self._start = start
        self._end = end
        self._label = label
        self._string_store = string_store or get_string_store()
        self._attrs: Dict[str, Any] = dict(kwargs)
        self._relations: Dict[str, List['Span']] = {}

    def __repr__(self) -> str:
        text = self.text
        preview = text[:20] + "..." if len(text) > 20 else text
        label_str = f", label=\"{self._label}\"" if self._label else ""
        return f"Span(\"{preview}\"{label_str}, {self._start}:{self._end})"

    def __eq__(self, other) -> bool:
        """判断两个Span是否相等（同一文档、相同偏移）"""
        if not isinstance(other, Span):
            return NotImplemented
        return (self._doc() is other._doc() and
                self._start == other._start and
                self._end == other._end)

    def __hash__(self) -> int:
        """Span的哈希值"""
        return hash((id(self._doc()), self._start, self._end))

    def __len__(self) -> int:
        """返回Span的字符长度"""
        return self._end - self._start

    def __getitem__(self, key: str) -> Any:
        """获取Span属性"""
        return self._attrs.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        """设置Span属性"""
        self._attrs[key] = value

    @property
    def doc(self) -> Optional[Doc]:
        """获取所属的Doc对象"""
        return self._doc()

    @property
    def start(self) -> int:
        """获取起始偏移"""
        return self._start

    @property
    def end(self) -> int:
        """获取结束偏移"""
        return self._end

    @property
    def label(self) -> str:
        """获取标签"""
        return self._label

    @label.setter
    def label(self, value: str) -> None:
        """设置标签"""
        self._label = value

    @property
    def text(self) -> str:
        """获取Span对应的文本"""
        doc = self._doc()
        if doc is None:
            return ""
        return doc.text[self._start:self._end]

    @property
    def start_char(self) -> int:
        """起始字符偏移"""
        return self._start

    @property
    def end_char(self) -> int:
        """结束字符偏移"""
        return self._end

    def get_attr(self, key: str, default: Any = None) -> Any:
        """
        获取属性值

        Args:
            key: 属性名
            default: 默认值

        Returns:
            属性值
        """
        return self._attrs.get(key, default)

    def set_attr(self, key: str, value: Any) -> None:
        """
        设置属性

        Args:
            key: 属性名
            value: 属性值
        """
        self._attrs[key] = value

    def has_attr(self, key: str) -> bool:
        """检查属性是否存在"""
        return key in self._attrs

    def as_span(self) -> 'Span':
        """返回自身（兼容接口）"""
        return self

    def add_relation(self, rel_type: str, target: 'Span') -> None:
        """
        添加与其他Span的关系

        Args:
            rel_type: 关系类型
            target: 目标Span
        """
        if rel_type not in self._relations:
            self._relations[rel_type] = []
        self._relations[rel_type].append(target)

    def get_relations(self, rel_type: Optional[str] = None) -> List['Span']:
        """
        获取关系

        Args:
            rel_type: 关系类型，None表示获取所有关系

        Returns:
            关联的Span列表
        """
        if rel_type is None:
            result = []
            for spans in self._relations.values():
                result.extend(spans)
            return result
        return self._relations.get(rel_type, [])

    def overlaps(self, other: 'Span') -> bool:
        """
        检查是否与另一个Span重叠

        Args:
            other: 另一个Span

        Returns:
            是否重叠
        """
        return self._start < other._end and other._start < self._end

    def contains(self, other: 'Span') -> bool:
        """
        检查是否包含另一个Span

        Args:
            other: 另一个Span

        Returns:
            是否完全包含
        """
        return self._start <= other._start and self._end >= other._end

    def similarity(self, other: 'Span') -> float:
        """
        计算与另一个Span的相似度（基于字符重叠）

        Args:
            other: 另一个Span

        Returns:
            相似度（0.0到1.0）
        """
        if self._doc() is not other._doc():
            return 0.0
        overlap_start = max(self._start, other._start)
        overlap_end = min(self._end, other._end)
        overlap = max(0, overlap_end - overlap_start)
        total = (self._end - self._start) + (other._end - other._start) - overlap
        return overlap / total if total > 0 else 0.0

    def slice(self, start: int, end: int) -> 'Span':
        """
        在当前Span内创建子Span

        Args:
            start: 相对于当前Span的起始偏移
            end: 相对于当前Span的结束偏移

        Returns:
            新的Span对象
        """
        doc = self._doc()
        if doc is None:
            raise RuntimeError("所属Doc对象已被回收")
        abs_start = self._start + start
        abs_end = self._start + end
        return Span(
            doc=doc,
            start=abs_start,
            end=abs_end,
            label=self._label,
            string_store=self._string_store
        )


# ============================================================
# 步骤57: Token对象实现 - 词元级别操作
# ============================================================

class Token:
    """
    词元对象 - 表示文档中的一个词元（token）

    Token是Span的特化版本，专门用于表示分词后的词元，
    提供词元级别的属性访问和关系链接。

    特性：
    - 词元属性（词性、词频等）
    - 关系链接（依存关系、共指关系等）
    - 属性继承
    - 便捷的属性访问接口
    """

    def __init__(
        self,
        doc: Doc,
        start: int,
        end: int,
        text: Optional[str] = None,
        string_store: Optional[StringStore] = None
    ):
        """
        初始化词元

        Args:
            doc: 所属的Doc对象
            start: 起始字符偏移
            end: 结束字符偏移
            text: 词元文本（可选，默认从doc中提取）
            string_store: 共享字符串存储池
        """
        self._doc = weakref.ref(doc)
        self._start = start
        self._end = end
        self._text = text
        self._string_store = string_store or get_string_store()
        self._pos: str = ""
        self._lemma: str = ""
        self._tag: str = ""
        self._dep: str = ""
        self._head: Optional['Token'] = None
        self._ner_label: str = ""
        self._attrs: Dict[str, Any] = {}
        self._relations: Dict[str, List[Union['Token', 'Span']]] = {}

    def __repr__(self) -> str:
        return f"Token(\"{self.text}\", pos=\"{self._pos}\")"

    def __eq__(self, other) -> bool:
        """判断两个Token是否相等"""
        if not isinstance(other, Token):
            return NotImplemented
        return (self._doc() is other._doc() and
                self._start == other._start and
                self._end == other._end)

    def __hash__(self) -> int:
        """Token的哈希值"""
        return hash((id(self._doc()), self._start, self._end))

    def __len__(self) -> int:
        """返回Token的字符长度"""
        return self._end - self._start

    def __getitem__(self, key: str) -> Any:
        """获取Token属性"""
        attr_map = {
            'pos': self._pos,
            'lemma': self._lemma,
            'tag': self._tag,
            'dep': self._dep,
            'ner': self._ner_label,
        }
        if key in attr_map:
            return attr_map[key]
        return self._attrs.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        """设置Token属性"""
        if key == 'pos':
            self._pos = value
        elif key == 'lemma':
            self._lemma = value
        elif key == 'tag':
            self._tag = value
        elif key == 'dep':
            self._dep = value
        elif key == 'ner':
            self._ner_label = value
        else:
            self._attrs[key] = value

    @property
    def doc(self) -> Optional[Doc]:
        """获取所属的Doc对象"""
        return self._doc()

    @property
    def text(self) -> str:
        """获取词元文本"""
        if self._text is not None:
            return self._text
        doc = self._doc()
        if doc is None:
            return ""
        return doc.text[self._start:self._end]

    @property
    def start(self) -> int:
        """获取起始偏移"""
        return self._start

    @property
    def end(self) -> int:
        """获取结束偏移"""
        return self._end

    @property
    def start_char(self) -> int:
        """起始字符偏移"""
        return self._start

    @property
    def end_char(self) -> int:
        """结束字符偏移"""
        return self._end

    @property
    def pos(self) -> str:
        """获取词性标注"""
        return self._pos

    @pos.setter
    def pos(self, value: str) -> None:
        """设置词性标注"""
        self._pos = value

    @property
    def lemma(self) -> str:
        """获取词元（原形）"""
        return self._lemma

    @lemma.setter
    def lemma(self, value: str) -> None:
        """设置词元"""
        self._lemma = value

    @property
    def tag(self) -> str:
        """获取细粒度词性标注"""
        return self._tag

    @tag.setter
    def tag(self, value: str) -> None:
        """设置细粒度词性标注"""
        self._tag = value

    @property
    def dep(self) -> str:
        """获取依存关系标签"""
        return self._dep

    @dep.setter
    def dep(self, value: str) -> None:
        """设置依存关系标签"""
        self._dep = value

    @property
    def head(self) -> Optional['Token']:
        """获取依存关系中的中心词"""
        return self._head

    @head.setter
    def head(self, value: Optional['Token']) -> None:
        """设置依存关系中的中心词"""
        self._head = value

    @property
    def ner_label(self) -> str:
        """获取命名实体标签"""
        return self._ner_label

    @ner_label.setter
    def ner_label(self, value: str) -> None:
        """设置命名实体标签"""
        self._ner_label = value

    def get_attr(self, key: str, default: Any = None) -> Any:
        """
        获取自定义属性

        Args:
            key: 属性名
            default: 默认值

        Returns:
            属性值
        """
        return self._attrs.get(key, default)

    def set_attr(self, key: str, value: Any) -> None:
        """
        设置自定义属性

        Args:
            key: 属性名
            value: 属性值
        """
        self._attrs[key] = value

    def has_attr(self, key: str) -> bool:
        """检查自定义属性是否存在"""
        return key in self._attrs

    def add_relation(self, rel_type: str, target: Union['Token', 'Span']) -> None:
        """
        添加与其他Token/Span的关系

        Args:
            rel_type: 关系类型
            target: 目标Token或Span
        """
        if rel_type not in self._relations:
            self._relations[rel_type] = []
        self._relations[rel_type].append(target)

    def get_relations(self, rel_type: Optional[str] = None) -> List[Union['Token', 'Span']]:
        """
        获取关系

        Args:
            rel_type: 关系类型，None表示获取所有关系

        Returns:
            关联的Token/Span列表
        """
        if rel_type is None:
            result = []
            for items in self._relations.values():
                result.extend(items)
            return result
        return self._relations.get(rel_type, [])

    def is_ancestor(self, other: 'Token') -> bool:
        """
        检查当前Token是否是另一个Token的祖先（依存树中）

        Args:
            other: 另一个Token

        Returns:
            是否是祖先
        """
        current = other.head
        visited = set()
        while current is not None:
            if current is self:
                return True
            token_id = id(current)
            if token_id in visited:
                break
            visited.add(token_id)
            current = current.head
        return False

    def get_ancestors(self) -> List['Token']:
        """
        获取依存树中的所有祖先节点

        Returns:
            祖先Token列表（从直接中心词到根节点）
        """
        ancestors = []
        current = self._head
        visited = set()
        while current is not None:
            token_id = id(current)
            if token_id in visited:
                break
            visited.add(token_id)
            ancestors.append(current)
            current = current.head
        return ancestors

    def get_descendants(self) -> List['Token']:
        """
        获取依存树中的所有后代节点

        Returns:
            后代Token列表
        """
        doc = self._doc()
        if doc is None:
            return []
        descendants = []
        for token in doc.tokens:
            if token.is_ancestor(self):
                descendants.append(token)
        return descendants

    def left_edge(self) -> 'Token':
        """获取依存子树的最左边界Token"""
        doc = self._doc()
        if doc is None:
            return self
        descendants = self.get_descendants()
        if not descendants:
            return self
        leftmost = self
        for t in [self] + descendants:
            if t.start < leftmost.start:
                leftmost = t
        return leftmost

    def right_edge(self) -> 'Token':
        """获取依存子树的最右边界Token"""
        doc = self._doc()
        if doc is None:
            return self
        descendants = self.get_descendants()
        if not descendants:
            return self
        rightmost = self
        for t in [self] + descendants:
            if t.end > rightmost.end:
                rightmost = t
        return rightmost

    def as_span(self) -> Span:
        """将Token转换为Span"""
        doc = self._doc()
        if doc is None:
            raise RuntimeError("所属Doc对象已被回收")
        return Span(
            doc=doc,
            start=self._start,
            end=self._end,
            label=self._ner_label,
            string_store=self._string_store
        )

    def subtree_text(self) -> str:
        """获取依存子树的完整文本"""
        doc = self._doc()
        if doc is None:
            return ""
        left = self.left_edge()
        right = self.right_edge()
        return doc.text[left.start:right.end]


# ============================================================
# 步骤51: Pipeline架构设计 - 模块化组件流水线
# ============================================================

class ComponentState(Enum):
    """组件状态枚举"""
    UNINITIALIZED = "uninitialized"   # 未初始化
    INITIALIZED = "initialized"       # 已初始化
    FROZEN = "frozen"                 # 已冻结
    TRAINING = "training"             # 训练中
    ERROR = "error"                   # 错误状态


class PipelineComponent(ABC):
    """
    流水线组件抽象基类

    所有Pipeline中的处理组件都需要继承此类，实现process方法。
    组件支持初始化、处理、冻结等生命周期管理。
    """

    def __init__(self, name: str = "", disabled: bool = False):
        """
        初始化流水线组件

        Args:
            name: 组件名称
            disabled: 是否禁用
        """
        self._name = name or self.__class__.__name__
        self._disabled = disabled
        self._state = ComponentState.UNINITIALIZED
        self._config: Dict[str, Any] = {}
        self._pipeline: Optional['Pipeline'] = None
        self._frozen = False
        self._requirements: List[str] = []
        self._provides: List[str] = []
        self._lock = threading.RLock()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name=\"{self._name}\", state={self._state.value})"

    @property
    def name(self) -> str:
        """获取组件名称"""
        return self._name

    @property
    def state(self) -> str:
        """获取组件状态"""
        return self._state.value

    @property
    def disabled(self) -> bool:
        """是否已禁用"""
        return self._disabled

    @property
    def frozen(self) -> bool:
        """是否已冻结"""
        return self._frozen

    @property
    def pipeline(self) -> Optional['Pipeline']:
        """获取所属Pipeline"""
        return self._pipeline

    def initialize(self, pipeline: 'Pipeline', config: Optional[Dict[str, Any]] = None) -> None:
        """
        初始化组件

        Args:
            pipeline: 所属Pipeline
            config: 组件配置
        """
        with self._lock:
            self._pipeline = pipeline
            self._config = config or {}
            self._state = ComponentState.INITIALIZED
            self.on_initialize()

    def on_initialize(self) -> None:
        """子类重写：初始化时的钩子方法"""
        pass

    @abstractmethod
    def process(self, doc: Doc) -> Doc:
        """
        处理文档（子类必须实现）

        Args:
            doc: 输入文档

        Returns:
            处理后的文档
        """
        pass

    def require(self, *attrs: str) -> None:
        """
        声明组件所需的前置属性

        Args:
            *attrs: 所需属性名列表
        """
        self._requirements.extend(attrs)

    def provide(self, *attrs: str) -> None:
        """
        声明组件提供的输出属性

        Args:
            *attrs: 提供的属性名列表
        """
        self._provides.extend(attrs)

    def get_requirements(self) -> List[str]:
        """获取前置依赖列表"""
        return list(self._requirements)

    def get_provides(self) -> List[str]:
        """获取输出属性列表"""
        return list(self._provides)

    def freeze(self) -> None:
        """冻结组件（参数锁定，不再更新）"""
        with self._lock:
            self._frozen = True
            self._state = ComponentState.FROZEN

    def unfreeze(self) -> None:
        """解冻组件"""
        with self._lock:
            self._frozen = False
            self._state = ComponentState.INITIALIZED

    def validate_requirements(self, doc: Doc) -> bool:
        """
        验证文档是否满足组件的前置要求

        Args:
            doc: 输入文档

        Returns:
            是否满足要求
        """
        for req in self._requirements:
            if not doc.has_attr(req) and req not in doc:
                return False
        return True

    def to_config(self) -> Dict[str, Any]:
        """将组件序列化为配置字典"""
        return {
            'name': self._name,
            'class': self.__class__.__module__ + '.' + self.__class__.__name__,
            'disabled': self._disabled,
            'frozen': self._frozen,
            'config': self._config,
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'PipelineComponent':
        """
        从配置字典创建组件

        Args:
            config: 配置字典

        Returns:
            组件实例
        """
        class_path = config.get('class', '')
        if class_path:
            parts = class_path.rsplit('.', 1)
            if len(parts) == 2:
                module_path, class_name = parts
                try:
                    module = importlib.import_module(module_path)
                    component_cls = getattr(module, class_name)
                    return component_cls(name=config.get('name', ''))
                except (ImportError, AttributeError):
                    pass
        return cls(name=config.get('name', ''))


class ConditionalBranch:
    """
    条件分支 - 根据条件决定是否执行组件

    支持基于文档属性、组件输出等条件的分支判断。
    """

    def __init__(
        self,
        condition: Callable[[Doc], bool],
        true_components: Optional[List[PipelineComponent]] = None,
        false_components: Optional[List[PipelineComponent]] = None
    ):
        """
        初始化条件分支

        Args:
            condition: 条件判断函数，接收Doc返回bool
            true_components: 条件为真时执行的组件列表
            false_components: 条件为假时执行的组件列表
        """
        self._condition = condition
        self._true_components = true_components or []
        self._false_components = false_components or []

    @property
    def condition(self) -> Callable[[Doc], bool]:
        """获取条件函数"""
        return self._condition

    @property
    def true_components(self) -> List[PipelineComponent]:
        """获取条件为真时的组件列表"""
        return self._true_components

    @property
    def false_components(self) -> List[PipelineComponent]:
        """获取条件为假时的组件列表"""
        return self._false_components

    def evaluate(self, doc: Doc) -> bool:
        """
        评估条件

        Args:
            doc: 输入文档

        Returns:
            条件结果
        """
        return self._condition(doc)

    def get_active_components(self, doc: Doc) -> List[PipelineComponent]:
        """
        获取当前条件下应激活的组件列表

        Args:
            doc: 输入文档

        Returns:
            应执行的组件列表
        """
        if self._condition(doc):
            return self._true_components
        return self._false_components


class Pipeline:
    """
    NLP处理流水线

    Pipeline是组件的有序集合，负责管理组件的生命周期、
    执行顺序和条件分支。

    特性：
    - 模块化组件管理
    - 组件顺序配置
    - 条件分支支持
    - 组件冻结机制
    - 批量处理
    """

    def __init__(
        self,
        name: str = "default",
        string_store: Optional[StringStore] = None,
        disable: Optional[List[str]] = None,
        enable: Optional[List[str]] = None
    ):
        """
        初始化流水线

        Args:
            name: 流水线名称
            string_store: 共享字符串存储池
            disable: 要禁用的组件名称列表
            enable: 要启用的组件名称列表
        """
        self._name = name
        self._string_store = string_store or get_string_store()
        self._components: List[Union[PipelineComponent, ConditionalBranch]] = []
        self._component_map: Dict[str, PipelineComponent] = {}
        self._disable_set: Set[str] = set(disable or [])
        self._enable_set: Set[str] = set(enable or [])
        self._initialized = False
        self._lock = threading.RLock()
        self._pre_processors: List[Callable[[Doc], Doc]] = []
        self._post_processors: List[Callable[[Doc], Doc]] = []
        self._error_handlers: List[Callable[[Doc, Exception], Doc]] = []

    def __repr__(self) -> str:
        return f"Pipeline(name=\"{self._name}\", components={len(self._components)})"

    @property
    def name(self) -> str:
        """获取流水线名称"""
        return self._name

    @property
    def string_store(self) -> StringStore:
        """获取共享字符串存储池"""
        return self._string_store

    @property
    def component_names(self) -> List[str]:
        """获取所有组件名称"""
        return [c.name for c in self._components if isinstance(c, PipelineComponent)]

    def add_component(self, component: PipelineComponent, after: Optional[str] = None, before: Optional[str] = None) -> None:
        """
        添加组件到流水线

        Args:
            component: 要添加的组件
            after: 插入到指定组件之后
            before: 插入到指定组件之前
        """
        with self._lock:
            if self._disable_set and component.name in self._disable_set:
                if component.name not in self._enable_set:
                    component._disabled = True

            if after is not None:
                idx = self._find_component_index(after)
                if idx is not None:
                    self._components.insert(idx + 1, component)
                else:
                    self._components.append(component)
            elif before is not None:
                idx = self._find_component_index(before)
                if idx is not None:
                    self._components.insert(idx, component)
                else:
                    self._components.append(component)
            else:
                self._components.append(component)

            self._component_map[component.name] = component

    def add_branch(self, branch: ConditionalBranch, after: Optional[str] = None) -> None:
        """
        添加条件分支到流水线

        Args:
            branch: 条件分支对象
            after: 插入到指定组件之后
        """
        with self._lock:
            if after is not None:
                idx = self._find_component_index(after)
                if idx is not None:
                    self._components.insert(idx + 1, branch)
                else:
                    self._components.append(branch)
            else:
                self._components.append(branch)

    def remove_component(self, name: str) -> bool:
        """
        移除组件

        Args:
            name: 组件名称

        Returns:
            是否成功移除
        """
        with self._lock:
            for i, c in enumerate(self._components):
                if isinstance(c, PipelineComponent) and c.name == name:
                    self._components.pop(i)
                    self._component_map.pop(name, None)
                    return True
            return False

    def get_component(self, name: str) -> Optional[PipelineComponent]:
        """
        获取组件

        Args:
            name: 组件名称

        Returns:
            组件实例，不存在则返回None
        """
        return self._component_map.get(name)

    def disable_component(self, name: str) -> bool:
        """
        禁用组件

        Args:
            name: 组件名称

        Returns:
            是否成功禁用
        """
        component = self._component_map.get(name)
        if component:
            component._disabled = True
            return True
        return False

    def enable_component(self, name: str) -> bool:
        """
        启用组件

        Args:
            name: 组件名称

        Returns:
            是否成功启用
        """
        component = self._component_map.get(name)
        if component:
            component._disabled = False
            return True
        return False

    def freeze_component(self, name: str) -> bool:
        """
        冻结组件（参数锁定）

        Args:
            name: 组件名称

        Returns:
            是否成功冻结
        """
        component = self._component_map.get(name)
        if component:
            component.freeze()
            return True
        return False

    def unfreeze_component(self, name: str) -> bool:
        """
        解冻组件

        Args:
            name: 组件名称

        Returns:
            是否成功解冻
        """
        component = self._component_map.get(name)
        if component:
            component.unfreeze()
            return True
        return False

    def _find_component_index(self, name: str) -> Optional[int]:
        """查找组件在列表中的索引"""
        for i, c in enumerate(self._components):
            if isinstance(c, PipelineComponent) and c.name == name:
                return i
        return None

    def initialize(self) -> None:
        """初始化所有组件"""
        with self._lock:
            for item in self._components:
                if isinstance(item, PipelineComponent) and not item._disabled:
                    item.initialize(self)
            self._initialized = True

    def process(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Doc:
        """
        处理文本

        Args:
            text: 输入文本
            metadata: 文档元数据

        Returns:
            处理后的Doc对象
        """
        if not self._initialized:
            self.initialize()

        doc = Doc(text=text, string_store=self._string_store, metadata=metadata)

        # 前处理
        for pre_proc in self._pre_processors:
            doc = pre_proc(doc)

        # 执行组件
        for item in self._components:
            if isinstance(item, PipelineComponent):
                if item._disabled:
                    continue
                try:
                    doc = item.process(doc)
                except Exception as e:
                    handled = False
                    for handler in self._error_handlers:
                        try:
                            doc = handler(doc, e)
                            handled = True
                            break
                        except Exception:
                            continue
                    if not handled:
                        raise
            elif isinstance(item, ConditionalBranch):
                active = item.get_active_components(doc)
                for comp in active:
                    if comp._disabled:
                        continue
                    try:
                        doc = comp.process(doc)
                    except Exception as e:
                        handled = False
                        for handler in self._error_handlers:
                            try:
                                doc = handler(doc, e)
                                handled = True
                                break
                            except Exception:
                                continue
                        if not handled:
                            raise

        # 后处理
        for post_proc in self._post_processors:
            doc = post_proc(doc)

        return doc

    def process_batch(
        self,
        texts: List[str],
        metadata_list: Optional[List[Optional[Dict[str, Any]]]] = None,
        batch_size: int = 32
    ) -> List[Doc]:
        """
        批量处理文本

        Args:
            texts: 文本列表
            metadata_list: 元数据列表
            batch_size: 批次大小

        Returns:
            Doc对象列表
        """
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_meta = None
            if metadata_list:
                batch_meta = metadata_list[i:i + batch_size]
            for j, text in enumerate(batch):
                meta = batch_meta[j] if batch_meta else None
                results.append(self.process(text, meta))
        return results

    def add_pre_processor(self, func: Callable[[Doc], Doc]) -> None:
        """添加前处理器"""
        self._pre_processors.append(func)

    def add_post_processor(self, func: Callable[[Doc], Doc]) -> None:
        """添加后处理器"""
        self._post_processors.append(func)

    def add_error_handler(self, func: Callable[[Doc, Exception], Doc]) -> None:
        """添加错误处理器"""
        self._error_handlers.append(func)

    def to_config(self) -> Dict[str, Any]:
        """将Pipeline序列化为配置"""
        components_config = []
        for item in self._components:
            if isinstance(item, PipelineComponent):
                components_config.append({
                    'type': 'component',
                    'config': item.to_config()
                })
            elif isinstance(item, ConditionalBranch):
                components_config.append({
                    'type': 'branch',
                    'true_components': [c.to_config() for c in item._true_components],
                    'false_components': [c.to_config() for c in item._false_components],
                })
        return {
            'name': self._name,
            'components': components_config,
        }

    def get_execution_order(self) -> List[str]:
        """获取组件执行顺序"""
        return [c.name for c in self._components if isinstance(c, PipelineComponent)]


# ============================================================
# 步骤52: 组件注册机制 - 插件化扩展
# ============================================================

class ComponentRegistry:
    """
    组件注册表 - 管理所有可用组件的注册和查找

    特性：
    - 装饰器注册
    - 按名称/类别查找
    - 自动发现机制
    - 工厂方法创建
    """

    def __init__(self):
        """初始化组件注册表"""
        self._registry: Dict[str, Type[PipelineComponent]] = {}
        self._categories: Dict[str, List[str]] = {}
        self._aliases: Dict[str, str] = {}
        self._lock = threading.RLock()

    def __repr__(self) -> str:
        return f"ComponentRegistry(registered={len(self._registry)})"

    def register(
        self,
        name: Optional[str] = None,
        category: str = "default",
        aliases: Optional[List[str]] = None
    ) -> Callable:
        """
        装饰器：注册组件类

        用法：
            @registry.register("my_component", category="tokenizer")
            class MyComponent(PipelineComponent):
                ...

        Args:
            name: 组件名称（默认使用类名）
            category: 组件类别
            aliases: 别名列表

        Returns:
            装饰器函数
        """
        def decorator(cls: Type[PipelineComponent]) -> Type[PipelineComponent]:
            comp_name = name or cls.__name__
            with self._lock:
                self._registry[comp_name] = cls
                if category not in self._categories:
                    self._categories[category] = []
                self._categories[category].append(comp_name)
                if aliases:
                    for alias in aliases:
                        self._aliases[alias] = comp_name
            return cls
        return decorator

    def register_class(
        self,
        cls: Type[PipelineComponent],
        name: Optional[str] = None,
        category: str = "default",
        aliases: Optional[List[str]] = None
    ) -> None:
        """
        直接注册组件类

        Args:
            cls: 组件类
            name: 组件名称
            category: 组件类别
            aliases: 别名列表
        """
        comp_name = name or cls.__name__
        with self._lock:
            self._registry[comp_name] = cls
            if category not in self._categories:
                self._categories[category] = []
            self._categories[category].append(comp_name)
            if aliases:
                for alias in aliases:
                    self._aliases[alias] = comp_name

    def get(self, name: str) -> Optional[Type[PipelineComponent]]:
        """
        获取注册的组件类

        Args:
            name: 组件名称或别名

        Returns:
            组件类，不存在则返回None
        """
        resolved = self._aliases.get(name, name)
        return self._registry.get(resolved)

    def create(self, name: str, **kwargs) -> Optional[PipelineComponent]:
        """
        创建组件实例

        Args:
            name: 组件名称或别名
            **kwargs: 传递给组件构造函数的参数

        Returns:
            组件实例，不存在则返回None
        """
        cls = self.get(name)
        if cls is None:
            return None
        return cls(**kwargs)

    def list_components(self, category: Optional[str] = None) -> List[str]:
        """
        列出所有已注册的组件名称

        Args:
            category: 按类别筛选

        Returns:
            组件名称列表
        """
        if category is not None:
            return list(self._categories.get(category, []))
        return list(self._registry.keys())

    def list_categories(self) -> List[str]:
        """列出所有组件类别"""
        return list(self._categories.keys())

    def unregister(self, name: str) -> bool:
        """
        取消注册组件

        Args:
            name: 组件名称

        Returns:
            是否成功取消
        """
        with self._lock:
            if name in self._registry:
                del self._registry[name]
                for cat, comps in self._categories.items():
                    if name in comps:
                        comps.remove(name)
                aliases_to_remove = [a for a, n in self._aliases.items() if n == name]
                for alias in aliases_to_remove:
                    del self._aliases[alias]
                return True
            return False

    def auto_discover(self, package_paths: Optional[List[str]] = None) -> int:
        """
        自动发现并注册组件

        扫描指定包路径下的模块，查找带有注册标记的组件类。

        Args:
            package_paths: 包路径列表

        Returns:
            新发现的组件数量
        """
        discovered = 0
        paths = package_paths or []
        for pkg_path in paths:
            try:
                package = importlib.import_module(pkg_path)
                pkg_dir = os.path.dirname(getattr(package, '__file__', ''))
                if not pkg_dir or not os.path.isdir(pkg_dir):
                    continue
                for filename in os.listdir(pkg_dir):
                    if filename.endswith('.py') and not filename.startswith('_'):
                        module_name = f"{pkg_path}.{filename[:-3]}"
                        try:
                            importlib.import_module(module_name)
                        except ImportError:
                            continue
            except ImportError:
                continue
        return discovered


# 全局组件注册表
_global_registry = ComponentRegistry()


def get_registry() -> ComponentRegistry:
    """获取全局组件注册表"""
    return _global_registry


def register_component(
    name: Optional[str] = None,
    category: str = "default",
    aliases: Optional[List[str]] = None
) -> Callable:
    """
    便捷装饰器：注册组件到全局注册表

    用法：
        @register_component("my_seg", category="tokenizer")
        class MySegmentor(PipelineComponent):
            ...
    """
    return _global_registry.register(name=name, category=category, aliases=aliases)


# ============================================================
# 步骤53: 组件配置系统 - JSON配置支持
# ============================================================

class ConfigValidationError(Exception):
    """配置验证错误"""
    pass


class ConfigSchema:
    """
    配置模式定义 - 描述配置项的类型和约束

    支持类型检查、默认值、必填项、范围约束等。
    """

    def __init__(self):
        """初始化配置模式"""
        self._fields: Dict[str, Dict[str, Any]] = {}

    def add_field(
        self,
        name: str,
        field_type: type = str,
        required: bool = False,
        default: Any = None,
        choices: Optional[List[Any]] = None,
        min_value: Optional[Union[int, float]] = None,
        max_value: Optional[Union[int, float]] = None,
        description: str = ""
    ) -> None:
        """
        添加配置字段定义

        Args:
            name: 字段名
            field_type: 字段类型
            required: 是否必填
            default: 默认值
            choices: 可选值列表
            min_value: 最小值
            max_value: 最大值
            description: 字段描述
        """
        self._fields[name] = {
            'type': field_type,
            'required': required,
            'default': default,
            'choices': choices,
            'min_value': min_value,
            'max_value': max_value,
            'description': description,
        }

    def validate(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        验证配置是否符合模式

        Args:
            config: 配置字典

        Returns:
            (是否有效, 错误消息列表)
        """
        errors = []
        for name, field_def in self._fields.items():
            if name not in config:
                if field_def['required']:
                    errors.append(f"缺少必填字段: {name}")
                continue

            value = config[name]
            expected_type = field_def['type']

            # 类型检查
            if not isinstance(value, expected_type):
                if expected_type == float and isinstance(value, int):
                    pass
                else:
                    errors.append(
                        f"字段 '{name}' 类型错误: 期望 {expected_type.__name__}, "
                        f"实际 {type(value).__name__}"
                    )
                    continue

            # 枚举检查
            if field_def['choices'] and value not in field_def['choices']:
                errors.append(
                    f"字段 '{name}' 值无效: {value}, "
                    f"可选值: {field_def['choices']}"
                )

            # 范围检查
            if field_def['min_value'] is not None and value < field_def['min_value']:
                errors.append(
                    f"字段 '{name}' 值过小: {value} < {field_def['min_value']}"
                )
            if field_def['max_value'] is not None and value > field_def['max_value']:
                errors.append(
                    f"字段 '{name}' 值过大: {value} > {field_def['max_value']}"
                )

        return len(errors) == 0, errors

    def apply_defaults(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        应用默认值到配置

        Args:
            config: 原始配置

        Returns:
            填充默认值后的配置
        """
        result = dict(config)
        for name, field_def in self._fields.items():
            if name not in result and field_def['default'] is not None:
                result[name] = field_def['default']
        return result

    def to_dict(self) -> Dict[str, Any]:
        """将模式导出为字典"""
        return {
            name: {
                'type': fdef['type'].__name__,
                'required': fdef['required'],
                'default': fdef['default'],
                'choices': fdef['choices'],
                'min_value': fdef['min_value'],
                'max_value': fdef['max_value'],
                'description': fdef['description'],
            }
            for name, fdef in self._fields.items()
        }


class PipelineConfig:
    """
    流水线配置管理器

    特性：
    - JSON配置加载/保存
    - 配置验证
    - 配置合并
    - 环境变量替换
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化配置管理器

        Args:
            config: 初始配置字典
        """
        self._config: Dict[str, Any] = config or {}
        self._schemas: Dict[str, ConfigSchema] = {}
        self._env_prefix = "AURORA_"

    def __repr__(self) -> str:
        return f"PipelineConfig(keys={list(self._config.keys())})"

    def __getitem__(self, key: str) -> Any:
        """获取配置值（支持点号分隔的嵌套键）"""
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    raise KeyError(f"配置项 '{key}' 不存在")
            else:
                raise KeyError(f"配置项 '{key}' 不存在")
        return value

    def __setitem__(self, key: str, value: Any) -> None:
        """设置配置值（支持点号分隔的嵌套键）"""
        keys = key.split('.')
        config = self._config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value

    def __contains__(self, key: str) -> bool:
        """检查配置项是否存在"""
        try:
            self[key]
            return True
        except KeyError:
            return False

    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值，带默认值

        Args:
            key: 配置键
            default: 默认值

        Returns:
            配置值
        """
        try:
            return self[key]
        except KeyError:
            return default

    @property
    def raw(self) -> Dict[str, Any]:
        """获取原始配置字典"""
        return dict(self._config)

    def add_schema(self, name: str, schema: ConfigSchema) -> None:
        """
        添加配置模式

        Args:
            name: 模式名称
            schema: ConfigSchema实例
        """
        self._schemas[name] = schema

    def validate(self, schema_name: Optional[str] = None) -> Tuple[bool, List[str]]:
        """
        验证配置

        Args:
            schema_name: 使用的模式名称

        Returns:
            (是否有效, 错误消息列表)
        """
        if schema_name and schema_name in self._schemas:
            return self._schemas[schema_name].validate(self._config)
        if not isinstance(self._config, dict):
            return False, ["配置必须是字典类型"]
        return True, []

    def apply_defaults(self, schema_name: Optional[str] = None) -> None:
        """
        应用默认值

        Args:
            schema_name: 使用的模式名称
        """
        if schema_name and schema_name in self._schemas:
            self._config = self._schemas[schema_name].apply_defaults(self._config)

    def merge(self, other: Union[Dict[str, Any], 'PipelineConfig'], overwrite: bool = True) -> None:
        """
        合并配置

        Args:
            other: 另一个配置
            overwrite: 是否覆盖已有值
        """
        if isinstance(other, PipelineConfig):
            other = other._config
        self._config = self._deep_merge(self._config, other, overwrite)

    @staticmethod
    def _deep_merge(base: Dict, override: Dict, overwrite: bool) -> Dict:
        """深度合并字典"""
        result = dict(base)
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = PipelineConfig._deep_merge(result[key], value, overwrite)
            elif overwrite or key not in result:
                result[key] = value
        return result

    def replace_env_vars(self) -> None:
        """替换配置中的环境变量引用（格式: ${ENV_VAR}）"""
        self._config = self._replace_env_recursive(self._config)

    def _replace_env_recursive(self, obj: Any) -> Any:
        """递归替换环境变量"""
        if isinstance(obj, str):
            if obj.startswith('${') and obj.endswith('}'):
                env_name = obj[2:-1]
                full_name = self._env_prefix + env_name
                return os.environ.get(full_name, os.environ.get(env_name, obj))
            return obj
        elif isinstance(obj, dict):
            return {k: self._replace_env_recursive(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._replace_env_recursive(item) for item in obj]
        return obj

    @classmethod
    def from_json(cls, json_str: str) -> 'PipelineConfig':
        """
        从JSON字符串加载配置

        Args:
            json_str: JSON格式的配置字符串

        Returns:
            PipelineConfig实例
        """
        config = json.loads(json_str)
        return cls(config=config)

    @classmethod
    def from_json_file(cls, file_path: str) -> 'PipelineConfig':
        """
        从JSON文件加载配置

        Args:
            file_path: JSON文件路径

        Returns:
            PipelineConfig实例
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return cls(config=config)

    def to_json(self, indent: int = 2) -> str:
        """
        将配置导出为JSON字符串

        Args:
            indent: 缩进空格数

        Returns:
            JSON字符串
        """
        return json.dumps(self._config, ensure_ascii=False, indent=indent)

    def save_json(self, file_path: str, indent: int = 2) -> None:
        """
        保存配置到JSON文件

        Args:
            file_path: 文件路径
            indent: 缩进空格数
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(self.to_json(indent))


# ============================================================
# 步骤54: 组件冻结机制 - 参数锁定
# ============================================================

class FreezableParams:
    """
    可冻结参数容器

    管理组件的参数，支持冻结/解冻操作。
    冻结后参数不可修改，适用于训练时锁定部分组件。

    特性：
    - 参数锁定
    - 冻结状态查询
    - 批量冻结/解冻
    - 参数快照
    """

    def __init__(self, initial_params: Optional[Dict[str, Any]] = None):
        """
        初始化可冻结参数

        Args:
            initial_params: 初始参数
        """
        self._params: Dict[str, Any] = dict(initial_params or {})
        self._frozen_params: Set[str] = set()
        self._lock = threading.RLock()
        self._snapshot: Optional[Dict[str, Any]] = None

    def __repr__(self) -> str:
        frozen_count = len(self._frozen_params)
        return f"FreezableParams(total={len(self._params)}, frozen={frozen_count})"

    def __getitem__(self, key: str) -> Any:
        """获取参数值"""
        return self._params.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        """设置参数值（冻结参数不可修改）"""
        with self._lock:
            if key in self._frozen_params:
                raise RuntimeError(f"参数 '{key}' 已冻结，不可修改")
            self._params[key] = value

    def __contains__(self, key: str) -> bool:
        """检查参数是否存在"""
        return key in self._params

    def __len__(self) -> int:
        """返回参数数量"""
        return len(self._params)

    def __iter__(self) -> Iterator[str]:
        """迭代参数名"""
        return iter(self._params)

    @property
    def frozen_keys(self) -> Set[str]:
        """获取所有冻结参数名"""
        return set(self._frozen_params)

    @property
    def unfrozen_keys(self) -> Set[str]:
        """获取所有未冻结参数名"""
        return set(self._params.keys()) - self._frozen_params

    def get(self, key: str, default: Any = None) -> Any:
        """
        获取参数值

        Args:
            key: 参数名
            default: 默认值

        Returns:
            参数值
        """
        return self._params.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """
        设置参数值

        Args:
            key: 参数名
            value: 参数值

        Raises:
            RuntimeError: 参数已冻结
        """
        self[key] = value

    def freeze(self, key: str) -> None:
        """
        冻结单个参数

        Args:
            key: 参数名
        """
        with self._lock:
            if key in self._params:
                self._frozen_params.add(key)

    def unfreeze(self, key: str) -> None:
        """
        解冻单个参数

        Args:
            key: 参数名
        """
        with self._lock:
            self._frozen_params.discard(key)

    def freeze_all(self) -> None:
        """冻结所有参数"""
        with self._lock:
            self._frozen_params = set(self._params.keys())

    def unfreeze_all(self) -> None:
        """解冻所有参数"""
        with self._lock:
            self._frozen_params.clear()

    def is_frozen(self, key: str) -> bool:
        """
        检查参数是否已冻结

        Args:
            key: 参数名

        Returns:
            是否已冻结
        """
        return key in self._frozen_params

    def batch_freeze(self, keys: Iterable[str]) -> None:
        """
        批量冻结参数

        Args:
            keys: 参数名列表
        """
        with self._lock:
            for key in keys:
                if key in self._params:
                    self._frozen_params.add(key)

    def batch_unfreeze(self, keys: Iterable[str]) -> None:
        """
        批量解冻参数

        Args:
            keys: 参数名列表
        """
        with self._lock:
            for key in keys:
                self._frozen_params.discard(key)

    def take_snapshot(self) -> None:
        """创建参数快照"""
        with self._lock:
            self._snapshot = copy.deepcopy(self._params)

    def restore_snapshot(self) -> bool:
        """
        从快照恢复参数

        Returns:
            是否成功恢复
        """
        with self._lock:
            if self._snapshot is None:
                return False
            self._params = copy.deepcopy(self._snapshot)
            return True

    def has_snapshot(self) -> bool:
        """检查是否有快照"""
        return self._snapshot is not None

    def to_dict(self) -> Dict[str, Any]:
        """导出参数为字典"""
        return dict(self._params)

    def update(self, params: Dict[str, Any]) -> None:
        """
        批量更新参数（跳过冻结参数）

        Args:
            params: 参数字典
        """
        with self._lock:
            for key, value in params.items():
                if key not in self._frozen_params:
                    self._params[key] = value


# ============================================================
# 步骤59: 模型版本管理
# ============================================================

class VersionInfo(NamedTuple):
    """版本信息命名元组"""
    major: int
    minor: int
    patch: int
    prerelease: str = ""
    build: str = ""


class ModelVersion:
    """
    模型版本管理器

    特性：
    - 语义化版本号
    - 版本比较
    - 兼容性检查
    - 版本元数据
    """

    def __init__(
        self,
        major: int = 0,
        minor: int = 1,
        patch: int = 0,
        prerelease: str = "",
        build: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        初始化模型版本

        Args:
            major: 主版本号
            minor: 次版本号
            patch: 修订号
            prerelease: 预发布标识
            build: 构建元数据
            metadata: 额外元数据
        """
        self._major = major
        self._minor = minor
        self._patch = patch
        self._prerelease = prerelease
        self._build = build
        self._metadata = metadata or {}
        self._created_at = time.time()
        self._description = ""

    def __repr__(self) -> str:
        return f"ModelVersion(\"{self.version_string}\")"

    def __str__(self) -> str:
        """版本字符串"""
        return self.version_string

    def __eq__(self, other) -> bool:
        """版本相等比较"""
        if not isinstance(other, ModelVersion):
            return NotImplemented
        return self._compare_tuple() == other._compare_tuple()

    def __lt__(self, other) -> bool:
        """版本小于比较"""
        if not isinstance(other, ModelVersion):
            return NotImplemented
        return self._compare_tuple() < other._compare_tuple()

    def __le__(self, other) -> bool:
        """版本小于等于比较"""
        if not isinstance(other, ModelVersion):
            return NotImplemented
        return self._compare_tuple() <= other._compare_tuple()

    def __gt__(self, other) -> bool:
        """版本大于比较"""
        if not isinstance(other, ModelVersion):
            return NotImplemented
        return self._compare_tuple() > other._compare_tuple()

    def __ge__(self, other) -> bool:
        """版本大于等于比较"""
        if not isinstance(other, ModelVersion):
            return NotImplemented
        return self._compare_tuple() >= other._compare_tuple()

    def _compare_tuple(self) -> Tuple:
        """生成用于比较的元组"""
        pre = (0, self._prerelease) if self._prerelease else (1, "")
        return (self._major, self._minor, self._patch, pre)

    @property
    def major(self) -> int:
        """主版本号"""
        return self._major

    @property
    def minor(self) -> int:
        """次版本号"""
        return self._minor

    @property
    def patch(self) -> int:
        """修订号"""
        return self._patch

    @property
    def prerelease(self) -> str:
        """预发布标识"""
        return self._prerelease

    @property
    def build(self) -> str:
        """构建元数据"""
        return self._build

    @property
    def version_string(self) -> str:
        """完整版本字符串"""
        version = f"{self._major}.{self._minor}.{self._patch}"
        if self._prerelease:
            version += f"-{self._prerelease}"
        if self._build:
            version += f"+{self._build}"
        return version

    @property
    def version_info(self) -> VersionInfo:
        """版本信息命名元组"""
        return VersionInfo(
            major=self._major,
            minor=self._minor,
            patch=self._patch,
            prerelease=self._prerelease,
            build=self._build
        )

    @property
    def metadata(self) -> Dict[str, Any]:
        """获取元数据"""
        return dict(self._metadata)

    @property
    def created_at(self) -> float:
        """创建时间戳"""
        return self._created_at

    @property
    def description(self) -> str:
        """获取描述"""
        return self._description

    @description.setter
    def description(self, value: str) -> None:
        """设置描述"""
        self._description = value

    def is_compatible(self, other: 'ModelVersion') -> bool:
        """
        检查与另一个版本的兼容性

        主版本号相同则兼容。

        Args:
            other: 另一个版本

        Returns:
            是否兼容
        """
        return self._major == other._major

    def is_breaking_change(self, other: 'ModelVersion') -> bool:
        """
        检查是否是破坏性变更

        Args:
            other: 另一个版本

        Returns:
            是否是破坏性变更
        """
        return self._major != other._major

    def bump_major(self) -> 'ModelVersion':
        """升级主版本号"""
        return ModelVersion(
            major=self._major + 1, minor=0, patch=0,
            metadata=dict(self._metadata)
        )

    def bump_minor(self) -> 'ModelVersion':
        """升级次版本号"""
        return ModelVersion(
            major=self._major, minor=self._minor + 1, patch=0,
            metadata=dict(self._metadata)
        )

    def bump_patch(self) -> 'ModelVersion':
        """升级修订号"""
        return ModelVersion(
            major=self._major, minor=self._minor, patch=self._patch + 1,
            metadata=dict(self._metadata)
        )

    def to_dict(self) -> Dict[str, Any]:
        """导出为字典"""
        return {
            'version': self.version_string,
            'major': self._major,
            'minor': self._minor,
            'patch': self._patch,
            'prerelease': self._prerelease,
            'build': self._build,
            'description': self._description,
            'metadata': self._metadata,
            'created_at': self._created_at,
        }

    @classmethod
    def parse(cls, version_str: str) -> 'ModelVersion':
        """
        从版本字符串解析

        支持格式: "1.2.3", "1.2.3-alpha", "1.2.3+build"

        Args:
            version_str: 版本字符串

        Returns:
            ModelVersion实例

        Raises:
            ValueError: 版本字符串格式无效
        """
        build = ""
        if '+' in version_str:
            version_str, build = version_str.split('+', 1)

        prerelease = ""
        if '-' in version_str:
            version_str, prerelease = version_str.split('-', 1)

        parts = version_str.split('.')
        if len(parts) != 3:
            raise ValueError(f"无效的版本字符串: {version_str}")

        try:
            major = int(parts[0])
            minor = int(parts[1])
            patch = int(parts[2])
        except ValueError:
            raise ValueError(f"无效的版本号: {version_str}")

        if major < 0 or minor < 0 or patch < 0:
            raise ValueError(f"版本号不能为负数: {version_str}")

        return cls(
            major=major, minor=minor, patch=patch,
            prerelease=prerelease, build=build
        )


class ModelLifecycle:
    """
    模型生命周期管理

    管理模型从创建到废弃的完整生命周期。
    """

    class Stage(Enum):
        """模型生命周期阶段"""
        DEVELOPMENT = "development"     # 开发中
        ALPHA = "alpha"                 # 内测
        BETA = "beta"                   # 公测
        STABLE = "stable"               # 稳定
        DEPRECATED = "deprecated"       # 已废弃
        ARCHIVED = "archived"           # 已归档

    def __init__(
        self,
        model_name: str,
        version: ModelVersion,
        stage: Optional[Stage] = None
    ):
        """
        初始化模型生命周期

        Args:
            model_name: 模型名称
            version: 模型版本
            stage: 生命周期阶段
        """
        self._model_name = model_name
        self._version = version
        self._stage = stage or self.Stage.DEVELOPMENT
        self._created_at = time.time()
        self._updated_at = self._created_at
        self._stage_history: List[Tuple[str, float]] = [
            (self._stage.value, self._created_at)
        ]
        self._checksum: str = ""
        self._file_path: str = ""
        self._size_bytes: int = 0

    def __repr__(self) -> str:
        return (
            f"ModelLifecycle(name=\"{self._model_name}\", "
            f"version={self._version}, stage={self._stage.value})"
        )

    @property
    def model_name(self) -> str:
        """获取模型名称"""
        return self._model_name

    @property
    def version(self) -> ModelVersion:
        """获取模型版本"""
        return self._version

    @property
    def stage(self) -> Stage:
        """获取当前阶段"""
        return self._stage

    @property
    def created_at(self) -> float:
        """获取创建时间"""
        return self._created_at

    @property
    def updated_at(self) -> float:
        """获取更新时间"""
        return self._updated_at

    def advance_stage(self, new_stage: Optional[Stage] = None) -> None:
        """
        推进到下一阶段

        Args:
            new_stage: 指定新阶段，None则自动推进到下一阶段
        """
        if new_stage is None:
            stage_order = list(self.Stage)
            current_idx = stage_order.index(self._stage)
            if current_idx < len(stage_order) - 1:
                new_stage = stage_order[current_idx + 1]
            else:
                return

        self._stage = new_stage
        self._updated_at = time.time()
        self._stage_history.append((new_stage.value, self._updated_at))

    def get_stage_history(self) -> List[Tuple[str, float]]:
        """获取阶段变更历史"""
        return list(self._stage_history)

    def set_checksum(self, checksum: str) -> None:
        """设置模型文件校验和"""
        self._checksum = checksum

    def set_file_info(self, file_path: str, size_bytes: int) -> None:
        """设置模型文件信息"""
        self._file_path = file_path
        self._size_bytes = size_bytes

    def compute_checksum(self, file_path: str) -> str:
        """
        计算模型文件的SHA256校验和

        Args:
            file_path: 文件路径

        Returns:
            SHA256校验和
        """
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        self._checksum = sha256.hexdigest()
        return self._checksum

    def to_dict(self) -> Dict[str, Any]:
        """导出为字典"""
        return {
            'model_name': self._model_name,
            'version': self._version.to_dict(),
            'stage': self._stage.value,
            'created_at': self._created_at,
            'updated_at': self._updated_at,
            'stage_history': self._stage_history,
            'checksum': self._checksum,
            'file_path': self._file_path,
            'size_bytes': self._size_bytes,
        }


# ============================================================
# 步骤60: 模型缓存机制 - LRU缓存策略
# ============================================================

class LRUCache:
    """
    LRU（最近最少使用）缓存

    特性：
    - 固定容量
    - 自动淘汰最近最少使用的条目
    - 线程安全
    - 内存使用统计
    """

    def __init__(self, capacity: int = 10):
        """
        初始化LRU缓存

        Args:
            capacity: 最大缓存条目数
        """
        self._capacity = max(1, capacity)
        self._cache: OrderedDict = OrderedDict()
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def __repr__(self) -> str:
        return (
            f"LRUCache(capacity={self._capacity}, "
            f"size={len(self._cache)}, "
            f"hits={self._hits}, misses={self._misses})"
        )

    def __len__(self) -> int:
        """返回当前缓存大小"""
        return len(self._cache)

    def __contains__(self, key) -> bool:
        """检查键是否在缓存中"""
        return key in self._cache

    @property
    def capacity(self) -> int:
        """获取缓存容量"""
        return self._capacity

    @property
    def size(self) -> int:
        """获取当前缓存大小"""
        return len(self._cache)

    @property
    def hits(self) -> int:
        """获取缓存命中次数"""
        return self._hits

    @property
    def misses(self) -> int:
        """获取缓存未命中次数"""
        return self._misses

    @property
    def evictions(self) -> int:
        """获取缓存淘汰次数"""
        return self._evictions

    @property
    def hit_rate(self) -> float:
        """获取缓存命中率"""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def get(self, key: Any) -> Optional[Any]:
        """
        获取缓存值

        Args:
            key: 缓存键

        Returns:
            缓存值，不存在则返回None
        """
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._hits += 1
                return self._cache[key]
            self._misses += 1
            return None

    def put(self, key: Any, value: Any) -> Optional[Any]:
        """
        放入缓存值

        如果缓存已满，淘汰最近最少使用的条目。

        Args:
            key: 缓存键
            value: 缓存值

        Returns:
            被淘汰的值，如果没有淘汰则返回None
        """
        with self._lock:
            evicted = None
            if key in self._cache:
                self._cache.move_to_end(key)
                self._cache[key] = value
            else:
                if len(self._cache) >= self._capacity:
                    evicted_key, evicted = self._cache.popitem(last=False)
                    self._evictions += 1
                self._cache[key] = value
            return evicted

    def delete(self, key: Any) -> bool:
        """
        删除缓存条目

        Args:
            key: 缓存键

        Returns:
            是否成功删除
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> None:
        """清空缓存"""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
            self._evictions = 0

    def resize(self, new_capacity: int) -> None:
        """
        调整缓存容量

        Args:
            new_capacity: 新的容量
        """
        with self._lock:
            self._capacity = max(1, new_capacity)
            while len(self._cache) > self._capacity:
                self._cache.popitem(last=False)
                self._evictions += 1

    def keys(self) -> List[Any]:
        """获取所有缓存键（按使用时间排序）"""
        return list(self._cache.keys())

    def values(self) -> List[Any]:
        """获取所有缓存值（按使用时间排序）"""
        return list(self._cache.values())

    def stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        return {
            'capacity': self._capacity,
            'size': len(self._cache),
            'hits': self._hits,
            'misses': self._misses,
            'evictions': self._evictions,
            'hit_rate': self.hit_rate,
        }


class ModelCache:
    """
    模型缓存管理器

    特性：
    - 基于LRU策略的模型缓存
    - 内存使用估算和管理
    - 模型版本感知
    - 自动清理
    """

    def __init__(self, max_models: int = 5, max_memory_mb: int = 512):
        """
        初始化模型缓存

        Args:
            max_models: 最大缓存模型数
            max_memory_mb: 最大内存使用（MB）
        """
        self._cache = LRUCache(capacity=max_models)
        self._max_memory_mb = max_memory_mb
        self._memory_usage: Dict[str, int] = {}
        self._lock = threading.RLock()
        self._load_times: Dict[str, float] = {}

    def __repr__(self) -> str:
        return (
            f"ModelCache(models={self._cache.size}/{self._cache.capacity}, "
            f"memory={self._total_memory_mb:.1f}/{self._max_memory_mb}MB)"
        )

    @property
    def _total_memory_mb(self) -> float:
        """计算总内存使用（MB）"""
        return sum(self._memory_usage.values()) / (1024 * 1024)

    def get(self, model_name: str, version: Optional[ModelVersion] = None) -> Optional[Any]:
        """
        获取缓存的模型

        Args:
            model_name: 模型名称
            version: 模型版本（可选）

        Returns:
            模型实例，不存在则返回None
        """
        cache_key = model_name
        if version:
            cache_key = f"{model_name}:{version.version_string}"
        return self._cache.get(cache_key)

    def put(
        self,
        model_name: str,
        model: Any,
        version: Optional[ModelVersion] = None,
        memory_mb: float = 0
    ) -> Optional[Any]:
        """
        放入模型到缓存

        Args:
            model_name: 模型名称
            model: 模型实例
            version: 模型版本
            memory_mb: 模型内存使用（MB）

        Returns:
            被淘汰的模型，如果没有则返回None
        """
        cache_key = model_name
        if version:
            cache_key = f"{model_name}:{version.version_string}"

        if memory_mb > 0 and self._total_memory_mb + memory_mb > self._max_memory_mb:
            self._evict_for_memory(memory_mb)

        evicted = self._cache.put(cache_key, model)
        if memory_mb > 0:
            self._memory_usage[cache_key] = int(memory_mb * 1024 * 1024)

        if evicted is not None:
            self._cleanup_memory()

        return evicted

    def _evict_for_memory(self, needed_mb: float) -> None:
        """淘汰模型以释放内存"""
        with self._lock:
            while (self._total_memory_mb + needed_mb > self._max_memory_mb
                   and self._cache.size > 0):
                oldest_key = self._cache.keys()[0]
                self._cache.delete(oldest_key)
                self._memory_usage.pop(oldest_key, None)

    def _cleanup_memory(self) -> None:
        """清理无效的内存记录"""
        valid_keys = set(self._cache.keys())
        stale_keys = [k for k in self._memory_usage if k not in valid_keys]
        for key in stale_keys:
            del self._memory_usage[key]

    def invalidate(self, model_name: str, version: Optional[ModelVersion] = None) -> bool:
        """
        使缓存失效

        Args:
            model_name: 模型名称
            version: 模型版本

        Returns:
            是否成功失效
        """
        cache_key = model_name
        if version:
            cache_key = f"{model_name}:{version.version_string}"
        return self._cache.delete(cache_key)

    def clear(self) -> None:
        """清空缓存"""
        self._cache.clear()
        self._memory_usage.clear()
        self._load_times.clear()

    def record_load_time(self, model_name: str, load_time: float) -> None:
        """记录模型加载时间"""
        self._load_times[model_name] = load_time

    def get_load_time(self, model_name: str) -> Optional[float]:
        """获取模型加载时间"""
        return self._load_times.get(model_name)

    def stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        return {
            'cache_stats': self._cache.stats(),
            'memory_usage_mb': self._total_memory_mb,
            'max_memory_mb': self._max_memory_mb,
            'load_times': dict(self._load_times),
        }


# ============================================================
# 步骤61: RESTful API设计 - HTTP服务接口
# ============================================================

class APIRequest:
    """API请求对象 - 封装HTTP请求信息"""

    def __init__(
        self,
        method: str = "GET",
        path: str = "/",
        headers: Optional[Dict[str, str]] = None,
        body: Optional[str] = None,
        query_params: Optional[Dict[str, str]] = None
    ):
        """
        初始化API请求

        Args:
            method: HTTP方法
            path: 请求路径
            headers: 请求头
            body: 请求体
            query_params: 查询参数
        """
        self.method = method.upper()
        self.path = path
        self.headers = headers or {}
        self.body = body or ""
        self.query_params = query_params or {}
        self._json_body: Optional[Any] = None

    @property
    def json(self) -> Any:
        """解析请求体为JSON"""
        if self._json_body is None and self.body:
            try:
                self._json_body = json.loads(self.body)
            except (json.JSONDecodeError, TypeError):
                self._json_body = None
        return self._json_body

    def get_header(self, name: str, default: str = "") -> str:
        """获取请求头"""
        return self.headers.get(name, self.headers.get(name.lower(), default))

    def get_param(self, name: str, default: str = "") -> str:
        """获取查询参数"""
        return self.query_params.get(name, default)


class APIResponse:
    """API响应对象 - 封装HTTP响应信息"""

    def __init__(
        self,
        status_code: int = 200,
        body: Optional[Any] = None,
        headers: Optional[Dict[str, str]] = None,
        content_type: str = "application/json"
    ):
        """
        初始化API响应

        Args:
            status_code: HTTP状态码
            body: 响应体
            headers: 响应头
            content_type: 内容类型
        """
        self.status_code = status_code
        self.body = body
        self.headers = headers or {}
        self.content_type = content_type

    @classmethod
    def ok(cls, data: Any = None, message: str = "success") -> 'APIResponse':
        """创建成功响应"""
        return cls(status_code=200, body={'status': 'ok', 'message': message, 'data': data})

    @classmethod
    def created(cls, data: Any = None) -> 'APIResponse':
        """创建资源创建响应"""
        return cls(status_code=201, body={'status': 'created', 'data': data})

    @classmethod
    def bad_request(cls, message: str = "Bad Request") -> 'APIResponse':
        """创建请求错误响应"""
        return cls(status_code=400, body={'status': 'error', 'message': message})

    @classmethod
    def not_found(cls, message: str = "Not Found") -> 'APIResponse':
        """创建未找到响应"""
        return cls(status_code=404, body={'status': 'error', 'message': message})

    @classmethod
    def server_error(cls, message: str = "Internal Server Error") -> 'APIResponse':
        """创建服务器错误响应"""
        return cls(status_code=500, body={'status': 'error', 'message': message})

    def to_json(self) -> str:
        """将响应序列化为JSON字符串"""
        if self.body is not None:
            return json.dumps(self.body, ensure_ascii=False)
        return ""


class RequestValidator:
    """请求验证器 - 验证API请求的合法性"""

    def __init__(self):
        """初始化请求验证器"""
        self._rules: Dict[str, Dict[str, Any]] = {}

    def add_rule(
        self,
        field_name: str,
        field_type: type = str,
        required: bool = False,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        pattern: Optional[str] = None,
        choices: Optional[List[Any]] = None
    ) -> None:
        """
        添加验证规则

        Args:
            field_name: 字段名
            field_type: 字段类型
            required: 是否必填
            min_length: 最小长度
            max_length: 最大长度
            pattern: 正则模式
            choices: 可选值列表
        """
        self._rules[field_name] = {
            'type': field_type,
            'required': required,
            'min_length': min_length,
            'max_length': max_length,
            'pattern': pattern,
            'choices': choices,
        }

    def validate(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        验证请求数据

        Args:
            data: 请求数据

        Returns:
            (是否有效, 错误消息列表)
        """
        errors = []
        for field_name, rule in self._rules.items():
            value = data.get(field_name)

            if value is None:
                if rule['required']:
                    errors.append(f"字段 '{field_name}' 为必填项")
                continue

            if not isinstance(value, rule['type']):
                errors.append(
                    f"字段 '{field_name}' 类型错误: 期望 {rule['type'].__name__}"
                )
                continue

            if isinstance(value, str):
                if rule['min_length'] is not None and len(value) < rule['min_length']:
                    errors.append(f"字段 '{field_name}' 长度过短: 最少 {rule['min_length']} 个字符")
                if rule['max_length'] is not None and len(value) > rule['max_length']:
                    errors.append(f"字段 '{field_name}' 长度过长: 最多 {rule['max_length']} 个字符")
                if rule['pattern'] is not None:
                    if not re.match(rule['pattern'], value):
                        errors.append(f"字段 '{field_name}' 格式不匹配")

            if rule['choices'] is not None and value not in rule['choices']:
                errors.append(f"字段 '{field_name}' 值无效: 可选值 {rule['choices']}")

        return len(errors) == 0, errors


class Route:
    """路由定义 - 将URL模式映射到处理函数"""

    def __init__(
        self,
        method: str,
        path_pattern: str,
        handler: Callable[[APIRequest], APIResponse],
        validator: Optional[RequestValidator] = None,
        description: str = ""
    ):
        """
        初始化路由

        Args:
            method: HTTP方法
            path_pattern: 路径模式（支持 {param} 参数）
            handler: 处理函数
            validator: 请求验证器
            description: 路由描述
        """
        self.method = method.upper()
        self.path_pattern = path_pattern
        self.handler = handler
        self.validator = validator
        self.description = description
        self._param_names = self._extract_params(path_pattern)

    def _extract_params(self, pattern: str) -> List[str]:
        """从路径模式中提取参数名"""
        return re.findall(r'\{(\w+)\}', pattern)

    def match(self, method: str, path: str) -> Optional[Dict[str, str]]:
        """
        匹配请求方法和路径

        Args:
            method: HTTP方法
            path: 请求路径

        Returns:
            匹配成功返回参数字典，否则返回None
        """
        if self.method != method.upper():
            return None

        regex_pattern = self.path_pattern
        for param in self._param_names:
            regex_pattern = regex_pattern.replace(f'{{{param}}}', f'(?P<{param}>[^/]+)')
        regex_pattern = f'^{regex_pattern}$'

        match = re.match(regex_pattern, path)
        if match:
            return match.groupdict()
        return None


class APIServer:
    """
    RESTful API服务器框架

    基于Python标准库http.server实现的轻量级API服务器。
    提供路由注册、请求验证、中间件支持等功能。

    注意：此为框架设计，不依赖flask/fastapi等外部库。
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8000):
        """
        初始化API服务器

        Args:
            host: 监听地址
            port: 监听端口
        """
        self._host = host
        self._port = port
        self._routes: List[Route] = []
        self._middleware: List[Callable] = []
        self._server: Optional[HTTPServer] = None
        self._pipeline: Optional[Pipeline] = None
        self._lock = threading.RLock()

    @property
    def host(self) -> str:
        """获取监听地址"""
        return self._host

    @property
    def port(self) -> int:
        """获取监听端口"""
        return self._port

    def set_pipeline(self, pipeline: Pipeline) -> None:
        """关联Pipeline"""
        self._pipeline = pipeline

    def add_route(
        self,
        method: str,
        path: str,
        handler: Callable[[APIRequest], APIResponse],
        validator: Optional[RequestValidator] = None,
        description: str = ""
    ) -> None:
        """添加路由"""
        route = Route(method, path, handler, validator, description)
        self._routes.append(route)

    def add_middleware(self, middleware: Callable) -> None:
        """添加中间件"""
        self._middleware.append(middleware)

    def _find_route(self, method: str, path: str) -> Optional[Tuple[Route, Dict[str, str]]]:
        """查找匹配的路由"""
        for route in self._routes:
            params = route.match(method, path)
            if params is not None:
                return route, params
        return None

    def handle_request(self, request: APIRequest) -> APIResponse:
        """
        处理API请求

        Args:
            request: API请求对象

        Returns:
            API响应对象
        """
        result = self._find_route(request.method, request.path)
        if result is None:
            return APIResponse.not_found(f"路径 {request.path} 不存在")

        route, params = result

        if route.validator is not None:
            data = request.json if request.json else {}
            data.update(params)
            is_valid, errors = route.validator.validate(data)
            if not is_valid:
                return APIResponse.bad_request("; ".join(errors))

        def execute_handler(req: APIRequest) -> APIResponse:
            try:
                return route.handler(req)
            except Exception as e:
                return APIResponse.server_error(str(e))

        handler = execute_handler
        for mw in reversed(self._middleware):
            handler = functools.partial(mw, handler=handler)

        return handler(request)

    def setup_default_routes(self) -> None:
        """设置默认路由"""
        def health_check(req: APIRequest) -> APIResponse:
            return APIResponse.ok({
                'status': 'healthy',
                'version': '1.0.0',
                'components': self._pipeline.component_names if self._pipeline else [],
            })
        self.add_route("GET", "/health", health_check, description="健康检查")

        def tokenize(req: APIRequest) -> APIResponse:
            if not self._pipeline:
                return APIResponse.server_error("Pipeline未初始化")
            data = req.json
            if not data or 'text' not in data:
                return APIResponse.bad_request("缺少 'text' 字段")
            doc = self._pipeline.process(data['text'])
            tokens = [t.text for t in doc.tokens]
            return APIResponse.ok({'tokens': tokens, 'text': doc.text})

        validator = RequestValidator()
        validator.add_rule('text', str, required=True, min_length=1, max_length=10000)
        self.add_route("POST", "/tokenize", tokenize, validator, description="分词处理")

        def batch_tokenize(req: APIRequest) -> APIResponse:
            if not self._pipeline:
                return APIResponse.server_error("Pipeline未初始化")
            data = req.json
            if not data or 'texts' not in data:
                return APIResponse.bad_request("缺少 'texts' 字段")
            texts = data['texts']
            if not isinstance(texts, list):
                return APIResponse.bad_request("'texts' 必须是数组")
            docs = self._pipeline.process_batch(texts)
            results = [{'tokens': [t.text for t in doc.tokens], 'text': doc.text} for doc in docs]
            return APIResponse.ok({'results': results, 'count': len(results)})

        self.add_route("POST", "/batch/tokenize", batch_tokenize, description="批量分词处理")

        def pipeline_info(req: APIRequest) -> APIResponse:
            if not self._pipeline:
                return APIResponse.server_error("Pipeline未初始化")
            return APIResponse.ok({
                'name': self._pipeline.name,
                'components': self._pipeline.get_execution_order(),
                'component_count': len(self._pipeline.component_names),
            })
        self.add_route("GET", "/pipeline/info", pipeline_info, description="获取Pipeline信息")

    def create_http_handler(self):
        """创建HTTP请求处理器类"""
        api_server = self

        class Handler(BaseHTTPRequestHandler):
            """HTTP请求处理器"""

            def do_GET(self):
                self._handle_any()

            def do_POST(self):
                self._handle_any()

            def do_PUT(self):
                self._handle_any()

            def do_DELETE(self):
                self._handle_any()

            def _handle_any(self):
                parsed = urlparse(self.path)
                query_params = {k: v[0] for k, v in parse_qs(parsed.query).items()}
                content_length = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(content_length).decode('utf-8') if content_length > 0 else ""
                headers = {k: v for k, v in self.headers.items()}

                request = APIRequest(
                    method=self.command,
                    path=parsed.path,
                    headers=headers,
                    body=body,
                    query_params=query_params,
                )
                response = api_server.handle_request(request)

                self.send_response(response.status_code)
                self.send_header('Content-Type', response.content_type)
                for key, value in response.headers.items():
                    self.send_header(key, value)
                self.end_headers()
                self.wfile.write(response.to_json().encode('utf-8'))

            def log_message(self, format, *args):
                pass

        return Handler

    def serve(self, blocking: bool = True) -> None:
        """启动HTTP服务器"""
        handler_class = self.create_http_handler()
        self._server = HTTPServer((self._host, self._port), handler_class)

        if blocking:
            print(f"AuroraNLP API服务器启动: http://{self._host}:{self._port}")
            try:
                self._server.serve_forever()
            except KeyboardInterrupt:
                print("\n服务器已停止")
            finally:
                self._server.server_close()
        else:
            thread = threading.Thread(target=self._server.serve_forever, daemon=True)
            thread.start()
            print(f"AuroraNLP API服务器启动（后台）: http://{self._host}:{self._port}")

    def shutdown(self) -> None:
        """关闭服务器"""
        if self._server:
            self._server.shutdown()
            self._server.server_close()
            self._server = None


# ============================================================
# 步骤62: gRPC接口实现 - 高性能RPC调用框架
# ============================================================

class RPCMessage:
    """RPC消息 - 封装RPC调用中的消息数据"""

    def __init__(self, payload: Optional[Dict[str, Any]] = None, metadata: Optional[Dict[str, str]] = None):
        """
        初始化RPC消息

        Args:
            payload: 消息载荷
            metadata: 消息元数据
        """
        self.payload = payload or {}
        self.metadata = metadata or {}
        self._timestamp = time.time()

    @property
    def timestamp(self) -> float:
        """获取消息时间戳"""
        return self._timestamp

    def to_json(self) -> str:
        """序列化为JSON"""
        return json.dumps({
            'payload': self.payload,
            'metadata': self.metadata,
            'timestamp': self._timestamp,
        }, ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> 'RPCMessage':
        """从JSON反序列化"""
        data = json.loads(json_str)
        msg = cls(payload=data.get('payload', {}), metadata=data.get('metadata', {}))
        msg._timestamp = data.get('timestamp', time.time())
        return msg


class RPCError(Exception):
    """RPC调用错误"""

    def __init__(self, code: int, message: str, details: str = ""):
        """
        初始化RPC错误

        Args:
            code: 错误码
            message: 错误消息
            details: 错误详情
        """
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details


class RPCStatusCode(IntEnum):
    """RPC状态码"""
    OK = 0
    CANCELLED = 1
    UNKNOWN = 2
    INVALID_ARGUMENT = 3
    DEADLINE_EXCEEDED = 4
    NOT_FOUND = 5
    ALREADY_EXISTS = 6
    PERMISSION_DENIED = 7
    RESOURCE_EXHAUSTED = 8
    FAILED_PRECONDITION = 9
    ABORTED = 10
    OUT_OF_RANGE = 11
    UNIMPLEMENTED = 12
    INTERNAL = 13
    UNAVAILABLE = 14
    DATA_LOSS = 15
    UNAUTHENTICATED = 16


class RPCServiceDescriptor:
    """RPC服务描述符 - 定义服务的接口规范"""

    def __init__(self, service_name: str, version: str = "1.0"):
        """
        初始化服务描述符

        Args:
            service_name: 服务名称
            version: 服务版本
        """
        self.service_name = service_name
        self.version = version
        self._methods: Dict[str, Dict[str, Any]] = {}

    def add_method(
        self,
        method_name: str,
        input_type: str = "RPCMessage",
        output_type: str = "RPCMessage",
        server_streaming: bool = False,
        client_streaming: bool = False,
        description: str = ""
    ) -> None:
        """添加方法定义"""
        self._methods[method_name] = {
            'input_type': input_type,
            'output_type': output_type,
            'server_streaming': server_streaming,
            'client_streaming': client_streaming,
            'description': description,
        }

    def get_methods(self) -> Dict[str, Dict[str, Any]]:
        """获取所有方法定义"""
        return dict(self._methods)


class RPCMethodHandler:
    """RPC方法处理器 - 处理单个RPC方法调用"""

    def __init__(
        self,
        name: str,
        handler: Callable[[RPCMessage], RPCMessage],
        input_validator: Optional[Callable[[RPCMessage], Tuple[bool, str]]] = None,
        description: str = ""
    ):
        """
        初始化方法处理器

        Args:
            name: 方法名
            handler: 处理函数
            input_validator: 输入验证函数
            description: 方法描述
        """
        self.name = name
        self.handler = handler
        self.input_validator = input_validator
        self.description = description


class RPCService:
    """
    RPC服务 - 提供一组相关的RPC方法

    注意：此为接口设计框架，不依赖grpc库。
    """

    def __init__(self, name: str, version: str = "1.0"):
        """
        初始化RPC服务

        Args:
            name: 服务名称
            version: 服务版本
        """
        self._name = name
        self._version = version
        self._handlers: Dict[str, RPCMethodHandler] = {}
        self._interceptors: List[Callable] = []
        self._descriptor = RPCServiceDescriptor(name, version)

    @property
    def name(self) -> str:
        """获取服务名称"""
        return self._name

    @property
    def version(self) -> str:
        """获取服务版本"""
        return self._version

    def register_method(
        self,
        method_name: str,
        handler: Callable[[RPCMessage], RPCMessage],
        input_validator: Optional[Callable[[RPCMessage], Tuple[bool, str]]] = None,
        description: str = "",
        server_streaming: bool = False,
        client_streaming: bool = False
    ) -> None:
        """注册RPC方法"""
        method_handler = RPCMethodHandler(
            name=method_name, handler=handler,
            input_validator=input_validator, description=description
        )
        self._handlers[method_name] = method_handler
        self._descriptor.add_method(
            method_name=method_name, server_streaming=server_streaming,
            client_streaming=client_streaming, description=description
        )

    def add_interceptor(self, interceptor: Callable) -> None:
        """添加拦截器"""
        self._interceptors.append(interceptor)

    def call_method(self, method_name: str, request: RPCMessage) -> RPCMessage:
        """
        调用RPC方法

        Args:
            method_name: 方法名
            request: 请求消息

        Returns:
            响应消息

        Raises:
            RPCError: 方法不存在或调用失败
        """
        if method_name not in self._handlers:
            raise RPCError(
                code=RPCStatusCode.UNIMPLEMENTED,
                message=f"方法 '{method_name}' 不存在",
                details=f"服务 '{self._name}' 可用方法: {list(self._handlers.keys())}"
            )

        method_handler = self._handlers[method_name]

        if method_handler.input_validator:
            is_valid, error_msg = method_handler.input_validator(request)
            if not is_valid:
                raise RPCError(code=RPCStatusCode.INVALID_ARGUMENT, message=error_msg)

        def execute(req: RPCMessage) -> RPCMessage:
            try:
                return method_handler.handler(req)
            except RPCError:
                raise
            except Exception as e:
                raise RPCError(code=RPCStatusCode.INTERNAL, message=str(e))

        handler = execute
        for interceptor in reversed(self._interceptors):
            handler = functools.partial(interceptor, handler=handler)

        return handler(request)

    def get_descriptor(self) -> RPCServiceDescriptor:
        """获取服务描述符"""
        return self._descriptor

    def list_methods(self) -> List[str]:
        """列出所有方法名"""
        return list(self._handlers.keys())


class RPCServer:
    """
    RPC服务器框架

    基于socket实现的轻量级RPC服务器。
    提供服务注册、方法调用、拦截器等功能。

    注意：此为接口设计框架，不依赖grpc库。
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 50051):
        """
        初始化RPC服务器

        Args:
            host: 监听地址
            port: 监听端口
        """
        self._host = host
        self._port = port
        self._services: Dict[str, RPCService] = {}
        self._socket: Optional[socket.socket] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        self._pipeline: Optional[Pipeline] = None

    @property
    def host(self) -> str:
        """获取监听地址"""
        return self._host

    @property
    def port(self) -> int:
        """获取监听端口"""
        return self._port

    def set_pipeline(self, pipeline: Pipeline) -> None:
        """关联Pipeline"""
        self._pipeline = pipeline

    def register_service(self, service: RPCService) -> None:
        """注册RPC服务"""
        with self._lock:
            self._services[service.name] = service

    def get_service(self, name: str) -> Optional[RPCService]:
        """获取RPC服务"""
        return self._services.get(name)

    def _handle_connection(self, conn: socket.socket, addr: Tuple[str, int]) -> None:
        """处理客户端连接"""
        try:
            buffer = BytesIO()
            while self._running:
                data = conn.recv(65536)
                if not data:
                    break
                buffer.write(data)

                buffer.seek(0)
                while True:
                    header = buffer.read(4)
                    if len(header) < 4:
                        break
                    msg_len = struct.unpack('!I', header)[0]
                    msg_data = buffer.read(msg_len)
                    if len(msg_data) < msg_len:
                        break

                    try:
                        request = RPCMessage.from_json(msg_data.decode('utf-8'))
                        response = self._dispatch(request)
                        resp_data = response.to_json().encode('utf-8')
                        conn.sendall(struct.pack('!I', len(resp_data)) + resp_data)
                    except Exception as e:
                        error_msg = RPCMessage(payload={'error': str(e)})
                        resp_data = error_msg.to_json().encode('utf-8')
                        conn.sendall(struct.pack('!I', len(resp_data)) + resp_data)

                remaining = buffer.read()
                buffer = BytesIO()
                buffer.write(remaining)
        except Exception:
            pass
        finally:
            conn.close()

    def _dispatch(self, request: RPCMessage) -> RPCMessage:
        """分发RPC请求"""
        service_name = request.metadata.get('service', '')
        method_name = request.metadata.get('method', '')

        if not service_name or not method_name:
            return RPCMessage(
                payload={'error': '缺少 service 或 method'},
                metadata={'status': str(RPCStatusCode.INVALID_ARGUMENT)}
            )

        service = self._services.get(service_name)
        if not service:
            return RPCMessage(
                payload={'error': f"服务 '{service_name}' 不存在"},
                metadata={'status': str(RPCStatusCode.NOT_FOUND)}
            )

        try:
            response = service.call_method(method_name, request)
            response.metadata['status'] = str(RPCStatusCode.OK)
            return response
        except RPCError as e:
            return RPCMessage(
                payload={'error': e.message, 'code': e.code},
                metadata={'status': str(e.code)}
            )

    def setup_nlp_service(self) -> RPCService:
        """创建NLP处理RPC服务"""
        service = RPCService("NLProcessor", version="1.0")

        def tokenize_handler(request: RPCMessage) -> RPCMessage:
            if not self._pipeline:
                raise RPCError(RPCStatusCode.INTERNAL, "Pipeline未初始化")
            text = request.payload.get('text', '')
            if not text:
                raise RPCError(RPCStatusCode.INVALID_ARGUMENT, "缺少 'text' 字段")
            doc = self._pipeline.process(text)
            tokens = [t.text for t in doc.tokens]
            return RPCMessage(payload={'tokens': tokens, 'text': doc.text})

        def segment_handler(request: RPCMessage) -> RPCMessage:
            if not self._pipeline:
                raise RPCError(RPCStatusCode.INTERNAL, "Pipeline未初始化")
            text = request.payload.get('text', '')
            doc = self._pipeline.process(text)
            return RPCMessage(payload={'segments': [t.text for t in doc.tokens]})

        def health_handler(request: RPCMessage) -> RPCMessage:
            return RPCMessage(payload={
                'status': 'healthy',
                'services': list(self._services.keys()),
            })

        service.register_method("Tokenize", tokenize_handler, description="分词处理")
        service.register_method("Segment", segment_handler, description="文本分段")
        service.register_method("Health", health_handler, description="健康检查")

        return service

    def serve(self, blocking: bool = True) -> None:
        """启动RPC服务器"""
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind((self._host, self._port))
        self._socket.listen(5)
        self._running = True

        print(f"AuroraNLP RPC服务器启动: {self._host}:{self._port}")

        if blocking:
            try:
                while self._running:
                    self._socket.settimeout(1.0)
                    try:
                        conn, addr = self._socket.accept()
                        thread = threading.Thread(
                            target=self._handle_connection, args=(conn, addr), daemon=True
                        )
                        thread.start()
                    except socket.timeout:
                        continue
            except KeyboardInterrupt:
                print("\nRPC服务器已停止")
            finally:
                self.shutdown()
        else:
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()

    def _run_loop(self) -> None:
        """服务器运行循环"""
        while self._running:
            try:
                self._socket.settimeout(1.0)
                conn, addr = self._socket.accept()
                thread = threading.Thread(
                    target=self._handle_connection, args=(conn, addr), daemon=True
                )
                thread.start()
            except socket.timeout:
                continue
            except Exception:
                break

    def shutdown(self) -> None:
        """关闭RPC服务器"""
        self._running = False
        if self._socket:
            try:
                self._socket.close()
            except Exception:
                pass
            self._socket = None


class RPCClient:
    """
    RPC客户端 - 用于调用RPC服务

    注意：此为接口设计框架，不依赖grpc库。
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 50051, timeout: float = 30.0):
        """
        初始化RPC客户端

        Args:
            host: 服务器地址
            port: 服务器端口
            timeout: 超时时间（秒）
        """
        self._host = host
        self._port = port
        self._timeout = timeout
        self._socket: Optional[socket.socket] = None
        self._connected = False

    @property
    def connected(self) -> bool:
        """是否已连接"""
        return self._connected

    def connect(self) -> None:
        """连接到RPC服务器"""
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.settimeout(self._timeout)
        self._socket.connect((self._host, self._port))
        self._connected = True

    def close(self) -> None:
        """关闭连接"""
        if self._socket:
            try:
                self._socket.close()
            except Exception:
                pass
            self._socket = None
            self._connected = False

    def call(
        self,
        service: str,
        method: str,
        payload: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, str]] = None
    ) -> RPCMessage:
        """
        调用RPC方法

        Args:
            service: 服务名
            method: 方法名
            payload: 请求载荷
            metadata: 请求元数据

        Returns:
            响应消息

        Raises:
            RPCError: 调用失败
        """
        if not self._connected:
            self.connect()

        meta = metadata or {}
        meta['service'] = service
        meta['method'] = method

        request = RPCMessage(payload=payload or {}, metadata=meta)
        request_data = request.to_json().encode('utf-8')

        try:
            self._socket.sendall(struct.pack('!I', len(request_data)) + request_data)

            header = self._recv_exact(4)
            if not header:
                raise RPCError(RPCStatusCode.UNAVAILABLE, "连接已断开")
            resp_len = struct.unpack('!I', header)[0]
            resp_data = self._recv_exact(resp_len)
            if not resp_data:
                raise RPCError(RPCStatusCode.UNAVAILABLE, "响应数据不完整")

            response = RPCMessage.from_json(resp_data.decode('utf-8'))
            status = int(response.metadata.get('status', RPCStatusCode.OK))

            if status != RPCStatusCode.OK:
                error_info = response.payload.get('error', '未知错误')
                raise RPCError(code=status, message=error_info)

            return response
        except socket.timeout:
            raise RPCError(RPCStatusCode.DEADLINE_EXCEEDED, "请求超时")
        except ConnectionError:
            self._connected = False
            raise RPCError(RPCStatusCode.UNAVAILABLE, "连接失败")

    def _recv_exact(self, n: int) -> Optional[bytes]:
        """精确接收n字节"""
        data = b''
        while len(data) < n:
            chunk = self._socket.recv(n - len(data))
            if not chunk:
                return None
            data += chunk
        return data

    def __enter__(self) -> 'RPCClient':
        """上下文管理器入口"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """上下文管理器出口"""
        self.close()


# ============================================================
# 步骤63: 异步处理支持 - asyncio集成
# ============================================================

class AsyncPipeline:
    """
    异步Pipeline - 支持asyncio的并发处理流水线

    特性：
    - 异步组件处理
    - 并发文档处理
    - 协程调度
    - 信号量控制并发度
    """

    def __init__(
        self,
        pipeline: Pipeline,
        max_concurrency: int = 10,
        timeout: Optional[float] = None
    ):
        """
        初始化异步Pipeline

        Args:
            pipeline: 同步Pipeline实例
            max_concurrency: 最大并发数
            timeout: 超时时间（秒）
        """
        self._pipeline = pipeline
        self._max_concurrency = max_concurrency
        self._timeout = timeout
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._executor: Optional[ThreadPoolExecutor] = None

    @property
    def pipeline(self) -> Pipeline:
        """获取底层Pipeline"""
        return self._pipeline

    async def _init_async(self) -> None:
        """初始化异步资源"""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self._max_concurrency)
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=self._max_concurrency)

    async def process(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Doc:
        """
        异步处理单个文本

        Args:
            text: 输入文本
            metadata: 文档元数据

        Returns:
            处理后的Doc对象
        """
        await self._init_async()
        async with self._semaphore:
            loop = asyncio.get_running_loop()
            doc = await loop.run_in_executor(
                self._executor, self._pipeline.process, text, metadata
            )
            return doc

    async def process_batch(
        self,
        texts: List[str],
        metadata_list: Optional[List[Optional[Dict[str, Any]]]] = None
    ) -> List[Doc]:
        """
        异步批量处理文本

        Args:
            texts: 文本列表
            metadata_list: 元数据列表

        Returns:
            Doc对象列表
        """
        await self._init_async()
        tasks = []
        for i, text in enumerate(texts):
            meta = metadata_list[i] if metadata_list and i < len(metadata_list) else None
            tasks.append(self.process(text, meta))

        if self._timeout:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self._timeout
            )
        else:
            results = await asyncio.gather(*tasks, return_exceptions=True)

        final_results = []
        for result in results:
            if isinstance(result, Exception):
                final_results.append(None)  # type: ignore
            else:
                final_results.append(result)
        return final_results  # type: ignore

    async def process_with_callback(
        self,
        text: str,
        callback: Callable[[Doc], Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Doc:
        """
        异步处理并回调

        Args:
            text: 输入文本
            callback: 回调函数
            metadata: 文档元数据

        Returns:
            处理后的Doc对象
        """
        doc = await self.process(text, metadata)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, callback, doc)
        return doc

    def shutdown(self) -> None:
        """关闭异步资源"""
        if self._executor:
            self._executor.shutdown(wait=False)
            self._executor = None


class AsyncTaskManager:
    """
    异步任务管理器 - 管理异步处理任务

    特性：
    - 任务队列
    - 优先级调度
    - 任务状态跟踪
    - 结果缓存
    """

    class TaskState(Enum):
        """任务状态"""
        PENDING = "pending"
        RUNNING = "running"
        COMPLETED = "completed"
        FAILED = "failed"
        CANCELLED = "cancelled"

    def __init__(self, max_workers: int = 4):
        """
        初始化任务管理器

        Args:
            max_workers: 最大工作线程数
        """
        self._max_workers = max_workers
        self._tasks: Dict[str, Dict[str, Any]] = {}
        self._queue: List[Tuple[int, str]] = []
        self._lock = threading.RLock()
        self._results: Dict[str, Any] = {}
        self._next_id = 0
        self._executor: Optional[ThreadPoolExecutor] = None

    def create_task(
        self,
        func: Callable,
        args: Tuple = (),
        kwargs: Optional[Dict[str, Any]] = None,
        priority: int = 0,
        callback: Optional[Callable] = None
    ) -> str:
        """
        创建异步任务

        Args:
            func: 任务函数
            args: 位置参数
            kwargs: 关键字参数
            priority: 优先级（数字越大优先级越高）
            callback: 完成回调

        Returns:
            任务ID
        """
        with self._lock:
            task_id = f"task_{self._next_id}"
            self._next_id += 1
            self._tasks[task_id] = {
                'func': func,
                'args': args,
                'kwargs': kwargs or {},
                'priority': priority,
                'callback': callback,
                'state': self.TaskState.PENDING,
                'created_at': time.time(),
                'started_at': None,
                'completed_at': None,
                'error': None,
            }
            self._queue.append((priority, task_id))
            self._queue.sort(key=lambda x: -x[0])
            return task_id

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务信息"""
        return self._tasks.get(task_id)

    def get_result(self, task_id: str) -> Optional[Any]:
        """获取任务结果"""
        return self._results.get(task_id)

    def get_task_state(self, task_id: str) -> Optional['AsyncTaskManager.TaskState']:
        """获取任务状态"""
        task = self._tasks.get(task_id)
        if task:
            return task['state']
        return None

    def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        with self._lock:
            task = self._tasks.get(task_id)
            if task and task['state'] == self.TaskState.PENDING:
                task['state'] = self.TaskState.CANCELLED
                self._queue = [(p, tid) for p, tid in self._queue if tid != task_id]
                return True
            return False

    def _execute_task(self, task_id: str) -> None:
        """执行单个任务"""
        task = self._tasks.get(task_id)
        if not task or task['state'] != self.TaskState.PENDING:
            return

        task['state'] = self.TaskState.RUNNING
        task['started_at'] = time.time()

        try:
            result = task['func'](*task['args'], **task['kwargs'])
            self._results[task_id] = result
            task['state'] = self.TaskState.COMPLETED
            if task['callback']:
                task['callback'](result)
        except Exception as e:
            task['error'] = str(e)
            task['state'] = self.TaskState.FAILED
        finally:
            task['completed_at'] = time.time()

    def process_queue(self, max_tasks: int = -1) -> int:
        """
        处理任务队列

        Args:
            max_tasks: 最大处理任务数，-1表示全部

        Returns:
            已处理的任务数
        """
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=self._max_workers)

        processed = 0
        while True:
            if max_tasks >= 0 and processed >= max_tasks:
                break

            with self._lock:
                if not self._queue:
                    break
                _, task_id = self._queue.pop(0)
                task = self._tasks.get(task_id)
                if not task or task['state'] != self.TaskState.PENDING:
                    continue

            self._executor.submit(self._execute_task, task_id)
            processed += 1

        return processed

    def pending_count(self) -> int:
        """获取待处理任务数"""
        with self._lock:
            return len(self._queue)

    def completed_count(self) -> int:
        """获取已完成任务数"""
        return sum(1 for t in self._tasks.values() if t['state'] == self.TaskState.COMPLETED)

    def stats(self) -> Dict[str, Any]:
        """获取任务统计信息"""
        states = {}
        for task in self._tasks.values():
            state = task['state'].value
            states[state] = states.get(state, 0) + 1
        return {
            'total_tasks': len(self._tasks),
            'pending': len(self._queue),
            'states': states,
        }

    def shutdown(self) -> None:
        """关闭任务管理器"""
        if self._executor:
            self._executor.shutdown(wait=False)
            self._executor = None


# ============================================================
# 步骤64: 流式处理实现 - 大文件流式分词
# ============================================================

class ProgressCallback:
    """进度回调 - 报告处理进度"""

    def __init__(
        self,
        total: Optional[int] = None,
        callback: Optional[Callable[[int, Optional[int], float], None]] = None,
        report_interval: float = 1.0
    ):
        """
        初始化进度回调

        Args:
            total: 总数量（已知时）
            callback: 回调函数(processed, total, elapsed_seconds)
            report_interval: 报告间隔（秒）
        """
        self._total = total
        self._callback = callback
        self._report_interval = report_interval
        self._processed = 0
        self._start_time = time.time()
        self._last_report_time = 0.0

    @property
    def processed(self) -> int:
        """已处理数量"""
        return self._processed

    @property
    def elapsed(self) -> float:
        """已用时间（秒）"""
        return time.time() - self._start_time

    @property
    def speed(self) -> float:
        """处理速度（条/秒）"""
        elapsed = self.elapsed
        return self._processed / elapsed if elapsed > 0 else 0.0

    def update(self, increment: int = 1) -> None:
        """更新进度"""
        self._processed += increment
        current_time = time.time()
        if (self._callback and
                current_time - self._last_report_time >= self._report_interval):
            self._callback(self._processed, self._total, self.elapsed)
            self._last_report_time = current_time

    def finish(self) -> None:
        """完成处理"""
        if self._callback:
            self._callback(self._processed, self._total, self.elapsed)


class StreamProcessor:
    """
    流式文本处理器 - 内存友好的大文件处理

    特性：
    - 逐行/逐块处理
    - 内存友好
    - 进度回调
    - 支持多种输入源
    - 结果流式输出
    """

    def __init__(
        self,
        pipeline: Pipeline,
        chunk_size: int = 1000,
        max_memory_lines: int = 10000
    ):
        """
        初始化流式处理器

        Args:
            pipeline: Pipeline实例
            chunk_size: 每批处理行数
            max_memory_lines: 最大内存中保持的行数
        """
        self._pipeline = pipeline
        self._chunk_size = chunk_size
        self._max_memory_lines = max_memory_lines

    @property
    def pipeline(self) -> Pipeline:
        """获取Pipeline"""
        return self._pipeline

    def _count_lines(self, file_path: str, encoding: str = "utf-8") -> int:
        """计算文件行数"""
        count = 0
        with open(file_path, 'r', encoding=encoding) as f:
            for _ in f:
                count += 1
        return count

    def _process_batch(
        self,
        texts: List[str],
        result_formatter: Optional[Callable[[Doc], str]] = None
    ) -> List[Doc]:
        """处理一批文本"""
        return self._pipeline.process_batch(texts)

    def process_file(
        self,
        file_path: str,
        encoding: str = "utf-8",
        output_path: Optional[str] = None,
        progress_callback: Optional[Callable[[int, Optional[int], float], None]] = None,
        line_filter: Optional[Callable[[str], bool]] = None,
        result_formatter: Optional[Callable[[Doc], str]] = None
    ) -> List[Doc]:
        """
        流式处理文本文件

        Args:
            file_path: 输入文件路径
            encoding: 文件编码
            output_path: 输出文件路径（可选）
            progress_callback: 进度回调函数
            line_filter: 行过滤函数（返回True则处理）
            result_formatter: 结果格式化函数

        Returns:
            处理结果列表
        """
        total_lines = self._count_lines(file_path, encoding)
        progress = ProgressCallback(total=total_lines, callback=progress_callback)
        results = []
        output_file = None

        try:
            if output_path:
                output_file = open(output_path, 'w', encoding=encoding)

            with open(file_path, 'r', encoding=encoding) as f:
                batch = []
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if line_filter and not line_filter(line):
                        progress.update()
                        continue

                    batch.append(line)

                    if len(batch) >= self._chunk_size:
                        batch_results = self._process_batch(batch, result_formatter)
                        results.extend(batch_results)

                        if output_file and result_formatter:
                            for doc in batch_results:
                                output_file.write(result_formatter(doc) + '\n')

                        if len(results) > self._max_memory_lines:
                            results = results[-self._max_memory_lines:]

                        progress.update(len(batch))
                        batch = []

                if batch:
                    batch_results = self._process_batch(batch, result_formatter)
                    results.extend(batch_results)
                    if output_file and result_formatter:
                        for doc in batch_results:
                            output_file.write(result_formatter(doc) + '\n')
                    progress.update(len(batch))

        finally:
            if output_file:
                output_file.close()

        progress.finish()
        return results

    def process_iterable(
        self,
        texts: Iterable[str],
        progress_callback: Optional[Callable[[int, Optional[int], float], None]] = None,
        result_formatter: Optional[Callable[[Doc], str]] = None
    ) -> Iterator[Doc]:
        """
        流式处理可迭代文本

        Args:
            texts: 文本可迭代对象
            progress_callback: 进度回调
            result_formatter: 结果格式化函数

        Yields:
            处理后的Doc对象
        """
        progress = ProgressCallback(callback=progress_callback)
        batch = []

        for text in texts:
            if not text or not text.strip():
                continue
            batch.append(text.strip())

            if len(batch) >= self._chunk_size:
                for doc in self._pipeline.process_batch(batch):
                    progress.update()
                    yield doc
                batch = []

        if batch:
            for doc in self._pipeline.process_batch(batch):
                progress.update()
                yield doc

        progress.finish()

    def process_stream(
        self,
        text_stream: Iterator[str],
        output_stream: Optional[Any] = None,
        progress_callback: Optional[Callable[[int, Optional[int], float], None]] = None,
        result_formatter: Optional[Callable[[Doc], str]] = None
    ) -> Iterator[Doc]:
        """
        流式处理文本流

        Args:
            text_stream: 文本流
            output_stream: 输出流
            progress_callback: 进度回调
            result_formatter: 结果格式化函数

        Yields:
            处理后的Doc对象
        """
        progress = ProgressCallback(callback=progress_callback)
        batch = []

        for text in text_stream:
            if not text or not text.strip():
                continue
            batch.append(text.strip())

            if len(batch) >= self._chunk_size:
                docs = self._pipeline.process_batch(batch)
                for doc in docs:
                    progress.update()
                    if output_stream and result_formatter:
                        output_stream.write(result_formatter(doc) + '\n')
                    yield doc
                batch = []

        if batch:
            docs = self._pipeline.process_batch(batch)
            for doc in docs:
                progress.update()
                if output_stream and result_formatter:
                    output_stream.write(result_formatter(doc) + '\n')
                yield doc

        progress.finish()

    def process_lines(
        self,
        lines: List[str],
        progress_callback: Optional[Callable[[int, Optional[int], float], None]] = None
    ) -> List[Doc]:
        """
        处理文本行列表

        Args:
            lines: 文本行列表
            progress_callback: 进度回调

        Returns:
            Doc对象列表
        """
        progress = ProgressCallback(total=len(lines), callback=progress_callback)
        results = []

        for i in range(0, len(lines), self._chunk_size):
            batch = lines[i:i + self._chunk_size]
            batch = [line.strip() for line in batch if line.strip()]
            if batch:
                batch_results = self._pipeline.process_batch(batch)
                results.extend(batch_results)
                progress.update(len(batch))

        progress.finish()
        return results


# ============================================================
# 步骤65: 插件系统设计 - 第三方扩展机制
# ============================================================

class PluginState(Enum):
    """插件状态"""
    DISCOVERED = "discovered"     # 已发现
    LOADED = "loaded"             # 已加载
    INITIALIZED = "initialized"   # 已初始化
    ENABLED = "enabled"           # 已启用
    DISABLED = "disabled"         # 已禁用
    ERROR = "error"               # 错误


class PluginInfo:
    """插件信息"""

    def __init__(
        self,
        name: str,
        version: str = "0.1.0",
        description: str = "",
        author: str = "",
        module_path: str = "",
        dependencies: Optional[List[str]] = None,
        entry_point: str = ""
    ):
        """
        初始化插件信息

        Args:
            name: 插件名称
            version: 插件版本
            description: 描述
            author: 作者
            module_path: 模块路径
            dependencies: 依赖列表
            entry_point: 入口点
        """
        self.name = name
        self.version = version
        self.description = description
        self.author = author
        self.module_path = module_path
        self.dependencies = dependencies or []
        self.entry_point = entry_point
        self.state = PluginState.DISCOVERED
        self.error_message = ""
        self.loaded_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """导出为字典"""
        return {
            'name': self.name,
            'version': self.version,
            'description': self.description,
            'author': self.author,
            'module_path': self.module_path,
            'dependencies': self.dependencies,
            'entry_point': self.entry_point,
            'state': self.state.value,
            'error_message': self.error_message,
            'loaded_at': self.loaded_at,
        }


class Plugin(ABC):
    """
    插件抽象基类

    所有AuroraNLP插件都需要继承此类并实现必要的方法。
    """

    def __init__(self):
        """初始化插件"""
        self._info: Optional[PluginInfo] = None
        self._pipeline: Optional[Pipeline] = None
        self._config: Dict[str, Any] = {}
        self._state = PluginState.DISCOVERED

    @property
    def info(self) -> PluginInfo:
        """获取插件信息"""
        if self._info is None:
            raise RuntimeError("插件信息未设置")
        return self._info

    @property
    def state(self) -> PluginState:
        """获取插件状态"""
        return self._state

    @property
    def name(self) -> str:
        """获取插件名称"""
        return self._info.name if self._info else "unknown"

    @abstractmethod
    def get_info(self) -> PluginInfo:
        """
        返回插件信息（子类必须实现）

        Returns:
            PluginInfo实例
        """
        pass

    def on_load(self) -> None:
        """插件加载时调用"""
        pass

    def on_initialize(self, pipeline: Pipeline, config: Optional[Dict[str, Any]] = None) -> None:
        """
        插件初始化时调用

        Args:
            pipeline: Pipeline实例
            config: 插件配置
        """
        self._pipeline = pipeline
        self._config = config or {}
        self._state = PluginState.INITIALIZED

    def on_enable(self) -> None:
        """插件启用时调用"""
        self._state = PluginState.ENABLED

    def on_disable(self) -> None:
        """插件禁用时调用"""
        self._state = PluginState.DISABLED

    def on_destroy(self) -> None:
        """插件销毁时调用"""
        pass

    def register_components(self, pipeline: Pipeline) -> None:
        """
        向Pipeline注册组件

        Args:
            pipeline: Pipeline实例
        """
        pass

    def unregister_components(self, pipeline: Pipeline) -> None:
        """
        从Pipeline注销组件

        Args:
            pipeline: Pipeline实例
        """
        pass


class PluginDependency:
    """插件依赖描述"""

    def __init__(self, name: str, min_version: str = "", max_version: str = ""):
        """
        初始化依赖描述

        Args:
            name: 依赖名称
            min_version: 最低版本
            max_version: 最高版本
        """
        self.name = name
        self.min_version = min_version
        self.max_version = max_version

    def is_compatible(self, version: str) -> bool:
        """
        检查版本是否兼容

        Args:
            version: 待检查版本

        Returns:
            是否兼容
        """
        if not self.min_version and not self.max_version:
            return True
        try:
            v = ModelVersion.parse(version)
            if self.min_version:
                min_v = ModelVersion.parse(self.min_version)
                if v < min_v:
                    return False
            if self.max_version:
                max_v = ModelVersion.parse(self.max_version)
                if v > max_v:
                    return False
            return True
        except ValueError:
            return False


class PluginManager:
    """
    插件管理器 - 管理插件的完整生命周期

    特性：
    - 插件发现和加载
    - 依赖管理
    - 生命周期控制
    - 插件注册表
    - 热加载/热卸载
    """

    def __init__(self, pipeline: Optional[Pipeline] = None):
        """
        初始化插件管理器

        Args:
            pipeline: 关联的Pipeline
        """
        self._pipeline = pipeline
        self._plugins: Dict[str, Plugin] = {}
        self._plugin_infos: Dict[str, PluginInfo] = {}
        self._plugin_dirs: List[str] = []
        self._lock = threading.RLock()
        self._hooks: Dict[str, List[Callable]] = {}

    @property
    def pipeline(self) -> Optional[Pipeline]:
        """获取关联的Pipeline"""
        return self._pipeline

    @pipeline.setter
    def pipeline(self, value: Optional[Pipeline]) -> None:
        """设置关联的Pipeline"""
        self._pipeline = value

    def add_plugin_dir(self, directory: str) -> None:
        """
        添加插件搜索目录

        Args:
            directory: 目录路径
        """
        if directory not in self._plugin_dirs:
            self._plugin_dirs.append(directory)

    def discover_plugins(self) -> List[PluginInfo]:
        """
        发现可用插件

        扫描插件目录，查找可用的插件模块。

        Returns:
            发现的插件信息列表
        """
        discovered = []
        for plugin_dir in self._plugin_dirs:
            if not os.path.isdir(plugin_dir):
                continue

            # 查找插件目录下的Python模块
            for filename in os.listdir(plugin_dir):
                if filename.endswith('.py') and not filename.startswith('_'):
                    module_name = filename[:-3]
                    module_path = os.path.join(plugin_dir, filename)
                    info = PluginInfo(
                        name=module_name,
                        module_path=module_path,
                        entry_point=module_name
                    )
                    self._plugin_infos[module_name] = info
                    discovered.append(info)

                # 查找插件包目录
                elif os.path.isdir(os.path.join(plugin_dir, filename)):
                    init_path = os.path.join(plugin_dir, filename, '__init__.py')
                    if os.path.exists(init_path):
                        info = PluginInfo(
                            name=filename,
                            module_path=init_path,
                            entry_point=filename
                        )
                        self._plugin_infos[filename] = info
                        discovered.append(info)

        return discovered

    def load_plugin(self, name: str, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        加载插件

        Args:
            name: 插件名称
            config: 插件配置

        Returns:
            是否加载成功
        """
        with self._lock:
            if name in self._plugins:
                return True

            info = self._plugin_infos.get(name)
            if info is None:
                return False

            try:
                # 动态导入插件模块
                if not info.module_path:
                    return False

                module_dir = os.path.dirname(info.module_path)
                module_file = os.path.basename(info.module_path)

                if module_dir and module_dir not in sys.path:
                    sys.path.insert(0, module_dir)

                if module_file.endswith('__init__.py'):
                    module_name = name
                else:
                    module_name = module_file[:-3] if module_file.endswith('.py') else module_file

                module = importlib.import_module(module_name)

                # 查找Plugin子类
                plugin_instance = None
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (inspect.isclass(attr) and
                            issubclass(attr, Plugin) and
                            attr is not Plugin):
                        plugin_instance = attr()
                        break

                if plugin_instance is None:
                    info.state = PluginState.ERROR
                    info.error_message = "未找到Plugin子类"
                    return False

                # 设置插件信息
                plugin_instance._info = info
                plugin_instance._info.state = PluginState.LOADED
                plugin_instance._info.loaded_at = time.time()

                # 调用加载钩子
                plugin_instance.on_load()

                self._plugins[name] = plugin_instance

                # 触发钩子
                self._trigger_hook('plugin_loaded', plugin=plugin_instance)

                return True

            except Exception as e:
                info.state = PluginState.ERROR
                info.error_message = str(e)
                return False

    def unload_plugin(self, name: str) -> bool:
        """
        卸载插件

        Args:
            name: 插件名称

        Returns:
            是否卸载成功
        """
        with self._lock:
            plugin = self._plugins.get(name)
            if plugin is None:
                return False

            try:
                plugin.on_disable()
                plugin.on_destroy()

                if self._pipeline:
                    plugin.unregister_components(self._pipeline)

                del self._plugins[name]
                self._trigger_hook('plugin_unloaded', name=name)
                return True
            except Exception:
                return False

    def initialize_plugin(self, name: str, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        初始化插件

        Args:
            name: 插件名称
            config: 插件配置

        Returns:
            是否初始化成功
        """
        plugin = self._plugins.get(name)
        if plugin is None:
            return False

        try:
            if self._pipeline:
                plugin.on_initialize(self._pipeline, config)
                plugin.register_components(self._pipeline)
            else:
                plugin.on_initialize(None, config)

            plugin.on_enable()
            self._trigger_hook('plugin_initialized', plugin=plugin)
            return True
        except Exception as e:
            plugin._state = PluginState.ERROR
            plugin._info.error_message = str(e)
            return False

    def enable_plugin(self, name: str) -> bool:
        """
        启用插件

        Args:
            name: 插件名称

        Returns:
            是否启用成功
        """
        plugin = self._plugins.get(name)
        if plugin is None:
            return False
        try:
            plugin.on_enable()
            self._trigger_hook('plugin_enabled', plugin=plugin)
            return True
        except Exception:
            return False

    def disable_plugin(self, name: str) -> bool:
        """
        禁用插件

        Args:
            name: 插件名称

        Returns:
            是否禁用成功
        """
        plugin = self._plugins.get(name)
        if plugin is None:
            return False
        try:
            plugin.on_disable()
            self._trigger_hook('plugin_disabled', plugin=plugin)
            return True
        except Exception:
            return False

    def get_plugin(self, name: str) -> Optional[Plugin]:
        """
        获取插件实例

        Args:
            name: 插件名称

        Returns:
            插件实例
        """
        return self._plugins.get(name)

    def list_plugins(self) -> List[PluginInfo]:
        """列出所有已加载的插件信息"""
        return [p.info for p in self._plugins.values()]

    def list_all_infos(self) -> List[PluginInfo]:
        """列出所有已知插件信息（包括未加载的）"""
        return list(self._plugin_infos.values())

    def check_dependencies(self, name: str) -> Tuple[bool, List[str]]:
        """
        检查插件依赖

        Args:
            name: 插件名称

        Returns:
            (是否满足, 缺少的依赖列表)
        """
        info = self._plugin_infos.get(name)
        if info is None:
            return False, [name]

        missing = []
        for dep_name in info.dependencies:
            if dep_name not in self._plugins:
                missing.append(dep_name)

        return len(missing) == 0, missing

    def register_hook(self, event: str, callback: Callable) -> None:
        """
        注册事件钩子

        Args:
            event: 事件名称
            callback: 回调函数
        """
        if event not in self._hooks:
            self._hooks[event] = []
        self._hooks[event].append(callback)

    def _trigger_hook(self, event: str, **kwargs) -> None:
        """触发事件钩子"""
        for callback in self._hooks.get(event, []):
            try:
                callback(**kwargs)
            except Exception:
                pass

    def load_all(self, config: Optional[Dict[str, Any]] = None) -> int:
        """
        加载所有已发现的插件

        Args:
            config: 全局配置

        Returns:
            成功加载的插件数量
        """
        loaded = 0
        for name in list(self._plugin_infos.keys()):
            if self.load_plugin(name, config):
                loaded += 1
        return loaded

    def initialize_all(self, config: Optional[Dict[str, Any]] = None) -> int:
        """
        初始化所有已加载的插件

        Args:
            config: 全局配置

        Returns:
            成功初始化的插件数量
        """
        initialized = 0
        for name in list(self._plugins.keys()):
            if self.initialize_plugin(name, config):
                initialized += 1
        return initialized

    def stats(self) -> Dict[str, Any]:
        """获取插件统计信息"""
        states = {}
        for plugin in self._plugins.values():
            state = plugin.state.value
            states[state] = states.get(state, 0) + 1
        return {
            'total_discovered': len(self._plugin_infos),
            'total_loaded': len(self._plugins),
            'states': states,
            'plugin_dirs': self._plugin_dirs,
        }


# ============================================================
# 示例组件 - 用于演示Pipeline的使用
# ============================================================

class SimpleTokenizerComponent(PipelineComponent):
    """
    简单分词组件示例

    基于最大正向匹配的简单分词器，用于演示Pipeline组件的编写方式。
    """

    def __init__(self, name: str = "simple_tokenizer", dictionary: Optional[Set[str]] = None):
        """
        初始化分词组件

        Args:
            name: 组件名称
            dictionary: 词典集合
        """
        super().__init__(name=name)
        self._dictionary = dictionary or set()
        self._max_word_len = 5
        self.provide('tokens')

    def on_initialize(self) -> None:
        """初始化时加载默认词典"""
        if not self._dictionary:
            self._dictionary = {
                '中国', '人民', '共和国', '北京', '上海', '广州', '深圳',
                '人工智能', '自然语言', '处理', '技术', '机器学习',
                '深度学习', '神经网络', '数据', '科学', '计算机',
            }

    def process(self, doc: Doc) -> Doc:
        """
        对文档进行分词

        Args:
            doc: 输入文档

        Returns:
            添加了分词结果的文档
        """
        text = doc.text
        tokens = []
        i = 0
        while i < len(text):
            matched = False
            for length in range(min(self._max_word_len, len(text) - i), 0, -1):
                word = text[i:i + length]
                if word in self._dictionary:
                    token = Token(
                        doc=doc,
                        start=i,
                        end=i + length,
                        string_store=doc.string_store
                    )
                    tokens.append(token)
                    i += length
                    matched = True
                    break
            if not matched:
                token = Token(
                    doc=doc,
                    start=i,
                    end=i + 1,
                    string_store=doc.string_store
                )
                tokens.append(token)
                i += 1

        for token in tokens:
            doc.add_token(token)

        return doc


class POSTaggerComponent(PipelineComponent):
    """
    简单词性标注组件示例

    为已分词的文档添加简单的词性标注。
    """

    def __init__(self, name: str = "pos_tagger"):
        """
        初始化词性标注组件

        Args:
            name: 组件名称
        """
        super().__init__(name=name)
        self.require('tokens')
        self.provide('pos_tags')

    def process(self, doc: Doc) -> Doc:
        """
        为文档中的词元添加词性标注

        Args:
            doc: 输入文档

        Returns:
            添加了词性标注的文档
        """
        for token in doc.tokens:
            text = token.text
            # 简单的词性标注规则
            if len(text) == 1 and text in '，。！？、；：""''（）【】':
                token.pos = 'w'
            elif text.isdigit():
                token.pos = 'm'
            elif any(c.isdigit() for c in text):
                token.pos = 'm'
            elif len(text) >= 2 and text[-1] in '们':
                token.pos = 'r'
            elif len(text) >= 2 and text[-1] in '的':
                token.pos = 'u'
            elif len(text) >= 2 and text[-1] in '了着过':
                token.pos = 'v'
            elif len(text) == 1:
                token.pos = 'x'
            else:
                token.pos = 'n'

        return doc


# ============================================================
# 模块导出
# ============================================================

__all__ = [
    # 步骤58: 词汇表共享
    "StringStore",
    "get_string_store",
    # 步骤55: Doc对象
    "Doc",
    # 步骤56: Span对象
    "Span",
    # 步骤57: Token对象
    "Token",
    # 步骤51: Pipeline架构
    "ComponentState",
    "PipelineComponent",
    "ConditionalBranch",
    "Pipeline",
    # 步骤52: 组件注册
    "ComponentRegistry",
    "get_registry",
    "register_component",
    # 步骤53: 配置系统
    "ConfigValidationError",
    "ConfigSchema",
    "PipelineConfig",
    # 步骤54: 组件冻结
    "FreezableParams",
    # 步骤59: 模型版本管理
    "VersionInfo",
    "ModelVersion",
    "ModelLifecycle",
    # 步骤60: 模型缓存
    "LRUCache",
    "ModelCache",
    # 步骤61: RESTful API
    "APIRequest",
    "APIResponse",
    "RequestValidator",
    "Route",
    "APIServer",
    # 步骤62: gRPC接口
    "RPCMessage",
    "RPCError",
    "RPCStatusCode",
    "RPCServiceDescriptor",
    "RPCMethodHandler",
    "RPCService",
    "RPCServer",
    "RPCClient",
    # 步骤63: 异步处理
    "AsyncPipeline",
    "AsyncTaskManager",
    # 步骤64: 流式处理
    "ProgressCallback",
    "StreamProcessor",
    # 步骤65: 插件系统
    "PluginState",
    "PluginInfo",
    "Plugin",
    "PluginDependency",
    "PluginManager",
    # 示例组件
    "SimpleTokenizerComponent",
    "POSTaggerComponent",
]