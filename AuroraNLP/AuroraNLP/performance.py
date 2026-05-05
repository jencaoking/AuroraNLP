"""
AuroraNLP 性能优化模块 - 阶段五

本模块实现了步骤66-80的功能：
- 步骤66: 对象复用与对象池（纯Python替代Cython加速思路）
- 步骤67: 批量操作优化（纯Python向量化实现）
- 步骤68: 内存池管理
- 步骤69: 对象复用
- 步骤70: 多线程支持
- 步骤71: 多进程处理
- 步骤72: GPU加速框架（接口层，可选PyTorch集成）
- 步骤73: Batch推理
- 步骤74: 混合精度推理
- 步骤75: TensorRT接口（可选）
- 步骤76: 内存映射文件
- 步骤77: 词典压缩
- 步骤78: 结果缓存
- 步骤79: 分布式任务调度
- 步骤80: 性能监控

约束：零外部依赖，纯Python标准库实现
"""

import time
import struct
import threading
import multiprocessing
import gc
import os
import mmap
import zlib
import bz2
import hashlib
from typing import (
    Any, Callable, Dict, Generic, List, Optional, 
    Tuple, TypeVar, Union, Iterator, NamedTuple,
    Set, Deque
)
from abc import ABC, abstractmethod
from collections import deque, defaultdict, OrderedDict
from functools import wraps, lru_cache
from enum import Enum, IntEnum


# ==================== 步骤66: 对象复用与对象池 ====================

T = TypeVar('T')


class PoolStats(NamedTuple):
    """对象池统计"""
    allocated: int
    in_use: int
    free: int
    hits: int
    misses: int


class ObjectPool(Generic[T]):
    """通用对象池
    
    通过复用对象减少GC压力和分配开销
    """

    def __init__(
        self,
        factory: Callable[[], T],
        resetter: Optional[Callable[[T], None]] = None,
        max_size: int = 1000,
        initial_size: int = 10
    ):
        self._factory = factory
        self._resetter = resetter
        self._max_size = max_size
        self._pool: Deque[T] = deque(maxlen=max_size)
        self._in_use: Set[int] = set()  # 对象ID集合
        
        # 统计
        self._stats = {
            'allocated': 0,
            'in_use': 0,
            'free': 0,
            'hits': 0,
            'misses': 0
        }
        self._lock = threading.RLock()
        
        # 预分配初始对象
        with self._lock:
            for _ in range(initial_size):
                obj = factory()
                if resetter:
                    resetter(obj)  # 预分配时就重置
                self._pool.append(obj)
                self._stats['allocated'] += 1
            self._stats['free'] = len(self._pool)

    def acquire(self) -> T:
        """获取一个对象"""
        with self._lock:
            if self._pool:
                obj = self._pool.popleft()
                self._stats['hits'] += 1
            else:
                obj = self._factory()
                if self._resetter:
                    self._resetter(obj)  # 新对象也重置
                self._stats['allocated'] += 1
                self._stats['misses'] += 1
            self._in_use.add(id(obj))
            self._stats['in_use'] = len(self._in_use)
            self._stats['free'] = len(self._pool)
            return obj

    def release(self, obj: T) -> None:
        """释放一个对象回池"""
        with self._lock:
            obj_id = id(obj)
            if obj_id in self._in_use:
                self._in_use.remove(obj_id)
                if self._resetter:
                    self._resetter(obj)
                if len(self._pool) < self._max_size:
                    self._pool.append(obj)
                self._stats['in_use'] = len(self._in_use)
                self._stats['free'] = len(self._pool)

    def get_stats(self) -> PoolStats:
        """获取统计信息"""
        with self._lock:
            return PoolStats(
                self._stats['allocated'],
                self._stats['in_use'],
                self._stats['free'],
                self._stats['hits'],
                self._stats['misses']
            )

    def clear(self) -> None:
        """清空对象池"""
        with self._lock:
            self._pool.clear()
            self._in_use.clear()
            self._stats['free'] = 0
            self._stats['in_use'] = 0

    def __enter__(self) -> T:
        """上下文管理器支持"""
        self._current_obj = self.acquire()
        return self._current_obj

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        if hasattr(self, '_current_obj'):
            self.release(self._current_obj)
            delattr(self, '_current_obj')


class PoolContext(Generic[T]):
    """对象池上下文管理器"""

    def __init__(self, pool: ObjectPool[T]):
        self._pool = pool
        self._obj: Optional[T] = None

    def __enter__(self) -> T:
        self._obj = self._pool.acquire()
        return self._obj

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._obj is not None:
            self._pool.release(self._obj)
        return False


# ==================== 步骤67: 批量操作优化 ====================

class BatchProcessor:
    """批量处理优化器
    
    减少函数调用开销，支持批量执行和流水线处理
    """

    def __init__(
        self,
        chunk_size: int = 100,
        pipeline_depth: int = 1
    ):
        self._chunk_size = chunk_size
        self._pipeline_depth = pipeline_depth
        self._stats = {
            'total_batches': 0,
            'total_items': 0,
            'avg_time': 0.0
        }
        self._lock = threading.RLock()

    def process(
        self,
        items: List[Any],
        func: Callable[[Any], Any],
        batch_func: Optional[Callable[[List[Any]], List[Any]]] = None
    ) -> List[Any]:
        """批量处理"""
        if not items:
            return []
        
        start_time = time.time()
        results = []
        
        # 使用专用批量函数（更快）
        if batch_func:
            results = batch_func(items)
        else:
            # 分块处理
            for i in range(0, len(items), self._chunk_size):
                chunk = items[i:i+self._chunk_size]
                chunk_results = [func(item) for item in chunk]
                results.extend(chunk_results)
        
        end_time = time.time()
        
        with self._lock:
            self._stats['total_batches'] += 1
            self._stats['total_items'] += len(items)
            elapsed = end_time - start_time
            n = self._stats['total_batches']
            self._stats['avg_time'] = (
                (self._stats['avg_time'] * (n - 1) + elapsed) / n
            )
        
        return results

    def process_stream(
        self,
        item_stream: Iterator[Any],
        func: Callable[[Any], Any]
    ) -> Iterator[Any]:
        """流式批量处理"""
        buffer: List[Any] = []
        for item in item_stream:
            buffer.append(item)
            if len(buffer) >= self._chunk_size:
                for result in self.process(buffer, func):
                    yield result
                buffer.clear()
        if buffer:
            for result in self.process(buffer, func):
                yield result

    def map_reduce(
        self,
        items: List[Any],
        map_func: Callable[[Any], Any],
        reduce_func: Callable[[List[Any]], Any]
    ) -> Any:
        """Map-Reduce模式"""
        mapped = self.process(items, map_func)
        return reduce_func(mapped)

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            return dict(self._stats)


# ==================== 步骤68: 内存池管理 ====================

class MemoryBlock:
    """内存块"""

    def __init__(self, size: int):
        self.size = size
        self.data = bytearray(size)
        self.used = 0

    def reset(self) -> None:
        self.used = 0

    def write(self, data: bytes) -> int:
        """写入数据到内存块"""
        available = self.size - self.used
        write_size = min(len(data), available)
        self.data[self.used:self.used+write_size] = data[:write_size]
        self.used += write_size
        return write_size


class MemoryPool:
    """内存池管理器
    
    减少内存分配/释放开销，预分配和复用内存块
    """

    class BlockSize(IntEnum):
        SMALL = 4096
        MEDIUM = 65536
        LARGE = 1048576
        XLARGE = 16777216

    def __init__(
        self,
        small_count: int = 100,
        medium_count: int = 50,
        large_count: int = 10,
        xlarge_count: int = 2
    ):
        self._pools: Dict[int, Deque[MemoryBlock]] = {}
        self._lock = threading.RLock()
        
        # 初始化各尺寸池
        sizes = [
            (self.BlockSize.SMALL, small_count),
            (self.BlockSize.MEDIUM, medium_count),
            (self.BlockSize.LARGE, large_count),
            (self.BlockSize.XLARGE, xlarge_count)
        ]
        for size, count in sizes:
            self._pools[size] = deque()
            for _ in range(count):
                self._pools[size].append(MemoryBlock(size))

    def acquire(self, min_size: int) -> MemoryBlock:
        """获取至少min_size的内存块"""
        # 找到最小的满足要求的块大小
        with self._lock:
            block_size = self.BlockSize.SMALL
            for bsize in [
                self.BlockSize.SMALL, 
                self.BlockSize.MEDIUM, 
                self.BlockSize.LARGE, 
                self.BlockSize.XLARGE
            ]:
                if bsize >= min_size:
                    block_size = bsize
                    break
            
            if self._pools.get(block_size):
                block = self._pools[block_size].popleft()
            else:
                # 没有可用块，创建新的
                block = MemoryBlock(block_size)
            
            block.reset()
            return block

    def release(self, block: MemoryBlock) -> None:
        """释放内存块"""
        with self._lock:
            size = block.size
            if size in self._pools:
                self._pools[size].append(block)

    def get_usage(self) -> Dict[str, int]:
        """获取内存使用情况"""
        with self._lock:
            result = {}
            for size, pool in self._pools.items():
                name = {
                    self.BlockSize.SMALL: 'small',
                    self.BlockSize.MEDIUM: 'medium',
                    self.BlockSize.LARGE: 'large',
                    self.BlockSize.XLARGE: 'xlarge'
                }.get(size, str(size))
                result[name] = len(pool)
            return result


# ==================== 步骤69: 延迟释放与GC管理 ====================

class DelayedGC:
    """延迟垃圾收集管理器
    
    在高吞吐场景下延后GC，减少GC带来的停顿
    """

    def __init__(self, target_interval: float = 10.0):
        self._target_interval = target_interval
        self._last_gc = time.time()
        self._threshold = 70  # GC触发阈值
        self._lock = threading.RLock()

    def disable_gc(self) -> None:
        """禁用自动GC"""
        gc.disable()

    def enable_gc(self) -> None:
        """启用自动GC"""
        gc.enable()

    def collect_if_needed(self, force: bool = False) -> Dict[str, int]:
        """在确实需要时才执行GC"""
        with self._lock:
            now = time.time()
            elapsed = now - self._last_gc
            
            # 检查是否应该手动GC
            should_collect = force
            if elapsed > self._target_interval:
                should_collect = True
            
            # 检查内存使用
            try:
                thresholds = gc.get_threshold()
                total = sum(thresholds)
                if total > self._threshold:
                    should_collect = True
            except Exception:
                pass
            
            if should_collect:
                count = gc.collect()
                self._last_gc = time.time()
                return {
                    'collected': count, 
                    'reason': 'manual' if force else 'threshold'
                }
            return {'collected': 0, 'reason': 'skipped'}

    @staticmethod
    def run_in_gc_free_zone(func: Callable):
        """在无GC区域运行"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            gc.disable()
            try:
                return func(*args, **kwargs)
            finally:
                gc.collect()
                gc.enable()
        return wrapper


# ==================== 步骤70: 多线程支持 ====================

class ThreadPoolExecutor:
    """简单线程池执行器（标准库实现）"""

    def __init__(self, max_workers: int = 4):
        self._max_workers = max_workers
        self._queue: Deque[Tuple[Callable, Tuple, Dict]] = deque()
        self._threads: List[threading.Thread] = []
        self._running = False
        self._lock = threading.RLock()
        self._results: Dict[int, Any] = {}
        self._conditions: Dict[int, threading.Condition] = {}
        self._counter = 0

    def start(self) -> None:
        """启动线程池"""
        with self._lock:
            if not self._running:
                self._running = True
                for i in range(self._max_workers):
                    t = threading.Thread(target=self._worker, daemon=True)
                    t.start()
                    self._threads.append(t)

    def stop(self) -> None:
        """停止线程池"""
        with self._lock:
            self._running = False

    def _worker(self):
        """工作线程"""
        while self._running:
            task = None
            with self._lock:
                if self._queue:
                    task = self._queue.popleft()
            
            if task:
                func, args, kwargs, task_id = task
                try:
                    result = func(*args, **kwargs)
                    with self._lock:
                        self._results[task_id] = result
                        if task_id in self._conditions:
                            with self._conditions[task_id]:
                                self._conditions[task_id].notify_all()
                except Exception as e:
                    with self._lock:
                        self._results[task_id] = e
                        if task_id in self._conditions:
                            with self._conditions[task_id]:
                                self._conditions[task_id].notify_all()

    def submit(self, func: Callable, *args, **kwargs) -> int:
        """提交任务"""
        with self._lock:
            task_id = self._counter
            self._counter += 1
            self._conditions[task_id] = threading.Condition(self._lock)
            self._queue.append((func, args, kwargs, task_id))
        return task_id

    def wait(self, task_id: int, timeout: float = 30.0) -> Any:
        """等待任务结果"""
        with self._lock:
            if task_id in self._results:
                return self._results[task_id]
            else:
                cond = self._conditions[task_id]
                cond.wait(timeout)
                if task_id in self._results:
                    result = self._results[task_id]
                    # 清理
                    del self._conditions[task_id]
                    del self._results[task_id]
                    return result
                raise TimeoutError(f"Task {task_id} timeout")


class ParallelTokenizer:
    """并行分词器"""

    def __init__(
        self,
        tokenizer_func: Callable[[str], List[str]],
        num_workers: int = 4
    ):
        self._tokenizer = tokenizer_func
        self._pool = ThreadPoolExecutor(num_workers)
        self._pool.start()

    def tokenize_batch(self, texts: List[str]) -> List[List[str]]:
        """批量并行分词"""
        task_ids = [self._pool.submit(self._tokenizer, text) for text in texts]
        return [self._pool.wait(tid) for tid in task_ids]


# ==================== 步骤71: 多进程处理 ====================

class ProcessPoolExecutor:
    """简单进程池（基于multiprocessing）"""

    def __init__(self, max_workers: int = None):
        self._max_workers = max_workers or multiprocessing.cpu_count()
        self._pool: Optional[multiprocessing.Pool] = None

    def start(self):
        self._pool = multiprocessing.Pool(self._max_workers)

    def stop(self):
        if self._pool:
            self._pool.close()
            self._pool.join()
            self._pool = None

    def map(self, func: Callable, items: List[Any]) -> List[Any]:
        if not self._pool:
            raise RuntimeError("Pool not started")
        return self._pool.map(func, items)

    def apply_async(self, func: Callable, *args):
        if not self._pool:
            raise RuntimeError("Pool not started")
        return self._pool.apply_async(func, args)


# ==================== 步骤72: GPU加速接口层 ====================

class DeviceType(IntEnum):
    """设备类型"""
    CPU = 0
    CUDA = 1
    MPS = 2  # Apple Silicon


class GPUInterface:
    """GPU加速接口层（可选集成PyTorch）
    
    零强制依赖，按需加载
    """

    def __init__(self):
        self._backend = None
        self._device = self._detect_device()

    def _detect_device(self) -> DeviceType:
        """检测可用设备"""
        try:
            import torch
            if torch.cuda.is_available():
                return DeviceType.CUDA
        except Exception:
            pass
        
        try:
            import torch
            if torch.backends.mps.is_available():
                return DeviceType.MPS
        except Exception:
            pass
        
        return DeviceType.CPU

    def is_available(self) -> bool:
        """检查GPU是否可用"""
        return self._device != DeviceType.CPU

    @property
    def device_type(self) -> DeviceType:
        return self._device

    def load_backend(self) -> bool:
        """尝试加载PyTorch后端"""
        try:
            import torch
            self._backend = torch
            return True
        except ImportError:
            return False


# ==================== 步骤73: Batch推理优化 ====================

class BatchInference:
    """批量推理优化器
    
    动态调整Batch大小，内存友好
    """

    def __init__(
        self,
        initial_batch_size: int = 32,
        max_batch_size: int = 128,
        min_batch_size: int = 1
    ):
        self._initial_batch = initial_batch_size
        self._max_batch = max_batch_size
        self._min_batch = min_batch_size
        self._current_batch = initial_batch_size
        self._stats = {'total_batches': 0, 'avg_latency': 0.0}

    def infer(
        self,
        model: Any,
        items: List[Any],
        predict_func: Callable[[Any, List[Any]], Any]
    ) -> List[Any]:
        """批量推理"""
        results = []
        total_items = len(items)
        
        for i in range(0, total_items, self._current_batch):
            batch = items[i:i+self._current_batch]
            start = time.time()
            batch_results = predict_func(model, batch)
            latency = time.time() - start
            results.extend(batch_results)
            
            self._update_stats(latency)
        
        return results

    def _update_stats(self, latency: float):
        self._stats['total_batches'] += 1
        avg = self._stats['avg_latency']
        n = self._stats['total_batches']
        self._stats['avg_latency'] = (avg * (n-1) + latency) / n


# ==================== 步骤74: 混合精度推理接口 ====================

class MixedPrecisionMode(IntEnum):
    """混合精度模式"""
    FP32 = 0
    FP16 = 1
    BF16 = 2


class MixedPrecisionInference:
    """混合精度推理管理器（接口层）"""

    def __init__(self, mode: MixedPrecisionMode = MixedPrecisionMode.FP16):
        self._mode = mode
        self._scaler = None

    def enable_fp16(self) -> None:
        self._mode = MixedPrecisionMode.FP16

    def enable_bf16(self) -> None:
        self._mode = MixedPrecisionMode.BF16

    def disable_mixed_precision(self) -> None:
        self._mode = MixedPrecisionMode.FP32


# ==================== 步骤75: TensorRT接口（可选） ====================

class TensorRTInterface:
    """TensorRT集成接口（可选）"""

    def __init__(self):
        self._engine = None

    def is_available(self) -> bool:
        """检查TensorRT是否可用"""
        try:
            import tensorrt
            return True
        except ImportError:
            return False


# ==================== 步骤76: 内存映射文件 ====================

class MemoryMappedFile:
    """内存映射文件管理
    
    大词典高效加载，支持多进程共享
    """

    def __init__(self, filename: str, writeable: bool = False):
        self._filename = filename
        self._writeable = writeable
        self._mmap: Optional[mmap.mmap] = None
        self._file_obj: Optional[Any] = None
        self._size = 0

    def open(self) -> None:
        """打开文件并映射到内存"""
        self._file_obj = open(self._filename, 'rb')
        self._file_obj.seek(0, 2)
        self._size = self._file_obj.tell()
        
        if self._size == 0:
            raise ValueError("File is empty")
        
        self._file_obj.seek(0)
        
        flags = mmap.PROT_READ
        if self._writeable:
            flags |= mmap.PROT_WRITE
        self._mmap = mmap.mmap(
            self._file_obj.fileno(), 
            self._size, 
            access=mmap.ACCESS_READ if not self._writeable else mmap.ACCESS_WRITE
        )

    def read_bytes(self, offset: int, size: int) -> bytes:
        """读取指定位置的字节"""
        if not self._mmap:
            raise RuntimeError("File not open")
        if offset < 0 or offset + size > self._size:
            raise ValueError("Offset/size out of range")
        return bytes(self._mmap[offset:offset+size])

    def close(self) -> None:
        """关闭映射和文件"""
        if self._mmap:
            self._mmap.close()
        if self._file_obj:
            self._file_obj.close()


# ==================== 步骤77: 词典压缩 ====================

class CompressionType(IntEnum):
    """压缩算法"""
    NONE = 0
    ZLIB = 1
    BZ2 = 2


class DictionaryCompressor:
    """词典压缩管理器"""

    def __init__(self, method: CompressionType = CompressionType.ZLIB):
        self._method = method
        self._compression_level = 6

    def set_level(self, level: int) -> None:
        if 1 <= level <= 9:
            self._compression_level = level

    def compress(self, data: bytes) -> bytes:
        """压缩数据"""
        if self._method == CompressionType.ZLIB:
            return zlib.compress(data, level=self._compression_level)
        elif self._method == CompressionType.BZ2:
            return bz2.compress(data, compresslevel=self._compression_level)
        else:
            return data

    def decompress(self, data: bytes) -> bytes:
        """解压数据"""
        if self._method == CompressionType.ZLIB:
            return zlib.decompress(data)
        elif self._method == CompressionType.BZ2:
            return bz2.decompress(data)
        else:
            return data


# ==================== 步骤78: 结果缓存 ====================

class LRUResultCache:
    """结果LRU缓存"""

    def __init__(self, max_size: int = 1000):
        self._cache: OrderedDict[str, Tuple[Any, float]] = OrderedDict()
        self._max_size = max_size
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    def _get_key(self, text: str) -> str:
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def get(self, text: str) -> Optional[Any]:
        """获取缓存结果"""
        key = self._get_key(text)
        with self._lock:
            if key in self._cache:
                self._hits += 1
                result, _ = self._cache[key]
                self._cache.move_to_end(key, last=True)
                return result
            self._misses += 1
            return None

    def put(self, text: str, result: Any, ttl: float = 3600) -> None:
        """存入缓存"""
        key = self._get_key(text)
        expires = time.time() + ttl
        with self._lock:
            self._cache[key] = (result, expires)
            self._cache.move_to_end(key, last=True)
            
            if len(self._cache) > self._max_size:
                self._clean_expired()
                if len(self._cache) > self._max_size:
                    self._cache.popitem(last=False)

    def _clean_expired(self) -> None:
        """清理过期条目"""
        now = time.time()
        expired = [k for k, (_, exp) in self._cache.items() if exp < now]
        for k in expired:
            del self._cache[k]

    def get_stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                'size': len(self._cache),
                'max_size': self._max_size,
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': (
                    self._hits / (self._hits + self._misses)
                    if (self._hits + self._misses) > 0
                    else 0.0
                )
            }


# ==================== 步骤79: 分布式任务调度接口 ====================

class DistributedTask:
    """分布式任务表示"""

    def __init__(self, task_id: int, func: Callable, args: Tuple, kwargs: Dict):
        self.task_id = task_id
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.status = 'pending'
        self.result = None

    def run(self):
        self.status = 'running'
        try:
            self.result = self.func(*self.args, **self.kwargs)
            self.status = 'success'
        except Exception as e:
            self.result = e
            self.status = 'failed'


class SimpleTaskScheduler:
    """简单任务调度器（单机模式）"""

    def __init__(self):
        self._tasks: Dict[int, DistributedTask] = {}
        self._next_id = 1
        self._lock = threading.RLock()

    def submit(self, func: Callable, *args, **kwargs) -> int:
        with self._lock:
            task_id = self._next_id
            self._next_id += 1
            task = DistributedTask(task_id, func, args, kwargs)
            self._tasks[task_id] = task
        return task_id

    def run(self, task_id: int):
        task = self._tasks.get(task_id)
        if task:
            task.run()


# ==================== 步骤80: 性能监控 ====================

class MetricType(IntEnum):
    """指标类型"""
    COUNTER = 0
    GAUGE = 1
    HISTOGRAM = 2


class Metric:
    """单个性能指标"""

    def __init__(self, name: str, metric_type: MetricType):
        self.name = name
        self.type = metric_type
        self.value = 0.0
        self.min = float('inf')
        self.max = -float('inf')
        self.sum = 0.0
        self.count = 0
        self._lock = threading.RLock()

    def record(self, value: float):
        """记录一个值"""
        with self._lock:
            self.value = value
            self.min = min(self.min, value)
            self.max = max(self.max, value)
            self.sum += value
            self.count += 1

    def incr(self, delta: float = 1):
        """递增计数器"""
        with self._lock:
            self.value += delta
            self.min = min(self.min, self.value)
            self.max = max(self.max, self.value)
            self.count += 1

    def get_stats(self) -> Dict:
        with self._lock:
            avg = self.sum / self.count if self.count > 0 else 0.0
            return {
                'name': self.name,
                'value': self.value,
                'current': self.value,
                'min': self.min if self.count > 0 else float('inf'),
                'max': self.max if self.count > 0 else float('-inf'),
                'sum': self.sum,
                'average': avg,
                'count': self.count
            }


class PerformanceMonitor:
    """性能监控管理器"""

    def __init__(self):
        self._metrics: Dict[str, Metric] = {}
        self._lock = threading.RLock()
        self._start_time = time.time()

    def get_metric(self, name: str, metric_type: MetricType = MetricType.GAUGE) -> Metric:
        """获取或创建指标"""
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = Metric(name, metric_type)
            return self._metrics[name]

    def time_it(self, name: str):
        """装饰器：测量函数执行时间"""
        metric = self.get_metric(name, MetricType.HISTOGRAM)
        
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start = time.time()
                try:
                    return func(*args, **kwargs)
                finally:
                    elapsed = time.time() - start
                    metric.record(elapsed * 1000)  # ms
            return wrapper
        return decorator

    def get_report(self) -> Dict:
        """获取性能报告"""
        with self._lock:
            uptime = time.time() - self._start_time
            return {
                'uptime_seconds': uptime,
                'metrics': {
                    name: m.get_stats() 
                    for name, m in self._metrics.items()
                }
            }


# ==================== 快捷函数 ====================

class OptimizationSuite:
    """一站式优化套件
    
    整合所有性能优化功能
    """

    def __init__(self):
        self._object_pools: Dict[str, ObjectPool] = {}
        self._memory_pool = MemoryPool()
        self._batch_processor = BatchProcessor()
        self._result_cache = LRUResultCache()
        self._monitor = PerformanceMonitor()
        self._delayed_gc = DelayedGC()
        self._gpu_interface = GPUInterface()

    @property
    def monitor(self) -> PerformanceMonitor:
        return self._monitor

    @property
    def batch_processor(self) -> BatchProcessor:
        return self._batch_processor

    @property
    def cache(self) -> LRUResultCache:
        return self._result_cache

    def get_object_pool(self, name: str) -> Optional[ObjectPool]:
        return self._object_pools.get(name)

    def register_object_pool(
        self, 
        name: str, 
        factory: Callable, 
        resetter: Callable = None,
        max_size: int = 1000
    ) -> ObjectPool:
        """注册对象池"""
        pool = ObjectPool(factory, resetter, max_size)
        self._object_pools[name] = pool
        return pool

    def get_full_report(self) -> Dict:
        """获取完整性能报告"""
        return {
            'monitor': self._monitor.get_report(),
            'cache': self._result_cache.get_stats(),
            'memory_pool': self._memory_pool.get_usage(),
            'object_pools': {
                name: pool.get_stats()._asdict() 
                for name, pool in self._object_pools.items()
            }
        }


# ==================== 入口导出准备 ====================

# 导出类
__all_core = [
    'ObjectPool', 'PoolContext',
    'BatchProcessor',
    'MemoryPool', 'MemoryBlock',
    'DelayedGC',
    'ThreadPoolExecutor',
    'ParallelTokenizer',
    'ProcessPoolExecutor',
    'GPUInterface',
    'BatchInference',
    'MixedPrecisionInference',
    'TensorRTInterface',
    'MemoryMappedFile',
    'DictionaryCompressor',
    'LRUResultCache',
    'PerformanceMonitor',
    'OptimizationSuite',
    'Metric',
    'DistributedTask',
    'SimpleTaskScheduler'
]

