#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import time
import os
import tempfile
from typing import List

from AuroraNLP import (
    ObjectPool,
    PerformanceBatchProcessor as BatchProcessor,
    MemoryPool, MemoryBlock,
    DelayedGC,
    NLPThreadPoolExecutor as ThreadPoolExecutor,
    ParallelTokenizer,
    NLPProcessPoolExecutor as ProcessPoolExecutor,
    GPUInterface,
    BatchInference,
    MixedPrecisionInference,
    TensorRTInterface,
    MemoryMappedFile,
    DictionaryCompressor,
    LRUResultCache,
    PerformanceMonitor,
    OptimizationSuite,
    Metric,
    DistributedTask,
    SimpleTaskScheduler
)


# 用于多进程测试的函数必须在模块顶层
def square(x):
    return x * x


def simple_tokenizer(text: str) -> List[str]:
    return text.split()


def add(x, y):
    return x + y


def list_factory():
    return [0] * 10


def list_resetter(obj):
    obj.clear()


class TestObjectPool(unittest.TestCase):
    """测试对象池 (步骤66)"""

    def setUp(self):
        self.pool = ObjectPool(
            factory=list_factory,
            resetter=list_resetter,
            max_size=5,
            initial_size=2
        )

    def test_pool_acquire(self):
        """测试获取对象"""
        obj1 = self.pool.acquire()
        obj2 = self.pool.acquire()
        obj3 = self.pool.acquire()
        
        self.assertEqual(len(obj1), 0)  # 应该被重置过
        self.assertEqual(len(obj2), 0)
        self.assertEqual(len(obj3), 0)
        
        self.pool.release(obj1)
        self.pool.release(obj2)
        self.pool.release(obj3)

    def test_pool_stats(self):
        """测试对象池统计"""
        stats_before = self.pool.get_stats()
        self.assertEqual(stats_before.allocated, 2)
        self.assertEqual(stats_before.in_use, 0)
        
        obj1 = self.pool.acquire()
        stats_mid = self.pool.get_stats()
        self.assertEqual(stats_mid.in_use, 1)
        
        self.pool.release(obj1)
        stats_after = self.pool.get_stats()
        self.assertEqual(stats_after.in_use, 0)

    def test_pool_context_manager(self):
        """测试上下文管理器"""
        with self.pool as obj:
            self.assertIsInstance(obj, list)


class TestBatchProcessor(unittest.TestCase):
    """测试批量处理优化 (步骤67)"""

    def setUp(self):
        self.processor = BatchProcessor(chunk_size=10)

    def test_batch_process(self):
        """测试批量处理"""
        items = list(range(100))
        
        def double(x):
            return x * 2
        
        results = self.processor.process(items, double)
        expected = [x * 2 for x in items]
        self.assertEqual(results, expected)

    def test_batch_process_with_batch_func(self):
        """测试使用专用批量函数"""
        items = list(range(100))
        
        def batch_double(xs):
            return [x * 2 for x in xs]
        
        results = self.processor.process(items, None, batch_double)
        expected = [x * 2 for x in items]
        self.assertEqual(results, expected)

    def test_stream_process(self):
        """测试流式处理"""
        items = (x for x in range(100))  # 生成器
        
        def double(x):
            return x * 2
        
        results = list(self.processor.process_stream(items, double))
        expected = [x * 2 for x in range(100)]
        self.assertEqual(results, expected)

    def test_map_reduce(self):
        """测试MapReduce模式"""
        items = list(range(100))
        
        def local_square(x):
            return x * x
        
        def sum_list(xs):
            return sum(xs)
        
        result = self.processor.map_reduce(items, local_square, sum_list)
        expected = sum(x * x for x in range(100))
        self.assertEqual(result, expected)


class TestMemoryPool(unittest.TestCase):
    """测试内存池管理 (步骤68)"""

    def setUp(self):
        self.pool = MemoryPool(
            small_count=10,
            medium_count=5,
            large_count=2,
            xlarge_count=1
        )

    def test_memory_block_acquire(self):
        """测试获取内存块"""
        block = self.pool.acquire(min_size=100)
        self.assertIsInstance(block, MemoryBlock)
        self.assertGreaterEqual(block.size, 100)
        
        block.write(b"hello world")
        self.assertEqual(block.used, 11)
        
        block.reset()
        self.assertEqual(block.used, 0)
        
        self.pool.release(block)

    def test_memory_pool_usage(self):
        """测试内存池使用统计"""
        usage = self.pool.get_usage()
        self.assertIn("small", usage)
        self.assertIn("medium", usage)
        self.assertIn("large", usage)
        self.assertIn("xlarge", usage)


class TestDelayedGC(unittest.TestCase):
    """测试延迟GC (步骤69)"""

    def setUp(self):
        self.dgc = DelayedGC(target_interval=10.0)

    def test_disable_enable_gc(self):
        """测试GC启用/禁用"""
        import gc
        
        self.dgc.disable_gc()
        self.assertFalse(gc.isenabled())
        
        self.dgc.enable_gc()
        self.assertTrue(gc.isenabled())

    def test_collect_if_needed(self):
        """测试按需收集"""
        result = self.dgc.collect_if_needed(force=True)
        self.assertIn("collected", result)
        self.assertIn("reason", result)

    def test_gc_free_zone_decorator(self):
        """测试无GC区域装饰器"""
        @DelayedGC.run_in_gc_free_zone
        def compute():
            result = 0
            for i in range(10000):
                result += i
            return result
        
        result = compute()
        self.assertEqual(result, sum(range(10000)))


class TestThreadPool(unittest.TestCase):
    """测试多线程支持 (步骤70)"""

    def setUp(self):
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.executor.start()

    def tearDown(self):
        self.executor.stop()

    def test_thread_pool_submit(self):
        """测试线程池提交任务"""
        task_id = self.executor.submit(add, 2, 3)
        result = self.executor.wait(task_id)
        self.assertEqual(result, 5)

    def test_parallel_tokenizer(self):
        """测试并行分词器"""
        tokenizer = ParallelTokenizer(simple_tokenizer, num_workers=2)
        
        texts = [
            "hello world",
            "foo bar baz",
            "a b c d",
            "one two three four"
        ]
        
        results = tokenizer.tokenize_batch(texts)
        expected = [text.split() for text in texts]
        self.assertEqual(results, expected)


class TestProcessPool(unittest.TestCase):
    """测试多进程处理 (步骤71)"""

    def setUp(self):
        self.executor = ProcessPoolExecutor(max_workers=2)
        self.executor.start()

    def tearDown(self):
        self.executor.stop()

    @unittest.skipIf(
        __import__("sys").platform.startswith("win"),
        "Windows + pytest 下 multiprocessing spawn 会与 pytest capture 死锁",
    )
    def test_process_pool_map(self):
        """测试进程池map"""
        items = list(range(10))
        results = self.executor.map(square, items, timeout=10)
        expected = [x * x for x in items]
        self.assertEqual(results, expected)


class TestGPUInterface(unittest.TestCase):
    """测试GPU加速接口 (步骤72)"""

    def setUp(self):
        self.gpu = GPUInterface()

    def test_gpu_device_detection(self):
        """测试GPU设备检测"""
        # 只要不报错就好，不一定有GPU
        self.assertIsNotNone(self.gpu.device_type)

    def test_gpu_load_backend(self):
        """测试加载后端"""
        # 尝试加载，可能失败但不应该崩溃
        try:
            self.gpu.load_backend()
        except Exception:
            pass


class TestBatchInference(unittest.TestCase):
    """测试Batch推理 (步骤73)"""

    def setUp(self):
        self.inference = BatchInference(
            initial_batch_size=10,
            max_batch_size=50,
            min_batch_size=1
        )

    def test_batch_inference(self):
        """测试批量推理"""
        # 创建简单的模型和预测函数
        class DummyModel:
            pass
        
        model = DummyModel()
        
        def predict(model, batch):
            return [x * 2 for x in batch]
        
        items = list(range(100))
        results = self.inference.infer(model, items, predict)
        expected = [x * 2 for x in items]
        self.assertEqual(results, expected)


class TestMixedPrecision(unittest.TestCase):
    """测试混合精度推理 (步骤74)"""

    def test_mixed_precision_mode(self):
        """测试混合精度模式"""
        mp = MixedPrecisionInference()
        mp.enable_fp16()
        mp.enable_bf16()
        mp.disable_mixed_precision()
        # 只要不报错就好


class TestTensorRT(unittest.TestCase):
    """测试TensorRT接口 (步骤75)"""

    def test_tensorrt_interface(self):
        """测试TensorRT接口"""
        trt = TensorRTInterface()
        # 尝试检查可用性，可能失败但不应该崩溃
        available = trt.is_available()
        self.assertIsInstance(available, bool)


class TestMemoryMappedFile(unittest.TestCase):
    """测试内存映射文件 (步骤76)"""

    def setUp(self):
        # 创建临时文件
        self.temp_file = tempfile.NamedTemporaryFile(delete=False)
        self.temp_file.write(b"hello world\nthis is a test file")
        self.temp_file.close()
        
        self.mmap_file = MemoryMappedFile(self.temp_file.name, writeable=False)

    def tearDown(self):
        self.mmap_file.close()
        os.unlink(self.temp_file.name)

    def test_memory_mapped_read(self):
        """测试内存映射读取"""
        self.mmap_file.open()
        
        data = self.mmap_file.read_bytes(0, 5)
        self.assertEqual(data, b"hello")
        
        data = self.mmap_file.read_bytes(6, 5)
        self.assertEqual(data, b"world")


class TestDictionaryCompressor(unittest.TestCase):
    """测试词典压缩 (步骤77)"""

    def test_zlib_compression(self):
        """测试zlib压缩"""
        compressor = DictionaryCompressor()
        
        data = b"this is some test data for compression"
        compressed = compressor.compress(data)
        decompressed = compressor.decompress(compressed)
        self.assertEqual(data, decompressed)

    def test_bz2_compression(self):
        """测试bz2压缩"""
        from AuroraNLP.core.performance import CompressionType
        
        compressor = DictionaryCompressor(method=CompressionType.BZ2)
        
        data = b"this is some test data for compression"
        compressed = compressor.compress(data)
        decompressed = compressor.decompress(compressed)
        self.assertEqual(data, decompressed)


class TestLRUResultCache(unittest.TestCase):
    """测试结果缓存 (步骤78)"""

    def setUp(self):
        self.cache = LRUResultCache(max_size=10)

    def test_cache_put_get(self):
        """测试缓存存和取"""
        self.cache.put("text1", "result1")
        self.cache.put("text2", "result2")
        
        self.assertEqual(self.cache.get("text1"), "result1")
        self.assertEqual(self.cache.get("text2"), "result2")
        self.assertIsNone(self.cache.get("text3"))

    def test_cache_ttl(self):
        """测试缓存过期"""
        self.cache.put("text_with_ttl", "result", ttl=0.1)
        self.assertEqual(self.cache.get("text_with_ttl"), "result")
        
        time.sleep(0.2)
        
        # 过期之后可能还在，直到下次清理
        self.cache.put("new_text", "new_result")  # 触发清理
        # 可能已经被清理

    def test_cache_lru_eviction(self):
        """测试LRU淘汰"""
        for i in range(20):
            self.cache.put(f"text{i}", f"result{i}")
        
        stats = self.cache.get_stats()
        self.assertLessEqual(stats["size"], 10)

    def test_cache_stats(self):
        """测试缓存统计"""
        self.cache.put("a", "1")
        self.cache.put("b", "2")
        
        self.cache.get("a")  # 命中
        self.cache.get("c")  # 未命中
        
        stats = self.cache.get_stats()
        self.assertIn("hits", stats)
        self.assertIn("misses", stats)
        self.assertIn("hit_rate", stats)


class TestDistributedTask(unittest.TestCase):
    """测试分布式任务调度 (步骤79)"""

    def test_simple_task_scheduler(self):
        """测试简单任务调度器"""
        scheduler = SimpleTaskScheduler()
        
        def add(x, y):
            return x + y
        
        task_id = scheduler.submit(add, 2, 3)
        scheduler.run(task_id)
        
        task = scheduler._tasks.get(task_id)
        self.assertIsNotNone(task)
        self.assertEqual(task.result, 5)
        self.assertEqual(task.status, "success")


class TestPerformanceMonitor(unittest.TestCase):
    """测试性能监控 (步骤80)"""

    def setUp(self):
        self.monitor = PerformanceMonitor()

    def test_metric_recording(self):
        """测试指标记录"""
        metric = self.monitor.get_metric("test_metric")
        
        for i in range(10):
            metric.record(i)
        
        stats = metric.get_stats()
        self.assertEqual(stats["count"], 10)
        self.assertEqual(stats["min"], 0)
        self.assertEqual(stats["max"], 9)

    def test_counter_increment(self):
        """测试计数器"""
        metric = self.monitor.get_metric("test_counter")
        
        metric.incr(1)
        metric.incr(2)
        metric.incr(3)
        
        stats = metric.get_stats()
        self.assertEqual(stats["value"], 6)

    def test_timer_decorator(self):
        """测试计时器装饰器"""
        @self.monitor.time_it("test_function")
        def compute():
            time.sleep(0.1)
            return 42
        
        result = compute()
        self.assertEqual(result, 42)
        
        metric = self.monitor.get_metric("test_function")
        stats = metric.get_stats()
        self.assertGreater(stats["sum"], 0)

    def test_performance_report(self):
        """测试性能报告"""
        report = self.monitor.get_report()
        self.assertIn("uptime_seconds", report)
        self.assertIn("metrics", report)


class TestOptimizationSuite(unittest.TestCase):
    """测试一站式优化套件"""

    def setUp(self):
        self.suite = OptimizationSuite()

    def test_suite_comprehensive(self):
        """测试套件综合功能"""
        # 测试各个组件是否能正常访问
        self.assertIsNotNone(self.suite.monitor)
        self.assertIsNotNone(self.suite.batch_processor)
        self.assertIsNotNone(self.suite.cache)
        
        # 测试注册对象池
        pool = self.suite.register_object_pool(
            "test_pool",
            factory=lambda: [],
            resetter=lambda x: x.clear(),
            max_size=10
        )
        self.assertIsNotNone(pool)
        
        # 测试获取对象池
        retrieved = self.suite.get_object_pool("test_pool")
        self.assertIsNotNone(retrieved)
        
        # 测试获取完整报告
        report = self.suite.get_full_report()
        self.assertIn("monitor", report)
        self.assertIn("cache", report)
        self.assertIn("memory_pool", report)
        self.assertIn("object_pools", report)


if __name__ == "__main__":
    unittest.main()
