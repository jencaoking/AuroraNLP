"""完整复现test_performance的卡死场景"""
import sys, time
sys.path.insert(0, r'J:\PROJECT\AuroraNLP\AuroraNLP')

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
    SimpleTaskScheduler,
)


def square(x):
    return x * x


if __name__ == '__main__':
    p = ProcessPoolExecutor(max_workers=2)
    print('start...', flush=True)
    p.start()
    print('started', flush=True)
    t = time.time()
    r = p.map(square, list(range(10)))
    print('result:', r, 'elapsed:', round(time.time()-t, 2), flush=True)
    p.stop()
    print('done', flush=True)