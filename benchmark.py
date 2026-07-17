import time
import functools
from typing import List, Callable, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class BenchmarkResult:
    operation: str
    total_time: float
    iterations: int
    avg_time: float
    min_time: float
    max_time: float
    ops_per_second: float


class PerformanceBenchmark:
    def __init__(self, segmentor):
        self.segmentor = segmentor

    def _run_benchmark(
        self,
        func: Callable,
        iterations: int,
        *args,
        operation: Optional[str] = None,
        **kwargs
    ) -> BenchmarkResult:
        times = []

        for _ in range(iterations):
            start = time.perf_counter()
            func(*args, **kwargs)
            end = time.perf_counter()
            times.append(end - start)

        total_time = sum(times)
        avg_time = total_time / iterations
        min_time = min(times)
        max_time = max(times)
        ops_per_second = iterations / total_time if total_time > 0 else 0

        operation_name = operation if operation is not None else func.__name__

        return BenchmarkResult(
            operation=operation_name,
            total_time=total_time,
            iterations=iterations,
            avg_time=avg_time,
            min_time=min_time,
            max_time=max_time,
            ops_per_second=ops_per_second
        )

    def benchmark_segment(
        self,
        text: str,
        mode: str = 'bidirectional',
        iterations: int = 100
    ) -> BenchmarkResult:
        return self._run_benchmark(
            self.segmentor.segment,
            iterations,
            text,
            mode
        )

    def benchmark_segment_with_pos(
        self,
        text: str,
        mode: str = 'bidirectional',
        iterations: int = 100
    ) -> BenchmarkResult:
        return self._run_benchmark(
            self.segmentor.segment_with_pos,
            iterations,
            text,
            mode
        )

    def benchmark_extract_keywords(
        self,
        text: str,
        method: str = 'tfidf',
        top_k: int = 10,
        iterations: int = 50
    ) -> BenchmarkResult:
        return self._run_benchmark(
            self.segmentor.extract_keywords,
            iterations,
            text,
            top_k,
            method
        )

    def benchmark_similarity(
        self,
        text1: str,
        text2: str,
        method: str = 'cosine',
        iterations: int = 100
    ) -> BenchmarkResult:
        return self._run_benchmark(
            self.segmentor.compute_similarity,
            iterations,
            text1,
            text2,
            method
        )

    def run_full_benchmark(
        self,
        test_texts: List[str],
        iterations: int = 100
    ) -> Dict[str, BenchmarkResult]:
        results = {}

        sample_text = test_texts[0] if test_texts else "今天天气很好"

        results['segment_forward'] = self.benchmark_segment(
            sample_text, 'forward', iterations
        )
        results['segment_backward'] = self.benchmark_segment(
            sample_text, 'backward', iterations
        )
        results['segment_bidirectional'] = self.benchmark_segment(
            sample_text, 'bidirectional', iterations
        )
        results['segment_with_pos'] = self.benchmark_segment_with_pos(
            sample_text, 'bidirectional', iterations
        )
        results['extract_keywords_tfidf'] = self.benchmark_extract_keywords(
            sample_text, 'tfidf', 10, iterations // 2
        )
        results['extract_keywords_textrank'] = self.benchmark_extract_keywords(
            sample_text, 'textrank', 10, iterations // 2
        )

        if len(test_texts) >= 2:
            results['similarity_cosine'] = self.benchmark_similarity(
                test_texts[0], test_texts[1], 'cosine', iterations
            )

        return results

    @staticmethod
    def format_result(result: BenchmarkResult) -> str:
        return (
            f"操作: {result.operation}\n"
            f"  总时间: {result.total_time:.4f}s\n"
            f"  迭代次数: {result.iterations}\n"
            f"  平均时间: {result.avg_time*1000:.4f}ms\n"
            f"  最小时间: {result.min_time*1000:.4f}ms\n"
            f"  最大时间: {result.max_time*1000:.4f}ms\n"
            f"  每秒操作数: {result.ops_per_second:.2f} ops/s"
        )

    @staticmethod
    def compare_results(
        baseline: BenchmarkResult,
        optimized: BenchmarkResult
    ) -> Dict[str, Any]:
        improvement = (
            (baseline.avg_time - optimized.avg_time) / baseline.avg_time * 100
            if baseline.avg_time > 0 else 0
        )
        speedup = (
            baseline.avg_time / optimized.avg_time
            if optimized.avg_time > 0 else float('inf')
        )

        return {
            'baseline_avg': baseline.avg_time,
            'optimized_avg': optimized.avg_time,
            'improvement_percent': improvement,
            'speedup_factor': speedup
        }


def measure_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} 执行时间: {(end - start) * 1000:.4f}ms")
        return result
    return wrapper


__all__ = ['PerformanceBenchmark', 'BenchmarkResult', 'measure_time']
