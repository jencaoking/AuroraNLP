"""PerformanceBenchmark 性能基准测试"""
import pytest
from AuroraNLP.core.benchmark import PerformanceBenchmark, BenchmarkResult, measure_time


class TestPerformanceBenchmark:
    """测试 PerformanceBenchmark 性能基准功能"""

    def test_benchmark_init(self, sample_segmentor):
        """初始化"""
        benchmark = PerformanceBenchmark(sample_segmentor)
        assert benchmark.segmentor is sample_segmentor

    def test_benchmark_segment(self, sample_segmentor):
        """分词基准测试"""
        benchmark = PerformanceBenchmark(sample_segmentor)
        result = benchmark.benchmark_segment("中国人在北京", iterations=10)
        assert isinstance(result, BenchmarkResult)
        assert result.operation == "segment"
        assert result.iterations == 10
        assert result.total_time > 0
        assert result.avg_time > 0
        assert result.min_time > 0
        assert result.max_time > 0
        assert result.ops_per_second > 0

    def test_benchmark_result_fields(self, sample_segmentor):
        """BenchmarkResult 字段完整性"""
        benchmark = PerformanceBenchmark(sample_segmentor)
        result = benchmark.benchmark_segment("自然语言处理", iterations=5)
        assert hasattr(result, 'operation')
        assert hasattr(result, 'total_time')
        assert hasattr(result, 'iterations')
        assert hasattr(result, 'avg_time')
        assert hasattr(result, 'min_time')
        assert hasattr(result, 'max_time')
        assert hasattr(result, 'ops_per_second')
        assert isinstance(result.operation, str)
        assert isinstance(result.total_time, float)
        assert isinstance(result.iterations, int)
        assert isinstance(result.avg_time, float)
        assert isinstance(result.min_time, float)
        assert isinstance(result.max_time, float)
        assert isinstance(result.ops_per_second, float)

    def test_measure_time(self):
        """measure_time 装饰器/函数"""
        @measure_time
        def simple_func(x, y):
            return x + y

        result = simple_func(3, 5)
        assert result == 8

    def test_measure_time_on_method(self):
        """measure_time 装饰方法"""
        class MyClass:
            @measure_time
            def compute(self, a, b):
                return a * b

        obj = MyClass()
        result = obj.compute(4, 6)
        assert result == 24
