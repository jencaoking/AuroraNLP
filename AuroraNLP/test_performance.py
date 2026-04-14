from AuroraNLP import Segmentor, PerformanceBenchmark

# 创建分词器实例
seg = Segmentor()

# 创建性能基准测试实例
bench = PerformanceBenchmark(seg)

# 测试文本
test_texts = [
    '今天天气很好，我想去公园散步',
    '自然语言处理是人工智能的重要分支',
    'Python是一种广泛使用的编程语言'
]

# 运行完整的基准测试
results = bench.run_full_benchmark(test_texts)

# 输出结果
for name, result in results.items():
    print(f'\n{name}:')
    print(PerformanceBenchmark.format_result(result))
