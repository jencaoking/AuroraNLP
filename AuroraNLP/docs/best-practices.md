# 最佳实践指南

本指南介绍使用 AuroraNLP 的最佳实践。

## 目录
- [性能优化](#性能优化)
- [生产环境部署](#生产环境部署)
- [常见问题](#常见问题)

## 性能优化

### 1. 使用对象池

对于频繁创建的对象，使用对象池复用：

```python
from AuroraNLP import ObjectPool

# 创建对象池
segmentor_pool = ObjectPool(
    creator=lambda: Segmentor(use_hybrid=True),
    max_size=10
)

# 使用
with segmentor_pool.acquire() as seg:
    words = seg.segment(text)
```

### 2. 批量处理

使用批量处理接口提高吞吐量：

```python
from AuroraNLP import BatchProcessor

processor = BatchProcessor(batch_size=100)
results = processor.process(texts, lambda text: seg.segment(text))
```

### 3. 启用缓存

为重复处理的文本启用缓存：

```python
from AuroraNLP import LRUResultCache

cache = LRUResultCache(max_size=10000)

def process(text):
    if text in cache:
        return cache[text]
    result = seg.segment(text)
    cache[text] = result
    return result
```

### 4. 多线程处理

对于 CPU 密集型任务：

```python
from AuroraNLP import ThreadPoolExecutor, ParallelTokenizer

executor = ThreadPoolExecutor(max_workers=4)
tokenizer = ParallelTokenizer(seg, executor)

results = tokenizer.tokenize_batch(texts)
```

### 5. 内存映射文件

对于大词典，使用内存映射：

```python
from AuroraNLP import MemoryMappedFile

mm_file = MemoryMappedFile("large_dict.txt", mode="r")
# 使用 mm_file
```

### 6. 延迟 GC

禁用自动 GC，手动控制：

```python
from AuroraNLP import DelayedGC

with DelayedGC():
    for text in many_texts:
        seg.segment(text)
```

## 生产环境部署

### 1. 使用 Pipeline 系统

```python
from AuroraNLP import Pipeline, Segmentor, POSTagger, NERRecognizer

nlp = Pipeline()
nlp.add_component(Segmentor(use_hybrid=True))
nlp.add_component(POSTagger())
nlp.add_component(NERRecognizer())

# 预热
nlp("预热文本")
```

### 2. 配置健康检查

```python
from AuroraNLP import HealthChecker, MemoryHealthCheck, DiskHealthCheck

checker = HealthChecker()
checker.add_check(MemoryHealthCheck(max_usage_percent=80))
checker.add_check(DiskHealthCheck(path="/", max_usage_percent=90))

# 集成到你的服务
@app.route("/health")
def health():
    status = checker.check()
    return jsonify(status.to_dict())
```

### 3. 使用结构化日志

```python
from AuroraNLP import LogManager, LogLevel, LogFormat

logger = LogManager.get_logger("aurora_service")
logger.set_level(LogLevel.INFO)
logger.set_format(LogFormat.JSON)  # JSON 格式日志

# 添加处理器
from AuroraNLP import FileLogHandler, ConsoleLogHandler

logger.add_handler(FileLogHandler("app.log"))
logger.add_handler(ConsoleLogHandler())

# 使用
logger.info("请求处理完成", extra={
    "text_length": len(text),
    "processing_time": 0.001
})
```

### 4. 配置限流熔断

```python
from AuroraNLP import TokenBucket, CircuitBreaker

# 限流
bucket = TokenBucket(capacity=1000, rate=100)

# 熔断
breaker = CircuitBreaker(
    failure_threshold=10,
    recovery_timeout=30,
    half_open_max_calls=5
)

# 使用
def process_request(text):
    if not bucket.try_consume():
        raise Exception("Too many requests")
    
    try:
        with breaker:
            return nlp(text)
    except Exception as e:
        logger.error("处理失败", error=str(e))
        raise
```

### 5. Docker 部署

使用生成的 Dockerfile：

```python
from AuroraNLP import generate_dockerfile_content

dockerfile_content = generate_dockerfile_content(
    base_image="python:3.10-slim",
    expose_ports=[8000],
    healthcheck=True
)

with open("Dockerfile", "w") as f:
    f.write(dockerfile_content)
```

### 6. Kubernetes 部署

```python
from AuroraNLP import (
    generate_k8s_deployment_content,
    generate_k8s_service_content,
    generate_k8s_ingress_content
)

# 生成部署文件
deployment = generate_k8s_deployment_content(
    name="aurora-nlp",
    image="aurora-nlp:latest",
    replicas=3,
    ports=[8000],
    liveness_probe=True,
    readiness_probe=True
)

service = generate_k8s_service_content(
    name="aurora-nlp",
    selector="aurora-nlp",
    ports=[(80, 8000)]
)

ingress = generate_k8s_ingress_content(
    name="aurora-nlp",
    host="nlp.example.com",
    service_name="aurora-nlp",
    service_port=80
)
```

## 常见问题

### Q: 分词速度太慢怎么办？

A: 
1. 使用混合分词而不是单一的深度学习模型
2. 启用缓存
3. 使用批量处理
4. 考虑使用轻量级模型

### Q: 如何处理领域特定的术语？

A:
1. 使用领域词典
2. 加载自定义用户词典
3. 考虑微调预训练模型

### Q: 内存占用太高怎么办？

A:
1. 按需加载词典，不要一次性加载所有
2. 使用内存映射文件
3. 启用词典压缩
4. 调整模型缓存大小

### Q: 如何提高 NER 准确率？

A:
1. 使用 BERT 等深度学习模型
2. 加载领域特定的词典
3. 在领域数据上微调模型
4. 使用实体链接消歧

### Q: 如何在生产环境保证高可用？

A:
1. 配置健康检查
2. 使用 Kubernetes 部署，设置多副本
3. 配置熔断限流
4. 设置数据备份
5. 配置灰度发布

更多问题请查看 [FAQ](faq.md)。
