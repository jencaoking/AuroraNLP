# 常见问题

## 安装相关

### Q: 支持哪些 Python 版本？

A: AuroraNLP 支持 Python 3.8 及更高版本。

### Q: 需要哪些依赖？

A: AuroraNLP 核心功能没有外部依赖。深度学习功能需要 PyTorch 或 TensorFlow。

### Q: 如何安装包含深度学习的版本？

A: 使用 `pip install aurora-nlp[full]`。

## 功能相关

### Q: 哪种分词策略最好？

A: 这取决于你的场景：
- 追求速度：规则 + HMM
- 追求准确率：CRF 或深度学习
- 推荐：混合策略

### Q: 如何添加自定义词典？

A:

```python
from AuroraNLP import UserDictionary

user_dict = UserDictionary()
user_dict.add_word("自定义词", "n", 10.0)

seg = Segmentor(dictionary=user_dict)
```

### Q: 支持繁体中文吗？

A: 支持，使用 `TraditionalChineseConverter`：

```python
from AuroraNLP import TraditionalChineseConverter

converter = TraditionalChineseConverter()
simplified = converter.to_simplified("繁體中文")
```

### Q: 如何处理新词？

A: 使用新词发现功能：

```python
from AuroraNLP import NewWordDetector

detector = NewWordDetector()
new_words = detector.detect(corpus)
```

### Q: 如何处理歧义？

A: 使用歧义检测功能：

```python
from AuroraNLP import AmbiguityDetector

detector = AmbiguityDetector()
ambiguities = detector.detect(text)
```

## 性能相关

### Q: 如何提高处理速度？

A:
1. 使用批量处理
2. 启用缓存
3. 使用对象池
4. 多线程/多进程
5. 选择更轻量的模型

### Q: 内存占用太高怎么办？

A:
1. 按需加载词典
2. 使用内存映射文件
3. 启用词典压缩
4. 调整模型缓存大小

### Q: 如何在生产环境优化？

A: 请参考 [最佳实践指南](best-practices.md)。

## 深度学习相关

### Q: 需要 GPU 吗？

A: 不需要，CPU 也可以运行。但 GPU 可以显著提高深度学习模型的速度。

### Q: 如何选择预训练模型？

A:
- 追求准确率：BERT 或 RoBERTa
- 追求速度：ALBERT Tiny 或 MiniLM
- 平衡：DistilBERT

### Q: 如何微调模型？

A: 参考 [用户手册](user-guide.md) 的"模型微调"章节。

## 企业级功能

### Q: 如何配置日志？

A:

```python
from AuroraNLP import LogManager, LogLevel

logger = LogManager.get_logger("my_app")
logger.set_level(LogLevel.INFO)
logger.add_handler(FileLogHandler("app.log"))
```

### Q: 如何监控服务？

A: 使用健康检查和 Prometheus 指标：

```python
from AuroraNLP import HealthChecker, PrometheusRegistry

checker = HealthChecker()
registry = PrometheusRegistry()
```

### Q: 如何部署到 Kubernetes？

A: 使用生成的配置文件，参考 [最佳实践指南](best-practices.md)。

## 故障排查

### Q: 分词结果不对怎么办？

A:
1. 检查是否加载了正确的词典
2. 尝试不同的分词策略
3. 添加用户词典
4. 检查是否有歧义

### Q: 内存泄漏怎么办？

A:
1. 检查是否有未释放的资源
2. 使用对象池复用对象
3. 启用延迟 GC
4. 检查词典是否无限增长

### Q: 遇到 bug 怎么办？

A:
1. 查看日志
2. 在 GitHub 上提交 Issue
3. 提供复现步骤
4. 如果可能，提供最小可复现代码

## 其他

### Q: 如何贡献代码？

A: 请参考 [CONTRIBUTING.md](../CONTRIBUTING.md)。

### Q: 有商业支持吗？

A: 请联系我们了解商业支持选项。

### Q:  roadmap 是什么？

A: 请参考 [ROADMAP.md](ROADMAP.md)。

如果你的问题在这里没有找到答案，请：
1. 查看其他文档
2. 在 GitHub Discussions 提问
3. 提交 Issue
