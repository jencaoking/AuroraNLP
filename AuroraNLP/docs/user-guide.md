# 用户手册

本手册详细介绍 AuroraNLP 的所有功能。

## 目录
- [分词](#分词)
- [词性标注](#词性标注)
- [命名实体识别](#命名实体识别)
- [句法分析](#句法分析)
- [关键词提取](#关键词提取)
- [文本相似度](#文本相似度)
- [词典管理](#词典管理)
- [Pipeline 系统](#pipeline-系统)
- [深度学习](#深度学习)
- [企业级功能](#企业级功能)

## 分词

### 基本用法

```python
from AuroraNLP import Segmentor

seg = Segmentor()
text = "中文分词是自然语言处理的基础任务"
words = seg.segment(text)
```

### 分词策略

#### HMM 分词
基于隐马尔可夫模型的统计分词方法：

```python
seg = Segmentor(use_hmm=True)
```

#### CRF 分词
基于条件随机场的分词方法：

```python
seg = Segmentor(use_crf=True)
```

#### 感知机分词
在线学习的感知机分词：

```python
seg = Segmentor(use_perceptron=True)
```

#### 词格分词
多路径分词，支持歧义消解：

```python
seg = Segmentor(use_lattice=True)
```

#### 混合分词（推荐）
融合多种策略：

```python
from AuroraNLP import HybridConfig, HybridStrategy

config = HybridConfig(
    strategies=[
        HybridStrategy.RULE_BASED,
        HybridStrategy.HMM,
        HybridStrategy.CRF
    ]
)
seg = Segmentor(use_hybrid=True, hybrid_config=config)
```

### 用户词典

```python
from AuroraNLP import UserDictionary

# 创建用户词典
user_dict = UserDictionary()
user_dict.add_word("自然语言处理", "n", 10.0)
user_dict.add_word("深度学习", "n", 10.0)

# 使用用户词典
seg = Segmentor(dictionary=user_dict)
```

### 领域词典

```python
from AuroraNLP import DomainDictionaryManager

manager = DomainDictionaryManager()
manager.load_domain("medical")    # 医疗
manager.load_domain("legal")       # 法律
manager.load_domain("ecommerce")  # 电商
manager.load_domain("news")        # 新闻

seg = Segmentor(dictionary=manager.combined_dictionary)
```

## 词性标注

```python
from AuroraNLP import HMMPOSTagger, CRFPOSTagger

# HMM 词性标注
pos_tagger = HMMPOSTagger()
tags = pos_tagger.tag(["我", "爱", "中国"])

# CRF 词性标注
pos_tagger = CRFPOSTagger()
tags = pos_tagger.tag(["我", "爱", "中国"])
```

## 命名实体识别

### 基本 NER

```python
from AuroraNLP import NERRecognizer

ner = NERRecognizer()
entities = ner.recognize("张三在北京的腾讯公司工作")
```

### 嵌套 NER

```python
from AuroraNLP import NestedNERRecognizer

nested_ner = NestedNERRecognizer()
entities = nested_ner.recognize("中华人民共和国北京市朝阳区")
```

### 实体链接

```python
from AuroraNLP import EntityLinker, KnowledgeBase

kb = KnowledgeBase()
kb.add_entity("张三", {"id": "1", "type": "PERSON"})

linker = EntityLinker(kb)
linked = linker.link(entities)
```

## 句法分析

### 依存分析

```python
from AuroraNLP import DependencyParser

parser = DependencyParser()
tree = parser.parse("我吃苹果")

for arc in tree.arcs:
    print(f"{arc.head} -> {arc.dependent}: {arc.relation}")
```

### 成分分析

```python
from AuroraNLP import ConstituentParser, CFG

cfg = CFG.from_rules([
    "S -> NP VP",
    "NP -> PN | NN",
    "VP -> V NP",
    "PN -> '我'",
    "NN -> '苹果'",
    "V -> '吃'"
])

parser = ConstituentParser(cfg)
tree = parser.parse("我吃苹果")
```

## 关键词提取

```python
from AuroraNLP import KeywordExtractor

extractor = KeywordExtractor()

# TF-IDF
keywords = extractor.extract_tfidf(text, top_k=10)

# TextRank
keywords = extractor.extract_textrank(text, top_k=10)
```

## 文本相似度

```python
from AuroraNLP import Similarity

sim = Similarity()

text1 = "我喜欢吃苹果"
text2 = "我爱吃苹果"

# 余弦相似度
cosine = sim.cosine(text1, text2)

# Jaccard 相似度
jaccard = sim.jaccard(text1, text2)

# 编辑距离
distance = sim.edit_distance(text1, text2)
```

## 词典管理

### 词典版本控制

```python
from AuroraNLP import VersionedDictionary

dict = VersionedDictionary()
dict.add_word("新词", "n")
version1 = dict.commit("添加新词", "user")

dict.add_word("另一个词", "n")
version2 = dict.commit("添加另一个词", "user")

# 回滚
dict.checkout(version1)
```

### 增量更新

```python
from AuroraNLP import IncrementalDictionary, HotUpdateDictionaryManager

dict = IncrementalDictionary()
manager = HotUpdateDictionaryManager()
manager.register_dictionary(dict)
manager.start_file_monitoring()
```

## Pipeline 系统

### 创建 Pipeline

```python
from AuroraNLP import Pipeline, PipelineComponent

class MyComponent(PipelineComponent):
    name = "my_component"
    
    def __call__(self, doc):
        doc.custom_attr = "processed"
        return doc

nlp = Pipeline()
nlp.add_component(MyComponent())

doc = nlp("测试文本")
print(doc.custom_attr)
```

### 配置 Pipeline

```python
from AuroraNLP import PipelineConfig

config = PipelineConfig({
    "components": [
        {"name": "segmentor", "enabled": True},
        {"name": "pos_tagger", "enabled": True},
        {"name": "ner", "enabled": True}
    ]
})

nlp = Pipeline(config=config)
```

## 深度学习

### BiLSTM-CRF

```python
from AuroraNLP.deep_learning import BiLSTMCRF

model = BiLSTMCRF()
model.fit(train_data)
preds = model.predict(test_data)
```

### 预训练 BERT

```python
from AuroraNLP.deep_learning import PreTrainedBERT, BERTChineseSegmentor

# BERT 分词
segmentor = BERTChineseSegmentor()
words = segmentor.segment(text)

# BERT NER
from AuroraNLP.deep_learning import BERTNER
ner = BERTNER()
entities = ner.recognize(text)
```

### 模型微调

```python
from AuroraNLP.deep_learning import FineTuningConfig, FineTuningTrainer

config = FineTuningConfig(
    learning_rate=2e-5,
    batch_size=32,
    epochs=3
)

trainer = FineTuningTrainer(config)
trainer.train(model, train_data, val_data)
```

## 企业级功能

### 日志系统

```python
from AuroraNLP import LogManager, LogLevel

logger = LogManager.get_logger("my_app")
logger.set_level(LogLevel.INFO)

logger.info("应用启动")
logger.error("发生错误")
```

### 健康检查

```python
from AuroraNLP import HealthChecker, MemoryHealthCheck, DiskHealthCheck

checker = HealthChecker()
checker.add_check(MemoryHealthCheck())
checker.add_check(DiskHealthCheck())

status = checker.check()
print(status)
```

### Prometheus 指标

```python
from AuroraNLP import PrometheusRegistry, PrometheusCounter, PrometheusGauge

registry = PrometheusRegistry()

counter = PrometheusCounter("requests_total", "总请求数")
gauge = PrometheusGauge("active_users", "活跃用户数")

registry.register(counter)
registry.register(gauge)

counter.inc()
gauge.set(100)

print(registry.export())
```

### 限流熔断

```python
from AuroraNLP import TokenBucket, CircuitBreaker

# 令牌桶限流
bucket = TokenBucket(capacity=100, rate=10)
if bucket.try_consume():
    process_request()

# 熔断器
breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
try:
    with breaker:
        risky_operation()
except Exception:
    pass
```

更多企业级功能请参考 [enterprise.md](enterprise.md)。
