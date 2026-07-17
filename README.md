# AuroraNLP

<p align="center">
  <em>专业级中文自然语言处理工具包</em>
</p>

<p align="center">
  <a href="https://pypi.org/project/aurora-nlp/"><img src="https://img.shields.io/pypi/v/aurora-nlp.svg" alt="PyPI"></a>
  <a href="https://pypi.org/project/aurora-nlp/"><img src="https://img.shields.io/pypi/pyversions/aurora-nlp.svg" alt="Python Versions"></a>
  <a href="https://github.com/AuroraNLP/AuroraNLP/actions"><img src="https://github.com/AuroraNLP/AuroraNLP/workflows/CI/badge.svg" alt="CI"></a>
  <a href="https://codecov.io/gh/AuroraNLP/AuroraNLP"><img src="https://codecov.io/gh/AuroraNLP/AuroraNLP/branch/main/graph/badge.svg" alt="Codecov"></a>
  <a href="https://github.com/AuroraNLP/AuroraNLP/blob/main/LICENSE"><img src="https://img.shields.io/github/license/AuroraNLP/AuroraNLP.svg" alt="License"></a>
</p>

## 简介

**AuroraNLP** 是一个专业级的中文自然语言处理工具包，提供从分词、词性标注、命名实体识别、句法分析到企业级支持的全栈功能。

## 特性

### 🚀 核心功能
- **分词**: 支持 HMM/CRF/感知机/词格/混合多种策略
- **词性标注**: 基于统计模型和深度学习的词性标注
- **命名实体识别**: 支持 PER/LOC/ORG 等实体类型，嵌套实体识别
- **句法分析**: 依存分析（ArcEager算法）和成分分析（CFG）
- **关键词提取**: TF-IDF/TextRank 算法
- **文本相似度**: 余弦/Jaccard/编辑距离
- **新词发现**: 互信息 + 左右熵

### 📚 丰富的词典资源
- 百万级通用中文词典
- 医疗/法律/电商/新闻等领域词库
- 人名词库、地名词库、机构名词库
- 搜狗细胞词库集成
- 开放词林（同义词/近义词）

### 🔬 深度学习支持
- BiLSTM-CRF 模型
- BERT 预训练模型集成
- 轻量级模型（ALBERT/DistilBERT）
- 模型微调、迁移学习、知识蒸馏
- 模型量化、ONNX 导出、热加载

### 🏗️ 企业级架构
- 类 spaCy 的 Pipeline 系统
- Doc/Span/Token 统一数据结构
- REST API / gRPC 接口
- 异步处理和流式处理
- 插件系统支持

### ⚡ 性能优化
- 对象池/内存池
- 批量处理
- 混合精度推理
- 内存映射文件
- 词典压缩
- 缓存机制

### 🏢 企业级功能
- 结构化日志系统
- 健康检查
- Prometheus 指标
- Docker & Kubernetes 支持
- 限流熔断
- 认证授权
- 配置管理
- 灰度发布
- 灾备方案

### 📦 轻量设计
- 纯 Python 实现，无强制外部依赖
- 零配置开箱即用
- Python 3.8+ 支持

## 安装

### 基本安装

```bash
pip install aurora-nlp
```

### 完整安装（包含深度学习支持）

```bash
pip install aurora-nlp[full]
```

### 开发安装

```bash
git clone https://github.com/AuroraNLP/AuroraNLP.git
cd AuroraNLP
pip install -e .[dev]
```

## 快速开始

### 基础分词

```python
from AuroraNLP import Segmentor

# 初始化分词器
seg = Segmentor(use_hybrid=True)

# 分词
text = "人工智能正在改变世界"
words = seg.segment(text)
print(words)  # ['人工智能', '正在', '改变', '世界']

# 词性标注
words_with_pos = seg.segment_with_pos(text)
print(words_with_pos)
# [('人工智能', 'n'), ('正在', 'd'), ('改变', 'v'), ('世界', 'n')]
```

### 命名实体识别

```python
from AuroraNLP import NERRecognizer

ner = NERRecognizer(use_bert=True)
text = "张三在阿里巴巴位于杭州的总部工作"
entities = ner.recognize(text)
print(entities)
# [
#   Entity(text='张三', type='PERSON', start=0, end=2),
#   Entity(text='阿里巴巴', type='ORGANIZATION', start=3, end=7),
#   Entity(text='杭州', type='LOCATION', start=9, end=11)
# ]
```

### Pipeline 系统

```python
from AuroraNLP import Pipeline, Segmentor, POSTagger, NERRecognizer

# 创建 Pipeline
nlp = Pipeline()
nlp.add_component(Segmentor())
nlp.add_component(POSTagger())
nlp.add_component(NERRecognizer())

# 处理文本
doc = nlp("中国北京是一座美丽的城市")

# 获取分词结果
print(doc.tokens)

# 获取词性标注
print([(token.text, token.pos) for token in doc.tokens])

# 获取命名实体
print(doc.ents)
```

### 使用领域词典

```python
from AuroraNLP import Segmentor, DomainDictionaryManager

# 加载医疗领域词典
dict_manager = DomainDictionaryManager()
dict_manager.load_domain("medical")

seg = Segmentor(dictionary=dict_manager.combined_dictionary)
text = "患者出现发热咳嗽症状，医生建议做CT检查"
words = seg.segment(text)
print(words)
```

## 文档

- [快速入门](docs/getting-started.md)
- [用户手册](docs/user-guide.md)
- [API 参考](docs/api.md)
- [最佳实践](docs/best-practices.md)
- [性能调优指南](docs/performance-guide.md)
- [FAQ](docs/faq.md)

## 示例

查看 [`examples/`](examples/) 目录获取更多示例：
- [基础分词](examples/basic_segmentation.py)
- [词性标注](examples/pos_tagging.py)
- [命名实体识别](examples/ner.py)
- [句法分析](examples/parsing.py)
- [情感分析](examples/sentiment.py)
- [Pipeline 使用](examples/pipeline.py)
- [企业级应用](examples/enterprise.py)

## 性能

| 功能 | 准确率 | 速度 |
|------|--------|------|
| 分词 | ~93% | 50万字/秒 |
| 词性标注 | ~92% | 40万字/秒 |
| NER | ~88% | 30万字/秒 |

## 贡献指南

欢迎贡献！请查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解详情。

## 许可证

本项目采用 Apache 2.0 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 致谢

- 参考项目：HanLP、spaCy、jieba、pkuseg、THULAC
- 数据资源：人民日报语料库、搜狗细胞词库、开放词林

## 联系方式

- GitHub: [AuroraNLP/AuroraNLP](https://github.com/AuroraNLP/AuroraNLP)
- Issues: [GitHub Issues](https://github.com/AuroraNLP/AuroraNLP/issues)
- 讨论: [GitHub Discussions](https://github.com/AuroraNLP/AuroraNLP/discussions)

## 路线图

查看 [ROADMAP.md](docs/ROADMAP.md) 了解我们的发展计划。

## 项目统计

![项目进度](https://progress-bar.dev/90/?title=V1.0进度)

已完成 90/100 步（90%）
