# AuroraNLP 1.0.0 发布公告

**发布日期：** 2026-05-05  
**版本代号：** coca

## 🎉 关于此发布

AuroraNLP 1.0.0 是该项目的首个正式稳定版本，标志着从轻量级工具包向专业级中文自然语言处理平台的转变。

## ✨ 核心特性

### 1. 强大的分词能力
- 支持多种分词策略：HMM、CRF、感知器、词格、混合策略
- 分词准确率约 93%，速度达 50万字/秒
- 支持用户词典和领域词典

### 2. 完整的 NLP 功能
- **词性标注** - 基于 HMM 和 CRF 的词性标注
- **命名实体识别** - 支持嵌套实体识别
- **句法分析** - 依存分析和成分分析
- **关键词提取** - TF-IDF 和 TextRank 算法
- **文本相似度计算** - 余弦、Jaccard、编辑距离

### 3. 深度学习集成
- BiLSTM-CRF 模型实现
- 预训练 BERT 支持
- ALBERT、DistilBERT 等轻量级模型
- 模型微调、迁移学习、知识蒸馏

### 4. 企业级架构
- 类似 spaCy 的 Pipeline 系统
- REST API 和 gRPC 接口
- 结构化日志和健康检查
- Prometheus 指标
- Docker 和 Kubernetes 支持
- 限流熔断和灰度发布

### 5. 丰富的词典资源
- 百万级通用中文词典
- 医疗、法律、电商、新闻等领域词库
- 人名词库、地名词库、机构名词库
- 搜狗细胞词库集成
- 开放词林（同义词/近义词）

## 📦 安装方式

### 基本安装
```bash
pip install aurora-nlp
```

### 完整安装（包含深度学习支持）
```bash
pip install aurora-nlp[all]
```

### 从源码安装
```bash
git clone https://github.com/AuroraNLP/AuroraNLP.git
cd AuroraNLP
pip install -e .[dev]
```

## 🚀 快速开始

### 基础分词
```python
from AuroraNLP import Segmentor

seg = Segmentor(use_hybrid=True)
text = "人工智能正在改变世界"
words = seg.segment(text)
print(words)  # ['人工智能', '正在', '改变', '世界']
```

### 使用 Pipeline
```python
from AuroraNLP import Pipeline, Segmentor, POSTagger, NERRecognizer

nlp = Pipeline()
nlp.add_component(Segmentor())
nlp.add_component(POSTagger())
nlp.add_component(NERRecognizer())

doc = nlp("张三在阿里巴巴位于杭州的总部工作")
print(doc.tokens)
print(doc.ents)
```

## 📚 文档资源

- **README** - 项目概览和快速开始
- **快速入门** ([getting-started.md](docs/getting-started.md)) - 详细入门教程
- **用户手册** ([user-guide.md](docs/user-guide.md)) - 完整功能文档
- **最佳实践** ([best-practices.md](docs/best-practices.md)) - 性能优化和生产部署指南
- **FAQ** ([faq.md](docs/faq.md)) - 常见问题解答

## 🧪 示例代码

查看 [examples/](examples/) 目录获取完整示例：
- [basic_segmentation.py](examples/basic_segmentation.py) - 基础分词
- [ner.py](examples/ner.py) - 命名实体识别
- [pipeline.py](examples/pipeline.py) - Pipeline 系统
- [enterprise.py](examples/enterprise.py) - 企业级功能

## 📈 性能数据

| 功能 | 准确率 | 速度 |
|------|--------|------|
| 分词 | ~93% | 50万字/秒 |
| 词性标注 | ~92% | 40万字/秒 |
| NER | ~88% | 30万字/秒 |

## 🤝 贡献指南

欢迎为 AuroraNLP 贡献代码！请查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解详情。

## 📄 许可证

本项目采用 Apache 2.0 许可证。

## 🙏 致谢

感谢所有为 AuroraNLP 做出贡献的人！

## 🔗 相关链接

- GitHub: https://github.com/AuroraNLP/AuroraNLP
- PyPI: https://pypi.org/project/aurora-nlp/
- Issues: https://github.com/AuroraNLP/AuroraNLP/issues

---

**AuroraNLP Team**
2026-05-05
