# AuroraNLP

<div align="center">

[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](https://github.com/AuroraNLP/AuroraNLP/releases)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-Production%20Stable-green.svg)]()

**AuroraNLP - 专业级中文自然语言处理工具包**

</div>

---

## 简介

AuroraNLP 是一个专业级的中文自然语言处理工具包，提供从分词、词性标注、命名实体识别、句法分析到企业级支持的全栈功能。项目采用纯 Python 实现，无重度依赖，易于安装和使用，同时具备企业级特性支持。

## 目录

- [简介](#简介)
- [快速开始](#快速开始)
- [功能特性](#功能特性)
- [项目结构](#项目结构)
- [使用示例](#使用示例)
- [配置说明](#配置说明)
- [贡献指南](#贡献指南)
- [许可证](#许可证)
- [联系方式](#联系方式)

---

## 快速开始

### 环境要求

- Python 3.8+
- 无其他强制依赖（深度学习功能需额外安装 PyTorch）

### 安装方式

```bash
# 通过 PyPI 安装（推荐）
pip install aurora-nlp

# 完整安装（包含深度学习支持）
pip install aurora-nlp[full]

# 开发安装
git clone https://github.com/AuroraNLP/AuroraNLP.git
cd AuroraNLP
pip install -e "AuroraNLP/[dev]"
```

### 基础使用

```python
from AuroraNLP import Segmentor

# 创建分词器
seg = Segmentor(use_hybrid=True)

# 基础分词
text = "我爱自然语言处理"
words = seg.segment(text)
print(words)  # ['我', '爱', '自然语言处理']

# 带词性标注的分词
words_with_pos = seg.segment_with_pos(text)
print(words_with_pos)  # [('我', 'r'), ('爱', 'v'), ('自然语言处理', 'n')]
```

---

## 功能特性

### 🚀 核心功能

- **分词** - 支持正向/逆向/双向最大匹配、HMM、CRF、感知器、Lattice、最短路径等多种分词方法
- **词性标注** - 基于 HMM 和 CRF 的词性标注
- **命名实体识别** - 支持 CRF-based NER，包括实体嵌套识别和实体链接（PER/LOC/ORG）
- **句法分析** - 支持依存句法分析（Arc-eager）和成分句法分析（PCFG/CKY）
- **关键词提取** - TF-IDF/TextRank 算法
- **文本相似度** - 余弦/Jaccard/编辑距离计算
- **新词发现** - 基于互信息和信息熵的新词发现算法
- **情感分析** - 支持情感极性分析和情感强度计算

### 📚 词典资源

- 百万级通用中文词典
- 医疗/法律/电商/新闻等领域专业词库
- 人名、地名、机构名词库
- 搜狗细胞词库集成
- 开放词林（同义词/近义词）

### 🔬 深度学习支持

- BiLSTM-CRF 模型
- BERT 预训练模型集成
- 轻量级模型（ALBERT/DistilBERT）
- 模型微调、迁移学习、知识蒸馏
- 模型量化、ONNX 导出、热加载

### 🏢 企业级功能

- 类 spaCy 的 Pipeline 系统
- REST API / gRPC 接口
- 异步处理和流式处理
- 结构化日志系统
- 健康检查和 Prometheus 指标
- Docker & Kubernetes 支持
- 限流熔断、认证授权、配置管理

---

## 项目结构

```
AuroraNLP/
├── AuroraNLP/                    # 核心源代码目录
│   ├── AuroraNLP/               # 主源码包
│   │   ├── __init__.py          # 包初始化
│   │   ├── data/                # 数据文件（词典、停用词、语料等）
│   │   ├── deep_learning/       # 深度学习模块
│   │   └── *.py                 # 各功能模块
│   ├── .github/                 # GitHub CI/CD 配置
│   ├── docs/                    # 文档目录
│   ├── examples/                # 示例代码
│   ├── tests/                   # 测试文件
│   ├── setup.py                 # 安装配置
│   └── requirements.txt         # 依赖列表
├── LICENSE                      # 许可证文件
└── README.md                    # 项目说明文档
```

---

## 使用示例

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

---

## 配置说明

### 配置文件

配置文件位于 `AuroraNLP/data/config.yaml`，支持以下配置项：

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `default_segmenter` | str | `hybrid` | 默认分词器类型 |
| `enable_cache` | bool | `true` | 是否启用缓存 |
| `cache_size` | int | `1000` | 缓存大小 |
| `log_level` | str | `INFO` | 日志级别 |
| `max_text_length` | int | `10000` | 最大处理文本长度 |

### 环境变量

| 变量名 | 说明 |
|--------|------|
| `AURORA_NLP_DATA_DIR` | 数据目录路径 |
| `AURORA_NLP_MODEL_DIR` | 模型目录路径 |
| `AURORA_NLP_LOG_LEVEL` | 日志级别 |

---

## 运行测试

```bash
cd AuroraNLP/AuroraNLP
pytest tests/ -v
```

---

## 贡献指南

欢迎贡献！请遵循以下流程：

1. Fork 项目仓库
2. 创建功能分支：`git checkout -b feature/your-feature`
3. 提交代码：`git commit -m "feat: add your feature"`
4. 推送到远程分支：`git push origin feature/your-feature`
5. 创建 Pull Request

### 代码规范

- 遵循 PEP 8 规范
- 使用类型提示
- 添加单元测试
- 编写清晰的文档

---

## 许可证

本项目采用 [Apache 2.0](LICENSE) 许可证。

---

## 致谢

- 参考项目：HanLP、spaCy、jieba、pkuseg、THULAC
- 数据资源：人民日报语料库、搜狗细胞词库、开放词林

---

## 联系方式

- GitHub: [AuroraNLP/AuroraNLP](https://github.com/AuroraNLP/AuroraNLP)
- Issues: [GitHub Issues](https://github.com/AuroraNLP/AuroraNLP/issues)
- 讨论: [GitHub Discussions](https://github.com/AuroraNLP/AuroraNLP/discussions)

---

<div align="center">

**如果这个项目对你有帮助，请给一个 ⭐️ Star！**

</div>