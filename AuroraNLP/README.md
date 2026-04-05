# AuroraNLP

<div align="center">

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-Alpha-orange.svg)]()

**AuroraNLP - 轻量级中文自然语言处理工具包**

[快速开始](#快速开始) • [功能特性](#功能特性) • [文档](#文档) • [示例](#示例)

</div>

---

## 简介

AuroraNLP 是一个轻量级的中文自然语言处理工具包，提供多种分词算法、关键词提取、文本相似度计算等功能。项目采用纯 Python 实现，无重度依赖，易于安装和使用。

### 核心特性

- 🔤 **多种分词算法** - 支持正向/逆向/双向最大匹配、HMM、CRF、感知器、Lattice 等多种分词方法
- 📚 **词典管理** - 支持系统词典和用户词典，支持优先级和权重配置
- 🔍 **关键词提取** - 支持 TF-IDF、TextRank、词频统计等多种关键词提取方法
- 📊 **文本相似度** - 支持余弦相似度、Jaccard、Dice、编辑距离等多种相似度计算
- 🏷️ **词性标注** - 基于词典的词性标注
- ⚡ **高性能** - 使用 Trie 树优化词典查询，支持批量处理

---

## 目录

- [快速开始](#快速开始)
- [安装](#安装)
- [功能特性](#功能特性)
- [示例](#示例)
- [API 文档](#api-文档)
- [项目结构](#项目结构)
- [路线图](#路线图)
- [贡献指南](#贡献指南)
- [许可证](#许可证)

---

## 快速开始

### 环境要求

- Python 3.8+
- 无其他强制依赖

### 安装

```bash
# 克隆仓库
git clone https://github.com/yourusername/AuroraNLP.git
cd AuroraNLP

# 安装
pip install -e .

# 或安装开发依赖
pip install -e ".[dev]"
```

### 基础使用

```python
from AuroraNLP import Segmentor

# 创建分词器
seg = Segmentor()

# 基础分词
text = "我爱自然语言处理"
words = seg.segment(text)
print(words)  # ['我', '爱', '自然语言处理']

# 带词性标注的分词
words_with_pos = seg.segment_with_pos(text)
print(words_with_pos)  # [('我', 'r'), ('爱', 'v'), ('自然语言处理', 'n')]

# 去除停用词
words_no_stop = seg.segment_without_stopwords(text)
print(words_no_stop)

# 关键词提取
keywords = seg.extract_keywords(text, top_k=5, method='tfidf')
print(keywords)

# 文本相似度计算
text1 = "我爱编程"
text2 = "我喜欢写代码"
similarity = seg.compute_similarity(text1, text2, method='cosine')
print(f"相似度: {similarity:.4f}")
```

---

## 功能特性

### 分词算法

AuroraNLP 支持多种分词算法，可根据场景选择：

| 算法 | 模式 | 特点 |
|------|------|------|
| 正向最大匹配 | `forward` | 基于词典，速度快 |
| 逆向最大匹配 | `backward` | 基于词典，准确率略高 |
| 双向最大匹配 | `bidirectional` | 默认模式，结合正逆向优点 |
| HMM | `hmm` | 统计模型，支持未登录词识别 |
| CRF | `crf` | 序列标注，准确率高 |
| 感知器 | `perceptron` | 在线学习，支持增量更新 |
| Lattice | `lattice` | 词格解码，支持歧义消解 |

```python
# 使用不同分词模式
seg = Segmentor()

# 词典分词（默认双向最大匹配）
words = seg.segment(text, mode='bidirectional')

# HMM 分词（需要先训练）
seg.train_hmm(corpus)
words = seg.segment(text, mode='hmm')

# CRF 分词（需要先训练）
seg.train_crf(corpus)
words = seg.segment(text, mode='crf')
```

### 词典管理

```python
# 添加自定义词汇
seg.add_word("自然语言处理", pos_tag='n', weight=10.0)

# 添加停用词
seg.add_stopword("的")

# 加载用户词典
seg.load_user_dictionary("custom_dict.txt")

# 创建多个用户词典
user_dict = seg.create_user_dictionary("my_dict", priority=100)
user_dict.add_word("深度学习", pos_tag='n', weight=15.0)
```

### 关键词提取

```python
# TF-IDF 方法
keywords = seg.extract_keywords(text, top_k=10, method='tfidf')

# TextRank 方法
keywords = seg.extract_keywords(text, top_k=10, method='textrank')

# 词频统计
keywords = seg.extract_keywords(text, top_k=10, method='freq')
```

### 文本相似度

```python
# 余弦相似度
sim = seg.compute_similarity(text1, text2, method='cosine')

# Jaccard 相似度
sim = seg.compute_similarity(text1, text2, method='jaccard')

# 编辑距离
sim = seg.compute_similarity(text1, text2, method='edit')

# 批量相似度计算
documents = ["文本1", "文本2", "文本3"]
results = seg.batch_similarity(query, documents, method='cosine')
```

### 命名实体识别

```python
# 基于词性标注的简单实体识别
entities = seg.recognize_entities("张三在北京工作")
print(entities)  # [('张三', 'PERSON'), ('北京', 'LOCATION')]

# 分词并标注实体类型
result = seg.segment_with_entities("张三在北京工作")
print(result)  # [('张三', 'nr', 'PERSON'), ('在', 'p', 'O'), ...]
```

---

## 示例

### HMM 分词示例

```python
from AuroraNLP import Segmentor

seg = Segmentor()

# 训练语料（词列表的列表）
corpus = [
    ['我', '爱', '自然语言处理'],
    ['今天', '天气', '很好'],
    ['北京', '是', '中国', '的', '首都']
]

# 训练 HMM 模型
seg.train_hmm(corpus)

# 分词
text = "我爱自然语言处理"
words = seg.segment(text, mode='hmm')
print(words)

# 保存模型
seg.save_hmm_model('hmm_model.pkl')

# 加载模型
seg.load_hmm_model('hmm_model.pkl')
```

### CRF 分词示例

```python
from AuroraNLP import Segmentor

seg = Segmentor()

# 训练语料
corpus = [
    ['我', '爱', '中国'],
    ['他', '是', '学生'],
    ['北京', '是', '首都']
]

# 训练 CRF 模型
seg.train_crf(corpus, max_iter=100, verbose=True)

# 分词
text = "我爱北京"
words = seg.segment(text, mode='crf')
print(words)

# 保存模型
seg.save_crf_model('crf_model.pkl')
```

### Lattice 分词示例

```python
from AuroraNLP import Segmentor

seg = Segmentor()

# 使用 Lattice 分词
words = seg.segment("研究生命的起源", mode='lattice')
print(words)

# 获取所有可能的分词结果
all_results = seg.get_all_lattice_segmentations("南京市长江大桥", max_results=10)
for result in all_results:
    print(result)

# 检测歧义
ambiguities = seg.detect_lattice_ambiguity("研究生命")
print(ambiguities)
```

---

## API 文档

### Segmentor 类

主要分词器类，整合所有分词功能。

#### 初始化参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `dictionary` | Dictionary | None | 自定义词典 |
| `load_default_dict` | bool | True | 是否加载默认词典 |
| `load_default_stopwords` | bool | True | 是否加载默认停用词 |
| `use_hmm` | bool | False | 是否启用 HMM |
| `use_crf` | bool | False | 是否启用 CRF |
| `use_perceptron` | bool | False | 是否启用感知器 |
| `use_lattice` | bool | False | 是否启用 Lattice |

#### 主要方法

| 方法 | 说明 |
|------|------|
| `segment(text, mode)` | 分词 |
| `segment_with_pos(text, mode)` | 分词并返回词性 |
| `segment_without_stopwords(text, mode)` | 分词并过滤停用词 |
| `extract_keywords(text, top_k, method)` | 关键词提取 |
| `compute_similarity(text1, text2, method)` | 文本相似度计算 |
| `add_word(word, pos_tag, weight)` | 添加词汇 |
| `add_stopword(word)` | 添加停用词 |
| `train_hmm(corpus)` | 训练 HMM 模型 |
| `train_crf(corpus)` | 训练 CRF 模型 |

---

## 项目结构

```
AuroraNLP/
├── AuroraNLP/              # 核心源代码
│   ├── __init__.py         # 包初始化
│   ├── segmentor.py        # 主分词器
│   ├── tokenizer.py        # 分词算法接口
│   ├── dictionary.py       # 词典管理
│   ├── trie.py             # Trie 树实现
│   ├── hmm.py              # HMM 模型
│   ├── crf.py              # CRF 模型
│   ├── perceptron.py       # 感知器模型
│   ├── lattice.py          # Lattice 分词
│   ├── ngram.py            # N-gram 语言模型
│   ├── keyword_extractor.py # 关键词提取
│   ├── similarity.py       # 文本相似度
│   ├── stopwords.py        # 停用词处理
│   ├── batch_processor.py  # 批量处理
│   ├── benchmark.py        # 性能测试
│   └── data/               # 数据文件
│       ├── dictionary.txt  # 默认词典
│       ├── stopwords.txt   # 默认停用词
│       └── train_corpus.txt # 训练语料
├── tests/                  # 测试文件
│   ├── test_crf.py
│   ├── test_hmm.py
│   ├── test_lattice.py
│   └── ...
├── examples/               # 示例代码
│   ├── demo_crf.py
│   ├── demo_hmm.py
│   └── demo_shortest_path.py
├── docs/                   # 文档
│   ├── CRF.md
│   ├── HMM.md
│   └── ROADMAP.md
├── setup.py                # 安装配置
├── requirements.txt        # 依赖列表
└── README.md               # 说明文档
```

---

## 路线图

详见 [ROADMAP.md](docs/ROADMAP.md)

### 当前版本 (v0.1.0)

- [x] 基础分词算法（最大匹配）
- [x] HMM 隐马尔可夫模型
- [x] CRF 条件随机场
- [x] 感知器分词器
- [x] Lattice 词格解码
- [x] N-gram 语言模型
- [x] 关键词提取
- [x] 文本相似度

### 计划功能

- [ ] 词性标注模型
- [ ] 命名实体识别
- [ ] 新词发现
- [ ] 深度学习模型集成
- [ ] 性能优化（Cython）

---

## 贡献指南

欢迎贡献代码、报告问题或提出建议！

### 开发环境

```bash
# 克隆仓库
git clone https://github.com/yourusername/AuroraNLP.git
cd AuroraNLP

# 安装开发依赖
pip install -e ".[dev]"

# 运行测试
pytest tests/
```

### 提交规范

- `feat`: 新功能
- `fix`: 修复 bug
- `docs`: 文档更新
- `test`: 测试相关
- `refactor`: 代码重构

---

## 许可证

本项目采用 [Apache 2.0](LICENSE) 许可证。

---

## 联系方式

- 项目主页: [GitHub](https://github.com/yourusername/AuroraNLP)
- 问题反馈: [Issues](https://github.com/yourusername/AuroraNLP/issues)

---

<div align="center">

**如果这个项目对你有帮助，请给一个 ⭐️ Star！**

</div>
