# AuroraNLP

<div align="center">

[![Version](https://img.shields.io/badge/version-0.3.0--beta-blue.svg)](https://github.com/yourusername/AuroraNLP/releases)
[![Codename](https://img.shields.io/badge/codename-coca-purple.svg)]()
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-Beta-yellow.svg)]()

**AuroraNLP - 轻量级中文自然语言处理工具包**

[快速开始](#快速开始) • [功能特性](#功能特性) • [文档](#文档) • [示例](#示例)

</div>

---

## 简介

AuroraNLP 是一个轻量级的中文自然语言处理工具包，提供多种分词算法、词性标注、命名实体识别、句法分析、关键词提取、文本相似度计算等功能。项目采用纯 Python 实现，无重度依赖，易于安装和使用。

### 核心特性

- 🔤 **多种分词算法** - 支持正向/逆向/双向最大匹配、HMM、CRF、感知器、Lattice、最短路径等多种分词方法
- 🏷️ **词性标注** - 基于 HMM 和 CRF 的词性标注
- 🔍 **命名实体识别** - 支持 CRF-based NER，包括实体嵌套识别和实体链接
- 🌳 **句法分析** - 支持依存句法分析（Arc-eager）和成分句法分析（PCFG/CKY）
- 📚 **词典管理** - 支持系统词典和用户词典，支持优先级和权重配置
- 🔎 **新词发现** - 基于互信息和信息熵的新词发现算法
- 📊 **歧义处理** - 支持交叉歧义和组合歧义检测与消解
- 🔍 **关键词提取** - 支持 TF-IDF、TextRank、词频统计等多种关键词提取方法
- 📐 **文本相似度** - 支持余弦相似度、Jaccard、Dice、编辑距离等多种相似度计算
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
| 最短路径 | `shortest_path` | Dijkstra算法，最优路径选择 |
| 混合模式 | `hybrid` | 规则+统计+深度学习融合 |

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

# 最短路径分词
words = seg.segment(text, mode='shortest_path')

# 混合分词
words = seg.segment(text, mode='hybrid')
```

### 词性标注

```python
# 基于 HMM 的词性标注
pos_tags = seg.tag_pos_hmm(text)

# 基于 CRF 的词性标注
pos_tags = seg.tag_pos_crf(text)

# 分词并标注词性
words_with_pos = seg.segment_with_pos(text)
```

### 命名实体识别

```python
# 基于 CRF 的 NER
entities = seg.recognize_entities_crf("张三在北京工作")
print(entities)  # [('张三', 'PERSON'), ('北京', 'LOCATION')]

# 实体嵌套识别
nested_entities = seg.recognize_nested_entities("中国科学院北京基因组研究所")
print(nested_entities)  # 支持多层级实体结构

# 实体链接
linked_entities = seg.link_entities("马云在阿里巴巴工作")
print(linked_entities)  # 实体归一化到知识库
```

### 句法分析

```python
# 依存句法分析（基于 Arc-eager 算法）
dependencies = seg.parse_dependency("我爱自然语言处理")
for dep in dependencies:
    print(f"{dep['word']} -> {dep['head']} ({dep['relation']})")

# 成分句法分析（基于 PCFG/CKY 算法）
constituency_tree = seg.parse_constituency("我爱自然语言处理")
print(constituency_tree)
```

### 歧义检测与消解

```python
# 检测交叉歧义
ambiguities = seg.detect_ambiguity("研究生命的起源")
print(ambiguities)  # [('研究', '生命'), ('研究生', '命')]

# 检测组合歧义
comb_ambiguities = seg.detect_combination_ambiguity("南京市长江大桥")

# Lattice 歧义消解
words = seg.segment("研究生命", mode='lattice')
```

### 新词发现

```python
# 基于互信息和信息熵的新词发现
new_words = seg.discover_new_words(corpus, min_pmi=3.0, min_entropy=1.0)
print(new_words)

# 自动扩充词典
seg.auto_expand_dictionary(corpus)
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

### 依存句法分析示例

```python
from AuroraNLP import Segmentor

seg = Segmentor()

# 依存句法分析
text = "我爱自然语言处理"
result = seg.parse_dependency(text)

for item in result:
    print(f"词语: {item['word']}, 词性: {item['pos']}, 依存关系: {item['relation']}, 指向: {item['head']}")
```

### 成分句法分析示例

```python
from AuroraNLP import Segmentor

seg = Segmentor()

# 成分句法分析
text = "我爱自然语言处理"
tree = seg.parse_constituency(text)

# 打印句法树
print(tree.pretty_print())
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
| `tag_pos_hmm(text)` | HMM 词性标注 |
| `tag_pos_crf(text)` | CRF 词性标注 |
| `recognize_entities_crf(text)` | CRF 命名实体识别 |
| `recognize_nested_entities(text)` | 嵌套实体识别 |
| `link_entities(text)` | 实体链接 |
| `parse_dependency(text)` | 依存句法分析 |
| `parse_constituency(text)` | 成分句法分析 |
| `detect_ambiguity(text)` | 歧义检测 |
| `discover_new_words(corpus)` | 新词发现 |
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
│   ├── viterbi.py          # Viterbi 算法
│   ├── crf.py              # CRF 模型
│   ├── perceptron.py       # 感知器模型
│   ├── lattice.py          # Lattice 分词
│   ├── shortest_path.py    # 最短路径分词
│   ├── ngram.py            # N-gram 语言模型
│   ├── pos_tagger.py       # 词性标注
│   ├── ner.py              # 命名实体识别
│   ├── entity_linker.py    # 实体链接
│   ├── dependency_parser.py # 依存句法分析
│   ├── constituency_parser.py # 成分句法分析
│   ├── ambiguity.py        # 歧义检测
│   ├── new_word_detector.py # 新词发现
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
│   ├── test_dependency_parser.py
│   ├── test_constituency_parser.py
│   └── ...
├── examples/               # 示例代码
│   ├── demo_crf.py
│   ├── demo_hmm.py
│   ├── demo_shortest_path.py
│   ├── demo_dependency_parser.py
│   └── demo_constituency_parser.py
├── docs/                   # 文档
│   ├── CRF.md
│   ├── HMM.md
│   ├── ROADMAP.md
│   ├── DEPENDENCY_PARSING.md
│   └── CONSTITUENCY_PARSING.md
├── setup.py                # 安装配置
├── requirements.txt        # 依赖列表
└── README.md               # 说明文档
```

---

## 版本信息

### 当前版本 (v0.3.0-beta "coca")

**版本号**: 0.3.0-beta  
**内部代号**: coca  
**发布日期**: 2026-04-05  
**开发状态**: Beta

#### 阶段一完成功能（1-20步）

**核心算法**
- ✅ HMM 隐马尔可夫模型（序列标注 B/M/E/S）
- ✅ Viterbi 算法（最优路径解码）
- ✅ N-gram 语言模型（词序列概率计算）
- ✅ 二元语法模型（相邻词搭配概率）
- ✅ CRF 条件随机场（精确序列标注）
- ✅ 感知器分词器（在线学习、增量更新）
- ✅ Word Lattice 词格解码（多路径搜索）
- ✅ 最短路径分词（Dijkstra算法）

**词典与歧义**
- ✅ 用户词典增强（优先级机制、权重配置）
- ✅ 歧义检测模块（交叉歧义、组合歧义识别）
- ✅ 新词发现算法（基于统计）
- ✅ 互信息计算（字符共现统计）
- ✅ 信息熵计算（左右熵、边界确定性）

**高级功能**
- ✅ 词性标注模型（HMM + CRF）
- ✅ 命名实体识别模型（CRF-based NER）
- ✅ 实体嵌套识别（多层级实体结构）
- ✅ 实体链接（实体归一化、知识库对接）
- ✅ 依存句法分析（Arc-eager算法）
- ✅ 成分句法分析（PCFG/CKY算法）
- ✅ 混合分词架构（规则+统计+深度学习融合）

#### 已实现功能

- ✅ 基础分词算法（正向/逆向/双向最大匹配）
- ✅ HMM 隐马尔可夫模型
- ✅ CRF 条件随机场
- ✅ 感知器分词器
- ✅ Lattice 词格解码
- ✅ 最短路径分词
- ✅ N-gram 语言模型
- ✅ 词性标注（HMM + CRF）
- ✅ 命名实体识别
- ✅ 实体嵌套识别
- ✅ 实体链接
- ✅ 依存句法分析
- ✅ 成分句法分析
- ✅ 歧义检测与消解
- ✅ 新词发现
- ✅ 关键词提取（TF-IDF, TextRank, 词频）
- ✅ 文本相似度计算（5 种算法）
- ✅ 用户词典管理
- ✅ 停用词过滤
- ✅ 批量处理接口
- ✅ 性能基准测试工具

---

## 路线图

详见 [ROADMAP.md](docs/ROADMAP.md)

### 开发进度

| 阶段 | 步骤范围 | 状态 | 核心目标 |
|------|----------|------|----------|
| 阶段一：算法升级 | 1-20 | ✅ 已完成 | 统计模型+深度学习混合架构 |
| 阶段二：数据资源扩展 | 21-35 | 📋 计划中 | 大规模词典和语料库 |
| 阶段三：深度学习集成 | 36-50 | 📋 计划中 | BERT等预训练模型 |
| 阶段四：架构重构 | 51-65 | 📋 计划中 | Pipeline模块化架构 |
| 阶段五：性能优化 | 66-80 | 📋 计划中 | Cython+GPU加速 |
| 阶段六：企业级功能 | 81-90 | 📋 计划中 | 可靠性+可观测性 |
| 阶段七：生态建设 | 91-100 | 📋 计划中 | 文档+社区+生态 |

### 下一版本计划（v0.4.0）

- [ ] 搜狗细胞词库整合
- [ ] 开放词林整合
- [ ] 人名/地名/机构名词库构建
- [ ] 专业术语库
- [ ] 网络新词库
- [ ] 情感词典

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
