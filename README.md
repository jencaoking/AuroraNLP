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

AuroraNLP 是一个专业级的中文自然语言处理工具包，提供多种分词算法、词性标注、命名实体识别、句法分析、关键词提取、文本相似度计算等功能。项目采用纯 Python 实现，无重度依赖，易于安装和使用，同时具备企业级特性支持。

## 核心特性

- 🔤 **多种分词算法** - 支持正向/逆向/双向最大匹配、HMM、CRF、感知器、Lattice、最短路径等多种分词方法
- 🏷️ **词性标注** - 基于 HMM 和 CRF 的词性标注
- 🔍 **命名实体识别** - 支持 CRF-based NER，包括实体嵌套识别和实体链接
- 🌳 **句法分析** - 支持依存句法分析（Arc-eager）和成分句法分析（PCFG/CKY）
- 📚 **词典管理** - 支持系统词典和用户词典，支持优先级和权重配置，支持版本管理和增量更新
- 🔎 **新词发现** - 基于互信息和信息熵的新词发现算法
- 📊 **歧义处理** - 支持交叉歧义和组合歧义检测与消解
- 📱 **领域词典** - 支持电商、法律、医疗、新闻等多个领域的专业词典
- 🧠 **情感分析** - 支持情感极性分析和情感强度计算
- 🌐 **网络词典** - 支持网络新词和流行词汇的识别
- 🏛️ **机构名识别** - 支持组织机构名称的识别和归一化
- 📍 **地名识别** - 支持地理位置名称的识别和归一化
- 👤 **人名识别** - 支持人名的识别和归一化
- 🔄 **繁体中文** - 支持繁简转换和地区差异处理
- 🚀 **企业级特性** - 日志系统、健康检查、Prometheus指标、限流熔断、认证授权、配置管理

## 快速开始

### 环境要求

- Python 3.8+
- 无其他强制依赖（深度学习功能需额外安装 PyTorch）

### 安装

```bash
# 克隆仓库
git clone https://github.com/AuroraNLP/AuroraNLP.git
cd AuroraNLP

# 安装基础版本
pip install -e AuroraNLP/

# 安装开发依赖
pip install -e "AuroraNLP/[dev]"

# 安装全部依赖
pip install -e "AuroraNLP/[all]"
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
```

## 项目结构

```
AuroraNLP/
├── AuroraNLP/              # 核心源代码
│   ├── .github/            # GitHub 配置
│   ├── AuroraNLP/          # 主要源码目录
│   │   ├── __init__.py     # 包初始化
│   │   ├── data/           # 数据文件（词典、停用词、语料等）
│   │   ├── deep_learning/  # 深度学习模块
│   │   └── *.py            # 各功能模块
│   ├── docs/               # 文档
│   ├── examples/           # 示例代码
│   ├── tests/              # 测试文件
│   ├── setup.py            # 安装配置
│   └── requirements.txt    # 依赖列表
├── LICENSE                 # 许可证
└── README.md               # 说明文档
```

## 详细文档

请查看 `AuroraNLP/docs/` 目录获取详细文档：
- [快速开始](AuroraNLP/docs/getting-started.md)
- [用户手册](AuroraNLP/docs/user-guide.md)
- [最佳实践](AuroraNLP/docs/best-practices.md)
- [FAQ](AuroraNLP/docs/faq.md)
- [CRF 文档](AuroraNLP/docs/CRF.md)
- [HMM 文档](AuroraNLP/docs/HMM.md)
- [路线图](AuroraNLP/docs/ROADMAP.md)

## 示例

查看 `AuroraNLP/examples/` 目录获取示例代码：
- [基础分词](AuroraNLP/examples/basic_segmentation.py)
- [BiLSTM-CRF 示例](AuroraNLP/examples/bilstm_crf_example.py)
- [命名实体识别](AuroraNLP/examples/ner.py)
- [Pipeline 使用](AuroraNLP/examples/pipeline.py)

## 运行测试

```bash
cd AuroraNLP
pytest tests/ -v
```

## 许可证

本项目采用 [Apache 2.0](LICENSE) 许可证。

---

<div align="center">

**如果这个项目对你有帮助，请给一个 ⭐️ Star！**

</div>