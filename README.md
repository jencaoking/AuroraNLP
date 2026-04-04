# NLP 分词工具包 (nlp_segment)

![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-Apache%202.0-green)
![Status](https://img.shields.io/badge/Status-Alpha-orange)

一个轻量级的中文文本分词工具包，基于词典的最大匹配算法实现，支持多种分词模式。

---

## 目录

- [项目概述与目标](#项目概述与目标)
- [核心功能与特性](#核心功能与特性)
- [技术栈与架构说明](#技术栈与架构说明)
- [环境配置与依赖安装](#环境配置与依赖安装)
- [详细使用指南](#详细使用指南)
- [API 接口文档](#api-接口文档)
- [目录结构](#目录结构)
- [贡献指南](#贡献指南)
- [许可证](#许可证)
- [联系方式与问题反馈](#联系方式与问题反馈)

---

## 项目概述与目标

### 项目简介

`nlp_segment` 是一个轻量级的中文自然语言处理分词工具包，采用基于词典的最大匹配算法（Maximum Matching）进行中文文本分词。工具包设计简洁、依赖少，易于集成到各类 NLP 应用场景中。

### 核心目标

- 提供高效、准确的中文分词能力
- 支持多种分词策略以适应不同场景需求
- 保持轻量级架构，便于二次开发和定制
- 为更复杂的 NLP 任务（如命名实体识别、关键词提取等）提供基础支持

### 应用场景

- 文本预处理与清洗
- 搜索引擎索引构建
- 文本分类前处理
- 教育领域的汉语分词教学
- 快速原型开发

---

## 核心功能与特性

### 主要功能

| 功能 | 说明 |
|------|------|
| 正向最大匹配 (FMM) | 从文本开头向后扫描，优先匹配最长词 |
| 逆向最大匹配 (BMM) | 从文本末尾向前扫描，优先匹配最长词 |
| 双向最大匹配 (BMMM) | 综合正逆向结果，选择最优分词方案 |
| 词典管理 | 支持加载自定义词典、动态添加词汇 |

### 核心特性

- **轻量级设计**：无重型依赖，仅需 Python 3.8+
- **多种分词模式**：支持正向、逆向、双向三种分词策略
- **可扩展词典**：支持从文件加载自定义词典
- **灵活配置**：可自定义最大词长、分词模式
- **易于集成**：简洁的 API 设计，便于嵌入其他项目

### 算法说明

#### 最大匹配算法原理

最大匹配算法是一种基于词典的贪婪分词方法，核心思想是：

1. 从待分词文本的起始位置开始
2. 尝试匹配尽可能长的词（最大长度限制内）
3. 匹配成功则将该词作为一个分词结果
4. 匹配失败则将单字作为未登录词处理
5. 重复步骤 1-4 直到文本结束

#### 正向最大匹配 (Forward Max Match)

```
文本: "今天天气很好"
词典: ["今天", "天气", "很好", "今天天气"]

步骤:
1. 尝试匹配 "今天天气" → 成功 → 输出 "今天天气"
2. 尝试匹配 "很好" → 成功 → 输出 "很好"
结果: ["今天天气", "很好"]
```

#### 双向最大匹配

当正逆向分词结果不一致时，通过以下策略选择最优结果：

1. 比较单字未登录词数量，选择单字较少的结果
2. 若单字数量相同，选择词数较少的结果

---

## 技术栈与架构说明

### 技术选型

| 类别 | 技术/工具 |
|------|----------|
| 编程语言 | Python 3.8+ |
| 分词算法 | 最大匹配算法 (Maximum Matching) |
| 测试框架 | pytest 7.0.0+ |
| 包管理 | setuptools |

### 架构设计

```
nlp_segment/
├── nlp_segment/          # 主包
│   ├── __init__.py       # 包导出
│   ├── segmentor.py      # 分词器核心类
│   ├── tokenizer.py      # 分词算法实现
│   ├── trie.py           # Trie 树实现
│   └── dictionary.py     # 词典管理类
├── tests/                # 测试用例
│   └── test_segment.py   # 单元测试
├── requirements.txt      # 依赖清单
└── setup.py              # 包配置
```

### 核心组件

#### Dictionary 类

词典管理组件，负责存储和管理词汇集合。

- `load_dictionary(path)`: 从文件加载词典
- `add_word(word)`: 添加新词
- `search_in_dict(word)`: 查询词是否存在
- `get_words()`: 获取所有词汇

#### Segmentor 类

分词器核心类，提供统一的分词接口。

- `segment(text, mode=None)`: 执行分词
- `set_mode(mode)`: 设置分词模式
- `load_dictionary(path)`: 加载词典文件

#### Tokenizer 模块

分词算法实现模块，包含三种分词算法：

- `forward_max_match(text, dictionary, max_len=15)`
- `backward_max_match(text, dictionary, max_len=15)`
- `bidirectional_max_match(text, dictionary, max_len=15)`

---

## 环境配置与依赖安装

### 系统要求

- Python 3.8 或更高版本
- 操作系统：Windows / macOS / Linux

### 安装步骤

#### 方式一：通过 pip 安装（待发布）

```bash
pip install nlp_segment
```

#### 方式二：源码安装

```bash
# 克隆项目
git clone https://github.com/jencaoking/alphaNLP.git
cd nlp_segment

# 安装依赖
pip install -e .

# 或仅安装核心依赖
pip install -e . --no-deps
```

#### 方式三：开发模式安装

```bash
# 安装包含开发依赖的完整版本
pip install -e ".[dev]"

# 运行测试
pytest tests/
```

### 验证安装

```bash
python -c "from nlp_segment import Segmentor; print('安装成功')"
```

---

## 详细使用指南

### 基础使用

#### 1. 快速开始

```python
from nlp_segment import Segmentor
from nlp_segment.dictionary import Dictionary

# 创建分词器（使用默认空词典）
segmentor = Segmentor()

# 执行分词
text = "今天天气很好"
result = segmentor.segment(text)
print(result)
# 输出: ['今天', '天气', '很', '好']
```

#### 2. 使用自定义词典

```python
from nlp_segment import Segmentor
from nlp_segment.dictionary import Dictionary

# 创建词典并添加词汇
dictionary = Dictionary()
dictionary.add_word("今天")
dictionary.add_word("天气")
dictionary.add_word("很好")

# 创建带词典的分词器
segmentor = Segmentor(dictionary)

# 执行分词
text = "今天天气很好"
result = segmentor.segment(text)
print(result)
# 输出: ['今天', '天气', '很好']
```

#### 3. 从文件加载词典

```python
from nlp_segment import Segmentor

# 创建分词器
segmentor = Segmentor()

# 从文件加载词典（词典文件每行一个词，UTF-8编码）
segmentor.load_dictionary("path/to/dictionary.txt")

# 执行分词
result = segmentor.segment("我爱中国")
print(result)
```

**词典文件格式示例** (`dictionary.txt`):

```
今天
天气
很好
中国
我爱
```

### 分词模式切换

```python
from nlp_segment import Segmentor
from nlp_segment.dictionary import Dictionary

# 创建词典
d = Dictionary()
d.add_word("研究")
d.add_word("研究生命")
d.add_word("生命")
d.add_word("起源")

# 创建分词器
seg = Segmentor(d)

# 使用不同模式
text = "研究生命起源"

# 正向最大匹配
result_fmm = seg.segment(text, mode='forward')
print(f"正向匹配: {result_fmm}")

# 逆向最大匹配
result_bmm = seg.segment(text, mode='backward')
print(f"逆向匹配: {result_bmm}")

# 双向最大匹配（默认）
result_bmmmm = seg.segment(text, mode='bidirectional')
print(f"双向匹配: {result_bmmmm}")
```

### 基础算法直接调用

```python
from nlp_segment.tokenizer import (
    forward_max_match,
    backward_max_match,
    bidirectional_max_match
)
from nlp_segment.dictionary import Dictionary

# 创建词典
d = Dictionary()
d.add_word("今天")
d.add_word("天气")
d.add_word("很好")

text = "今天天气很好"

# 直接调用算法函数
result1 = forward_max_match(text, d)
result2 = backward_max_match(text, d)
result3 = bidirectional_max_match(text, d)

print(f"正向: {result1}")
print(f"逆向: {result2}")
print(f"双向: {result3}")
```

### 完整使用示例

```python
"""
NLP 分词工具完整使用示例
"""

from nlp_segment import Segmentor
from nlp_segment.dictionary import Dictionary

def main():
    # 初始化词典
    dictionary = Dictionary()

    # 方式一：逐个添加词汇
    words_to_add = [
        "自然语言处理",
        "分词",
        "中文",
        "工具包",
        "最大匹配",
        "正向匹配",
        "逆向匹配"
    ]
    for word in words_to_add:
        dictionary.add_word(word)

    # 方式二：从文件加载（可选）
    # dictionary.load_dictionary("custom_dictionary.txt")

    # 创建分词器
    segmentor = Segmentor(dictionary)

    # 测试文本
    test_texts = [
        "自然语言处理是人工智能的重要分支",
        "中文分词是NLP的基础任务",
        "最大匹配算法是一种经典的分词方法"
    ]

    # 执行分词
    print("=" * 50)
    print("分词结果展示")
    print("=" * 50)

    for text in test_texts:
        result = segmentor.segment(text)
        print(f"\n原文: {text}")
        print(f"分词: {' | '.join(result)}")

    # 模式切换示例
    print("\n" + "=" * 50)
    print("分词模式对比")
    print("=" * 50)

    text = "自然语言处理很重要"
    print(f"\n原文: {text}")
    print(f"正向匹配: {segmentor.segment(text, mode='forward')}")
    print(f"逆向匹配: {segmentor.segment(text, mode='backward')}")
    print(f"双向匹配: {segmentor.segment(text, mode='bidirectional')}")

if __name__ == "__main__":
    main()
```

---

## API 接口文档

### Segmentor 类

分词器核心类，提供统一的分词接口。

#### 构造函数

```python
Segmentor(dictionary=None)
```

**参数说明：**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| dictionary | Dictionary | 否 | None | 词典实例，若不提供则创建空词典 |

**示例：**

```python
# 使用空词典
seg1 = Segmentor()

# 使用自定义词典
d = Dictionary()
d.add_word("词1")
d.add_word("词2")
seg2 = Segmentor(d)
```

#### segment() 方法

执行文本分词。

```python
segment(text, mode=None)
```

**参数说明：**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| text | str | 是 | - | 待分词的中文文本 |
| mode | str | 否 | None | 分词模式：'forward'、'backward'、'bidirectional' |

**返回值：** `List[str]` - 分词结果列表

**分词模式说明：**

| 模式 | 说明 |
|------|------|
| forward | 正向最大匹配 |
| backward | 逆向最大匹配 |
| bidirectional | 双向最大匹配（默认） |

**示例：**

```python
seg = Segmentor()
result = seg.segment("今天天气很好")
# ['今天', '天气', '很', '好']

result = seg.segment("今天天气很好", mode='forward')
# ['今天', '天气', '很', '好']
```

#### set_mode() 方法

设置默认分词模式。

```python
set_mode(mode)
```

**参数说明：**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| mode | str | 是 | 分词模式 |

**示例：**

```python
seg = Segmentor()
seg.set_mode('forward')  # 设置默认模式为正向匹配
result = seg.segment("今天天气很好")  # 使用正向匹配
```

#### load_dictionary() 方法

从文件加载词典。

```python
load_dictionary(path)
```

**参数说明：**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| path | str | 是 | 词典文件路径 |

**文件格式：** 每行一个词语，UTF-8 编码

**示例：**

```python
seg = Segmentor()
seg.load_dictionary("dictionary.txt")
```

---

### Dictionary 类

词典管理类，负责存储和管理词汇集合。

#### 构造函数

```python
Dictionary()
```

**示例：**

```python
d = Dictionary()
```

#### load_dictionary() 方法

从文件加载词典。

```python
load_dictionary(path)
```

**参数说明：**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| path | str | 是 | 词典文件路径 |

#### add_word() 方法

添加新词到词典。

```python
add_word(word)
```

**参数说明：**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| word | str | 是 | 要添加的词语 |

#### search_in_dict() 方法

查询词是否存在于词典中。

```python
search_in_dict(word)
```

**返回值：** `bool` - 词存在返回 True，否则返回 False

#### get_words() 方法

返回词典中所有词汇。

```python
get_words()
```

**返回值：** `set` - 词汇集合

---

### tokenizer 模块函数

#### forward_max_match()

正向最大匹配算法。

```python
forward_max_match(text, dictionary, max_len=15)
```

**参数说明：**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| text | str | 是 | - | 待分词文本 |
| dictionary | Dictionary | 是 | - | 词典实例 |
| max_len | int | 否 | 15 | 最大词长 |

**返回值：** `List[str]` - 分词结果

#### backward_max_match()

逆向最大匹配算法。

```python
backward_max_match(text, dictionary, max_len=15)
```

#### bidirectional_max_match()

双向最大匹配算法。

```python
bidirectional_max_match(text, dictionary, max_len=15)
```

---

## 目录结构

```
nlp_segment/
├── nlp_segment/                 # 主包目录
│   ├── __init__.py              # 包导出，公开 Segmentor 类
│   ├── segmentor.py             # 分词器核心类
│   ├── tokenizer.py             # 分词算法实现
│   ├── trie.py                  # Trie 树实现
│   └── dictionary.py            # 词典管理类
├── tests/                        # 测试目录
│   ├── __init__.py
│   └── test_segment.py          # 单元测试
├── nlp_segment.egg-info/         # 包安装信息
├── .pytest_cache/               # pytest 缓存
├── requirements.txt             # 依赖清单
├── setup.py                     # 包配置
└── README.md                    # 项目文档
```

---

## 贡献指南

### 开发环境搭建

```bash
# 1. 克隆项目
git clone <repository-url>
cd nlp_segment

# 2. 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或
venv\Scripts\activate  # Windows

# 3. 安装开发依赖
pip install -e ".[dev]"

# 4. 运行测试
pytest tests/ -v
```

### 代码规范

- 遵循 PEP 8 代码风格
- 使用有意义的变量和函数命名
- 为公共 API 添加 docstring 文档
- 新功能需包含对应的单元测试

### 分支管理

- `main`: 主分支，稳定版本
- `develop`: 开发分支
- `feature/*`: 功能分支
- `fix/*`: 修复分支

### 提交规范

```
<类型>(<模块>): <描述>

可选的详细说明

[可选的脚注]
```

**类型标识：**

| 标识 | 说明 |
|------|------|
| feat | 新功能 |
| fix | 修复 Bug |
| docs | 文档更新 |
| style | 代码格式调整 |
| refactor | 重构 |
| test | 测试相关 |
| chore | 构建/工具相关 |

**提交示例：**

```
feat(segmentor): 添加双向匹配算法

- 实现双向最大匹配算法
- 添加结果选择策略
- 更新文档说明
```

### Pull Request 流程

1. Fork 项目并创建功能分支
2. 在分支中完成开发并添加测试
3. 确保所有测试通过
4. 提交 PR 并描述变更内容
5. 等待代码审查
6. 合并后删除分支

---

## 许可证

本项目采用 Apache License 2.0 许可证开源。详见 [LICENSE](LICENSE) 文件。

```
Copyright 2024 NLP Team

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

---

## 联系方式与问题反馈

### 问题反馈

如果您在使用过程中遇到问题或有功能建议，欢迎通过以下方式反馈：

- **提交 Issue**: 在项目仓库提交 Bug 报告或功能请求
- **邮件联系**: 发送邮件至项目维护团队

### 参与贡献

我们欢迎所有形式的贡献，包括但不限于：

- 提交代码改进
- 完善文档
- 报告和修复 Bug
- 分享使用经验

### 致谢

感谢所有为项目做出贡献的开发者。

---

<div align="center">

**NLP 分词工具包** - 轻量、简单、易用

</div>
