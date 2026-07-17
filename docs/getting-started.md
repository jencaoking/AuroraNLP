# 快速入门

欢迎使用 AuroraNLP！本指南将帮助你快速上手。

## 安装

首先，确保你已经安装了 Python 3.8 或更高版本。

### 通过 pip 安装

```bash
pip install aurora-nlp
```

### 验证安装

运行以下命令验证安装：

```python
import AuroraNLP
print(AuroraNLP.__version__)
```

## 第一个例子：分词

让我们从最基础的功能开始——中文分词：

```python
from AuroraNLP import Segmentor

# 创建分词器
seg = Segmentor()

# 分词
text = "今天天气真不错，我们去公园散步吧！"
words = seg.segment(text)
print(words)
```

输出：

```
['今天', '天气', '真', '不错', '，', '我们', '去', '公园', '散步', '吧', '！']
```

## 使用不同的分词策略

AuroraNLP 支持多种分词策略，你可以根据需要选择：

```python
# 使用 HMM 分词
seg_hmm = Segmentor(use_hmm=True)

# 使用 CRF 分词
seg_crf = Segmentor(use_crf=True)

# 使用感知机分词
seg_perceptron = Segmentor(use_perceptron=True)

# 使用词格分词
seg_lattice = Segmentor(use_lattice=True)

# 使用混合分词（推荐）
seg_hybrid = Segmentor(use_hybrid=True)
```

## 词性标注

除了分词，你还可以获取每个词的词性：

```python
seg = Segmentor()
words_with_pos = seg.segment_with_pos("我爱北京天安门")
print(words_with_pos)
```

输出：

```
[('我', 'r'), ('爱', 'v'), ('北京', 'ns'), ('天安门', 'ns')]
```

词性说明：

| 标签 | 说明 |
|------|------|
| n | 名词 |
| v | 动词 |
| a | 形容词 |
| d | 副词 |
| r | 代词 |
| ns | 地名 |
| nr | 人名 |
| nt | 机构名 |
| ... | ... |

## 命名实体识别

```python
from AuroraNLP import NERRecognizer

ner = NERRecognizer()
text = "阿里巴巴集团的张勇董事长今天在杭州出席了会议"
entities = ner.recognize(text)

for entity in entities:
    print(f"{entity.text} ({entity.type}): {entity.start}-{entity.end}")
```

## 使用 Pipeline 系统

对于更复杂的任务，推荐使用 Pipeline 系统：

```python
from AuroraNLP import Pipeline, Segmentor, POSTagger, NERRecognizer

# 创建 Pipeline
nlp = Pipeline()
nlp.add_component(Segmentor())
nlp.add_component(POSTagger())
nlp.add_component(NERRecognizer())

# 处理文本
doc = nlp("张三在百度公司工作，住在北京市海淀区")

# 访问结果
print("分词:", doc.tokens)
print("词性:", [(t.text, t.pos) for t in doc.tokens])
print("实体:", doc.ents)
```

## 下一步

- 查看 [用户手册](user-guide.md) 深入了解所有功能
- 参考 [API 参考](api.md) 了解完整的 API 文档
- 浏览 [示例代码](../examples/) 学习各种使用场景
- 如有问题，请查看 [FAQ](faq.md)
