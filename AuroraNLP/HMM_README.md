# HMM 隐马尔可夫模型中文分词

## 概述

HMM (Hidden Markov Model，隐马尔可夫模型) 是一种统计模型，已成功集成到 AuroraNLP 中用于中文分词任务。该实现支持序列标注和未登录词识别。

## 核心功能

### 1. 序列标注 (B/M/E/S)

HMM 模型使用四个状态对汉字进行标注：
- **B (Begin)**: 词的开始
- **M (Middle)**: 词的中间
- **E (End)**: 词的结束
- **S (Single)**: 单字词

### 2. 未登录词识别

HMM 基于统计模型，能够识别训练语料中未出现的词汇，提高分词的泛化能力。

## 使用方法

### 基本使用

```python
from AuroraNLP import Segmentor

# 创建分词器
seg = Segmentor()

# 准备训练语料（已分词的句子列表）
corpus = [
    ['我', '爱', '自然语言处理'],
    ['今天', '天气', '很好'],
    ['我们', '在', '学习', '中文', '分词']
]

# 训练 HMM 模型
seg.train_hmm(corpus)

# 使用 HMM 模式分词
text = "我爱自然语言处理"
words = seg.segment(text, mode='hmm')
print(words)  # ['我', '爱', '自然语言处理']
```

### 从文件训练

```python
from AuroraNLP import Segmentor

seg = Segmentor()

# 从文件加载训练语料
# 文件格式：每行一个已分词的句子，词之间用空格分隔
seg.train_hmm_from_file('train_corpus.txt')

# 分词
words = seg.segment("今天天气很好", mode='hmm')
```

### 查看状态标注

```python
# 获取每个字的状态标注
states = seg.segment_with_hmm_states("我爱编程")
# 返回: [('我', 'S'), ('爱', 'B'), ('编', 'M'), ('程', 'E')]
```

### 保存和加载模型

```python
# 保存训练好的模型
seg.save_hmm_model('hmm_model.pkl')

# 加载模型
seg2 = Segmentor()
seg2.load_hmm_model('hmm_model.pkl')

# 使用加载的模型分词
words = seg2.segment("自然语言处理", mode='hmm')
```

### 查看模型信息

```python
info = seg.get_hmm_model_info()
print(info)
# {
#     'trained': True,
#     'total_states': 831,
#     'state_counts': {'S': 60, 'B': 315, 'M': 141, 'E': 315},
#     'vocabulary_sizes': {'B': 162, 'M': 85, 'E': 148, 'S': 16}
# }
```

## 技术实现

### 核心组件

1. **初始概率 (π)**: 句子第一个字的状态概率
2. **转移概率 (A)**: 从一个状态转移到另一个状态的概率
3. **发射概率 (B)**: 在某个状态下生成某个汉字的概率
4. **Viterbi 算法**: 解码最优状态序列

### 平滑处理

使用拉普拉斯平滑处理未登录词，避免零概率问题。

## 性能特点

### 优势
- ✓ 能够识别未登录词 (OOV)
- ✓ 基于统计模型，泛化能力强
- ✓ 可以处理歧义切分问题
- ✓ 不依赖大规模词典

### 局限
- 需要标注语料进行训练
- 对训练语料的质量和规模有依赖
- 无法利用词典中的先验知识

## 与词典分词对比

| 特性 | 词典分词 | HMM 分词 |
|------|---------|---------|
| 依赖词典 | 是 | 否 |
| 未登录词识别 | 否 | 是 |
| 训练需求 | 否 | 是 |
| 分词速度 | 快 | 较快 |
| 准确率 | 依赖词典质量 | 依赖训练语料 |

## 最佳实践

1. **训练语料质量**: 使用高质量、领域相关的分词语料
2. **语料规模**: 建议至少 10 万词以上的训练语料
3. **混合使用**: 可以结合词典分词和 HMM 分词的优势
4. **模型保存**: 训练后保存模型，避免重复训练

## 示例代码

完整示例请参考 `demo_hmm.py` 文件。

## 测试

运行测试用例：

```bash
cd AuroraNLP
python -m pytest tests/test_hmm.py -v
```

## 参考文献

- Rabiner, L.R. (1989). A tutorial on hidden Markov models and selected applications in speech recognition.
- 刘群. 中文分词技术回顾与展望.
