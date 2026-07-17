# CRF (Conditional Random Field) 条件随机场

## 概述

条件随机场（Conditional Random Field, CRF）是一种判别式概率图模型，特别适用于序列标注任务。与HMM相比，CRF具有以下优势：

1. **更丰富的特征**：可以使用任意复杂的特征函数
2. **无独立性假设**：不需要观测序列的独立性假设
3. **全局最优**：通过全局归一化避免标签偏置问题
4. **更强的表达能力**：可以捕捉长距离依赖关系

## 架构设计

### 核心组件

AuroraNLP的CRF模块包含三个核心类：

#### 1. CRFFeatureTemplate - 特征模板

特征模板定义了从输入序列和标签序列中提取特征的方法。

```python
from AuroraNLP import CRFFeatureTemplate

template = CRFFeatureTemplate()

# 添加特征
template.add_unigram_feature('word', 0)      # 当前词
template.add_bigram_feature('bigram', 0)     # 当前词和下一个词
template.add_transition_feature()             # 转移特征
template.add_start_feature()                  # 句子开始特征
template.add_end_feature()                    # 句子结束特征
template.add_char_shape_feature(0)           # 字符形状特征
template.add_length_feature(0)               # 词长度特征
template.add_prefix_feature(2, 0)            # 前缀特征
template.add_suffix_feature(2, 0)            # 后缀特征
```

#### 2. CRFModel - CRF模型

CRF模型实现了条件随机场的核心算法：

- **训练**：使用梯度下降算法优化模型参数
- **推理**：使用Viterbi算法进行最优路径解码
- **保存/加载**：支持模型的持久化

```python
from AuroraNLP import CRFModel

model = CRFModel(tags=['B', 'M', 'E', 'S'])

# 训练模型
corpus = [
    (['我', '爱', '中', '国'], ['S', 'S', 'B', 'E']),
    (['他', '是', '学', '生'], ['S', 'S', 'B', 'E'])
]

model.train(
    corpus,
    learning_rate=0.1,
    l2_reg=0.01,
    max_iter=100,
    verbose=True
)

# 预测
tokens = ['我', '爱', '北', '京']
tags = model.predict(tokens)
```

#### 3. CRFSegmentor - CRF分词器

CRFSegmentor封装了CRF模型，专门用于中文分词任务：

```python
from AuroraNLP import CRFSegmentor

segmentor = CRFSegmentor()

# 训练
corpus = [
    ['我', '爱', '中国'],
    ['他', '是', '学生']
]

segmentor.train(corpus, max_iter=100, verbose=True)

# 分词
text = '我爱北京'
words = segmentor.segment(text)

# 查看状态序列
states = segmentor.segment_with_states(text)
```

## 使用方法

### 方法1：直接使用CRFSegmentor

```python
from AuroraNLP import CRFSegmentor

# 创建分词器
segmentor = CRFSegmentor()

# 准备训练语料（已分词的文本）
corpus = [
    ['我', '爱', '中国'],
    ['北京', '是', '首都'],
    ['自然语言', '处理', '很', '有趣']
]

# 训练模型
segmentor.train(
    corpus,
    learning_rate=0.1,
    l2_reg=0.01,
    max_iter=100,
    verbose=True
)

# 分词
text = '我爱自然语言处理'
words = segmentor.segment(text)
print(' / '.join(words))
# 输出: 我 / 爱 / 自然语言 / 处理

# 保存模型
segmentor.save_model('crf_model.pkl')

# 加载模型
new_segmentor = CRFSegmentor()
new_segmentor.load_model('crf_model.pkl')
```

### 方法2：使用Segmentor统一接口

```python
from AuroraNLP import Segmentor

# 创建分词器
segmentor = Segmentor()

# 训练CRF模型
corpus = [
    ['我', '爱', '中国'],
    ['他', '是', '学生']
]

segmentor.train_crf(corpus, max_iter=100, verbose=True)

# 使用CRF模式分词
text = '我爱学习'
words = segmentor.segment(text, mode='crf')
print(' / '.join(words))

# 设置默认模式为CRF
segmentor.set_mode('crf')
words = segmentor.segment('我爱中国')

# 查看模型信息
info = segmentor.get_crf_model_info()
print(f"特征数量: {info['num_features']}")
```

### 方法3：从文件训练

```python
from AuroraNLP import Segmentor

segmentor = Segmentor()

# 从文件加载训练语料
# 文件格式：每行一个已分词的句子，词之间用空格分隔
segmentor.train_crf_from_file('corpus.txt')

# 分词
words = segmentor.segment('我爱中国', mode='crf')
```

## 特征模板详解

### 默认特征模板

CRFModel在训练时会自动设置默认特征模板，包括：

1. **转移特征**
   - `TRANS:prev_tag->curr_tag`：标签转移特征
   - `START:tag`：句子开始特征
   - `END:tag`：句子结束特征

2. **词特征**
   - `unigram_word_{-2..2}`：当前位置前后2个位置的词
   - `bigram_bigram_{-1..0}`：当前位置前后1个位置的词对

3. **字符特征**
   - `char_shape_{-1..1}`：字符形状特征（大小写、数字等）
   - `length_{-1..1}`：词长度特征

4. **词缀特征**
   - `prefix{1,2,3}_0`：前缀特征（长度1-3）
   - `suffix{1,2,3}_0`：后缀特征（长度1-3）

### 自定义特征模板

```python
from AuroraNLP import CRFModel, CRFFeatureTemplate

# 创建自定义特征模板
template = CRFFeatureTemplate()

# 添加基本特征
template.add_transition_feature()
template.add_start_feature()
template.add_end_feature()

# 添加词特征
template.add_unigram_feature('word', 0)
template.add_unigram_feature('word', -1)
template.add_unigram_feature('word', 1)

# 添加词对特征
template.add_bigram_feature('bigram', 0)

# 添加字符特征
template.add_char_shape_feature(0)
template.add_length_feature(0)

# 创建模型并设置特征模板
model = CRFModel(tags=['B', 'M', 'E', 'S'])
model.feature_template = template

# 训练模型
model.train(corpus, max_iter=100)
```

## 训练参数

### 学习率 (learning_rate)

- 默认值：0.1
- 作用：控制每次参数更新的步长
- 建议：较小的学习率（0.01-0.1）通常更稳定

### L2正则化 (l2_reg)

- 默认值：0.01
- 作用：防止过拟合
- 建议：根据数据集大小调整，数据集小可增大正则化系数

### 最大迭代次数 (max_iter)

- 默认值：100
- 作用：控制训练的最大迭代次数
- 建议：观察损失函数收敛情况，通常50-200次迭代足够

### 收敛阈值 (epsilon)

- 默认值：1e-6
- 作用：判断训练是否收敛
- 建议：使用默认值即可

## 性能优化

### 1. 特征选择

- 只使用对任务有帮助的特征
- 避免冗余特征
- 根据数据特点调整特征窗口大小

### 2. 训练数据

- 使用足够大的训练语料
- 确保数据质量
- 数据预处理（清洗、规范化）

### 3. 超参数调优

```python
# 网格搜索最佳参数
best_score = 0
best_params = {}

for lr in [0.01, 0.05, 0.1]:
    for l2 in [0.001, 0.01, 0.1]:
        segmentor = CRFSegmentor()
        segmentor.train(
            train_corpus,
            learning_rate=lr,
            l2_reg=l2,
            max_iter=100,
            verbose=False
        )
        
        # 在验证集上评估
        score = evaluate(segmentor, dev_corpus)
        
        if score > best_score:
            best_score = score
            best_params = {'lr': lr, 'l2': l2}
```

## 与HMM对比

| 特性 | HMM | CRF |
|------|-----|-----|
| 模型类型 | 生成式模型 | 判别式模型 |
| 特征表示 | 简单 | 丰富灵活 |
| 独立性假设 | 需要 | 不需要 |
| 训练复杂度 | 低 | 高 |
| 推理速度 | 快 | 较慢 |
| 准确率 | 较好 | 更好 |
| 适用场景 | 简单任务 | 复杂任务 |

## 应用场景

1. **中文分词**：基于字符的序列标注
2. **词性标注**：为每个词标注词性
3. **命名实体识别**：识别文本中的人名、地名、机构名等
4. **语义角色标注**：识别句子中的语义角色

## 示例

完整示例请参考：[examples/demo_crf.py](../examples/demo_crf.py)

## 参考文献

1. Lafferty, J., McCallum, A., & Pereira, F. C. (2001). Conditional random fields: Probabilistic models for segmenting and labeling sequence data.
2. Sutton, C., & McCallum, A. (2012). An introduction to conditional random fields.
3. Peng, N., & Dredze, M. (2016). Improving named entity recognition for chinese social media with word segmentation representation learning.
