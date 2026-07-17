# AuroraNLP

AuroraNLP 是一个专业级的中文自然语言处理工具包，使用纯 Python 标准库实现，无强制第三方依赖，覆盖从基础分词到企业级部署的完整链路。

## 特性

- **分词与词典**：多策略分词（正向最大匹配、HMM、CRF、感知机、词格、混合），支持用户词典、领域词典、网络词典、版本化词典与热更新增量词典。
- **词性标注与统计模型**：HMM / CRF / 感知机词性标注，N-Gram 语言模型。
- **命名实体识别与实体链接**：CRF-NER、嵌套实体识别、知识库实体链接与归一化。
- **语法分析**：依存句法分析（Arc-Eager）、短语结构句法分析（PCFG / CKY）。
- **混合分词**：多分词器投票、加权、级联、置信度融合等策略。
- **语义与文本挖掘**：情感分析、关键词抽取、文本相似度、同义词典、人名 / 地名 / 机构名识别、术语库、繁简转换。
- **语料工具**：语料构建、标注、主动学习与标注质量评估。
- **Pipeline 与性能**：可组合的 Pipeline（组件注册、异步、流式、插件）、对象池、批处理、并行分词、GPU / 混合精度 / TensorRT 推理、LRU 缓存与压缩。
- **企业级能力**：结构化日志、健康检查、Prometheus 指标、限流熔断、鉴权、配置中心、灰度发布与故障转移、K8s / Docker 部署模板。
- **深度学习**：BiLSTM-CRF 模型，支持 PyTorch / TensorFlow 后端。

## 安装

```bash
pip install .
```

或安装开发 / 测试 / 代码检查所需的额外依赖：

```bash
pip install ".[all]"
```

## 快速开始

```python
from AuroraNLP import Segmentor, NERRecognizer

# 中文分词
seg = Segmentor()
print(seg.segment("今天天气真不错，我们去公园散步吧！"))

# 命名实体识别
ner = NERRecognizer()
text = "阿里巴巴集团的张勇董事长今天在杭州出席了会议"
for entity in ner.recognize(text):
    print(f"{entity.text} ({entity.type}): {entity.start}-{entity.end}")
```

更复杂的场景推荐使用 `Pipeline` 系统（分词 + 词性 + 实体联合处理），详见 `examples/`。

## 项目结构

```
AuroraNLP/
├── __init__.py            # 包入口，统一聚合所有公开 API
├── *.py                   # 各功能模块（分词、词典、NER、分析器、Pipeline 等）
├── deep_learning/         # BiLSTM-CRF 与深度学习后端
├── data/                  # 词典、停用词、领域词典等数据资源
├── docs/                  # 使用文档（快速入门、用户手册、FAQ 等）
├── examples/              # 示例代码
├── tests/                 # 单元与集成测试（pytest）
├── setup.py               # 打包配置
└── requirements.txt       # 依赖说明
```

## 文档

详见 `docs/` 目录：

- [快速入门](docs/getting-started.md)
- [用户手册](docs/user-guide.md)
- [最佳实践](docs/best-practices.md)
- [FAQ](docs/faq.md)
- [HMM 分词](docs/HMM.md)
- [CRF 分词](docs/CRF.md)

## 许可证

本项目基于 `LICENSE` 文件中的许可证发布（Apache-2.0）。
