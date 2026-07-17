# 贡献指南

感谢你对 AuroraNLP 的关注！我们欢迎所有形式的贡献。

## 代码贡献流程

### 1. 提交 Issue（可选但推荐）
- 如果这是一个 Bug，请提交 Bug Report
- 如果这是一个新功能请求，请提交 Feature Request

### 2. Fork 仓库
- 点击 GitHub 上的 Fork 按钮
- 克隆你 fork 后的仓库到本地

```bash
git clone https://github.com/你的用户名/AuroraNLP.git
cd AuroraNLP
```

### 3. 创建分支
为你的功能或修复创建一个新分支：

```bash
git checkout -b feature/你的功能名称
# 或
git checkout -b fix/修复的问题
```

### 4. 安装开发依赖
```bash
pip install -e .[dev,test]
```

### 5. 进行修改
- 遵循现有代码风格
- 添加或更新测试
- 更新文档

### 6. 运行测试
确保所有测试通过：

```bash
pytest
```

### 7. 提交和推送
```bash
git add .
git commit -m "描述你的更改"
git push origin 你的分支名
```

### 8. 创建 Pull Request
- 向 main 分支提交 PR
- 填写 PR 模板
- 等待审核

## 代码风格

我们遵循以下约定：
- 使用 Black 进行代码格式化
- 使用 isort 整理导入
- 使用 Ruff 进行代码检查

```bash
# 格式化代码
black .
isort .

# 检查代码
ruff .
```

## 测试指南

- 所有新功能都应该有对应的测试
- 确保修改不破坏现有功能
- 运行完整测试套件验证

```bash
# 运行所有测试
pytest

# 运行特定文件的测试
pytest tests/test_your_test.py

# 查看覆盖率
pytest --cov=AuroraNLP
```

## 文档贡献

- 更新或改进文档
- 添加新的示例
- 修正错别字或语言问题

文档位于 `docs/` 目录。

## 社区指南

- 尊重所有贡献者
- 保持讨论主题相关
- 帮助新贡献者

## 版本发布

发布流程由维护者管理：
- 更新版本号
- 更新 CHANGELOG
- 创建标签
- 发布到 PyPI

## 获取帮助

如果你有任何问题：
- 查看 README.md 和文档
- 搜索现有的 Issues/Discussions
- 创建新的 Issue

感谢你的贡献！
