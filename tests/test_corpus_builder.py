"""CorpusBuilder 语料构建测试"""
import pytest
from AuroraNLP.corpus.corpus_builder import CorpusBuilder, CorpusManager


class TestCorpusBuilder:
    """测试 CorpusBuilder 语料构建功能"""

    def test_corpus_builder_init(self):
        """初始化"""
        builder = CorpusBuilder()
        assert builder is not None
        assert builder.data_dir is not None
        assert isinstance(builder.corpus_types, dict)

    def test_corpus_builder_init_with_data_dir(self, tmp_path):
        """指定数据目录初始化"""
        builder = CorpusBuilder(data_dir=str(tmp_path))
        assert builder.data_dir == str(tmp_path)

    def test_corpus_builder_load_corpus(self, tmp_path):
        """加载语料"""
        corpus_file = tmp_path / "test_corpus.txt"
        corpus_file.write_text(
            "自然 语言 处理\n"
            "机器 学习 是 人工智能\n"
            "深度 学习 发展 迅速\n",
            encoding='utf-8'
        )
        builder = CorpusBuilder()
        lines = builder.load_corpus("custom", str(corpus_file))
        assert isinstance(lines, list)
        assert len(lines) == 3

    def test_corpus_builder_preprocess_corpus(self):
        """预处理语料"""
        builder = CorpusBuilder()
        lines = ["自然 语言 处理", "机器 学习 是 人工智能", ""]
        result = builder.preprocess_corpus(lines, "msra")
        assert isinstance(result, list)
        assert len(result) == 2
        assert "" not in result

    def test_corpus_builder_split_corpus(self, tmp_path):
        """分割语料"""
        corpus_file = tmp_path / "split_corpus.txt"
        corpus_file.write_text(
            "自然 语言 处理\n"
            "机器 学习 是 人工智能\n"
            "深度 学习 发展 迅速\n"
            "自然 语言 处理 技术\n"
            "知识 图谱 应用 广泛\n"
            "语料 库 构建 完成\n"
            "分词 系统 性能 优秀\n"
            "词性 标注 准确 率高\n"
            "命名 实体 识别 技术\n"
            "情感 分析 研究 热门\n",
            encoding='utf-8'
        )
        builder = CorpusBuilder()
        train_path, val_path, test_path = builder.split_corpus(str(corpus_file), train_ratio=0.8, val_ratio=0.1)
        assert train_path.endswith("_train.txt")
        assert val_path.endswith("_val.txt")
        assert test_path.endswith("_test.txt")

    def test_corpus_builder_statistics(self, tmp_path):
        """语料统计"""
        corpus_file = tmp_path / "stats_corpus.txt"
        corpus_file.write_text(
            "自然 语言 处理\n"
            "机器 学习 是 人工智能\n"
            "深度 学习 发展 迅速\n",
            encoding='utf-8'
        )
        builder = CorpusBuilder()
        stats = builder.statistics(str(corpus_file))
        assert isinstance(stats, dict)
        assert stats["total_lines"] == 3
        assert stats["total_tokens"] > 0
        assert stats["total_chars"] > 0
        assert stats["avg_tokens_per_line"] > 0
        assert stats["avg_chars_per_line"] > 0


class TestCorpusManager:
    """测试 CorpusManager 语料管理功能"""

    def test_corpus_manager_init(self, tmp_path):
        """管理器初始化"""
        manager = CorpusManager(data_dir=str(tmp_path))
        assert manager is not None
        assert manager.data_dir == str(tmp_path)
        assert isinstance(manager.registry, dict)
        assert "corpora" in manager.registry

    def test_corpus_manager_register_corpus(self, tmp_path):
        """注册语料"""
        corpus_file = tmp_path / "reg_corpus.txt"
        corpus_file.write_text("自然 语言 处理\n", encoding='utf-8')
        manager = CorpusManager(data_dir=str(tmp_path))
        manager.register_corpus("test_corpus", "custom", str(corpus_file), "测试语料")
        corpora = manager.list_corpora()
        assert len(corpora) == 1
        assert corpora[0]["name"] == "test_corpus"
        assert corpora[0]["type"] == "custom"

    def test_corpus_manager_build_combined(self, tmp_path):
        """构建组合语料"""
        corpus_file1 = tmp_path / "combined1.txt"
        corpus_file1.write_text("自然 语言 处理\n", encoding='utf-8')
        corpus_file2 = tmp_path / "combined2.txt"
        corpus_file2.write_text("机器 学习 技术\n", encoding='utf-8')

        manager = CorpusManager(data_dir=str(tmp_path))
        manager.register_corpus("corpus_a", "custom", str(corpus_file1))
        manager.register_corpus("corpus_b", "custom", str(corpus_file2))

        result_path = manager.build_combined_corpus("combined_test", ["corpus_a", "corpus_b"])
        assert result_path is not None
        assert "combined_test_corpus.txt" in result_path
