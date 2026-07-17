"""词格分词测试"""
import pytest
from AuroraNLP.lattice import Lattice, LatticeBuilder, LatticeSegmentor, LatticeNode, LatticeEdge, PathScorer


class TestLatticeInit:
    """Lattice 初始化测试"""

    def test_lattice_init(self):
        """测试 Lattice 初始化"""
        lattice = Lattice("中国")
        assert lattice.text == "中国"
        assert lattice.length == 2
        assert len(lattice.nodes) == 3  # 位置 0, 1, 2
        assert len(lattice.edges) == 0


class TestLatticeBuild:
    """Lattice 构建测试"""

    def test_lattice_build(self, sample_dictionary):
        """测试构建词格"""
        builder = LatticeBuilder(sample_dictionary)
        lattice = builder.build("中国")
        assert lattice.text == "中国"
        assert lattice.length == 2
        assert len(lattice.edges) > 0
        assert lattice.has_path() is True


class TestLatticeSegment:
    """Lattice 分词测试"""

    def test_lattice_segment(self, sample_dictionary):
        """测试词格分词"""
        segmentor = LatticeSegmentor(sample_dictionary)
        result = segmentor.segment("中国")
        assert isinstance(result, list)
        assert len(result) > 0
        assert ''.join(result) == "中国"


class TestLatticeAllSegmentations:
    """Lattice 所有分词结果测试"""

    def test_lattice_get_all_segmentations(self, sample_dictionary):
        """测试获取所有分词结果"""
        segmentor = LatticeSegmentor(sample_dictionary)
        results = segmentor.get_all_segmentations("中国", max_results=10)
        assert isinstance(results, list)
        assert len(results) > 0
        for seg in results:
            assert ''.join(seg) == "中国"


class TestLatticeAmbiguity:
    """Lattice 歧义检测测试"""

    def test_lattice_detect_ambiguity(self, sample_dictionary):
        """测试检测歧义"""
        segmentor = LatticeSegmentor(sample_dictionary)
        ambiguities = segmentor.detect_ambiguity("中国人")
        assert isinstance(ambiguities, list)


class TestLatticeScoring:
    """Lattice 路径评分测试"""

    def test_lattice_scoring(self, sample_dictionary):
        """测试路径评分"""
        builder = LatticeBuilder(sample_dictionary)
        lattice = builder.build("中国")
        paths = lattice.get_all_paths()
        assert len(paths) > 0

        path = paths[0]
        length_score = PathScorer.score_by_length(path)
        assert isinstance(length_score, (int, float))

        word_len_score = PathScorer.score_by_word_length(path)
        assert isinstance(word_len_score, (int, float))

        freq_score = PathScorer.score_by_frequency(path)
        assert isinstance(freq_score, (int, float))

        weight_score = PathScorer.score_by_weight(path)
        assert isinstance(weight_score, (int, float))
