"""HMM 模型测试"""
import pytest
from AuroraNLP.hmm import HMMSegmentor, train_from_file


class TestHMMInit:
    """HMM 初始化测试"""

    def test_hmm_init(self):
        """测试 HMMSegmentor 初始化状态"""
        model = HMMSegmentor()
        assert model.STATES == ['B', 'M', 'E', 'S']
        assert model.init_prob == {}
        assert model.trans_prob == {}
        assert model.emit_prob == {}
        assert model.is_trained() is False


class TestHMMTrain:
    """HMM 训练测试"""

    def test_hmm_train(self, sample_corpus):
        """测试 HMM 模型训练"""
        model = HMMSegmentor()
        model.train(sample_corpus)
        assert model.is_trained() is True
        assert model.total_states > 0
        assert len(model.init_prob) == 4
        assert len(model.trans_prob) == 4


class TestHMMSegment:
    """HMM 分词测试"""

    def test_hmm_segment(self, sample_corpus):
        """测试训练后分词"""
        model = HMMSegmentor()
        model.train(sample_corpus)
        result = model.segment("我爱中国")
        assert isinstance(result, list)
        assert len(result) > 0
        assert ''.join(result) == "我爱中国"

    def test_hmm_viterbi(self, sample_corpus):
        """测试 Viterbi 解码"""
        model = HMMSegmentor()
        model.train(sample_corpus)
        states = model.viterbi("中国")
        assert isinstance(states, list)
        assert len(states) == 2
        assert all(s in model.STATES for s in states)

    def test_hmm_segment_with_states(self, sample_corpus):
        """测试分词带状态"""
        model = HMMSegmentor()
        model.train(sample_corpus)
        result = model.segment_with_states("中国")
        assert isinstance(result, list)
        assert len(result) == 2
        for char, state in result:
            assert isinstance(char, str)
            assert state in model.STATES


class TestHMMNotTrained:
    """HMM 未训练异常测试"""

    def test_hmm_not_trained(self):
        """测试未训练时分词抛异常"""
        model = HMMSegmentor()
        with pytest.raises(RuntimeError, match="not been trained"):
            model.segment("测试")


class TestHMMSaveLoad:
    """HMM 保存和加载测试"""

    def test_hmm_save_and_load_model(self, sample_corpus, tmp_path):
        """测试保存和加载模型"""
        model = HMMSegmentor()
        model.train(sample_corpus)

        model_path = str(tmp_path / "hmm_model.bin")
        model.save_model(model_path)

        new_model = HMMSegmentor()
        new_model.load_model(model_path)
        assert new_model.is_trained() is True
        assert new_model.total_states == model.total_states
        assert new_model.init_prob == model.init_prob


class TestHMMModelInfo:
    """HMM 模型信息测试"""

    def test_hmm_model_info(self, sample_corpus):
        """测试获取模型信息"""
        model = HMMSegmentor()
        info = model.get_model_info()
        assert info['trained'] is False

        model.train(sample_corpus)
        info = model.get_model_info()
        assert info['trained'] is True
        assert 'total_states' in info
        assert 'state_counts' in info
        assert 'vocabulary_sizes' in info
        assert 'smooth' in info


class TestHMMIsTrained:
    """HMM 训练状态检查测试"""

    def test_hmm_is_trained(self, sample_corpus):
        """测试训练状态检查"""
        model = HMMSegmentor()
        assert model.is_trained() is False
        model.train(sample_corpus)
        assert model.is_trained() is True


class TestTrainFromFile:
    """从文件训练测试"""

    def test_train_from_file(self, sample_corpus, tmp_path):
        """测试从文件训练"""
        corpus_file = tmp_path / "corpus.txt"
        lines = [' '.join(sentence) for sentence in sample_corpus]
        corpus_file.write_text('\n'.join(lines), encoding='utf-8')

        model = HMMSegmentor()
        train_from_file(model, str(corpus_file))
        assert model.is_trained() is True
        assert model.total_states > 0
