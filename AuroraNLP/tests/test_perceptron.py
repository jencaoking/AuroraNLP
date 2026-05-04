"""感知器模型测试"""
import pytest
from AuroraNLP.perceptron import StructuredPerceptron, PerceptronSegmentor, PerceptronFeatureTemplate


class TestPerceptronInit:
    """感知器初始化测试"""

    def test_perceptron_init(self):
        """测试 StructuredPerceptron 初始化"""
        model = StructuredPerceptron()
        assert model.tags == []
        assert model.is_trained() is False

    def test_perceptron_segmentor_init(self):
        """测试 PerceptronSegmentor 初始化"""
        segmentor = PerceptronSegmentor()
        assert segmentor.STATES == ['B', 'M', 'E', 'S']
        assert segmentor.is_trained() is False

    def test_perceptron_feature_template_init(self):
        """测试 PerceptronFeatureTemplate 初始化"""
        template = PerceptronFeatureTemplate()
        assert template.feature_functions == []
        assert template.feature_names == []


class TestPerceptronTrain:
    """感知器训练测试"""

    @pytest.mark.slow
    def test_perceptron_train(self, sample_corpus):
        """测试感知器模型训练"""
        segmentor = PerceptronSegmentor()
        segmentor.train(sample_corpus, max_iter=5, verbose=False)
        assert segmentor.is_trained() is True


class TestPerceptronSegment:
    """感知器分词测试"""

    @pytest.mark.slow
    def test_perceptron_segment(self, sample_corpus):
        """测试训练后分词"""
        segmentor = PerceptronSegmentor()
        segmentor.train(sample_corpus, max_iter=5, verbose=False)
        result = segmentor.segment("我爱中国")
        assert isinstance(result, list)
        assert len(result) > 0
        assert ''.join(result) == "我爱中国"

    @pytest.mark.slow
    def test_perceptron_segment_with_states(self, sample_corpus):
        """测试分词带状态"""
        segmentor = PerceptronSegmentor()
        segmentor.train(sample_corpus, max_iter=5, verbose=False)
        result = segmentor.segment_with_states("中国")
        assert isinstance(result, list)
        assert len(result) == 2
        for char, state in result:
            assert isinstance(char, str)
            assert state in segmentor.STATES


class TestPerceptronNotTrained:
    """感知器未训练异常测试"""

    def test_perceptron_not_trained(self):
        """测试未训练时抛异常"""
        segmentor = PerceptronSegmentor()
        with pytest.raises(RuntimeError, match="not been trained"):
            segmentor.segment("测试")


class TestPerceptronSaveLoad:
    """感知器保存和加载测试"""

    @pytest.mark.slow
    def test_perceptron_save_and_load(self, sample_corpus, tmp_path):
        """测试保存和加载模型"""
        segmentor = PerceptronSegmentor()
        segmentor.train(sample_corpus, max_iter=5, verbose=False)

        model_path = str(tmp_path / "perceptron_model.bin")
        segmentor.save_model(model_path)

        new_segmentor = PerceptronSegmentor()
        new_segmentor.load_model(model_path)
        assert new_segmentor.is_trained() is True


class TestPerceptronOnlineTraining:
    """感知器在线学习测试"""

    @pytest.mark.slow
    def test_perceptron_online_training(self, sample_corpus):
        """测试在线学习"""
        segmentor = PerceptronSegmentor()
        segmentor.train(sample_corpus, max_iter=5, verbose=False)

        is_correct, accuracy = segmentor.train_online(["我", "爱", "中国"])
        assert isinstance(is_correct, bool)
        assert 0.0 <= accuracy <= 1.0


class TestPerceptronModelInfo:
    """感知器模型信息测试"""

    @pytest.mark.slow
    def test_perceptron_model_info(self, sample_corpus):
        """测试模型信息"""
        segmentor = PerceptronSegmentor()
        info = segmentor.get_model_info()
        assert info['trained'] is False

        segmentor.train(sample_corpus, max_iter=5, verbose=False)
        info = segmentor.get_model_info()
        assert info['trained'] is True
        assert 'num_tags' in info
        assert 'tags' in info
        assert 'num_features' in info
