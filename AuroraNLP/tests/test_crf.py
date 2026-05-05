"""CRF 模型测试"""
import pytest
from AuroraNLP.crf import CRFModel, CRFSegmentor, CRFFeatureTemplate


class TestCRFModelInit:
    """CRF 模型初始化测试"""

    def test_crf_model_init(self):
        """测试 CRFModel 初始化"""
        model = CRFModel()
        assert model.tags == []
        assert model.is_trained() is False

    def test_crf_segmentor_init(self):
        """测试 CRFSegmentor 初始化"""
        segmentor = CRFSegmentor()
        assert segmentor.STATES == ['B', 'M', 'E', 'S']
        assert segmentor.is_trained() is False

    def test_crf_feature_template_init(self):
        """测试 CRFFeatureTemplate 初始化"""
        template = CRFFeatureTemplate()
        assert template.feature_functions == []
        assert template.feature_names == []


class TestCRFTrain:
    """CRF 训练测试"""

    @pytest.mark.slow
    def test_crf_train(self, sample_corpus):
        """测试 CRF 模型训练"""
        segmentor = CRFSegmentor()
        segmentor.train(sample_corpus, max_iter=5, verbose=False)
        assert segmentor.is_trained() is True


class TestCRFSegment:
    """CRF 分词测试"""

    @pytest.mark.slow
    def test_crf_segment(self, sample_corpus):
        """测试训练后分词"""
        segmentor = CRFSegmentor()
        segmentor.train(sample_corpus, max_iter=5, verbose=False)
        result = segmentor.segment("我爱中国")
        assert isinstance(result, list)
        assert len(result) > 0
        assert ''.join(result) == "我爱中国"

    @pytest.mark.slow
    def test_crf_segment_with_states(self, sample_corpus):
        """测试分词带状态"""
        segmentor = CRFSegmentor()
        segmentor.train(sample_corpus, max_iter=5, verbose=False)
        result = segmentor.segment_with_states("中国")
        assert isinstance(result, list)
        assert len(result) == 2
        for char, state in result:
            assert isinstance(char, str)
            assert state in segmentor.STATES


class TestCRFNotTrained:
    """CRF 未训练异常测试"""

    def test_crf_not_trained(self):
        """测试未训练时分词抛异常"""
        segmentor = CRFSegmentor()
        with pytest.raises(RuntimeError, match="not been trained"):
            segmentor.segment("测试")


class TestCRFSaveLoad:
    """CRF 保存和加载测试"""

    @pytest.mark.slow
    def test_crf_save_and_load(self, sample_corpus, tmp_path):
        """测试保存和加载模型"""
        segmentor = CRFSegmentor()
        segmentor.train(sample_corpus, max_iter=5, verbose=False)

        model_path = str(tmp_path / "crf_model.bin")
        segmentor.save_model(model_path)

        new_segmentor = CRFSegmentor()
        new_segmentor.load_model(model_path)
        assert new_segmentor.is_trained() is True


class TestCRFModelInfo:
    """CRF 模型信息测试"""

    @pytest.mark.slow
    def test_crf_model_info(self, sample_corpus):
        """测试模型信息"""
        segmentor = CRFSegmentor()
        info = segmentor.get_model_info()
        assert info['trained'] is False

        segmentor.train(sample_corpus, max_iter=5, verbose=False)
        info = segmentor.get_model_info()
        assert info['trained'] is True
        assert 'num_tags' in info
        assert 'tags' in info
        assert 'num_features' in info


class TestCRFIsTrained:
    """CRF 训练状态测试"""

    @pytest.mark.slow
    def test_crf_is_trained(self, sample_corpus):
        """测试训练状态"""
        segmentor = CRFSegmentor()
        assert segmentor.is_trained() is False
        segmentor.train(sample_corpus, max_iter=5, verbose=False)
        assert segmentor.is_trained() is True
