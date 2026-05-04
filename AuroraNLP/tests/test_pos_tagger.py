"""测试词性标注模块"""

import pytest

from AuroraNLP.pos_tagger import HMMPOSTagger, CRFPOSTagger, POS_TAGS, DEFAULT_TAGS


@pytest.fixture
def pos_corpus():
    """创建带词性标注的训练语料

    格式为 List[Tuple[List[str], List[str]]]
    """
    return [
        (["我", "爱", "中国"], ["r", "v", "ns"]),
        (["中国", "是", "伟大", "的", "国家"], ["ns", "v", "a", "u", "n"]),
        (["北京", "是", "中国", "的", "首都"], ["ns", "v", "ns", "u", "n"]),
        (["我", "在", "北京", "大学", "学习"], ["r", "p", "ns", "n", "v"]),
        (["自然语言", "处理", "是", "人工智能", "的", "方向"], ["n", "v", "v", "n", "u", "n"]),
        (["我", "爱", "自然语言", "处理"], ["r", "v", "n", "v"]),
        (["中国", "人", "很", "好"], ["ns", "n", "d", "a"]),
        (["研究", "生", "在", "研究", "生命"], ["v", "n", "p", "v", "n"]),
    ]


class TestHMMPOSTaggerInit:
    """测试 HMM 词性标注器初始化"""

    def test_hmm_pos_tagger_init(self):
        """测试初始化"""
        tagger = HMMPOSTagger()
        assert tagger is not None
        assert tagger.is_trained() is False
        assert isinstance(tagger.tags, list)
        assert len(tagger.tags) > 0


class TestHMMPOSTaggerTrain:
    """测试 HMM 词性标注器训练"""

    @pytest.mark.slow
    def test_hmm_pos_tagger_train(self, pos_corpus):
        """测试训练"""
        tagger = HMMPOSTagger()
        tagger.train(pos_corpus)
        assert tagger.is_trained() is True
        info = tagger.get_model_info()
        assert info['trained'] is True
        assert info['num_tags'] > 0
        assert info['word_count'] > 0


class TestHMMPOSTaggerTag:
    """测试 HMM 词性标注器标注"""

    @pytest.mark.slow
    def test_hmm_pos_tagger_tag(self, pos_corpus):
        """测试标注"""
        tagger = HMMPOSTagger()
        tagger.train(pos_corpus)
        words = ["我", "爱", "中国"]
        tags = tagger.tag(words)
        assert isinstance(tags, list)
        assert len(tags) == len(words)
        for tag in tags:
            assert isinstance(tag, str)
            assert len(tag) > 0

        # 测试 tag_sentence
        result = tagger.tag_sentence(words)
        assert isinstance(result, list)
        assert len(result) == len(words)
        for word, tag in result:
            assert isinstance(word, str)
            assert isinstance(tag, str)


class TestPOSTagsConstants:
    """测试词性标注常量"""

    def test_pos_tags_defined(self):
        """测试 POS_TAGS 常量存在"""
        assert POS_TAGS is not None
        assert isinstance(POS_TAGS, dict)
        assert len(POS_TAGS) > 0
        # 验证常见标签
        assert 'n' in POS_TAGS
        assert 'v' in POS_TAGS
        assert 'a' in POS_TAGS

    def test_default_tags_defined(self):
        """测试 DEFAULT_TAGS 常量存在"""
        assert DEFAULT_TAGS is not None
        assert isinstance(DEFAULT_TAGS, list)
        assert len(DEFAULT_TAGS) > 0
        # DEFAULT_TAGS 应该是 POS_TAGS 的键列表
        assert set(DEFAULT_TAGS) == set(POS_TAGS.keys())
