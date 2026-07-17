"""测试命名实体识别模块"""

import pytest

from AuroraNLP.ner.ner import (
    NERRecognizer,
    CRFNERModel,
    Entity,
    NER_TAGS,
    DEFAULT_NER_TAGS,
    NER_ENTITY_TYPES,
)


@pytest.fixture
def ner_corpus():
    """创建 NER 训练语料

    格式为 List[Tuple[str, List[Entity]]]
    """
    return [
        (
            "张三在北京清华大学学习",
            [
                Entity("张三", "PER", 0, 2),
                Entity("北京", "LOC", 3, 5),
                Entity("清华大学", "ORG", 5, 9),
            ]
        ),
        (
            "李四在上海交通大学工作",
            [
                Entity("李四", "PER", 0, 2),
                Entity("上海", "LOC", 3, 5),
                Entity("交通大学", "ORG", 5, 9),
            ]
        ),
        (
            "王五于二零二三年加入阿里巴巴集团",
            [
                Entity("王五", "PER", 0, 2),
                Entity("阿里巴巴集团", "ORG", 8, 14),
            ]
        ),
        (
            "小明在广州的腾讯公司上班",
            [
                Entity("小明", "PER", 0, 2),
                Entity("广州", "LOC", 3, 5),
                Entity("腾讯公司", "ORG", 6, 10),
            ]
        ),
        (
            "刘医生在北京协和医院工作",
            [
                Entity("刘医生", "PER", 0, 3),
                Entity("北京", "LOC", 4, 6),
                Entity("协和医院", "ORG", 6, 10),
            ]
        ),
    ]


class TestNERRecognizerInit:
    """测试 NER 识别器初始化"""

    def test_ner_init(self):
        """测试初始化"""
        recognizer = NERRecognizer()
        assert recognizer is not None
        assert recognizer.is_trained() is False
        assert recognizer.model is not None


class TestNERRecognizerTrain:
    """测试 NER 识别器训练"""

    @pytest.mark.slow
    def test_ner_train(self, ner_corpus):
        """测试训练"""
        recognizer = NERRecognizer()
        recognizer.train(ner_corpus, max_iter=3, verbose=False)
        assert recognizer.is_trained() is True


class TestNERRecognizerRecognize:
    """测试 NER 识别器识别"""

    @pytest.mark.slow
    def test_ner_recognize(self, ner_corpus):
        """测试识别实体"""
        recognizer = NERRecognizer()
        recognizer.train(ner_corpus, max_iter=3, verbose=False)
        entities = recognizer.recognize("张三在北京清华大学学习")
        assert isinstance(entities, list)
        for entity in entities:
            assert isinstance(entity, Entity)
            assert hasattr(entity, 'text')
            assert hasattr(entity, 'entity_type')
            assert hasattr(entity, 'start')
            assert hasattr(entity, 'end')


class TestNERTagsConstants:
    """测试 NER 常量"""

    def test_ner_tags_defined(self):
        """测试 NER_TAGS 常量存在"""
        assert NER_TAGS is not None
        assert isinstance(NER_TAGS, list)
        assert len(NER_TAGS) > 0
        # 应包含 O 标签和 B-/I-/E-/S- 标签
        assert 'O' in NER_TAGS
        assert any(tag.startswith('B-') for tag in NER_TAGS)
        assert any(tag.startswith('I-') for tag in NER_TAGS)

    def test_ner_entity_types_defined(self):
        """测试 NER_ENTITY_TYPES 常量存在"""
        assert NER_ENTITY_TYPES is not None
        assert isinstance(NER_ENTITY_TYPES, dict)
        assert len(NER_ENTITY_TYPES) > 0
        # 验证常见实体类型
        assert 'PER' in NER_ENTITY_TYPES
        assert 'LOC' in NER_ENTITY_TYPES
        assert 'ORG' in NER_ENTITY_TYPES


class TestEntity:
    """测试 Entity 对象"""

    def test_entity_creation(self):
        """测试 Entity 对象创建"""
        entity = Entity(text="北京", entity_type="LOC", start=0, end=2, confidence=0.95)
        assert entity.text == "北京"
        assert entity.entity_type == "LOC"
        assert entity.start == 0
        assert entity.end == 2
        assert entity.confidence == 0.95
        assert entity.length == 2

    def test_entity_to_dict(self):
        """测试 Entity 转字典"""
        entity = Entity(text="北京", entity_type="LOC", start=0, end=2)
        d = entity.to_dict()
        assert isinstance(d, dict)
        assert d['text'] == "北京"
        assert d['entity_type'] == "LOC"
        assert d['start'] == 0
        assert d['end'] == 2

    def test_entity_equality(self):
        """测试 Entity 相等性"""
        e1 = Entity(text="北京", entity_type="LOC", start=0, end=2)
        e2 = Entity(text="北京", entity_type="LOC", start=0, end=2)
        e3 = Entity(text="上海", entity_type="LOC", start=0, end=2)
        assert e1 == e2
        assert e1 != e3
