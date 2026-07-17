"""测试人名识别模块"""

import pytest

from AuroraNLP.ner.person_name import (
    PersonNameDictionary,
    PersonNameManager,
    PersonName,
    Surname,
    NameChar,
    Gender,
    SurnameType,
    NAME_CHAR_CATEGORIES,
)


class TestPersonNameDictionary:
    """PersonNameDictionary 测试"""

    def test_person_name_dict_init(self):
        """测试初始化"""
        db = PersonNameDictionary(load_default=False)
        assert db is not None
        assert db.get_surname_count() == 0
        assert db.get_name_char_count() == 0
        assert not db.is_loaded()

    def test_person_name_dict_add_surname(self):
        """测试添加姓氏"""
        db = PersonNameDictionary(load_default=False)
        db.add_surname("张", frequency=0.1, surname_type=SurnameType.SINGLE)
        assert db.is_surname("张")
        assert db.get_surname("张") is not None
        assert db.get_surname("张").name == "张"

    def test_person_name_dict_add_name_char(self):
        """测试添加名字用字"""
        db = PersonNameDictionary(load_default=False)
        db.add_name_char("明", male_freq=0.5, female_freq=0.3, neutral_freq=0.2)
        assert db.is_name_char("明")
        assert db.get_name_char("明") is not None
        assert db.get_name_char("明").char == "明"


class TestPersonNameManager:
    """PersonNameManager 测试"""

    def test_person_name_manager_init(self):
        """测试管理器初始化"""
        manager = PersonNameManager(load_default=False)
        assert manager is not None
        assert manager.get_dictionary() is None


class TestEnums:
    """枚举测试"""

    def test_gender_enum(self):
        """测试性别枚举"""
        assert Gender.MALE.value == "male"
        assert Gender.FEMALE.value == "female"
        assert Gender.NEUTRAL.value == "neutral"
        assert Gender.UNKNOWN.value == "unknown"

    def test_surname_type_enum(self):
        """测试姓氏类型枚举"""
        assert SurnameType.SINGLE.value == "single"
        assert SurnameType.COMPOUND.value == "compound"


class TestConstants:
    """常量测试"""

    def test_name_char_categories(self):
        """测试名字用字分类常量"""
        assert isinstance(NAME_CHAR_CATEGORIES, dict)
        assert len(NAME_CHAR_CATEGORIES) > 0
        assert "virtue" in NAME_CHAR_CATEGORIES
        assert "nature" in NAME_CHAR_CATEGORIES


class TestPersonName:
    """PersonName 对象测试"""

    def test_person_name_creation(self):
        """测试 PersonName 对象创建"""
        name = PersonName(
            full_name="张三",
            surname="张",
            given_name="三",
            gender=Gender.MALE,
            confidence=0.8,
        )
        assert name.full_name == "张三"
        assert name.surname == "张"
        assert name.given_name == "三"
        assert name.gender == Gender.MALE
        assert name.confidence == 0.8
        assert str(name) == "张三"
