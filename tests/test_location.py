"""测试地名识别模块"""

import pytest

from AuroraNLP.ner.location import (
    LocationDatabase,
    LocationManager,
    Location,
    AdminLevel,
)


class TestLocationDatabase:
    """LocationDatabase 测试"""

    def test_location_db_init(self):
        """测试初始化"""
        db = LocationDatabase(load_default=False)
        assert db is not None
        assert db.get_location_count() == 0
        assert not db.is_loaded()

    def test_location_db_add_location(self):
        """测试添加地名"""
        db = LocationDatabase(load_default=False)
        db.add_location(
            code="110000",
            name="北京市",
            level=AdminLevel.PROVINCE,
            latitude=39.9042,
            longitude=116.4074,
        )
        assert db.get_location_count() == 1
        loc = db.get_by_code("110000")
        assert loc is not None
        assert loc.name == "北京市"

    def test_location_db_search(self):
        """测试搜索地名"""
        db = LocationDatabase(load_default=False)
        db.add_location(
            code="110000",
            name="北京市",
            level=AdminLevel.PROVINCE,
        )
        db.add_location(
            code="310000",
            name="上海市",
            level=AdminLevel.PROVINCE,
        )
        results = db.search("北京")
        assert len(results) >= 1
        assert any(loc.name == "北京市" for loc in results)


class TestLocationManager:
    """LocationManager 测试"""

    def test_location_manager_init(self):
        """测试管理器初始化"""
        manager = LocationManager(load_default=False)
        assert manager is not None
        assert manager.get_database() is None


class TestAdminLevel:
    """AdminLevel 枚举测试"""

    def test_admin_level_enum(self):
        """测试行政级别枚举"""
        assert AdminLevel.PROVINCE.value == 1
        assert AdminLevel.CITY.value == 2
        assert AdminLevel.COUNTY.value == 3
        assert AdminLevel.TOWN.value == 4
        assert AdminLevel.VILLAGE.value == 5


class TestLocation:
    """Location 对象测试"""

    def test_location_creation(self):
        """测试 Location 对象创建"""
        loc = Location(
            code="110000",
            name="北京市",
            level=AdminLevel.PROVINCE,
            latitude=39.9042,
            longitude=116.4074,
            aliases=["北京"],
        )
        assert loc.code == "110000"
        assert loc.name == "北京市"
        assert loc.level == AdminLevel.PROVINCE
        assert loc.latitude == 39.9042
        assert loc.longitude == 116.4074
        assert "北京" in loc.aliases
        assert str(loc) == "北京市"
