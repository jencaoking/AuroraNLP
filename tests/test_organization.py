"""测试机构名识别模块"""

import pytest

from AuroraNLP.ner.organization import (
    OrganizationDatabase,
    OrganizationManager,
    Organization,
    Enterprise,
    School,
    Hospital,
    Government,
    OrgType,
    EnterpriseType,
    SchoolType,
    HospitalType,
    GovType,
    ENTERPRISE_SUFFIXES,
    SCHOOL_SUFFIXES,
    HOSPITAL_SUFFIXES,
    GOV_SUFFIXES,
)


class TestOrganizationDatabase:
    """OrganizationDatabase 测试"""

    def test_org_db_init(self):
        """测试初始化"""
        db = OrganizationDatabase(load_default=False)
        assert db is not None
        assert db.get_org_count() == 0
        assert not db.is_loaded()

    def test_org_db_add_org(self):
        """测试添加机构"""
        db = OrganizationDatabase(load_default=False)
        org = Organization(
            org_id="ORG001",
            name="测试公司",
            org_type=OrgType.ENTERPRISE,
            region="北京",
        )
        db.add_organization(org)
        assert db.get_org_count() == 1
        assert db.get_by_id("ORG001") is not None

    def test_org_db_search(self):
        """测试搜索机构"""
        db = OrganizationDatabase(load_default=False)
        org = Organization(
            org_id="ORG001",
            name="清华大学",
            org_type=OrgType.SCHOOL,
        )
        db.add_organization(org)
        results = db.search("清华")
        assert len(results) >= 1
        assert any(o.name == "清华大学" for o in results)


class TestOrganizationManager:
    """OrganizationManager 测试"""

    def test_org_manager_init(self):
        """测试管理器初始化"""
        manager = OrganizationManager(load_default=False)
        assert manager is not None
        assert manager.get_database() is None


class TestEnums:
    """枚举测试"""

    def test_org_type_enum(self):
        """测试机构类型枚举"""
        assert OrgType.ENTERPRISE.value == "enterprise"
        assert OrgType.SCHOOL.value == "school"
        assert OrgType.HOSPITAL.value == "hospital"
        assert OrgType.GOVERNMENT.value == "government"
        assert OrgType.OTHER.value == "other"


class TestSuffixes:
    """后缀常量测试"""

    def test_enterprise_suffixes(self):
        """测试企业后缀常量存在"""
        assert isinstance(ENTERPRISE_SUFFIXES, list)
        assert len(ENTERPRISE_SUFFIXES) > 0
        assert "有限公司" in ENTERPRISE_SUFFIXES

    def test_school_suffixes(self):
        """测试学校后缀常量存在"""
        assert isinstance(SCHOOL_SUFFIXES, list)
        assert len(SCHOOL_SUFFIXES) > 0
        assert "大学" in SCHOOL_SUFFIXES

    def test_hospital_suffixes(self):
        """测试医院后缀常量存在"""
        assert isinstance(HOSPITAL_SUFFIXES, list)
        assert len(HOSPITAL_SUFFIXES) > 0
        assert "医院" in HOSPITAL_SUFFIXES

    def test_gov_suffixes(self):
        """测试政府后缀常量存在"""
        assert isinstance(GOV_SUFFIXES, list)
        assert len(GOV_SUFFIXES) > 0
        assert "政府" in GOV_SUFFIXES


class TestOrganization:
    """Organization 对象测试"""

    def test_organization_creation(self):
        """测试 Organization 对象创建"""
        org = Organization(
            org_id="ORG001",
            name="测试机构",
            org_type=OrgType.ENTERPRISE,
            region="北京",
            aliases=["测试"],
        )
        assert org.org_id == "ORG001"
        assert org.name == "测试机构"
        assert org.org_type == OrgType.ENTERPRISE
        assert org.region == "北京"
        assert "测试" in org.aliases
        assert str(org) == "测试机构"
