"""
机构名词库模块测试
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from AuroraNLP.organization import (
    OrgType,
    EnterpriseType,
    SchoolType,
    HospitalType,
    GovType,
    ENTERPRISE_SUFFIXES,
    SCHOOL_SUFFIXES,
    HOSPITAL_SUFFIXES,
    GOV_SUFFIXES,
    Organization,
    Enterprise,
    School,
    Hospital,
    Government,
    OrganizationDatabase,
    OrganizationManager,
)


class TestOrgType(unittest.TestCase):
    def test_org_type_values(self):
        self.assertEqual(OrgType.ENTERPRISE.value, "enterprise")
        self.assertEqual(OrgType.SCHOOL.value, "school")
        self.assertEqual(OrgType.HOSPITAL.value, "hospital")
        self.assertEqual(OrgType.GOVERNMENT.value, "government")
    
    def test_org_type_get_name(self):
        self.assertEqual(OrgType.ENTERPRISE.get_name(), "企业")
        self.assertEqual(OrgType.SCHOOL.get_name(), "学校")
        self.assertEqual(OrgType.HOSPITAL.get_name(), "医院")
        self.assertEqual(OrgType.GOVERNMENT.get_name(), "政府机构")


class TestEnterpriseType(unittest.TestCase):
    def test_enterprise_type_values(self):
        self.assertEqual(EnterpriseType.STATE_OWNED.value, "state_owned")
        self.assertEqual(EnterpriseType.PRIVATE.value, "private")
        self.assertEqual(EnterpriseType.FOREIGN.value, "foreign")
    
    def test_enterprise_type_get_name(self):
        self.assertEqual(EnterpriseType.STATE_OWNED.get_name(), "国有企业")
        self.assertEqual(EnterpriseType.PRIVATE.get_name(), "民营企业")
        self.assertEqual(EnterpriseType.FOREIGN.get_name(), "外资企业")


class TestSchoolType(unittest.TestCase):
    def test_school_type_values(self):
        self.assertEqual(SchoolType.UNIVERSITY.value, "university")
        self.assertEqual(SchoolType.HIGH_SCHOOL.value, "high_school")
        self.assertEqual(SchoolType.PRIMARY_SCHOOL.value, "primary_school")
    
    def test_school_type_get_name(self):
        self.assertEqual(SchoolType.UNIVERSITY.get_name(), "大学")
        self.assertEqual(SchoolType.HIGH_SCHOOL.get_name(), "高中")
        self.assertEqual(SchoolType.PRIMARY_SCHOOL.get_name(), "小学")


class TestHospitalType(unittest.TestCase):
    def test_hospital_type_values(self):
        self.assertEqual(HospitalType.GENERAL.value, "general")
        self.assertEqual(HospitalType.SPECIALIZED.value, "specialized")
        self.assertEqual(HospitalType.TCM.value, "tcm")
    
    def test_hospital_type_get_name(self):
        self.assertEqual(HospitalType.GENERAL.get_name(), "综合医院")
        self.assertEqual(HospitalType.SPECIALIZED.get_name(), "专科医院")
        self.assertEqual(HospitalType.TCM.get_name(), "中医院")


class TestGovType(unittest.TestCase):
    def test_gov_type_values(self):
        self.assertEqual(GovType.CENTRAL.value, "central")
        self.assertEqual(GovType.PROVINCE.value, "province")
        self.assertEqual(GovType.CITY.value, "city")
    
    def test_gov_type_get_name(self):
        self.assertEqual(GovType.CENTRAL.get_name(), "中央级")
        self.assertEqual(GovType.PROVINCE.get_name(), "省级")
        self.assertEqual(GovType.CITY.get_name(), "市级")


class TestOrganization(unittest.TestCase):
    def test_organization_creation(self):
        org = Organization(
            org_id="TEST001",
            name="测试机构",
            org_type=OrgType.ENTERPRISE,
            region="北京",
            aliases=["测试", "测试公司"],
        )
        self.assertEqual(org.org_id, "TEST001")
        self.assertEqual(org.name, "测试机构")
        self.assertEqual(org.org_type, OrgType.ENTERPRISE)
        self.assertEqual(org.region, "北京")
        self.assertEqual(len(org.aliases), 2)
    
    def test_organization_get_all_names(self):
        org = Organization(
            org_id="TEST001",
            name="测试机构",
            org_type=OrgType.ENTERPRISE,
            aliases=["测试", "测试公司"],
        )
        names = org.get_all_names()
        self.assertEqual(len(names), 3)
        self.assertIn("测试机构", names)
        self.assertIn("测试", names)
    
    def test_organization_matches(self):
        org = Organization(
            org_id="TEST001",
            name="测试机构",
            org_type=OrgType.ENTERPRISE,
            aliases=["测试", "测试公司"],
        )
        self.assertTrue(org.matches("测试机构"))
        self.assertTrue(org.matches("TEST001"))
        self.assertTrue(org.matches("测试"))
        self.assertFalse(org.matches("其他机构"))
    
    def test_organization_type_checks(self):
        enterprise = Organization(
            org_id="ENT001",
            name="企业",
            org_type=OrgType.ENTERPRISE,
        )
        school = Organization(
            org_id="SCH001",
            name="学校",
            org_type=OrgType.SCHOOL,
        )
        hospital = Organization(
            org_id="HOS001",
            name="医院",
            org_type=OrgType.HOSPITAL,
        )
        government = Organization(
            org_id="GOV001",
            name="政府",
            org_type=OrgType.GOVERNMENT,
        )
        
        self.assertTrue(enterprise.is_enterprise())
        self.assertFalse(enterprise.is_school())
        self.assertTrue(school.is_school())
        self.assertFalse(school.is_hospital())
        self.assertTrue(hospital.is_hospital())
        self.assertFalse(hospital.is_government())
        self.assertTrue(government.is_government())


class TestEnterprise(unittest.TestCase):
    def test_enterprise_creation(self):
        ent = Enterprise(
            org_id="ENT001",
            name="测试企业",
            org_type=OrgType.ENTERPRISE,
            enterprise_type=EnterpriseType.PRIVATE,
            industry="互联网",
            region="北京",
        )
        self.assertEqual(ent.org_type, OrgType.ENTERPRISE)
        self.assertEqual(ent.enterprise_type, EnterpriseType.PRIVATE)
        self.assertEqual(ent.industry, "互联网")


class TestSchool(unittest.TestCase):
    def test_school_creation(self):
        school = School(
            org_id="SCH001",
            name="测试学校",
            school_type=SchoolType.UNIVERSITY,
            is_public=True,
            level="985",
            region="北京",
        )
        self.assertEqual(school.org_type, OrgType.SCHOOL)
        self.assertEqual(school.school_type, SchoolType.UNIVERSITY)
        self.assertTrue(school.is_public)
        self.assertEqual(school.level, "985")


class TestHospital(unittest.TestCase):
    def test_hospital_creation(self):
        hospital = Hospital(
            org_id="HOS001",
            name="测试医院",
            hospital_type=HospitalType.GENERAL,
            level="三级甲等",
            is_public=True,
            region="北京",
        )
        self.assertEqual(hospital.org_type, OrgType.HOSPITAL)
        self.assertEqual(hospital.hospital_type, HospitalType.GENERAL)
        self.assertEqual(hospital.level, "三级甲等")


class TestGovernment(unittest.TestCase):
    def test_government_creation(self):
        gov = Government(
            org_id="GOV001",
            name="测试政府",
            gov_type=GovType.CITY,
            department="教育",
            region="北京",
        )
        self.assertEqual(gov.org_type, OrgType.GOVERNMENT)
        self.assertEqual(gov.gov_type, GovType.CITY)
        self.assertEqual(gov.department, "教育")


class TestOrganizationDatabase(unittest.TestCase):
    def setUp(self):
        self.db = OrganizationDatabase(load_default=True)
    
    def test_load_data(self):
        self.assertTrue(self.db.is_loaded())
        self.assertGreater(self.db.get_org_count(), 0)
    
    def test_get_by_id(self):
        org = self.db.get_by_id("ENT001")
        self.assertIsNotNone(org)
        self.assertEqual(org.name, "阿里巴巴集团")
    
    def test_get_by_name(self):
        orgs = self.db.get_by_name("北京大学")
        self.assertGreater(len(orgs), 0)
        self.assertEqual(orgs[0].name, "北京大学")
    
    def test_get_by_alias(self):
        org = self.db.get_by_alias("阿里")
        self.assertIsNotNone(org)
        self.assertEqual(org.name, "阿里巴巴集团")
        
        org = self.db.get_by_alias("北大")
        self.assertIsNotNone(org)
        self.assertEqual(org.name, "北京大学")
    
    def test_search(self):
        results = self.db.search("大学")
        self.assertGreater(len(results), 0)
    
    def test_get_by_type(self):
        enterprises = self.db.get_enterprises()
        self.assertGreater(len(enterprises), 0)
        
        schools = self.db.get_schools()
        self.assertGreater(len(schools), 0)
        
        hospitals = self.db.get_hospitals()
        self.assertGreater(len(hospitals), 0)
        
        governments = self.db.get_governments()
        self.assertGreater(len(governments), 0)
    
    def test_get_by_region(self):
        beijing_orgs = self.db.get_by_region("北京")
        self.assertGreater(len(beijing_orgs), 0)
    
    def test_is_organization(self):
        self.assertTrue(self.db.is_organization("阿里巴巴集团"))
        self.assertTrue(self.db.is_organization("阿里"))
        self.assertTrue(self.db.is_organization("北京大学"))
        self.assertFalse(self.db.is_organization("不存在的机构"))
    
    def test_is_enterprise(self):
        self.assertTrue(self.db.is_enterprise("阿里巴巴集团"))
        self.assertFalse(self.db.is_enterprise("北京大学"))
    
    def test_is_school(self):
        self.assertTrue(self.db.is_school("北京大学"))
        self.assertFalse(self.db.is_school("阿里巴巴集团"))
    
    def test_is_hospital(self):
        self.assertTrue(self.db.is_hospital("北京协和医院"))
        self.assertFalse(self.db.is_hospital("北京大学"))
    
    def test_is_government(self):
        self.assertTrue(self.db.is_government("中华人民共和国国务院"))
        self.assertFalse(self.db.is_government("北京大学"))
    
    def test_recognize_organizations(self):
        text = "阿里巴巴集团和腾讯控股有限公司都是知名互联网企业，北京大学和清华大学是中国顶尖学府。"
        results = self.db.recognize_organizations(text)
        self.assertGreater(len(results), 0)
        
        names = [org.name for org, _, _ in results]
        self.assertIn("阿里巴巴集团", names)
        self.assertIn("腾讯控股有限公司", names)
    
    def test_guess_org_type(self):
        self.assertEqual(self.db.guess_org_type("某某有限公司"), OrgType.ENTERPRISE)
        self.assertEqual(self.db.guess_org_type("某某大学"), OrgType.SCHOOL)
        self.assertEqual(self.db.guess_org_type("某某医院"), OrgType.HOSPITAL)
        self.assertEqual(self.db.guess_org_type("某某人民政府"), OrgType.GOVERNMENT)
    
    def test_get_statistics(self):
        stats = self.db.get_statistics()
        self.assertTrue(stats["loaded"])
        self.assertGreater(stats["total_count"], 0)
        self.assertGreater(stats["enterprise_count"], 0)
        self.assertGreater(stats["school_count"], 0)
        self.assertGreater(stats["hospital_count"], 0)
        self.assertGreater(stats["government_count"], 0)
    
    def test_add_organization(self):
        initial_count = self.db.get_org_count()
        
        org = Organization(
            org_id="TEST_ADD_001",
            name="新增测试机构",
            org_type=OrgType.OTHER,
            region="测试地区",
        )
        self.db.add_organization(org)
        
        self.assertEqual(self.db.get_org_count(), initial_count + 1)
        self.assertIsNotNone(self.db.get_by_id("TEST_ADD_001"))
    
    def test_add_enterprise(self):
        self.db.add_enterprise(
            org_id="TEST_ENT_001",
            name="新增测试企业",
            enterprise_type=EnterpriseType.PRIVATE,
            industry="测试行业",
            region="测试地区",
        )
        
        org = self.db.get_by_id("TEST_ENT_001")
        self.assertIsNotNone(org)
        self.assertIsInstance(org, Enterprise)
        self.assertEqual(org.name, "新增测试企业")
    
    def test_add_school(self):
        self.db.add_school(
            org_id="TEST_SCH_001",
            name="新增测试学校",
            school_type=SchoolType.UNIVERSITY,
            is_public=True,
            region="测试地区",
        )
        
        org = self.db.get_by_id("TEST_SCH_001")
        self.assertIsNotNone(org)
        self.assertIsInstance(org, School)
        self.assertEqual(org.name, "新增测试学校")
    
    def test_add_hospital(self):
        self.db.add_hospital(
            org_id="TEST_HOS_001",
            name="新增测试医院",
            hospital_type=HospitalType.GENERAL,
            level="三级甲等",
            region="测试地区",
        )
        
        org = self.db.get_by_id("TEST_HOS_001")
        self.assertIsNotNone(org)
        self.assertIsInstance(org, Hospital)
        self.assertEqual(org.name, "新增测试医院")
    
    def test_add_government(self):
        self.db.add_government(
            org_id="TEST_GOV_001",
            name="新增测试政府",
            gov_type=GovType.CITY,
            region="测试地区",
        )
        
        org = self.db.get_by_id("TEST_GOV_001")
        self.assertIsNotNone(org)
        self.assertIsInstance(org, Government)
        self.assertEqual(org.name, "新增测试政府")
    
    def test_contains(self):
        self.assertIn("阿里巴巴集团", self.db)
        self.assertIn("北大", self.db)
        self.assertNotIn("不存在的机构", self.db)
    
    def test_getitem(self):
        org = self.db["ENT001"]
        self.assertIsNotNone(org)
        self.assertEqual(org.name, "阿里巴巴集团")
    
    def test_len(self):
        self.assertGreater(len(self.db), 0)


class TestOrganizationManager(unittest.TestCase):
    def setUp(self):
        self.manager = OrganizationManager(load_default=True)
    
    def test_get_database(self):
        db = self.manager.get_database()
        self.assertIsNotNone(db)
    
    def test_is_loaded(self):
        self.assertTrue(self.manager.is_loaded())
    
    def test_get_by_id(self):
        org = self.manager.get_by_id("ENT001")
        self.assertIsNotNone(org)
        self.assertEqual(org.name, "阿里巴巴集团")
    
    def test_get_by_name(self):
        orgs = self.manager.get_by_name("北京大学")
        self.assertGreater(len(orgs), 0)
    
    def test_get_by_alias(self):
        org = self.manager.get_by_alias("阿里")
        self.assertIsNotNone(org)
    
    def test_search(self):
        results = self.manager.search("大学")
        self.assertGreater(len(results), 0)
    
    def test_get_by_type(self):
        enterprises = self.manager.get_enterprises()
        self.assertGreater(len(enterprises), 0)
        
        schools = self.manager.get_schools()
        self.assertGreater(len(schools), 0)
    
    def test_is_organization(self):
        self.assertTrue(self.manager.is_organization("阿里巴巴集团"))
        self.assertTrue(self.manager.is_organization("北京大学"))
    
    def test_is_enterprise(self):
        self.assertTrue(self.manager.is_enterprise("阿里巴巴集团"))
        self.assertFalse(self.manager.is_enterprise("北京大学"))
    
    def test_is_school(self):
        self.assertTrue(self.manager.is_school("北京大学"))
        self.assertFalse(self.manager.is_school("阿里巴巴集团"))
    
    def test_is_hospital(self):
        self.assertTrue(self.manager.is_hospital("北京协和医院"))
    
    def test_is_government(self):
        self.assertTrue(self.manager.is_government("中华人民共和国国务院"))
    
    def test_recognize_organizations(self):
        text = "阿里巴巴和腾讯是中国互联网巨头，北京协和医院是知名医院。"
        results = self.manager.recognize_organizations(text)
        self.assertGreater(len(results), 0)
    
    def test_guess_org_type(self):
        self.assertEqual(self.manager.guess_org_type("某某有限公司"), OrgType.ENTERPRISE)
        self.assertEqual(self.manager.guess_org_type("某某大学"), OrgType.SCHOOL)
    
    def test_get_statistics(self):
        stats = self.manager.get_statistics()
        self.assertTrue(stats["loaded"])


class TestSuffixes(unittest.TestCase):
    def test_enterprise_suffixes(self):
        self.assertIn("有限公司", ENTERPRISE_SUFFIXES)
        self.assertIn("集团", ENTERPRISE_SUFFIXES)
        self.assertIn("公司", ENTERPRISE_SUFFIXES)
    
    def test_school_suffixes(self):
        self.assertIn("大学", SCHOOL_SUFFIXES)
        self.assertIn("学院", SCHOOL_SUFFIXES)
        self.assertIn("中学", SCHOOL_SUFFIXES)
        self.assertIn("小学", SCHOOL_SUFFIXES)
    
    def test_hospital_suffixes(self):
        self.assertIn("医院", HOSPITAL_SUFFIXES)
        self.assertIn("人民医院", HOSPITAL_SUFFIXES)
        self.assertIn("中医院", HOSPITAL_SUFFIXES)
    
    def test_gov_suffixes(self):
        self.assertIn("人民政府", GOV_SUFFIXES)
        self.assertIn("政府", GOV_SUFFIXES)
        self.assertIn("厅", GOV_SUFFIXES)
        self.assertIn("局", GOV_SUFFIXES)


if __name__ == "__main__":
    unittest.main()
