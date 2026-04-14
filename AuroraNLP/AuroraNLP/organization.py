"""
机构名词库模块 - 机构名词库构建

提供企业、学校、医院、政府机构等机构名词的识别、查询和管理功能。

功能：
- 企业名称库：包含知名企业及其信息
- 学校名称库：包含各类学校及其信息
- 医院名称库：包含各类医院及其信息
- 政府机构库：包含各级政府机构及其信息
- 机构识别：从文本中识别机构名
- 别名处理：支持机构别名和简称

数据格式说明：
- 机构格式：机构ID\t名称\t类型\t子类型\t地区\t别名(逗号分隔)\t网址\t描述
"""

import os
import re
from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum


class OrgType(Enum):
    ENTERPRISE = "enterprise"
    SCHOOL = "school"
    HOSPITAL = "hospital"
    GOVERNMENT = "government"
    OTHER = "other"
    
    def get_name(self) -> str:
        names = {
            self.ENTERPRISE: "企业",
            self.SCHOOL: "学校",
            self.HOSPITAL: "医院",
            self.GOVERNMENT: "政府机构",
            self.OTHER: "其他",
        }
        return names.get(self, "未知")


class EnterpriseType(Enum):
    STATE_OWNED = "state_owned"
    PRIVATE = "private"
    FOREIGN = "foreign"
    JOINT_VENTURE = "joint_venture"
    LISTED = "listed"
    STARTUP = "startup"
    OTHER = "other"
    
    def get_name(self) -> str:
        names = {
            self.STATE_OWNED: "国有企业",
            self.PRIVATE: "民营企业",
            self.FOREIGN: "外资企业",
            self.JOINT_VENTURE: "合资企业",
            self.LISTED: "上市企业",
            self.STARTUP: "创业公司",
            self.OTHER: "其他企业",
        }
        return names.get(self, "未知")


class SchoolType(Enum):
    UNIVERSITY = "university"
    COLLEGE = "college"
    HIGH_SCHOOL = "high_school"
    MIDDLE_SCHOOL = "middle_school"
    PRIMARY_SCHOOL = "primary_school"
    KINDERGARTEN = "kindergarten"
    VOCATIONAL = "vocational"
    OTHER = "other"
    
    def get_name(self) -> str:
        names = {
            self.UNIVERSITY: "大学",
            self.COLLEGE: "学院",
            self.HIGH_SCHOOL: "高中",
            self.MIDDLE_SCHOOL: "初中",
            self.PRIMARY_SCHOOL: "小学",
            self.KINDERGARTEN: "幼儿园",
            self.VOCATIONAL: "职业学校",
            self.OTHER: "其他学校",
        }
        return names.get(self, "未知")


class HospitalType(Enum):
    GENERAL = "general"
    SPECIALIZED = "specialized"
    TCM = "tcm"
    COMMUNITY = "community"
    CLINIC = "clinic"
    MATERNITY = "maternity"
    CHILDREN = "children"
    OTHER = "other"
    
    def get_name(self) -> str:
        names = {
            self.GENERAL: "综合医院",
            self.SPECIALIZED: "专科医院",
            self.TCM: "中医院",
            self.COMMUNITY: "社区医院",
            self.CLINIC: "诊所",
            self.MATERNITY: "妇幼保健院",
            self.CHILDREN: "儿童医院",
            self.OTHER: "其他医院",
        }
        return names.get(self, "未知")


class GovType(Enum):
    CENTRAL = "central"
    PROVINCE = "province"
    CITY = "city"
    COUNTY = "county"
    TOWN = "town"
    OTHER = "other"
    
    def get_name(self) -> str:
        names = {
            self.CENTRAL: "中央级",
            self.PROVINCE: "省级",
            self.CITY: "市级",
            self.COUNTY: "县级",
            self.TOWN: "乡镇级",
            self.OTHER: "其他",
        }
        return names.get(self, "未知")


ENTERPRISE_SUFFIXES: List[str] = [
    "有限公司", "股份有限公司", "集团", "公司", "有限责任公司",
    "集团股份有限公司", "控股集团", "投资集团", "科技集团",
    "Co.", "Ltd.", "Inc.", "Corp.", "Corporation", "Company",
]

SCHOOL_SUFFIXES: List[str] = [
    "大学", "学院", "学校", "中学", "小学", "幼儿园",
    "高级中学", "初级中学", "实验小学", "附属中学", "附属小学",
    "职业技术学院", "专科学校", "研究院", "研究生院",
]

HOSPITAL_SUFFIXES: List[str] = [
    "医院", "人民医院", "中心医院", "附属医院", "专科医院",
    "中医院", "中西医结合医院", "妇幼保健院", "儿童医院",
    "第一医院", "第二医院", "第三医院", "诊所", "卫生院",
]

GOV_SUFFIXES: List[str] = [
    "人民政府", "政府", "厅", "局", "委员会", "部", "署",
    "办公室", "中心", "站", "所", "院", "司", "处",
    "省委", "市委", "县委", "区委", "镇政府", "乡政府",
]


@dataclass
class Organization:
    org_id: str
    name: str
    org_type: OrgType
    sub_type: Optional[str] = None
    region: Optional[str] = None
    aliases: List[str] = field(default_factory=list)
    website: Optional[str] = None
    description: Optional[str] = None
    founded: Optional[str] = None
    address: Optional[str] = None
    
    def __str__(self) -> str:
        return self.name
    
    def __repr__(self) -> str:
        return (
            f"Organization(id='{self.org_id}', name='{self.name}', "
            f"type={self.org_type.get_name()})"
        )
    
    def get_all_names(self) -> List[str]:
        names = [self.name]
        names.extend(self.aliases)
        return names
    
    def matches(self, text: str) -> bool:
        if text == self.name:
            return True
        if text == self.org_id:
            return True
        if text in self.aliases:
            return True
        return False
    
    def is_enterprise(self) -> bool:
        return self.org_type == OrgType.ENTERPRISE
    
    def is_school(self) -> bool:
        return self.org_type == OrgType.SCHOOL
    
    def is_hospital(self) -> bool:
        return self.org_type == OrgType.HOSPITAL
    
    def is_government(self) -> bool:
        return self.org_type == OrgType.GOVERNMENT


@dataclass
class Enterprise(Organization):
    enterprise_type: EnterpriseType = EnterpriseType.OTHER
    industry: Optional[str] = None
    stock_code: Optional[str] = None
    employees: Optional[int] = None
    org_type: OrgType = OrgType.ENTERPRISE
    
    def __post_init__(self):
        self.org_type = OrgType.ENTERPRISE


@dataclass
class School(Organization):
    school_type: SchoolType = SchoolType.OTHER
    is_public: bool = True
    level: Optional[str] = None
    org_type: OrgType = OrgType.SCHOOL
    
    def __post_init__(self):
        self.org_type = OrgType.SCHOOL


@dataclass
class Hospital(Organization):
    hospital_type: HospitalType = HospitalType.OTHER
    level: Optional[str] = None
    is_public: bool = True
    bed_count: Optional[int] = None
    org_type: OrgType = OrgType.HOSPITAL
    
    def __post_init__(self):
        self.org_type = OrgType.HOSPITAL


@dataclass
class Government(Organization):
    gov_type: GovType = GovType.OTHER
    department: Optional[str] = None
    org_type: OrgType = OrgType.GOVERNMENT
    
    def __post_init__(self):
        self.org_type = OrgType.GOVERNMENT


class OrganizationDatabase:
    DEFAULT_DATA_PATH = os.path.join(
        os.path.dirname(__file__), 'data', 'organizations.txt'
    )
    
    def __init__(self, load_default: bool = True):
        self._organizations: Dict[str, Organization] = {}
        self._name_index: Dict[str, Set[str]] = {}
        self._alias_index: Dict[str, str] = {}
        self._type_index: Dict[OrgType, Set[str]] = {
            OrgType.ENTERPRISE: set(),
            OrgType.SCHOOL: set(),
            OrgType.HOSPITAL: set(),
            OrgType.GOVERNMENT: set(),
            OrgType.OTHER: set(),
        }
        self._region_index: Dict[str, Set[str]] = {}
        self._loaded: bool = False
        self._org_count: int = 0
        
        if load_default:
            self._load_default_data()
    
    def _load_default_data(self) -> None:
        if os.path.exists(self.DEFAULT_DATA_PATH):
            self.load_data(self.DEFAULT_DATA_PATH)
    
    def load_data(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"机构数据文件不存在: {path}")
        
        self._organizations.clear()
        self._name_index.clear()
        self._alias_index.clear()
        for org_type in self._type_index:
            self._type_index[org_type] = set()
        self._region_index.clear()
        self._org_count = 0
        
        current_section = None
        
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    if line.startswith('# @'):
                        section = line[3:].strip()
                        if section in ['enterprises', 'schools', 'hospitals', 'governments']:
                            current_section = section
                    continue
                
                self._parse_organization(line, current_section)
        
        self._loaded = True
    
    def _parse_organization(self, line: str, section: Optional[str]) -> None:
        parts = line.split('\t')
        if len(parts) < 3:
            return
        
        org_id = parts[0].strip()
        name = parts[1].strip()
        
        org_type_str = parts[2].strip().lower()
        org_type = self._parse_org_type(org_type_str)
        
        sub_type = parts[3].strip() if len(parts) > 3 and parts[3].strip() else None
        region = parts[4].strip() if len(parts) > 4 and parts[4].strip() else None
        
        aliases = []
        if len(parts) > 5 and parts[5].strip():
            aliases = [a.strip() for a in parts[5].split(',') if a.strip()]
        
        website = parts[6].strip() if len(parts) > 6 and parts[6].strip() else None
        description = parts[7].strip() if len(parts) > 7 and parts[7].strip() else None
        founded = parts[8].strip() if len(parts) > 8 and parts[8].strip() else None
        address = parts[9].strip() if len(parts) > 9 and parts[9].strip() else None
        
        org: Organization
        if org_type == OrgType.ENTERPRISE:
            enterprise_type = self._parse_enterprise_type(sub_type) if sub_type else EnterpriseType.OTHER
            industry = parts[10].strip() if len(parts) > 10 and parts[10].strip() else None
            org = Enterprise(
                org_id=org_id,
                name=name,
                org_type=org_type,
                enterprise_type=enterprise_type,
                industry=industry,
                region=region,
                aliases=aliases,
                website=website,
                description=description,
                founded=founded,
                address=address,
            )
        elif org_type == OrgType.SCHOOL:
            school_type = self._parse_school_type(sub_type) if sub_type else SchoolType.OTHER
            is_public = parts[10].strip().lower() != 'private' if len(parts) > 10 and parts[10].strip() else True
            level = parts[11].strip() if len(parts) > 11 and parts[11].strip() else None
            org = School(
                org_id=org_id,
                name=name,
                org_type=org_type,
                school_type=school_type,
                is_public=is_public,
                level=level,
                region=region,
                aliases=aliases,
                website=website,
                description=description,
                founded=founded,
                address=address,
            )
        elif org_type == OrgType.HOSPITAL:
            hospital_type = self._parse_hospital_type(sub_type) if sub_type else HospitalType.OTHER
            level = parts[10].strip() if len(parts) > 10 and parts[10].strip() else None
            is_public = parts[11].strip().lower() != 'private' if len(parts) > 11 and parts[11].strip() else True
            org = Hospital(
                org_id=org_id,
                name=name,
                org_type=org_type,
                hospital_type=hospital_type,
                level=level,
                is_public=is_public,
                region=region,
                aliases=aliases,
                website=website,
                description=description,
                founded=founded,
                address=address,
            )
        elif org_type == OrgType.GOVERNMENT:
            gov_type = self._parse_gov_type(sub_type) if sub_type else GovType.OTHER
            department = parts[10].strip() if len(parts) > 10 and parts[10].strip() else None
            org = Government(
                org_id=org_id,
                name=name,
                org_type=org_type,
                gov_type=gov_type,
                department=department,
                region=region,
                aliases=aliases,
                website=website,
                description=description,
                founded=founded,
                address=address,
            )
        else:
            org = Organization(
                org_id=org_id,
                name=name,
                org_type=org_type,
                sub_type=sub_type,
                region=region,
                aliases=aliases,
                website=website,
                description=description,
                founded=founded,
                address=address,
            )
        
        self._add_organization(org)
    
    def _parse_org_type(self, type_str: str) -> OrgType:
        mapping = {
            'enterprise': OrgType.ENTERPRISE,
            '企业': OrgType.ENTERPRISE,
            'school': OrgType.SCHOOL,
            '学校': OrgType.SCHOOL,
            'hospital': OrgType.HOSPITAL,
            '医院': OrgType.HOSPITAL,
            'government': OrgType.GOVERNMENT,
            '政府': OrgType.GOVERNMENT,
            '政府机构': OrgType.GOVERNMENT,
        }
        return mapping.get(type_str.lower(), OrgType.OTHER)
    
    def _parse_enterprise_type(self, type_str: str) -> EnterpriseType:
        mapping = {
            'state_owned': EnterpriseType.STATE_OWNED,
            '国企': EnterpriseType.STATE_OWNED,
            '国有企业': EnterpriseType.STATE_OWNED,
            'private': EnterpriseType.PRIVATE,
            '民企': EnterpriseType.PRIVATE,
            '民营企业': EnterpriseType.PRIVATE,
            'foreign': EnterpriseType.FOREIGN,
            '外企': EnterpriseType.FOREIGN,
            '外资企业': EnterpriseType.FOREIGN,
            'joint_venture': EnterpriseType.JOINT_VENTURE,
            '合资': EnterpriseType.JOINT_VENTURE,
            '合资企业': EnterpriseType.JOINT_VENTURE,
            'listed': EnterpriseType.LISTED,
            '上市': EnterpriseType.LISTED,
            '上市企业': EnterpriseType.LISTED,
            'startup': EnterpriseType.STARTUP,
            '创业公司': EnterpriseType.STARTUP,
        }
        return mapping.get(type_str.lower(), EnterpriseType.OTHER)
    
    def _parse_school_type(self, type_str: str) -> SchoolType:
        mapping = {
            'university': SchoolType.UNIVERSITY,
            '大学': SchoolType.UNIVERSITY,
            'college': SchoolType.COLLEGE,
            '学院': SchoolType.COLLEGE,
            'high_school': SchoolType.HIGH_SCHOOL,
            '高中': SchoolType.HIGH_SCHOOL,
            '高级中学': SchoolType.HIGH_SCHOOL,
            'middle_school': SchoolType.MIDDLE_SCHOOL,
            '初中': SchoolType.MIDDLE_SCHOOL,
            '初级中学': SchoolType.MIDDLE_SCHOOL,
            'primary_school': SchoolType.PRIMARY_SCHOOL,
            '小学': SchoolType.PRIMARY_SCHOOL,
            'kindergarten': SchoolType.KINDERGARTEN,
            '幼儿园': SchoolType.KINDERGARTEN,
            'vocational': SchoolType.VOCATIONAL,
            '职业': SchoolType.VOCATIONAL,
            '职业学校': SchoolType.VOCATIONAL,
        }
        return mapping.get(type_str.lower(), SchoolType.OTHER)
    
    def _parse_hospital_type(self, type_str: str) -> HospitalType:
        mapping = {
            'general': HospitalType.GENERAL,
            '综合': HospitalType.GENERAL,
            '综合医院': HospitalType.GENERAL,
            'specialized': HospitalType.SPECIALIZED,
            '专科': HospitalType.SPECIALIZED,
            '专科医院': HospitalType.SPECIALIZED,
            'tcm': HospitalType.TCM,
            '中医': HospitalType.TCM,
            '中医院': HospitalType.TCM,
            'community': HospitalType.COMMUNITY,
            '社区': HospitalType.COMMUNITY,
            '社区医院': HospitalType.COMMUNITY,
            'clinic': HospitalType.CLINIC,
            '诊所': HospitalType.CLINIC,
            'maternity': HospitalType.MATERNITY,
            '妇幼': HospitalType.MATERNITY,
            '妇幼保健院': HospitalType.MATERNITY,
            'children': HospitalType.CHILDREN,
            '儿童': HospitalType.CHILDREN,
            '儿童医院': HospitalType.CHILDREN,
        }
        return mapping.get(type_str.lower(), HospitalType.OTHER)
    
    def _parse_gov_type(self, type_str: str) -> GovType:
        mapping = {
            'central': GovType.CENTRAL,
            '中央': GovType.CENTRAL,
            '中央级': GovType.CENTRAL,
            'province': GovType.PROVINCE,
            '省': GovType.PROVINCE,
            '省级': GovType.PROVINCE,
            'city': GovType.CITY,
            '市': GovType.CITY,
            '市级': GovType.CITY,
            'county': GovType.COUNTY,
            '县': GovType.COUNTY,
            '县级': GovType.COUNTY,
            'town': GovType.TOWN,
            '乡镇': GovType.TOWN,
            '乡镇级': GovType.TOWN,
        }
        return mapping.get(type_str.lower(), GovType.OTHER)
    
    def _add_organization(self, org: Organization) -> None:
        self._organizations[org.org_id] = org
        self._org_count += 1
        
        if org.name not in self._name_index:
            self._name_index[org.name] = set()
        self._name_index[org.name].add(org.org_id)
        
        for alias in org.aliases:
            self._alias_index[alias] = org.org_id
        
        self._type_index[org.org_type].add(org.org_id)
        
        if org.region:
            if org.region not in self._region_index:
                self._region_index[org.region] = set()
            self._region_index[org.region].add(org.org_id)
    
    def get_by_id(self, org_id: str) -> Optional[Organization]:
        return self._organizations.get(org_id)
    
    def get_by_name(self, name: str) -> List[Organization]:
        codes = self._name_index.get(name, set())
        return [self._organizations[code] for code in codes if code in self._organizations]
    
    def get_by_alias(self, alias: str) -> Optional[Organization]:
        code = self._alias_index.get(alias)
        if code:
            return self._organizations.get(code)
        return None
    
    def search(self, query: str) -> List[Organization]:
        results: List[Organization] = []
        
        if query in self._organizations:
            results.append(self._organizations[query])
        
        results.extend(self.get_by_name(query))
        
        org = self.get_by_alias(query)
        if org and org not in results:
            results.append(org)
        
        for name, codes in self._name_index.items():
            if query in name and name != query:
                for code in codes:
                    o = self._organizations.get(code)
                    if o and o not in results:
                        results.append(o)
        
        return results
    
    def get_by_type(self, org_type: OrgType) -> List[Organization]:
        codes = self._type_index.get(org_type, set())
        return [self._organizations[code] for code in codes if code in self._organizations]
    
    def get_enterprises(self) -> List[Organization]:
        return self.get_by_type(OrgType.ENTERPRISE)
    
    def get_schools(self) -> List[Organization]:
        return self.get_by_type(OrgType.SCHOOL)
    
    def get_hospitals(self) -> List[Organization]:
        return self.get_by_type(OrgType.HOSPITAL)
    
    def get_governments(self) -> List[Organization]:
        return self.get_by_type(OrgType.GOVERNMENT)
    
    def get_by_region(self, region: str) -> List[Organization]:
        codes = self._region_index.get(region, set())
        return [self._organizations[code] for code in codes if code in self._organizations]
    
    def get_by_type_and_region(self, org_type: OrgType, region: str) -> List[Organization]:
        type_codes = self._type_index.get(org_type, set())
        region_codes = self._region_index.get(region, set())
        codes = type_codes & region_codes
        return [self._organizations[code] for code in codes if code in self._organizations]
    
    def is_organization(self, text: str) -> bool:
        if text in self._organizations:
            return True
        if text in self._name_index:
            return True
        if text in self._alias_index:
            return True
        return False
    
    def is_enterprise(self, text: str) -> bool:
        orgs = self.get_by_name(text)
        if any(o.is_enterprise() for o in orgs):
            return True
        org = self.get_by_alias(text)
        if org and org.is_enterprise():
            return True
        return False
    
    def is_school(self, text: str) -> bool:
        orgs = self.get_by_name(text)
        if any(o.is_school() for o in orgs):
            return True
        org = self.get_by_alias(text)
        if org and org.is_school():
            return True
        return False
    
    def is_hospital(self, text: str) -> bool:
        orgs = self.get_by_name(text)
        if any(o.is_hospital() for o in orgs):
            return True
        org = self.get_by_alias(text)
        if org and org.is_hospital():
            return True
        return False
    
    def is_government(self, text: str) -> bool:
        orgs = self.get_by_name(text)
        if any(o.is_government() for o in orgs):
            return True
        org = self.get_by_alias(text)
        if org and org.is_government():
            return True
        return False
    
    def recognize_organizations(self, text: str) -> List[Tuple[Organization, int, int]]:
        results: List[Tuple[Organization, int, int]] = []
        used_positions: Set[int] = set()
        
        sorted_names = sorted(
            set(self._name_index.keys()) | set(self._alias_index.keys()),
            key=len,
            reverse=True
        )
        
        for name in sorted_names:
            start = 0
            while True:
                pos = text.find(name, start)
                if pos == -1:
                    break
                
                end = pos + len(name)
                overlap = False
                for i in range(pos, end):
                    if i in used_positions:
                        overlap = True
                        break
                
                if not overlap:
                    if name in self._alias_index:
                        code = self._alias_index[name]
                        org = self._organizations.get(code)
                    else:
                        codes = self._name_index.get(name, set())
                        org = self._organizations.get(next(iter(codes))) if codes else None
                    
                    if org:
                        results.append((org, pos, end))
                        for i in range(pos, end):
                            used_positions.add(i)
                
                start = pos + 1
        
        results.sort(key=lambda x: x[1])
        return results
    
    def guess_org_type(self, text: str) -> Optional[OrgType]:
        for suffix in ENTERPRISE_SUFFIXES:
            if text.endswith(suffix):
                return OrgType.ENTERPRISE
        
        for suffix in SCHOOL_SUFFIXES:
            if text.endswith(suffix):
                return OrgType.SCHOOL
        
        for suffix in HOSPITAL_SUFFIXES:
            if text.endswith(suffix):
                return OrgType.HOSPITAL
        
        for suffix in GOV_SUFFIXES:
            if text.endswith(suffix):
                return OrgType.GOVERNMENT
        
        return None
    
    def get_all_organizations(self) -> List[Organization]:
        return list(self._organizations.values())
    
    def get_org_count(self) -> int:
        return self._org_count
    
    def is_loaded(self) -> bool:
        return self._loaded
    
    def get_statistics(self) -> Dict[str, Any]:
        return {
            "loaded": self._loaded,
            "total_count": self._org_count,
            "enterprise_count": len(self._type_index[OrgType.ENTERPRISE]),
            "school_count": len(self._type_index[OrgType.SCHOOL]),
            "hospital_count": len(self._type_index[OrgType.HOSPITAL]),
            "government_count": len(self._type_index[OrgType.GOVERNMENT]),
            "alias_count": len(self._alias_index),
            "region_count": len(self._region_index),
        }
    
    def add_organization(self, org: Organization) -> None:
        self._add_organization(org)
    
    def add_enterprise(
        self,
        org_id: str,
        name: str,
        enterprise_type: EnterpriseType = EnterpriseType.OTHER,
        industry: Optional[str] = None,
        region: Optional[str] = None,
        aliases: Optional[List[str]] = None,
        website: Optional[str] = None,
        description: Optional[str] = None,
    ) -> None:
        org = Enterprise(
            org_id=org_id,
            name=name,
            org_type=OrgType.ENTERPRISE,
            enterprise_type=enterprise_type,
            industry=industry,
            region=region,
            aliases=aliases or [],
            website=website,
            description=description,
        )
        self._add_organization(org)
    
    def add_school(
        self,
        org_id: str,
        name: str,
        school_type: SchoolType = SchoolType.OTHER,
        is_public: bool = True,
        level: Optional[str] = None,
        region: Optional[str] = None,
        aliases: Optional[List[str]] = None,
        website: Optional[str] = None,
        description: Optional[str] = None,
    ) -> None:
        org = School(
            org_id=org_id,
            name=name,
            org_type=OrgType.SCHOOL,
            school_type=school_type,
            is_public=is_public,
            level=level,
            region=region,
            aliases=aliases or [],
            website=website,
            description=description,
        )
        self._add_organization(org)
    
    def add_hospital(
        self,
        org_id: str,
        name: str,
        hospital_type: HospitalType = HospitalType.OTHER,
        level: Optional[str] = None,
        is_public: bool = True,
        region: Optional[str] = None,
        aliases: Optional[List[str]] = None,
        website: Optional[str] = None,
        description: Optional[str] = None,
    ) -> None:
        org = Hospital(
            org_id=org_id,
            name=name,
            org_type=OrgType.HOSPITAL,
            hospital_type=hospital_type,
            level=level,
            is_public=is_public,
            region=region,
            aliases=aliases or [],
            website=website,
            description=description,
        )
        self._add_organization(org)
    
    def add_government(
        self,
        org_id: str,
        name: str,
        gov_type: GovType = GovType.OTHER,
        department: Optional[str] = None,
        region: Optional[str] = None,
        aliases: Optional[List[str]] = None,
        website: Optional[str] = None,
        description: Optional[str] = None,
    ) -> None:
        org = Government(
            org_id=org_id,
            name=name,
            org_type=OrgType.GOVERNMENT,
            gov_type=gov_type,
            department=department,
            region=region,
            aliases=aliases or [],
            website=website,
            description=description,
        )
        self._add_organization(org)
    
    def save_data(self, path: str) -> None:
        with open(path, 'w', encoding='utf-8') as f:
            f.write("# 机构名词库数据文件\n")
            f.write("# 格式说明:\n")
            f.write("# 机构ID\\t名称\\t类型\\t子类型\\t地区\\t别名\\t网址\\t描述\\t成立时间\\t地址\\t扩展字段\n")
            f.write("#\n")
            
            f.write("\n# @enterprises\n")
            for org_id in sorted(self._type_index[OrgType.ENTERPRISE]):
                org = self._organizations.get(org_id)
                if org and isinstance(org, Enterprise):
                    parts = [
                        org.org_id,
                        org.name,
                        "enterprise",
                        org.enterprise_type.value if org.enterprise_type else "",
                        org.region or "",
                        ','.join(org.aliases) if org.aliases else "",
                        org.website or "",
                        org.description or "",
                        org.founded or "",
                        org.address or "",
                        org.industry or "",
                    ]
                    f.write('\t'.join(parts) + '\n')
            
            f.write("\n# @schools\n")
            for org_id in sorted(self._type_index[OrgType.SCHOOL]):
                org = self._organizations.get(org_id)
                if org and isinstance(org, School):
                    parts = [
                        org.org_id,
                        org.name,
                        "school",
                        org.school_type.value if org.school_type else "",
                        org.region or "",
                        ','.join(org.aliases) if org.aliases else "",
                        org.website or "",
                        org.description or "",
                        org.founded or "",
                        org.address or "",
                        "public" if org.is_public else "private",
                        org.level or "",
                    ]
                    f.write('\t'.join(parts) + '\n')
            
            f.write("\n# @hospitals\n")
            for org_id in sorted(self._type_index[OrgType.HOSPITAL]):
                org = self._organizations.get(org_id)
                if org and isinstance(org, Hospital):
                    parts = [
                        org.org_id,
                        org.name,
                        "hospital",
                        org.hospital_type.value if org.hospital_type else "",
                        org.region or "",
                        ','.join(org.aliases) if org.aliases else "",
                        org.website or "",
                        org.description or "",
                        org.founded or "",
                        org.address or "",
                        org.level or "",
                        "public" if org.is_public else "private",
                    ]
                    f.write('\t'.join(parts) + '\n')
            
            f.write("\n# @governments\n")
            for org_id in sorted(self._type_index[OrgType.GOVERNMENT]):
                org = self._organizations.get(org_id)
                if org and isinstance(org, Government):
                    parts = [
                        org.org_id,
                        org.name,
                        "government",
                        org.gov_type.value if org.gov_type else "",
                        org.region or "",
                        ','.join(org.aliases) if org.aliases else "",
                        org.website or "",
                        org.description or "",
                        org.founded or "",
                        org.address or "",
                        org.department or "",
                    ]
                    f.write('\t'.join(parts) + '\n')
    
    def __len__(self) -> int:
        return self._org_count
    
    def __contains__(self, text: str) -> bool:
        return self.is_organization(text)
    
    def __getitem__(self, org_id: str) -> Optional[Organization]:
        return self.get_by_id(org_id)
    
    def __repr__(self) -> str:
        return (
            f"OrganizationDatabase(organizations={self._org_count}, "
            f"loaded={self._loaded})"
        )


class OrganizationManager:
    def __init__(self, load_default: bool = True):
        self._database: Optional[OrganizationDatabase] = None
        if load_default:
            self._database = OrganizationDatabase(load_default=True)
    
    def load(self, path: Optional[str] = None) -> None:
        if path:
            self._database = OrganizationDatabase(load_default=False)
            self._database.load_data(path)
        else:
            self._database = OrganizationDatabase(load_default=True)
    
    def get_database(self) -> Optional[OrganizationDatabase]:
        return self._database
    
    def get_by_id(self, org_id: str) -> Optional[Organization]:
        if self._database is None:
            return None
        return self._database.get_by_id(org_id)
    
    def get_by_name(self, name: str) -> List[Organization]:
        if self._database is None:
            return []
        return self._database.get_by_name(name)
    
    def get_by_alias(self, alias: str) -> Optional[Organization]:
        if self._database is None:
            return None
        return self._database.get_by_alias(alias)
    
    def search(self, query: str) -> List[Organization]:
        if self._database is None:
            return []
        return self._database.search(query)
    
    def get_by_type(self, org_type: OrgType) -> List[Organization]:
        if self._database is None:
            return []
        return self._database.get_by_type(org_type)
    
    def get_enterprises(self) -> List[Organization]:
        if self._database is None:
            return []
        return self._database.get_enterprises()
    
    def get_schools(self) -> List[Organization]:
        if self._database is None:
            return []
        return self._database.get_schools()
    
    def get_hospitals(self) -> List[Organization]:
        if self._database is None:
            return []
        return self._database.get_hospitals()
    
    def get_governments(self) -> List[Organization]:
        if self._database is None:
            return []
        return self._database.get_governments()
    
    def get_by_region(self, region: str) -> List[Organization]:
        if self._database is None:
            return []
        return self._database.get_by_region(region)
    
    def is_organization(self, text: str) -> bool:
        if self._database is None:
            return False
        return self._database.is_organization(text)
    
    def is_enterprise(self, text: str) -> bool:
        if self._database is None:
            return False
        return self._database.is_enterprise(text)
    
    def is_school(self, text: str) -> bool:
        if self._database is None:
            return False
        return self._database.is_school(text)
    
    def is_hospital(self, text: str) -> bool:
        if self._database is None:
            return False
        return self._database.is_hospital(text)
    
    def is_government(self, text: str) -> bool:
        if self._database is None:
            return False
        return self._database.is_government(text)
    
    def recognize_organizations(self, text: str) -> List[Tuple[Organization, int, int]]:
        if self._database is None:
            return []
        return self._database.recognize_organizations(text)
    
    def guess_org_type(self, text: str) -> Optional[OrgType]:
        if self._database is None:
            return None
        return self._database.guess_org_type(text)
    
    def get_statistics(self) -> Dict[str, Any]:
        if self._database is None:
            return {"loaded": False}
        return self._database.get_statistics()
    
    def is_loaded(self) -> bool:
        return self._database is not None and self._database.is_loaded()
