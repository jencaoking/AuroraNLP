"""
地名数据库模块 - 地名数据库构建

提供中国五级行政区划（省/市/县/镇/村）地名查询、经纬度信息和别名处理功能。

功能：
- 五级行政区划：省、市、县、镇、村
- 经纬度信息：支持经纬度查询和距离计算
- 别名处理：支持地名别名和简称
- 行政区划代码：支持行政区划代码查询
- 地名识别：从文本中识别地名
- 地理查询：按层级、区域查询地名

数据格式说明：
- 地名格式：行政区划代码\t名称\t等级\t纬度\t经度\t别名(逗号分隔)\t上级代码
"""

import os
import re
from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import math


class AdminLevel(Enum):
    PROVINCE = 1
    CITY = 2
    COUNTY = 3
    TOWN = 4
    VILLAGE = 5
    
    @classmethod
    def from_code_length(cls, code_length: int) -> Optional['AdminLevel']:
        mapping = {
            2: cls.PROVINCE,
            4: cls.CITY,
            6: cls.COUNTY,
            9: cls.TOWN,
            12: cls.VILLAGE,
        }
        return mapping.get(code_length)
    
    def get_name(self) -> str:
        names = {
            self.PROVINCE: "省级",
            self.CITY: "市级",
            self.COUNTY: "县级",
            self.TOWN: "乡镇级",
            self.VILLAGE: "村级",
        }
        return names.get(self, "未知")


@dataclass
class Location:
    code: str
    name: str
    level: AdminLevel
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    aliases: List[str] = field(default_factory=list)
    parent_code: Optional[str] = None
    pinyin: Optional[str] = None
    area_code: Optional[str] = None
    zip_code: Optional[str] = None
    
    def __str__(self) -> str:
        return self.name
    
    def __repr__(self) -> str:
        return (
            f"Location(code='{self.code}', name='{self.name}', "
            f"level={self.level.get_name()}, lat={self.latitude}, lon={self.longitude})"
        )
    
    def has_coordinates(self) -> bool:
        return self.latitude is not None and self.longitude is not None
    
    def distance_to(self, other: 'Location') -> Optional[float]:
        if not self.has_coordinates() or not other.has_coordinates():
            return None
        
        lat1, lon1 = math.radians(self.latitude), math.radians(self.longitude)
        lat2, lon2 = math.radians(other.latitude), math.radians(other.longitude)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.asin(math.sqrt(a))
        
        r = 6371.0
        return c * r
    
    def get_all_names(self) -> List[str]:
        names = [self.name]
        names.extend(self.aliases)
        return names
    
    def matches(self, text: str) -> bool:
        if text == self.name:
            return True
        if text == self.code:
            return True
        if text in self.aliases:
            return True
        return False


class LocationDatabase:
    DEFAULT_DATA_PATH = os.path.join(
        os.path.dirname(__file__), 'data', 'locations.txt'
    )
    
    def __init__(self, load_default: bool = True):
        self._locations: Dict[str, Location] = {}
        self._name_index: Dict[str, Set[str]] = {}
        self._alias_index: Dict[str, str] = {}
        self._level_index: Dict[AdminLevel, Set[str]] = {
            AdminLevel.PROVINCE: set(),
            AdminLevel.CITY: set(),
            AdminLevel.COUNTY: set(),
            AdminLevel.TOWN: set(),
            AdminLevel.VILLAGE: set(),
        }
        self._parent_index: Dict[str, Set[str]] = {}
        self._loaded: bool = False
        self._location_count: int = 0
        
        if load_default:
            self._load_default_data()
    
    def _load_default_data(self) -> None:
        if os.path.exists(self.DEFAULT_DATA_PATH):
            self.load_data(self.DEFAULT_DATA_PATH)
    
    def load_data(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"地名数据文件不存在: {path}")
        
        self._locations.clear()
        self._name_index.clear()
        self._alias_index.clear()
        for level in self._level_index:
            self._level_index[level] = set()
        self._parent_index.clear()
        self._location_count = 0
        
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                self._parse_location(line)
        
        self._loaded = True
    
    def _parse_location(self, line: str) -> None:
        parts = line.split('\t')
        if len(parts) < 3:
            return
        
        code = parts[0].strip()
        name = parts[1].strip()
        
        try:
            level_value = int(parts[2].strip())
            level = AdminLevel(level_value)
        except (ValueError, KeyError):
            return
        
        latitude = None
        longitude = None
        aliases = []
        parent_code = None
        pinyin = None
        area_code = None
        zip_code = None
        
        if len(parts) > 3 and parts[3].strip():
            try:
                latitude = float(parts[3].strip())
            except ValueError:
                pass
        
        if len(parts) > 4 and parts[4].strip():
            try:
                longitude = float(parts[4].strip())
            except ValueError:
                pass
        
        if len(parts) > 5 and parts[5].strip():
            aliases = [a.strip() for a in parts[5].split(',') if a.strip()]
        
        if len(parts) > 6 and parts[6].strip():
            parent_code = parts[6].strip()
        
        if len(parts) > 7 and parts[7].strip():
            pinyin = parts[7].strip()
        
        if len(parts) > 8 and parts[8].strip():
            area_code = parts[8].strip()
        
        if len(parts) > 9 and parts[9].strip():
            zip_code = parts[9].strip()
        
        location = Location(
            code=code,
            name=name,
            level=level,
            latitude=latitude,
            longitude=longitude,
            aliases=aliases,
            parent_code=parent_code,
            pinyin=pinyin,
            area_code=area_code,
            zip_code=zip_code
        )
        
        self._add_location(location)
    
    def _add_location(self, location: Location) -> None:
        self._locations[location.code] = location
        self._location_count += 1
        
        if location.name not in self._name_index:
            self._name_index[location.name] = set()
        self._name_index[location.name].add(location.code)
        
        for alias in location.aliases:
            self._alias_index[alias] = location.code
        
        self._level_index[location.level].add(location.code)
        
        if location.parent_code:
            if location.parent_code not in self._parent_index:
                self._parent_index[location.parent_code] = set()
            self._parent_index[location.parent_code].add(location.code)
    
    def get_by_code(self, code: str) -> Optional[Location]:
        return self._locations.get(code)
    
    def get_by_name(self, name: str) -> List[Location]:
        codes = self._name_index.get(name, set())
        return [self._locations[code] for code in codes if code in self._locations]
    
    def get_by_alias(self, alias: str) -> Optional[Location]:
        code = self._alias_index.get(alias)
        if code:
            return self._locations.get(code)
        return None
    
    def search(self, query: str) -> List[Location]:
        results: List[Location] = []
        
        if query in self._locations:
            results.append(self._locations[query])
        
        results.extend(self.get_by_name(query))
        
        location = self.get_by_alias(query)
        if location and location not in results:
            results.append(location)
        
        for name, codes in self._name_index.items():
            if query in name and name != query:
                for code in codes:
                    loc = self._locations.get(code)
                    if loc and loc not in results:
                        results.append(loc)
        
        return results
    
    def get_by_level(self, level: AdminLevel) -> List[Location]:
        codes = self._level_index.get(level, set())
        return [self._locations[code] for code in codes if code in self._locations]
    
    def get_provinces(self) -> List[Location]:
        return self.get_by_level(AdminLevel.PROVINCE)
    
    def get_cities(self, province_code: Optional[str] = None) -> List[Location]:
        if province_code:
            return self.get_children(province_code)
        return self.get_by_level(AdminLevel.CITY)
    
    def get_counties(self, city_code: Optional[str] = None) -> List[Location]:
        if city_code:
            return self.get_children(city_code)
        return self.get_by_level(AdminLevel.COUNTY)
    
    def get_towns(self, county_code: Optional[str] = None) -> List[Location]:
        if county_code:
            return self.get_children(county_code)
        return self.get_by_level(AdminLevel.TOWN)
    
    def get_villages(self, town_code: Optional[str] = None) -> List[Location]:
        if town_code:
            return self.get_children(town_code)
        return self.get_by_level(AdminLevel.VILLAGE)
    
    def get_parent(self, code: str) -> Optional[Location]:
        location = self._locations.get(code)
        if location and location.parent_code:
            return self._locations.get(location.parent_code)
        return None
    
    def get_children(self, code: str) -> List[Location]:
        codes = self._parent_index.get(code, set())
        return [self._locations[c] for c in codes if c in self._locations]
    
    def get_ancestors(self, code: str) -> List[Location]:
        ancestors: List[Location] = []
        current = self._locations.get(code)
        
        while current and current.parent_code:
            parent = self._locations.get(current.parent_code)
            if parent:
                ancestors.append(parent)
                current = parent
            else:
                break
        
        return ancestors
    
    def get_descendants(self, code: str) -> List[Location]:
        descendants: List[Location] = []
        children = self.get_children(code)
        
        for child in children:
            descendants.append(child)
            descendants.extend(self.get_descendants(child.code))
        
        return descendants
    
    def get_full_path(self, code: str) -> List[Location]:
        path = self.get_ancestors(code)
        path.reverse()
        location = self._locations.get(code)
        if location:
            path.append(location)
        return path
    
    def get_full_name(self, code: str, separator: str = "") -> str:
        path = self.get_full_path(code)
        return separator.join(loc.name for loc in path)
    
    def is_location(self, text: str) -> bool:
        if text in self._locations:
            return True
        if text in self._name_index:
            return True
        if text in self._alias_index:
            return True
        return False
    
    def recognize_locations(self, text: str) -> List[Tuple[Location, int, int]]:
        results: List[Tuple[Location, int, int]] = []
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
                        location = self._locations.get(code)
                    else:
                        codes = self._name_index.get(name, set())
                        location = self._locations.get(next(iter(codes), ''))
                    
                    if location:
                        results.append((location, pos, end))
                        for i in range(pos, end):
                            used_positions.add(i)
                
                start = pos + 1
        
        results.sort(key=lambda x: x[1])
        return results
    
    def find_nearby(
        self,
        latitude: float,
        longitude: float,
        radius_km: float = 50.0,
        level: Optional[AdminLevel] = None
    ) -> List[Tuple[Location, float]]:
        results: List[Tuple[Location, float]] = []
        
        locations = self.get_by_level(level) if level else list(self._locations.values())
        
        for location in locations:
            if location.has_coordinates():
                temp_loc = Location(
                    code="temp",
                    name="temp",
                    level=AdminLevel.PROVINCE,
                    latitude=latitude,
                    longitude=longitude
                )
                distance = location.distance_to(temp_loc)
                if distance is not None and distance <= radius_km:
                    results.append((location, distance))
        
        results.sort(key=lambda x: x[1])
        return results
    
    def find_in_bounds(
        self,
        min_lat: float,
        max_lat: float,
        min_lon: float,
        max_lon: float,
        level: Optional[AdminLevel] = None
    ) -> List[Location]:
        results: List[Location] = []
        
        locations = self.get_by_level(level) if level else list(self._locations.values())
        
        for location in locations:
            if location.has_coordinates():
                if (min_lat <= location.latitude <= max_lat and
                    min_lon <= location.longitude <= max_lon):
                    results.append(location)
        
        return results
    
    def get_all_locations(self) -> List[Location]:
        return list(self._locations.values())
    
    def get_location_count(self) -> int:
        return self._location_count
    
    def is_loaded(self) -> bool:
        return self._loaded
    
    def get_statistics(self) -> Dict[str, Any]:
        return {
            "loaded": self._loaded,
            "total_count": self._location_count,
            "province_count": len(self._level_index[AdminLevel.PROVINCE]),
            "city_count": len(self._level_index[AdminLevel.CITY]),
            "county_count": len(self._level_index[AdminLevel.COUNTY]),
            "town_count": len(self._level_index[AdminLevel.TOWN]),
            "village_count": len(self._level_index[AdminLevel.VILLAGE]),
            "alias_count": len(self._alias_index),
        }
    
    def add_location(
        self,
        code: str,
        name: str,
        level: AdminLevel,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        aliases: Optional[List[str]] = None,
        parent_code: Optional[str] = None,
        pinyin: Optional[str] = None,
        area_code: Optional[str] = None,
        zip_code: Optional[str] = None
    ) -> None:
        location = Location(
            code=code,
            name=name,
            level=level,
            latitude=latitude,
            longitude=longitude,
            aliases=aliases or [],
            parent_code=parent_code,
            pinyin=pinyin,
            area_code=area_code,
            zip_code=zip_code
        )
        self._add_location(location)
    
    def save_data(self, path: str) -> None:
        with open(path, 'w', encoding='utf-8') as f:
            f.write("# 地名数据库文件\n")
            f.write("# 格式: 代码\\t名称\\t等级\\t纬度\\t经度\\t别名\\t上级代码\\t拼音\\t区号\\t邮编\n")
            f.write("# 等级: 1省 2市 3县 4镇 5村\n")
            f.write("#\n")
            
            for level in AdminLevel:
                for code in sorted(self._level_index[level]):
                    location = self._locations.get(code)
                    if location:
                        parts = [
                            location.code,
                            location.name,
                            str(location.level.value),
                            str(location.latitude) if location.latitude else "",
                            str(location.longitude) if location.longitude else "",
                            ','.join(location.aliases) if location.aliases else "",
                            location.parent_code or "",
                            location.pinyin or "",
                            location.area_code or "",
                            location.zip_code or ""
                        ]
                        f.write('\t'.join(parts) + '\n')
    
    def __len__(self) -> int:
        return self._location_count
    
    def __contains__(self, text: str) -> bool:
        return self.is_location(text)
    
    def __getitem__(self, code: str) -> Optional[Location]:
        return self.get_by_code(code)
    
    def __repr__(self) -> str:
        return (
            f"LocationDatabase(locations={self._location_count}, "
            f"loaded={self._loaded})"
        )


class LocationManager:
    def __init__(self, load_default: bool = True):
        self._database: Optional[LocationDatabase] = None
        if load_default:
            self._database = LocationDatabase(load_default=True)
    
    def load(self, path: Optional[str] = None) -> None:
        if path:
            self._database = LocationDatabase(load_default=False)
            self._database.load_data(path)
        else:
            self._database = LocationDatabase(load_default=True)
    
    def get_database(self) -> Optional[LocationDatabase]:
        return self._database
    
    def get_by_code(self, code: str) -> Optional[Location]:
        if self._database is None:
            return None
        return self._database.get_by_code(code)
    
    def get_by_name(self, name: str) -> List[Location]:
        if self._database is None:
            return []
        return self._database.get_by_name(name)
    
    def get_by_alias(self, alias: str) -> Optional[Location]:
        if self._database is None:
            return None
        return self._database.get_by_alias(alias)
    
    def search(self, query: str) -> List[Location]:
        if self._database is None:
            return []
        return self._database.search(query)
    
    def get_provinces(self) -> List[Location]:
        if self._database is None:
            return []
        return self._database.get_provinces()
    
    def get_cities(self, province_code: Optional[str] = None) -> List[Location]:
        if self._database is None:
            return []
        return self._database.get_cities(province_code)
    
    def get_counties(self, city_code: Optional[str] = None) -> List[Location]:
        if self._database is None:
            return []
        return self._database.get_counties(city_code)
    
    def get_towns(self, county_code: Optional[str] = None) -> List[Location]:
        if self._database is None:
            return []
        return self._database.get_towns(county_code)
    
    def get_villages(self, town_code: Optional[str] = None) -> List[Location]:
        if self._database is None:
            return []
        return self._database.get_villages(town_code)
    
    def get_parent(self, code: str) -> Optional[Location]:
        if self._database is None:
            return None
        return self._database.get_parent(code)
    
    def get_children(self, code: str) -> List[Location]:
        if self._database is None:
            return []
        return self._database.get_children(code)
    
    def get_ancestors(self, code: str) -> List[Location]:
        if self._database is None:
            return []
        return self._database.get_ancestors(code)
    
    def get_full_path(self, code: str) -> List[Location]:
        if self._database is None:
            return []
        return self._database.get_full_path(code)
    
    def get_full_name(self, code: str, separator: str = "") -> str:
        if self._database is None:
            return ""
        return self._database.get_full_name(code, separator)
    
    def is_location(self, text: str) -> bool:
        if self._database is None:
            return False
        return self._database.is_location(text)
    
    def recognize_locations(self, text: str) -> List[Tuple[Location, int, int]]:
        if self._database is None:
            return []
        return self._database.recognize_locations(text)
    
    def find_nearby(
        self,
        latitude: float,
        longitude: float,
        radius_km: float = 50.0,
        level: Optional[AdminLevel] = None
    ) -> List[Tuple[Location, float]]:
        if self._database is None:
            return []
        return self._database.find_nearby(latitude, longitude, radius_km, level)
    
    def calculate_distance(self, code1: str, code2: str) -> Optional[float]:
        if self._database is None:
            return None
        loc1 = self._database.get_by_code(code1)
        loc2 = self._database.get_by_code(code2)
        if loc1 and loc2:
            return loc1.distance_to(loc2)
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        if self._database is None:
            return {"loaded": False}
        return self._database.get_statistics()
    
    def is_loaded(self) -> bool:
        return self._database is not None and self._database.is_loaded()
