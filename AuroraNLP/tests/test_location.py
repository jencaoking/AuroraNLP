"""
地名数据库模块测试
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from AuroraNLP.location import (
    AdminLevel,
    Location,
    LocationDatabase,
    LocationManager
)


class TestAdminLevel(unittest.TestCase):
    
    def test_from_code_length(self):
        self.assertEqual(AdminLevel.from_code_length(2), AdminLevel.PROVINCE)
        self.assertEqual(AdminLevel.from_code_length(4), AdminLevel.CITY)
        self.assertEqual(AdminLevel.from_code_length(6), AdminLevel.COUNTY)
        self.assertEqual(AdminLevel.from_code_length(9), AdminLevel.TOWN)
        self.assertEqual(AdminLevel.from_code_length(12), AdminLevel.VILLAGE)
        self.assertIsNone(AdminLevel.from_code_length(5))
    
    def test_get_name(self):
        self.assertEqual(AdminLevel.PROVINCE.get_name(), "省级")
        self.assertEqual(AdminLevel.CITY.get_name(), "市级")
        self.assertEqual(AdminLevel.COUNTY.get_name(), "县级")
        self.assertEqual(AdminLevel.TOWN.get_name(), "乡镇级")
        self.assertEqual(AdminLevel.VILLAGE.get_name(), "村级")


class TestLocation(unittest.TestCase):
    
    def test_location_creation(self):
        loc = Location(
            code="11",
            name="北京市",
            level=AdminLevel.PROVINCE,
            latitude=39.9042,
            longitude=116.4074,
            aliases=["北京"]
        )
        
        self.assertEqual(loc.code, "11")
        self.assertEqual(loc.name, "北京市")
        self.assertEqual(loc.level, AdminLevel.PROVINCE)
        self.assertEqual(loc.latitude, 39.9042)
        self.assertEqual(loc.longitude, 116.4074)
        self.assertEqual(loc.aliases, ["北京"])
    
    def test_has_coordinates(self):
        loc1 = Location(code="11", name="北京市", level=AdminLevel.PROVINCE, latitude=39.9042, longitude=116.4074)
        loc2 = Location(code="12", name="天津市", level=AdminLevel.PROVINCE)
        
        self.assertTrue(loc1.has_coordinates())
        self.assertFalse(loc2.has_coordinates())
    
    def test_distance_to(self):
        beijing = Location(code="11", name="北京市", level=AdminLevel.PROVINCE, latitude=39.9042, longitude=116.4074)
        shanghai = Location(code="31", name="上海市", level=AdminLevel.PROVINCE, latitude=31.2304, longitude=121.4737)
        
        distance = beijing.distance_to(shanghai)
        self.assertIsNotNone(distance)
        self.assertGreater(distance, 1000)
        self.assertLess(distance, 1500)
    
    def test_distance_to_no_coordinates(self):
        loc1 = Location(code="11", name="北京市", level=AdminLevel.PROVINCE, latitude=39.9042, longitude=116.4074)
        loc2 = Location(code="12", name="天津市", level=AdminLevel.PROVINCE)
        
        self.assertIsNone(loc1.distance_to(loc2))
    
    def test_get_all_names(self):
        loc = Location(code="11", name="北京市", level=AdminLevel.PROVINCE, aliases=["北京", "京城"])
        
        all_names = loc.get_all_names()
        self.assertEqual(len(all_names), 3)
        self.assertIn("北京市", all_names)
        self.assertIn("北京", all_names)
        self.assertIn("京城", all_names)
    
    def test_matches(self):
        loc = Location(code="11", name="北京市", level=AdminLevel.PROVINCE, aliases=["北京"])
        
        self.assertTrue(loc.matches("北京市"))
        self.assertTrue(loc.matches("11"))
        self.assertTrue(loc.matches("北京"))
        self.assertFalse(loc.matches("上海"))


class TestLocationDatabase(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.db = LocationDatabase(load_default=True)
    
    def test_load_data(self):
        self.assertTrue(self.db.is_loaded())
        self.assertGreater(self.db.get_location_count(), 0)
    
    def test_get_by_code(self):
        loc = self.db.get_by_code("11")
        self.assertIsNotNone(loc)
        self.assertEqual(loc.name, "北京市")
    
    def test_get_by_name(self):
        locations = self.db.get_by_name("北京市")
        self.assertGreater(len(locations), 0)
        self.assertEqual(locations[0].code, "11")
    
    def test_get_by_alias(self):
        loc = self.db.get_by_alias("北京")
        self.assertIsNotNone(loc)
        self.assertEqual(loc.name, "北京市")
    
    def test_search(self):
        results = self.db.search("北京")
        self.assertGreater(len(results), 0)
        
        names = [r.name for r in results]
        self.assertIn("北京市", names)
    
    def test_get_by_level(self):
        provinces = self.db.get_by_level(AdminLevel.PROVINCE)
        self.assertGreater(len(provinces), 0)
        
        province_names = [p.name for p in provinces]
        self.assertIn("北京市", province_names)
        self.assertIn("上海市", province_names)
    
    def test_get_provinces(self):
        provinces = self.db.get_provinces()
        self.assertGreater(len(provinces), 0)
    
    def test_get_cities(self):
        cities = self.db.get_cities()
        self.assertGreater(len(cities), 0)
        
        beijing_cities = self.db.get_cities("11")
        self.assertGreater(len(beijing_cities), 0)
    
    def test_get_parent(self):
        parent = self.db.get_parent("1101")
        self.assertIsNotNone(parent)
        self.assertEqual(parent.code, "11")
    
    def test_get_children(self):
        children = self.db.get_children("11")
        self.assertGreater(len(children), 0)
        
        child_codes = [c.code for c in children]
        self.assertIn("1101", child_codes)
    
    def test_get_ancestors(self):
        ancestors = self.db.get_ancestors("110105")
        self.assertGreater(len(ancestors), 0)
        
        ancestor_codes = [a.code for a in ancestors]
        self.assertIn("11", ancestor_codes)
        self.assertIn("1101", ancestor_codes)
    
    def test_get_full_path(self):
        path = self.db.get_full_path("110105")
        self.assertGreater(len(path), 0)
        
        names = [p.name for p in path]
        self.assertIn("北京市", names)
    
    def test_get_full_name(self):
        full_name = self.db.get_full_name("110105")
        self.assertIn("北京", full_name)
    
    def test_is_location(self):
        self.assertTrue(self.db.is_location("北京市"))
        self.assertTrue(self.db.is_location("北京"))
        self.assertTrue(self.db.is_location("11"))
        self.assertFalse(self.db.is_location("不存在的地方"))
    
    def test_recognize_locations(self):
        text = "我从北京市出发，经过上海市，最后到达广州市。"
        results = self.db.recognize_locations(text)
        
        self.assertGreater(len(results), 0)
        
        location_names = [r[0].name for r in results]
        self.assertIn("北京市", location_names)
        self.assertIn("上海市", location_names)
        self.assertIn("广州市", location_names)
    
    def test_find_nearby(self):
        beijing_lat, beijing_lon = 39.9042, 116.4074
        
        nearby = self.db.find_nearby(beijing_lat, beijing_lon, radius_km=100)
        self.assertGreater(len(nearby), 0)
        
        for loc, dist in nearby:
            self.assertLessEqual(dist, 100)
    
    def test_get_statistics(self):
        stats = self.db.get_statistics()
        
        self.assertTrue(stats["loaded"])
        self.assertGreater(stats["total_count"], 0)
        self.assertGreater(stats["province_count"], 0)
    
    def test_add_location(self):
        test_code = "999999"
        self.db.add_location(
            code=test_code,
            name="测试地名",
            level=AdminLevel.COUNTY,
            latitude=30.0,
            longitude=120.0,
            aliases=["测试"]
        )
        
        loc = self.db.get_by_code(test_code)
        self.assertIsNotNone(loc)
        self.assertEqual(loc.name, "测试地名")
    
    def test_contains(self):
        self.assertIn("北京市", self.db)
        self.assertIn("北京", self.db)
        self.assertNotIn("不存在的地方", self.db)
    
    def test_getitem(self):
        loc = self.db["11"]
        self.assertIsNotNone(loc)
        self.assertEqual(loc.name, "北京市")


class TestLocationManager(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.manager = LocationManager(load_default=True)
    
    def test_get_database(self):
        db = self.manager.get_database()
        self.assertIsNotNone(db)
    
    def test_get_by_code(self):
        loc = self.manager.get_by_code("11")
        self.assertIsNotNone(loc)
        self.assertEqual(loc.name, "北京市")
    
    def test_get_by_name(self):
        locations = self.manager.get_by_name("上海市")
        self.assertGreater(len(locations), 0)
    
    def test_search(self):
        results = self.manager.search("杭州")
        self.assertGreater(len(results), 0)
    
    def test_get_provinces(self):
        provinces = self.manager.get_provinces()
        self.assertGreater(len(provinces), 0)
    
    def test_get_parent(self):
        parent = self.manager.get_parent("3201")
        self.assertIsNotNone(parent)
        self.assertEqual(parent.code, "32")
    
    def test_get_children(self):
        children = self.manager.get_children("32")
        self.assertGreater(len(children), 0)
    
    def test_get_full_path(self):
        path = self.manager.get_full_path("330102")
        self.assertGreater(len(path), 0)
    
    def test_is_location(self):
        self.assertTrue(self.manager.is_location("北京市"))
        self.assertFalse(self.manager.is_location("不存在的地方"))
    
    def test_recognize_locations(self):
        text = "我在南京市玄武区工作。"
        results = self.manager.recognize_locations(text)
        self.assertGreater(len(results), 0)
    
    def test_calculate_distance(self):
        distance = self.manager.calculate_distance("11", "31")
        self.assertIsNotNone(distance)
        self.assertGreater(distance, 1000)
    
    def test_get_statistics(self):
        stats = self.manager.get_statistics()
        self.assertTrue(stats["loaded"])


if __name__ == "__main__":
    unittest.main()
