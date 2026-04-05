import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from AuroraNLP.ambiguity import (
    AmbiguityType, AmbiguityRegion, AmbiguityResult, AmbiguityDetector
)
from AuroraNLP.dictionary import Dictionary
from AuroraNLP.lattice import Lattice, LatticeBuilder
from AuroraNLP import Segmentor


class TestAmbiguityType(unittest.TestCase):
    def test_enum_values(self):
        self.assertEqual(AmbiguityType.CROSS.value, "cross")
        self.assertEqual(AmbiguityType.COMBINATION.value, "combination")
        self.assertEqual(AmbiguityType.OVERLAP.value, "overlap")


class TestAmbiguityRegion(unittest.TestCase):
    def test_region_creation(self):
        region = AmbiguityRegion(
            start=0,
            end=3,
            text="研究生",
            ambiguity_type=AmbiguityType.CROSS,
            segmentations=[["研究生"], ["研究", "生"]],
            edges=[(0, 3, "研究生"), (0, 2, "研究")],
            confidence=0.5
        )
        
        self.assertEqual(region.start, 0)
        self.assertEqual(region.end, 3)
        self.assertEqual(region.text, "研究生")
        self.assertEqual(region.ambiguity_type, AmbiguityType.CROSS)
        self.assertEqual(len(region.segmentations), 2)
    
    def test_region_to_dict(self):
        region = AmbiguityRegion(
            start=0,
            end=3,
            text="研究生",
            ambiguity_type=AmbiguityType.CROSS,
            segmentations=[["研究生"]],
            confidence=0.8
        )
        
        result = region.to_dict()
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result['start'], 0)
        self.assertEqual(result['end'], 3)
        self.assertEqual(result['text'], "研究生")
        self.assertEqual(result['type'], "cross")


class TestAmbiguityResult(unittest.TestCase):
    def test_result_creation(self):
        result = AmbiguityResult(
            text="研究生",
            total_ambiguities=1,
            cross_count=1,
            combination_count=0,
            overlap_count=0
        )
        
        self.assertEqual(result.text, "研究生")
        self.assertEqual(result.total_ambiguities, 1)
        self.assertTrue(result.has_ambiguity())
    
    def test_result_no_ambiguity(self):
        result = AmbiguityResult(
            text="测试",
            total_ambiguities=0,
            cross_count=0,
            combination_count=0,
            overlap_count=0
        )
        
        self.assertFalse(result.has_ambiguity())
    
    def test_get_regions_by_type(self):
        region1 = AmbiguityRegion(
            start=0, end=3, text="研究生",
            ambiguity_type=AmbiguityType.CROSS
        )
        region2 = AmbiguityRegion(
            start=5, end=8, text="南京市",
            ambiguity_type=AmbiguityType.COMBINATION
        )
        
        result = AmbiguityResult(
            text="研究生南京市",
            total_ambiguities=2,
            cross_count=1,
            combination_count=1,
            overlap_count=0,
            regions=[region1, region2]
        )
        
        cross_regions = result.get_regions_by_type(AmbiguityType.CROSS)
        self.assertEqual(len(cross_regions), 1)
        self.assertEqual(cross_regions[0].text, "研究生")
        
        comb_regions = result.get_regions_by_type(AmbiguityType.COMBINATION)
        self.assertEqual(len(comb_regions), 1)
    
    def test_result_to_dict(self):
        result = AmbiguityResult(
            text="研究生",
            total_ambiguities=1,
            cross_count=1,
            combination_count=0,
            overlap_count=0
        )
        
        result_dict = result.to_dict()
        
        self.assertIsInstance(result_dict, dict)
        self.assertEqual(result_dict['text'], "研究生")
        self.assertEqual(result_dict['total_ambiguities'], 1)


class TestAmbiguityDetector(unittest.TestCase):
    def setUp(self):
        self.dict = Dictionary(load_default=False)
        self.dict.add_word("研究", "v")
        self.dict.add_word("研究生", "n")
        self.dict.add_word("生命", "n")
        self.dict.add_word("的", "u")
        self.dict.add_word("起源", "n")
        self.dict.add_word("南京市", "ns")
        self.dict.add_word("南京", "ns")
        self.dict.add_word("市长", "n")
        self.dict.add_word("长江", "ns")
        self.dict.add_word("大桥", "n")
        self.dict.add_word("江大桥", "nr")
        self.detector = AmbiguityDetector(self.dict)
    
    def test_detector_creation(self):
        self.assertIsNotNone(self.detector)
        self.assertEqual(self.detector.max_word_len, 15)
    
    def test_detect_cross_ambiguity(self):
        result = self.detector.detect("研究生")
        
        self.assertTrue(result.has_ambiguity())
        self.assertGreater(result.cross_count, 0)
        
        cross_regions = result.get_regions_by_type(AmbiguityType.CROSS)
        self.assertGreater(len(cross_regions), 0)
    
    def test_detect_no_ambiguity(self):
        self.dict.add_word("测试", "v")
        result = self.detector.detect("测试")
        
        self.assertFalse(result.has_ambiguity())
    
    def test_detect_with_multiple_ambiguities(self):
        result = self.detector.detect("研究生命的起源")
        
        self.assertTrue(result.has_ambiguity())
    
    def test_get_ambiguity_statistics(self):
        stats = self.detector.get_ambiguity_statistics("研究生")
        
        self.assertIn('text_length', stats)
        self.assertIn('total_ambiguities', stats)
        self.assertIn('ambiguity_density', stats)
        self.assertIn('by_type', stats)
        self.assertIn('avg_confidence', stats)
    
    def test_detect_from_lattice(self):
        builder = LatticeBuilder(self.dict)
        lattice = builder.build("研究生")
        
        result = self.detector.detect_from_lattice(lattice)
        
        self.assertTrue(result.has_ambiguity())


class TestAmbiguityDetectorClassicCases(unittest.TestCase):
    def setUp(self):
        self.dict = Dictionary(load_default=False)
        self.dict.add_word("南京市", "ns")
        self.dict.add_word("南京", "ns")
        self.dict.add_word("市长", "n")
        self.dict.add_word("长江", "ns")
        self.dict.add_word("大桥", "n")
        self.dict.add_word("江大桥", "nr")
        self.dict.add_word("研究", "v")
        self.dict.add_word("研究生", "n")
        self.dict.add_word("生活", "n")
        self.dict.add_word("生", "n")
        self.dict.add_word("命", "n")
        self.dict.add_word("生命", "n")
        self.dict.add_word("的", "u")
        self.dict.add_word("起源", "n")
        self.dict.add_word("和", "c")
        self.dict.add_word("发展", "v")
        self.dict.add_word("结婚", "v")
        self.dict.add_word("和", "c")
        self.dict.add_word("尚未", "d")
        self.dict.add_word("结婚", "v")
        self.dict.add_word("的", "u")
        self.dict.add_word("青年", "n")
        self.detector = AmbiguityDetector(self.dict)
    
    def test_nanjin_changjiang_bridge(self):
        result = self.detector.detect("南京市长江大桥")
        
        self.assertTrue(result.has_ambiguity())
    
    def test_yanjiusheng(self):
        result = self.detector.detect("研究生")
        
        self.assertTrue(result.has_ambiguity())
        
        cross_regions = result.get_regions_by_type(AmbiguityType.CROSS)
        self.assertGreater(len(cross_regions), 0)
        
        found_yanjiusheng = False
        found_yanjiu_sheng = False
        for region in cross_regions:
            for seg in region.segmentations:
                if seg == ["研究生"]:
                    found_yanjiusheng = True
                if seg == ["研究", "生"]:
                    found_yanjiu_sheng = True
        
        self.assertTrue(found_yanjiusheng or found_yanjiu_sheng)
    
    def test_shengming_qiyuan(self):
        result = self.detector.detect("研究生命的起源")
        
        self.assertTrue(result.has_ambiguity())


class TestSegmentorAmbiguityIntegration(unittest.TestCase):
    def setUp(self):
        self.segmentor = Segmentor(load_default_dict=False)
        self.segmentor.add_word("研究", "v")
        self.segmentor.add_word("研究生", "n")
        self.segmentor.add_word("生命", "n")
        self.segmentor.add_word("的", "u")
        self.segmentor.add_word("起源", "n")
        self.segmentor.add_word("南京市", "ns")
        self.segmentor.add_word("南京", "ns")
        self.segmentor.add_word("市长", "n")
        self.segmentor.add_word("长江", "ns")
        self.segmentor.add_word("大桥", "n")
    
    def test_detect_ambiguity(self):
        result = self.segmentor.detect_ambiguity("研究生")
        
        self.assertIsInstance(result, AmbiguityResult)
        self.assertTrue(result.has_ambiguity())
    
    def test_detect_ambiguity_from_lattice(self):
        result = self.segmentor.detect_ambiguity_from_lattice("研究生")
        
        self.assertIsInstance(result, AmbiguityResult)
        self.assertTrue(result.has_ambiguity())
    
    def test_has_ambiguity(self):
        self.assertTrue(self.segmentor.has_ambiguity("研究生"))
    
    def test_get_cross_ambiguities(self):
        regions = self.segmentor.get_cross_ambiguities("研究生")
        
        self.assertIsInstance(regions, list)
        self.assertGreater(len(regions), 0)
        for region in regions:
            self.assertEqual(region.ambiguity_type, AmbiguityType.CROSS)
    
    def test_get_combination_ambiguities(self):
        regions = self.segmentor.get_combination_ambiguities("研究生")
        
        self.assertIsInstance(regions, list)
        for region in regions:
            self.assertEqual(region.ambiguity_type, AmbiguityType.COMBINATION)
    
    def test_get_ambiguity_statistics(self):
        stats = self.segmentor.get_ambiguity_statistics("研究生")
        
        self.assertIsInstance(stats, dict)
        self.assertIn('total_ambiguities', stats)
        self.assertIn('by_type', stats)
    
    def test_resolve_ambiguity(self):
        result = self.segmentor.resolve_ambiguity("研究生", method='shortest')
        
        self.assertIsInstance(result, list)
        self.assertEqual("".join(result), "研究生")


class TestAmbiguityEdgeCases(unittest.TestCase):
    def setUp(self):
        self.dict = Dictionary(load_default=False)
        self.detector = AmbiguityDetector(self.dict)
    
    def test_empty_text(self):
        result = self.detector.detect("")
        
        self.assertFalse(result.has_ambiguity())
        self.assertEqual(result.total_ambiguities, 0)
    
    def test_single_char(self):
        result = self.detector.detect("测")
        
        self.assertFalse(result.has_ambiguity())
    
    def test_no_dictionary_matches(self):
        result = self.detector.detect("xyzabc")
        
        self.assertFalse(result.has_ambiguity())
    
    def test_single_word_no_ambiguity(self):
        self.dict.add_word("测试", "v")
        result = self.detector.detect("测试")
        
        self.assertFalse(result.has_ambiguity())


if __name__ == '__main__':
    unittest.main()
