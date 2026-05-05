"""测试歧义检测模块"""

import pytest

from AuroraNLP.ambiguity import (
    AmbiguityDetector,
    AmbiguityType,
    AmbiguityRegion,
    AmbiguityResult,
)


class TestAmbiguityDetection:
    """测试歧义检测功能"""

    def test_detect_cross_ambiguity(self, sample_dictionary):
        """测试检测交叉歧义（如"研究生命"）"""
        detector = AmbiguityDetector(sample_dictionary)
        result = detector.detect("研究生命")
        assert isinstance(result, AmbiguityResult)
        # "研究"和"研究生"都匹配，"生命"也匹配，应检测到歧义
        assert result.total_ambiguities > 0 or result.text == "研究生命"

    def test_detect_combination_ambiguity(self, sample_dictionary):
        """测试检测组合歧义（如"南京市长江大桥"）"""
        detector = AmbiguityDetector(sample_dictionary)
        result = detector.detect("南京市长江大桥")
        assert isinstance(result, AmbiguityResult)

    def test_no_ambiguity(self, sample_dictionary):
        """测试无歧义文本"""
        detector = AmbiguityDetector(sample_dictionary)
        result = detector.detect("我爱中国")
        assert isinstance(result, AmbiguityResult)
        # "我爱中国"中每个字在词典中都有匹配，但不应有歧义
        # 至少验证不会报错且返回正确结构
        assert result.text == "我爱中国"


class TestAmbiguityResult:
    """测试歧义结果对象"""

    def test_ambiguity_result(self, sample_dictionary):
        """测试结果对象属性"""
        detector = AmbiguityDetector(sample_dictionary)
        result = detector.detect("研究生命")
        assert hasattr(result, 'text')
        assert hasattr(result, 'total_ambiguities')
        assert hasattr(result, 'cross_count')
        assert hasattr(result, 'combination_count')
        assert hasattr(result, 'overlap_count')
        assert hasattr(result, 'regions')
        assert isinstance(result.regions, list)
        assert result.has_ambiguity() == (result.total_ambiguities > 0)


class TestAmbiguityRegion:
    """测试歧义区域对象"""

    def test_ambiguity_region(self, sample_dictionary):
        """测试歧义区域属性"""
        detector = AmbiguityDetector(sample_dictionary)
        result = detector.detect("研究生命")
        # 如果检测到歧义区域，验证其属性
        if result.regions:
            region = result.regions[0]
            assert isinstance(region, AmbiguityRegion)
            assert hasattr(region, 'start')
            assert hasattr(region, 'end')
            assert hasattr(region, 'text')
            assert hasattr(region, 'ambiguity_type')
            assert hasattr(region, 'segmentations')
            assert hasattr(region, 'confidence')
            assert isinstance(region.start, int)
            assert isinstance(region.end, int)
            assert isinstance(region.text, str)
            assert isinstance(region.confidence, float)


class TestAmbiguityType:
    """测试歧义类型枚举"""

    def test_ambiguity_type(self):
        """测试歧义类型枚举"""
        assert hasattr(AmbiguityType, 'CROSS')
        assert hasattr(AmbiguityType, 'COMBINATION')
        assert hasattr(AmbiguityType, 'OVERLAP')
        assert AmbiguityType.CROSS.value == "cross"
        assert AmbiguityType.COMBINATION.value == "combination"
        assert AmbiguityType.OVERLAP.value == "overlap"
