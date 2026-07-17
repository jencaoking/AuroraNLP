"""TraditionalChinese 繁体中文测试"""
import pytest
from AuroraNLP.traditional_chinese import TraditionalChineseConverter, TraditionalChineseDictionary


class TestTraditionalChineseConverter:
    """测试 TraditionalChineseConverter"""

    def test_converter_init(self):
        """转换器初始化"""
        converter = TraditionalChineseConverter()
        assert converter is not None

    def test_converter_init_no_default(self):
        """不加载默认映射的初始化"""
        converter = TraditionalChineseConverter(load_default=False)
        assert converter is not None

    def test_converter_simplified_to_traditional(self):
        """简转繁"""
        converter = TraditionalChineseConverter()
        result = converter.simplified_to_traditional("中国")
        assert isinstance(result, str)
        # "中" 无对应繁体，"国" -> "國"
        assert "國" in result

    def test_converter_traditional_to_simplified(self):
        """繁转简"""
        converter = TraditionalChineseConverter()
        result = converter.traditional_to_simplified("國")
        assert isinstance(result, str)
        assert result == "国"

    def test_converter_detect_variant(self):
        """检测变体"""
        converter = TraditionalChineseConverter()
        # 台湾标记词
        result_tw = converter.detect_language_variant("臺灣總統在臺北")
        assert result_tw == "tw"
        # 香港标记词
        result_hk = converter.detect_language_variant("香港特首在九龍")
        assert result_hk == "hk"
        # 无标记词
        result_none = converter.detect_language_variant("今天天气很好")
        assert result_none is None

    def test_converter_get_traditional_chars(self):
        """获取繁体字符集"""
        converter = TraditionalChineseConverter()
        chars = converter.get_traditional_chars()
        assert isinstance(chars, set)
        assert len(chars) > 0
        # 确认一些已知的繁体字存在
        assert "國" in chars
        assert "愛" in chars


class TestTraditionalChineseDictionary:
    """测试 TraditionalChineseDictionary"""

    def test_traditional_dict_init(self):
        """繁体词典初始化"""
        td = TraditionalChineseDictionary()
        assert td is not None

    def test_traditional_dict_init_with_converter(self):
        """带转换器的繁体词典初始化"""
        converter = TraditionalChineseConverter()
        td = TraditionalChineseDictionary(converter=converter)
        assert td is not None

    def test_traditional_dict_add_word(self):
        """添加词"""
        td = TraditionalChineseDictionary()
        td.add_word("人工智能")
        assert td.contains("人工智能")

    def test_traditional_dict_add_word_with_traditional(self):
        """添加词（指定繁体形式）"""
        td = TraditionalChineseDictionary()
        td.add_word("人工智能", traditional="人工智慧")
        assert td.contains("人工智慧")

    def test_traditional_dict_load_from_file(self, tmp_path):
        """从文件加载"""
        td = TraditionalChineseDictionary()
        dict_file = tmp_path / "traditional_dict.txt"
        dict_file.write_text("自然語言處理\n機器學習\n深度學習\n", encoding='utf-8')
        count = td.load_from_file(str(dict_file))
        assert count == 3
        assert td.contains("自然語言處理")
        assert td.contains("機器學習")
        assert td.contains("深度學習")

    def test_traditional_dict_save_to_file(self, tmp_path):
        """保存到文件"""
        td = TraditionalChineseDictionary()
        td.add_word("人工智能")
        td.add_word("机器学习")
        output_file = tmp_path / "output_dict.txt"
        td.save_to_file(str(output_file))
        content = output_file.read_text(encoding='utf-8')
        assert len(content.strip()) > 0
        lines = [line for line in content.strip().split('\n') if line]
        assert len(lines) == 2
