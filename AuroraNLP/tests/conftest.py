"""AuroraNLP 测试套件 - 公共 fixtures"""
import sys
import os
import pytest

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from AuroraNLP.trie import Trie
from AuroraNLP.dictionary import Dictionary, UserDictionary
from AuroraNLP.stopwords import StopWords
from AuroraNLP.segmentor import Segmentor


@pytest.fixture
def empty_trie():
    """创建空 Trie 树"""
    return Trie()


@pytest.fixture
def sample_trie():
    """创建包含示例词的 Trie 树"""
    trie = Trie()
    trie.insert("中国", "ns", 1.0, 0)
    trie.insert("中国人", "ns", 1.0, 0)
    trie.insert("中文", "nz", 1.0, 0)
    trie.insert("自然语言", "n", 1.0, 0)
    trie.insert("自然语言处理", "n", 1.5, 0)
    trie.insert("处理", "v", 1.0, 0)
    trie.insert("爱", "v", 1.0, 0)
    trie.insert("我", "r", 1.0, 0)
    trie.insert("北京", "ns", 1.0, 0)
    trie.insert("北京大学", "nt", 1.0, 0)
    trie.insert("大学", "n", 1.0, 0)
    trie.insert("研究", "v", 1.0, 0)
    trie.insert("研究生", "n", 1.0, 0)
    trie.insert("生命", "n", 1.0, 0)
    trie.insert("的", "u", 1.0, 0)
    trie.insert("是", "v", 1.0, 0)
    trie.insert("在", "p", 1.0, 0)
    trie.insert("工作", "v", 1.0, 0)
    trie.insert("好", "a", 1.0, 0)
    trie.insert("很好", "a", 1.2, 0)
    return trie


@pytest.fixture
def empty_dictionary():
    """创建空词典"""
    return Dictionary(load_default=False)


@pytest.fixture
def sample_dictionary():
    """创建包含示例词的词典"""
    d = Dictionary(load_default=False)
    d.add_word("中国", "ns", 1.0)
    d.add_word("中国人", "ns", 1.0)
    d.add_word("中文", "nz", 1.0)
    d.add_word("自然语言", "n", 1.0)
    d.add_word("自然语言处理", "n", 1.5)
    d.add_word("处理", "v", 1.0)
    d.add_word("爱", "v", 1.0)
    d.add_word("我", "r", 1.0)
    d.add_word("北京", "ns", 1.0)
    d.add_word("北京大学", "nt", 1.0)
    d.add_word("大学", "n", 1.0)
    d.add_word("研究", "v", 1.0)
    d.add_word("研究生", "n", 1.0)
    d.add_word("生命", "n", 1.0)
    d.add_word("的", "u", 1.0)
    d.add_word("是", "v", 1.0)
    d.add_word("在", "p", 1.0)
    d.add_word("工作", "v", 1.0)
    d.add_word("好", "a", 1.0)
    d.add_word("很好", "a", 1.2)
    return d


@pytest.fixture
def sample_user_dictionary():
    """创建用户词典"""
    ud = UserDictionary(name="test_user", priority=100)
    ud.add_word("人工智能", "n", 10.0)
    ud.add_word("深度学习", "n", 10.0)
    ud.add_word("机器学习", "n", 10.0)
    return ud


@pytest.fixture
def sample_stopwords():
    """创建停用词对象"""
    sw = StopWords(load_default=False)
    sw.add_stopword("的")
    sw.add_stopword("了")
    sw.add_stopword("在")
    sw.add_stopword("是")
    sw.add_stopword("我")
    return sw


@pytest.fixture
def sample_segmentor(sample_dictionary):
    """创建使用示例词典的分词器"""
    return Segmentor(dictionary=sample_dictionary, load_default_dict=False, load_default_stopwords=False)


@pytest.fixture
def sample_corpus():
    """示例训练语料"""
    return [
        ['我', '爱', '中国'],
        ['中国', '是', '伟大', '的', '国家'],
        ['北京', '是', '中国', '的', '首都'],
        ['我', '在', '北京', '大学', '学习'],
        ['自然语言', '处理', '是', '人工智能', '的', '方向'],
        ['我', '爱', '自然语言', '处理'],
        ['中国', '人', '很', '好'],
        ['研究', '生', '在', '研究', '生命'],
    ]


@pytest.fixture
def tmp_dict_file(tmp_path):
    """创建临时词典文件"""
    dict_file = tmp_path / "test_dict.txt"
    dict_file.write_text("人工智能 n 1.5 0\n深度学习 n 1.5 0\n机器学习 n 1.5 0\n神经网络 n 1.0 0\n", encoding='utf-8')
    return str(dict_file)


@pytest.fixture
def tmp_stopwords_file(tmp_path):
    """创建临时停用词文件"""
    sw_file = tmp_path / "test_stopwords.txt"
    sw_file.write_text("的\n了\n在\n是\n我\n", encoding='utf-8')
    return str(sw_file)
