import unittest
from AuroraNLP import DomainDictionary, DomainDictionaryManager, DictionaryManager


class TestDomainDictionary(unittest.TestCase):
    def test_creation(self):
        """测试领域词典的创建"""
        # 测试新闻领域词典
        news_dict = DomainDictionary('news')
        self.assertEqual(news_dict.domain, 'news')
        self.assertEqual(news_dict.domain_name, '新闻领域')
        self.assertTrue(len(news_dict) > 0)
        
        # 测试医疗领域词典
        medical_dict = DomainDictionary('medical')
        self.assertEqual(medical_dict.domain, 'medical')
        self.assertEqual(medical_dict.domain_name, '医疗领域')
        self.assertTrue(len(medical_dict) > 0)
        
        # 测试法律领域词典
        legal_dict = DomainDictionary('legal')
        self.assertEqual(legal_dict.domain, 'legal')
        self.assertEqual(legal_dict.domain_name, '法律领域')
        self.assertTrue(len(legal_dict) > 0)
        
        # 测试电商领域词典
        ecommerce_dict = DomainDictionary('ecommerce')
        self.assertEqual(ecommerce_dict.domain, 'ecommerce')
        self.assertEqual(ecommerce_dict.domain_name, '电商领域')
        self.assertTrue(len(ecommerce_dict) > 0)
    
    def test_supported_domains(self):
        """测试支持的领域列表"""
        domains = DomainDictionary.get_supported_domains()
        self.assertIn('news', domains)
        self.assertIn('medical', domains)
        self.assertIn('legal', domains)
        self.assertIn('ecommerce', domains)
    
    def test_get_domain_info(self):
        """测试获取领域信息"""
        news_dict = DomainDictionary('news')
        info = news_dict.get_domain_info()
        self.assertEqual(info['domain'], 'news')
        self.assertEqual(info['domain_name'], '新闻领域')
        self.assertEqual(info['name'], 'domain_news')
        self.assertTrue(info['word_count'] > 0)
    
    def test_create_domain_dictionary(self):
        """测试通过类方法创建领域词典"""
        news_dict = DomainDictionary.create_domain_dictionary('news')
        self.assertEqual(news_dict.domain, 'news')
        self.assertTrue(len(news_dict) > 0)


class TestDomainDictionaryManager(unittest.TestCase):
    def test_register_domain_dictionary(self):
        """测试注册领域词典"""
        manager = DomainDictionaryManager()
        news_dict = DomainDictionary('news')
        manager.register_domain_dictionary(news_dict)
        self.assertEqual(len(manager), 1)
        self.assertIsNotNone(manager.get_domain_dictionary('news'))
    
    def test_unregister_domain_dictionary(self):
        """测试注销领域词典"""
        manager = DomainDictionaryManager()
        news_dict = DomainDictionary('news')
        manager.register_domain_dictionary(news_dict)
        self.assertEqual(len(manager), 1)
        result = manager.unregister_domain_dictionary('news')
        self.assertTrue(result)
        self.assertEqual(len(manager), 0)
        self.assertIsNone(manager.get_domain_dictionary('news'))
    
    def test_load_all_domains(self):
        """测试加载所有领域词典"""
        manager = DomainDictionaryManager()
        manager.load_all_domains()
        self.assertEqual(len(manager), 4)
        self.assertIsNotNone(manager.get_domain_dictionary('news'))
        self.assertIsNotNone(manager.get_domain_dictionary('medical'))
        self.assertIsNotNone(manager.get_domain_dictionary('legal'))
        self.assertIsNotNone(manager.get_domain_dictionary('ecommerce'))
    
    def test_get_all_domains_info(self):
        """测试获取所有领域词典信息"""
        manager = DomainDictionaryManager()
        manager.load_all_domains()
        info_list = manager.get_all_domains_info()
        self.assertEqual(len(info_list), 4)
        for info in info_list:
            self.assertEqual(info['type'], 'domain')
            self.assertIn(info['domain'], ['news', 'medical', 'legal', 'ecommerce'])


class TestDictionaryManagerWithDomain(unittest.TestCase):
    def test_register_domain_dictionary_in_manager(self):
        """测试在DictionaryManager中注册领域词典"""
        manager = DictionaryManager()
        news_dict = DomainDictionary('news')
        manager.register_domain_dictionary(news_dict)
        # 测试获取领域词典
        domain_dict = manager.get_domain_dictionary('news')
        self.assertIsNotNone(domain_dict)
        self.assertEqual(domain_dict.domain, 'news')
        # 测试获取所有词典信息
        info_list = manager.get_all_dictionaries_info()
        domain_info = next((info for info in info_list if info['type'] == 'domain'), None)
        self.assertIsNotNone(domain_info)
        self.assertEqual(domain_info['domain'], 'news')
    
    def test_unregister_domain_dictionary_in_manager(self):
        """测试在DictionaryManager中注销领域词典"""
        manager = DictionaryManager()
        news_dict = DomainDictionary('news')
        manager.register_domain_dictionary(news_dict)
        result = manager.unregister_dictionary('news')
        self.assertTrue(result)
        self.assertIsNone(manager.get_domain_dictionary('news'))


if __name__ == '__main__':
    unittest.main()