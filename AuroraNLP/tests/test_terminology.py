"""
专业术语库模块测试
"""

import os
import pytest
from auroranlp.terminology import (
    TermDomain,
    Term,
    TerminologyDatabase,
    TerminologyManager,
)


class TestTermDomain:
    def test_domain_name(self):
        assert TermDomain.MEDICAL.get_name() == "医学"
        assert TermDomain.LEGAL.get_name() == "法律"
        assert TermDomain.FINANCE.get_name() == "金融"
        assert TermDomain.IT.get_name() == "IT"
        assert TermDomain.OTHER.get_name() == "其他"


class TestTerm:
    def test_term_creation(self):
        term = Term(
            term_id="TEST001",
            name="测试术语",
            domain=TermDomain.IT,
            english="test term",
            aliases=["测试词", "测试名"],
            definition="这是一个测试术语",
        )
        
        assert term.term_id == "TEST001"
        assert term.name == "测试术语"
        assert term.domain == TermDomain.IT
        assert term.english == "test term"
        assert "测试词" in term.aliases
        assert term.definition == "这是一个测试术语"
    
    def test_term_get_all_names(self):
        term = Term(
            term_id="TEST001",
            name="测试术语",
            domain=TermDomain.IT,
            english="test term",
            aliases=["测试词"],
        )
        
        names = term.get_all_names()
        assert "测试术语" in names
        assert "测试词" in names
        assert "test term" in names
    
    def test_term_matches(self):
        term = Term(
            term_id="TEST001",
            name="测试术语",
            domain=TermDomain.IT,
            english="test term",
            aliases=["测试词"],
        )
        
        assert term.matches("测试术语")
        assert term.matches("TEST001")
        assert term.matches("测试词")
        assert term.matches("test term")
        assert term.matches("TEST TERM")
        assert not term.matches("其他术语")


class TestTerminologyDatabase:
    @pytest.fixture
    def db(self):
        return TerminologyDatabase(load_default=True, load_sogou=False)
    
    def test_database_creation(self, db):
        assert db is not None
        assert db.is_loaded()
    
    def test_get_by_name(self, db):
        terms = db.get_by_name("高血压")
        assert len(terms) > 0
        assert terms[0].domain == TermDomain.MEDICAL
    
    def test_get_by_english(self, db):
        term = db.get_by_english("hypertension")
        assert term is not None
        assert term.name == "高血压"
    
    def test_get_by_alias(self, db):
        term = db.get_by_alias("血压高")
        assert term is not None
        assert term.name == "高血压"
    
    def test_search(self, db):
        results = db.search("合同")
        assert len(results) > 0
    
    def test_get_by_domain(self, db):
        medical_terms = db.get_by_domain(TermDomain.MEDICAL)
        assert len(medical_terms) > 0
        
        legal_terms = db.get_by_domain(TermDomain.LEGAL)
        assert len(legal_terms) > 0
        
        finance_terms = db.get_by_domain(TermDomain.FINANCE)
        assert len(finance_terms) > 0
        
        it_terms = db.get_by_domain(TermDomain.IT)
        assert len(it_terms) > 0
    
    def test_get_medical_terms(self, db):
        terms = db.get_medical_terms()
        assert len(terms) > 0
        for term in terms:
            assert term.domain == TermDomain.MEDICAL
    
    def test_get_legal_terms(self, db):
        terms = db.get_legal_terms()
        assert len(terms) > 0
        for term in terms:
            assert term.domain == TermDomain.LEGAL
    
    def test_get_finance_terms(self, db):
        terms = db.get_finance_terms()
        assert len(terms) > 0
        for term in terms:
            assert term.domain == TermDomain.FINANCE
    
    def test_get_it_terms(self, db):
        terms = db.get_it_terms()
        assert len(terms) > 0
        for term in terms:
            assert term.domain == TermDomain.IT
    
    def test_is_term(self, db):
        assert db.is_term("高血压")
        assert db.is_term("合同法")
        assert db.is_term("股票")
        assert db.is_term("算法")
        assert not db.is_term("不存在的术语")
    
    def test_get_term_domain(self, db):
        domain = db.get_term_domain("高血压")
        assert domain == TermDomain.MEDICAL
        
        domain = db.get_term_domain("合同法")
        assert domain == TermDomain.LEGAL
        
        domain = db.get_term_domain("股票")
        assert domain == TermDomain.FINANCE
        
        domain = db.get_term_domain("算法")
        assert domain == TermDomain.IT
    
    def test_recognize_terms(self, db):
        text = "高血压患者需要注意饮食，合同法规定了违约责任。"
        results = db.recognize_terms(text)
        
        assert len(results) >= 2
        
        term_names = [t.name for t, _, _ in results]
        assert "高血压" in term_names
        assert "合同法" in term_names
    
    def test_recognize_terms_by_domain(self, db):
        text = "高血压和糖尿病都是慢性疾病"
        results = db.recognize_terms_by_domain(text, TermDomain.MEDICAL)
        
        assert len(results) >= 1
        
        for term, _, _ in results:
            assert term.domain == TermDomain.MEDICAL
    
    def test_add_term(self, db):
        initial_count = db.get_term_count()
        
        term = Term(
            term_id="NEW001",
            name="新术语",
            domain=TermDomain.OTHER,
        )
        db.add_term(term)
        
        assert db.get_term_count() == initial_count + 1
        assert db.is_term("新术语")
    
    def test_add_term_simple(self, db):
        initial_count = db.get_term_count()
        
        db.add_term_simple(
            name="简单术语",
            domain=TermDomain.IT,
            english="simple term",
            aliases=["简单词"],
            definition="这是一个简单术语",
        )
        
        assert db.get_term_count() == initial_count + 1
        assert db.is_term("简单术语")
        assert db.get_by_english("simple term") is not None
    
    def test_get_statistics(self, db):
        stats = db.get_statistics()
        
        assert stats["loaded"] is True
        assert stats["total_count"] > 0
        assert stats["medical_count"] > 0
        assert stats["legal_count"] > 0
        assert stats["finance_count"] > 0
        assert stats["it_count"] > 0
    
    def test_contains(self, db):
        assert "高血压" in db
        assert "合同法" in db
        assert "股票" in db
        assert "算法" in db
    
    def test_len(self, db):
        assert len(db) > 0


class TestTerminologyManager:
    @pytest.fixture
    def manager(self):
        return TerminologyManager(load_default=True, load_sogou=False)
    
    def test_manager_creation(self, manager):
        assert manager is not None
        assert manager.is_loaded()
    
    def test_get_database(self, manager):
        db = manager.get_database()
        assert db is not None
        assert isinstance(db, TerminologyDatabase)
    
    def test_get_by_name(self, manager):
        terms = manager.get_by_name("高血压")
        assert len(terms) > 0
    
    def test_search(self, manager):
        results = manager.search("合同")
        assert len(results) > 0
    
    def test_get_by_domain(self, manager):
        terms = manager.get_by_domain(TermDomain.MEDICAL)
        assert len(terms) > 0
    
    def test_is_term(self, manager):
        assert manager.is_term("高血压")
        assert manager.is_term("合同法")
    
    def test_recognize_terms(self, manager):
        text = "高血压患者需要服用抗生素治疗"
        results = manager.recognize_terms(text)
        
        assert len(results) >= 1
    
    def test_get_statistics(self, manager):
        stats = manager.get_statistics()
        assert stats["loaded"] is True


class TestSogouIntegration:
    @pytest.fixture
    def db_with_sogou(self):
        sogou_path = os.path.join(
            os.path.dirname(__file__), 
            '..', 'AuroraNLP', 'data', 'sogou'
        )
        if not os.path.exists(sogou_path):
            pytest.skip("Sogou data directory not found")
        
        return TerminologyDatabase(load_default=True, load_sogou=True)
    
    def test_sogou_loading(self, db_with_sogou):
        assert db_with_sogou.is_sogou_loaded()
        
        stats = db_with_sogou.get_statistics()
        assert stats["sogou_loaded"] is True
    
    def test_legal_terms_from_sogou(self, db_with_sogou):
        legal_terms = db_with_sogou.get_legal_terms()
        assert len(legal_terms) > 0
    
    def test_finance_terms_from_sogou(self, db_with_sogou):
        finance_terms = db_with_sogou.get_finance_terms()
        assert len(finance_terms) > 0
    
    def test_it_terms_from_sogou(self, db_with_sogou):
        it_terms = db_with_sogou.get_it_terms()
        assert len(it_terms) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
