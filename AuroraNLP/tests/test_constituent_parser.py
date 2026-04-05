#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
成分句法分析测试
"""

import unittest
import tempfile
import os

from AuroraNLP.constituent_parser import (
    ConstituentTree,
    ConstituentNode,
    GrammarRule,
    PCFG,
    CKYParser,
    ConstituentParser,
    create_sample_constituent_trees,
    CONSTITUENT_LABELS,
    POS_LABELS,
)


class TestGrammarRule(unittest.TestCase):
    
    def test_rule_creation(self):
        rule = GrammarRule(lhs='S', rhs=('NP', 'VP'), probability=0.5)
        self.assertEqual(rule.lhs, 'S')
        self.assertEqual(rule.rhs, ('NP', 'VP'))
        self.assertEqual(rule.probability, 0.5)
    
    def test_rule_types(self):
        unary = GrammarRule(lhs='NP', rhs=('NN',))
        binary = GrammarRule(lhs='S', rhs=('NP', 'VP'))
        
        self.assertTrue(unary.is_unary())
        self.assertFalse(unary.is_binary())
        
        self.assertFalse(binary.is_unary())
        self.assertTrue(binary.is_binary())
    
    def test_rule_from_string(self):
        rule = GrammarRule.from_string('S -> NP VP', probability=0.3)
        self.assertEqual(rule.lhs, 'S')
        self.assertEqual(rule.rhs, ('NP', 'VP'))
        self.assertEqual(rule.probability, 0.3)
    
    def test_rule_equality(self):
        rule1 = GrammarRule(lhs='S', rhs=('NP', 'VP'))
        rule2 = GrammarRule(lhs='S', rhs=('NP', 'VP'))
        rule3 = GrammarRule(lhs='S', rhs=('NP', 'PP'))
        
        self.assertEqual(rule1, rule2)
        self.assertNotEqual(rule1, rule3)
    
    def test_rule_hash(self):
        rule1 = GrammarRule(lhs='S', rhs=('NP', 'VP'))
        rule2 = GrammarRule(lhs='S', rhs=('NP', 'VP'))
        
        rules_set = {rule1, rule2}
        self.assertEqual(len(rules_set), 1)


class TestConstituentNode(unittest.TestCase):
    
    def test_terminal_node(self):
        node = ConstituentNode(label='NN', word='苹果', start=0, end=1)
        
        self.assertTrue(node.is_terminal())
        self.assertFalse(node.is_preterminal())
        self.assertEqual(node.get_words(), ['苹果'])
        self.assertEqual(node.get_text(), '苹果')
        self.assertEqual(node.get_height(), 0)
    
    def test_preterminal_node(self):
        word_node = ConstituentNode(label='苹果', word='苹果', start=0, end=1)
        pos_node = ConstituentNode(label='NN', children=[word_node], start=0, end=1)
        
        self.assertTrue(pos_node.is_preterminal())
        self.assertFalse(pos_node.is_terminal())
        self.assertEqual(pos_node.get_words(), ['苹果'])
        self.assertEqual(pos_node.get_height(), 1)
    
    def test_phrase_node(self):
        word1 = ConstituentNode(label='我', word='我', start=0, end=1)
        pos1 = ConstituentNode(label='NR', children=[word1], start=0, end=1)
        
        word2 = ConstituentNode(label='爱', word='爱', start=1, end=2)
        pos2 = ConstituentNode(label='VV', children=[word2], start=1, end=2)
        
        np = ConstituentNode(label='NP', children=[pos1], start=0, end=1)
        vp = ConstituentNode(label='VP', children=[pos2], start=1, end=2)
        s = ConstituentNode(label='S', children=[np, vp], start=0, end=2)
        
        self.assertFalse(s.is_terminal())
        self.assertFalse(s.is_preterminal())
        self.assertEqual(s.get_words(), ['我', '爱'])
        self.assertEqual(s.get_text(), '我爱')
        self.assertEqual(s.get_height(), 3)
    
    def test_get_phrases(self):
        word1 = ConstituentNode(label='我', word='我', start=0, end=1)
        pos1 = ConstituentNode(label='NR', children=[word1], start=0, end=1)
        np = ConstituentNode(label='NP', children=[pos1], start=0, end=1)
        
        word2 = ConstituentNode(label='爱', word='爱', start=1, end=2)
        pos2 = ConstituentNode(label='VV', children=[word2], start=1, end=2)
        vp = ConstituentNode(label='VP', children=[pos2], start=1, end=2)
        
        s = ConstituentNode(label='S', children=[np, vp], start=0, end=2)
        
        nps = s.get_noun_phrases()
        vps = s.get_verb_phrases()
        
        self.assertEqual(len(nps), 1)
        self.assertEqual(len(vps), 1)
    
    def test_to_lisp_string(self):
        word = ConstituentNode(label='苹果', word='苹果', start=0, end=1)
        pos = ConstituentNode(label='NN', children=[word], start=0, end=1)
        
        self.assertEqual(pos.to_lisp_string(), '(NN 苹果)')
    
    def test_to_dict_and_from_dict(self):
        word = ConstituentNode(label='苹果', word='苹果', start=0, end=1)
        pos = ConstituentNode(label='NN', children=[word], start=0, end=1)
        
        data = pos.to_dict()
        restored = ConstituentNode.from_dict(data)
        
        self.assertEqual(restored.label, 'NN')
        self.assertEqual(len(restored.children), 1)
        self.assertEqual(restored.children[0].word, '苹果')


class TestConstituentTree(unittest.TestCase):
    
    def test_from_penn_treebank_simple(self):
        tree_str = "(S (NP (NR 我)) (VP (VV 爱) (NP (NR 中国))))"
        tree = ConstituentTree.from_penn_treebank(tree_str)
        
        self.assertIsNotNone(tree.root)
        self.assertEqual(tree.root.label, 'S')
        self.assertEqual(tree.get_text(), '我爱中国')
        self.assertEqual(tree.get_words(), ['我', '爱', '中国'])
    
    def test_from_penn_treebank_multiline(self):
        tree_str = """
        (S
            (NP (NR 我))
            (VP (VV 爱)
                (NP (NR 中国))))
        """
        tree = ConstituentTree.from_penn_treebank(tree_str)
        
        self.assertIsNotNone(tree.root)
        self.assertEqual(tree.get_text(), '我爱中国')
    
    def test_tree_length(self):
        tree_str = "(S (NP (NR 我)) (VP (VV 爱) (NP (NR 中国))))"
        tree = ConstituentTree.from_penn_treebank(tree_str)
        
        self.assertEqual(len(tree), 3)
    
    def test_tree_iteration(self):
        tree_str = "(S (NP (NR 我)) (VP (VV 爱)))"
        tree = ConstituentTree.from_penn_treebank(tree_str)
        
        nodes = list(tree)
        self.assertEqual(len(nodes), 7)
    
    def test_get_pos_tags(self):
        tree_str = "(S (NP (NR 我)) (VP (VV 爱)))"
        tree = ConstituentTree.from_penn_treebank(tree_str)
        
        pos_tags = tree.get_pos_tags()
        self.assertEqual(len(pos_tags), 2)
        self.assertEqual(pos_tags[0], ('NR', '我'))
        self.assertEqual(pos_tags[1], ('VV', '爱'))
    
    def test_extract_rules(self):
        tree_str = "(S (NP (NR 我)) (VP (VV 爱)))"
        tree = ConstituentTree.from_penn_treebank(tree_str)
        
        rules = tree.extract_rules()
        
        rule_strs = [f"{r.lhs} -> {' '.join(r.rhs)}" for r in rules]
        
        self.assertIn('S -> NP VP', rule_strs)
        self.assertIn('NP -> NR', rule_strs)
        self.assertIn('NR -> 我', rule_strs)
    
    def test_to_penn_treebank(self):
        tree_str = "(S (NP (NR 我)) (VP (VV 爱)))"
        tree = ConstituentTree.from_penn_treebank(tree_str)
        
        result = tree.to_penn_treebank()
        self.assertIn('S', result)
        self.assertIn('NP', result)
        self.assertIn('VP', result)


class TestPCFG(unittest.TestCase):
    
    def test_pcfg_train(self):
        trees = create_sample_constituent_trees()
        
        pcfg = PCFG()
        pcfg.train(trees)
        
        self.assertTrue(pcfg._trained)
        self.assertGreater(len(pcfg.rules), 0)
        self.assertGreater(len(pcfg.non_terminals), 0)
        self.assertGreater(len(pcfg.terminals), 0)
    
    def test_pcfg_add_rule(self):
        pcfg = PCFG()
        
        rule1 = GrammarRule(lhs='S', rhs=('NP', 'VP'), probability=0.5, count=1)
        rule2 = GrammarRule(lhs='S', rhs=('NP', 'VP'), probability=0.3, count=1)
        
        pcfg.add_rule(rule1)
        pcfg.add_rule(rule2)
        
        rules = pcfg.get_rules('S')
        self.assertEqual(len(rules), 1)
        self.assertEqual(rules[0].count, 2)
    
    def test_pcfg_get_binary_rules(self):
        pcfg = PCFG()
        
        pcfg.add_rule(GrammarRule(lhs='S', rhs=('NP', 'VP')))
        pcfg.add_rule(GrammarRule(lhs='NP', rhs=('NN',)))
        
        binary = pcfg.get_binary_rules('S')
        unary = pcfg.get_unary_rules('NP')
        
        self.assertEqual(len(binary), 1)
        self.assertEqual(len(unary), 1)
    
    def test_pcfg_compute_probability(self):
        tree_str = "(S (NP (NR 我)) (VP (VV 爱)))"
        tree = ConstituentTree.from_penn_treebank(tree_str)
        
        pcfg = PCFG()
        pcfg.train([tree])
        
        prob = pcfg.compute_probability(tree)
        self.assertGreater(prob, 0)
        self.assertLessEqual(prob, 1)
    
    def test_pcfg_save_load(self):
        trees = create_sample_constituent_trees()
        
        pcfg = PCFG()
        pcfg.train(trees)
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            temp_path = f.name
        
        try:
            pcfg.save(temp_path)
            
            pcfg2 = PCFG()
            pcfg2.load(temp_path)
            
            self.assertEqual(pcfg2._trained, True)
            self.assertEqual(len(pcfg2.rules), len(pcfg.rules))
        finally:
            os.unlink(temp_path)


class TestCKYParser(unittest.TestCase):
    
    def test_cky_parse(self):
        trees = create_sample_constituent_trees()
        
        pcfg = PCFG()
        pcfg.train(trees)
        
        cky = CKYParser(pcfg)
        
        tree = cky.parse(['我', '爱', '中国'])
        
        if tree:
            self.assertEqual(tree.get_text(), '我爱中国')
    
    def test_cky_parse_empty(self):
        pcfg = PCFG()
        cky = CKYParser(pcfg)
        
        tree = cky.parse([])
        self.assertIsNone(tree)
    
    def test_cky_parse_k_best(self):
        trees = create_sample_constituent_trees()
        
        pcfg = PCFG()
        pcfg.train(trees)
        
        cky = CKYParser(pcfg)
        
        results = cky.parse_k_best(['我', '爱', '中国'], k=3)
        
        self.assertIsInstance(results, list)


class TestConstituentParser(unittest.TestCase):
    
    def test_parser_train(self):
        parser = ConstituentParser()
        trees = create_sample_constituent_trees()
        
        parser.train(trees)
        
        self.assertTrue(parser.is_trained())
    
    def test_parser_parse(self):
        parser = ConstituentParser()
        trees = create_sample_constituent_trees()
        parser.train(trees)
        
        tree = parser.parse(['我', '爱', '中国'])
        
        if tree:
            self.assertEqual(tree.get_text(), '我爱中国')
    
    def test_parser_parse_untrained(self):
        parser = ConstituentParser()
        
        with self.assertRaises(RuntimeError):
            parser.parse(['我', '爱'])
    
    def test_parser_save_load(self):
        parser = ConstituentParser()
        trees = create_sample_constituent_trees()
        parser.train(trees)
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            temp_path = f.name
        
        try:
            parser.save_model(temp_path)
            
            parser2 = ConstituentParser()
            parser2.load_model(temp_path)
            
            self.assertTrue(parser2.is_trained())
        finally:
            os.unlink(temp_path)
    
    def test_parser_get_model_info(self):
        parser = ConstituentParser()
        
        info = parser.get_model_info()
        self.assertFalse(info['trained'])
        
        trees = create_sample_constituent_trees()
        parser.train(trees)
        
        info = parser.get_model_info()
        self.assertTrue(info['trained'])
        self.assertIn('grammar', info)


class TestSampleTrees(unittest.TestCase):
    
    def test_create_sample_trees(self):
        trees = create_sample_constituent_trees()
        
        self.assertGreater(len(trees), 0)
        
        for tree in trees:
            self.assertIsNotNone(tree.root)
            self.assertGreater(len(tree), 0)


if __name__ == '__main__':
    unittest.main()
