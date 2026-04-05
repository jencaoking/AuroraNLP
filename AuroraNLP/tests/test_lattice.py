import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from AuroraNLP.lattice import (
    Lattice, LatticeEdge, LatticeNode, 
    LatticeBuilder, LatticeSegmentor, PathScorer
)
from AuroraNLP.dictionary import Dictionary
from AuroraNLP.ngram import BigramModel
from AuroraNLP import Segmentor


class TestLatticeEdge(unittest.TestCase):
    def test_edge_creation(self):
        edge = LatticeEdge(word="测试", start=0, end=2, pos_tag="n", weight=0.5)
        self.assertEqual(edge.word, "测试")
        self.assertEqual(edge.start, 0)
        self.assertEqual(edge.end, 2)
        self.assertEqual(edge.pos_tag, "n")
        self.assertEqual(edge.weight, 0.5)
    
    def test_edge_hash_and_equality(self):
        edge1 = LatticeEdge(word="测试", start=0, end=2)
        edge2 = LatticeEdge(word="测试", start=0, end=2)
        edge3 = LatticeEdge(word="测试", start=0, end=3)
        
        self.assertEqual(edge1, edge2)
        self.assertNotEqual(edge1, edge3)
        self.assertEqual(hash(edge1), hash(edge2))


class TestLatticeNode(unittest.TestCase):
    def test_node_creation(self):
        node = LatticeNode(position=0)
        self.assertEqual(node.position, 0)
        self.assertEqual(len(node.incoming_edges), 0)
        self.assertEqual(len(node.outgoing_edges), 0)
    
    def test_add_edges(self):
        node = LatticeNode(position=0)
        edge1 = LatticeEdge(word="测", start=0, end=1)
        edge2 = LatticeEdge(word="测试", start=0, end=2)
        
        node.add_outgoing(edge1)
        node.add_outgoing(edge2)
        
        self.assertEqual(len(node.outgoing_edges), 2)
        self.assertIn(edge1, node.outgoing_edges)
        self.assertIn(edge2, node.outgoing_edges)


class TestLattice(unittest.TestCase):
    def test_lattice_creation(self):
        lattice = Lattice("测试文本")
        self.assertEqual(lattice.length, 4)
        self.assertEqual(len(lattice.nodes), 5)
    
    def test_add_edge(self):
        lattice = Lattice("测试")
        edge = lattice.add_edge("测", 0, 1, pos_tag="v")
        
        self.assertEqual(len(lattice.edges), 1)
        self.assertEqual(lattice.nodes[0].outgoing_edges[0], edge)
        self.assertEqual(lattice.nodes[1].incoming_edges[0], edge)
    
    def test_invalid_edge(self):
        lattice = Lattice("测试")
        with self.assertRaises(ValueError):
            lattice.add_edge("测试", 0, 5)
        with self.assertRaises(ValueError):
            lattice.add_edge("测试", -1, 2)
    
    def test_get_all_paths(self):
        lattice = Lattice("测试")
        lattice.add_edge("测", 0, 1)
        lattice.add_edge("试", 1, 2)
        lattice.add_edge("测试", 0, 2)
        
        paths = lattice.get_all_paths()
        self.assertEqual(len(paths), 2)
        
        path_words = [lattice.get_path_words(p) for p in paths]
        self.assertIn(["测", "试"], path_words)
        self.assertIn(["测试"], path_words)
    
    def test_has_path(self):
        lattice = Lattice("测试")
        self.assertFalse(lattice.has_path())
        
        lattice.add_edge("测试", 0, 2)
        self.assertTrue(lattice.has_path())
    
    def test_is_fully_connected(self):
        lattice = Lattice("测试")
        lattice.add_edge("测", 0, 1)
        lattice.add_edge("试", 1, 2)
        self.assertTrue(lattice.is_fully_connected())
        
        lattice2 = Lattice("测试")
        lattice2.add_edge("测", 0, 1)
        self.assertFalse(lattice2.is_fully_connected())
    
    def test_get_statistics(self):
        lattice = Lattice("测试")
        lattice.add_edge("测", 0, 1)
        lattice.add_edge("试", 1, 2)
        lattice.add_edge("测试", 0, 2)
        
        stats = lattice.get_statistics()
        self.assertEqual(stats['text_length'], 2)
        self.assertEqual(stats['total_edges'], 3)
        self.assertTrue(stats['is_fully_connected'])


class TestLatticeBuilder(unittest.TestCase):
    def setUp(self):
        self.dict = Dictionary(load_default=False)
        self.dict.add_word("研究", "v")
        self.dict.add_word("研究生", "n")
        self.dict.add_word("生命", "n")
        self.dict.add_word("命", "n")
        self.dict.add_word("的", "u")
        self.dict.add_word("起源", "n")
        self.builder = LatticeBuilder(self.dict)
    
    def test_build_basic(self):
        lattice = self.builder.build("研究")
        self.assertTrue(lattice.has_path())
        self.assertGreaterEqual(len(lattice.edges), 1)
    
    def test_build_with_ambiguity(self):
        lattice = self.builder.build("研究生")
        
        self.assertTrue(lattice.has_path())
        self.assertGreater(len(lattice.edges), 1)
        
        paths = lattice.get_all_paths()
        path_words = [lattice.get_path_words(p) for p in paths]
        self.assertIn(["研究生"], path_words)
        self.assertIn(["研究", "生"], path_words)
    
    def test_build_with_freq(self):
        word_freq = {"研究": 100, "研究生": 50}
        lattice = self.builder.build_with_freq("研究生", word_freq)
        
        for edge in lattice.edges:
            if edge.word == "研究":
                self.assertEqual(edge.freq, 100)
            elif edge.word == "研究生":
                self.assertEqual(edge.freq, 50)


class TestLatticeSegmentor(unittest.TestCase):
    def setUp(self):
        self.dict = Dictionary(load_default=False)
        self.dict.add_word("研究", "v")
        self.dict.add_word("研究生", "n")
        self.dict.add_word("生命", "n")
        self.dict.add_word("的", "u")
        self.dict.add_word("起源", "n")
        self.dict.add_word("我", "r")
        self.dict.add_word("喜欢", "v")
        self.dict.add_word("学习", "v")
        self.segmentor = LatticeSegmentor(self.dict)
    
    def test_segment_basic(self):
        result = self.segmentor.segment("研究")
        self.assertEqual(result, ["研究"])
    
    def test_segment_with_ambiguity(self):
        result = self.segmentor.segment("研究生")
        self.assertIsInstance(result, list)
        self.assertEqual("".join(result), "研究生")
    
    def test_segment_with_pos(self):
        result = self.segmentor.segment_with_pos("研究")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], "研究")
        self.assertEqual(result[0][1], "v")
    
    def test_shortest_path(self):
        lattice = self.segmentor.build_lattice("研究生")
        path = self.segmentor.shortest_path(lattice)
        
        self.assertIsInstance(path, list)
        self.assertEqual("".join(e.word for e in path), "研究生")
    
    def test_detect_ambiguity(self):
        ambiguities = self.segmentor.detect_ambiguity("研究生")
        self.assertGreater(len(ambiguities), 0)
    
    def test_get_all_segmentations(self):
        all_segs = self.segmentor.get_all_segmentations("研究生", max_results=10)
        self.assertGreater(len(all_segs), 0)
        
        for seg in all_segs:
            self.assertEqual("".join(seg), "研究生")
    
    def test_segment_with_lattice(self):
        words, lattice = self.segmentor.segment_with_lattice("研究")
        self.assertEqual(words, ["研究"])
        self.assertIsInstance(lattice, Lattice)
    
    def test_scoring_methods(self):
        self.segmentor.set_scoring_method('shortest')
        result1 = self.segmentor.segment("研究生")
        
        self.segmentor.set_scoring_method('longest_word')
        result2 = self.segmentor.segment("研究生")
        
        self.assertEqual("".join(result1), "研究生")
        self.assertEqual("".join(result2), "研究生")


class TestLatticeSegmentorWithNgram(unittest.TestCase):
    def setUp(self):
        self.dict = Dictionary(load_default=False)
        self.dict.add_word("我", "r")
        self.dict.add_word("喜欢", "v")
        self.dict.add_word("学习", "v")
        self.dict.add_word("研究", "v")
        self.dict.add_word("研究生", "n")
        
        self.ngram = BigramModel()
        corpus = [
            ["我", "喜欢", "学习"],
            ["我", "喜欢", "研究"],
            ["研究生", "学习"],
            ["研究", "生", "活"]
        ]
        self.ngram.train(corpus)
        
        self.segmentor = LatticeSegmentor(self.dict)
    
    def test_ngram_scoring(self):
        self.segmentor.set_ngram_model(self.ngram)
        
        result = self.segmentor.segment("我喜欢学习")
        self.assertEqual("".join(result), "我喜欢学习")


class TestPathScorer(unittest.TestCase):
    def test_score_by_length(self):
        edges = [
            LatticeEdge("我", 0, 1),
            LatticeEdge("喜欢", 1, 3),
            LatticeEdge("学习", 3, 5)
        ]
        score = PathScorer.score_by_length(edges)
        self.assertEqual(score, -3)
    
    def test_score_by_word_length(self):
        edges = [
            LatticeEdge("我", 0, 1),
            LatticeEdge("喜欢", 1, 3),
            LatticeEdge("学习", 3, 5)
        ]
        score = PathScorer.score_by_word_length(edges)
        self.assertEqual(score, 5/3)
    
    def test_score_by_frequency(self):
        edges = [
            LatticeEdge("我", 0, 1, freq=100),
            LatticeEdge("喜欢", 1, 3, freq=50),
            LatticeEdge("学习", 3, 5, freq=30)
        ]
        score = PathScorer.score_by_frequency(edges)
        self.assertEqual(score, 180)


class TestSegmentorIntegration(unittest.TestCase):
    def setUp(self):
        self.segmentor = Segmentor(load_default_dict=False)
        self.segmentor.add_word("研究", "v")
        self.segmentor.add_word("研究生", "n")
        self.segmentor.add_word("生命", "n")
        self.segmentor.add_word("的", "u")
        self.segmentor.add_word("起源", "n")
    
    def test_lattice_mode(self):
        result = self.segmentor.segment("研究", mode='lattice')
        self.assertEqual(result, ["研究"])
    
    def test_detect_ambiguity(self):
        ambiguities = self.segmentor.detect_lattice_ambiguity("研究生")
        self.assertIsInstance(ambiguities, list)
    
    def test_get_all_segmentations(self):
        all_segs = self.segmentor.get_all_lattice_segmentations("研究生")
        self.assertGreater(len(all_segs), 0)
    
    def test_build_lattice(self):
        lattice = self.segmentor.build_lattice("研究")
        self.assertIsInstance(lattice, Lattice)
        self.assertTrue(lattice.has_path())
    
    def test_find_k_best_paths(self):
        paths = self.segmentor.find_k_best_paths("研究生", k=3)
        self.assertGreater(len(paths), 0)
        self.assertLessEqual(len(paths), 3)


if __name__ == '__main__':
    unittest.main()
