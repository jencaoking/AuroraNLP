import unittest
import tempfile
import os
from AuroraNLP.dependency_parser import (
    Transition,
    DEPENDENCY_RELATIONS,
    DEFAULT_RELATIONS,
    DependencyArc,
    DependencyNode,
    DependencyTree,
    ParserState,
    ArcEagerOracle,
    DependencyFeatureExtractor,
    DependencyParser,
    create_sample_dependency_corpus,
    train_dependency_parser_from_file,
)


class TestDependencyArc(unittest.TestCase):
    def test_arc_creation(self):
        arc = DependencyArc(head=1, dependent=2, relation='nsubj')
        self.assertEqual(arc.head, 1)
        self.assertEqual(arc.dependent, 2)
        self.assertEqual(arc.relation, 'nsubj')
    
    def test_arc_equality(self):
        arc1 = DependencyArc(head=1, dependent=2, relation='nsubj')
        arc2 = DependencyArc(head=1, dependent=2, relation='nsubj')
        arc3 = DependencyArc(head=1, dependent=2, relation='obj')
        arc4 = DependencyArc(head=2, dependent=1, relation='nsubj')
        
        self.assertEqual(arc1, arc2)
        self.assertNotEqual(arc1, arc3)
        self.assertNotEqual(arc1, arc4)
    
    def test_arc_hash(self):
        arc1 = DependencyArc(head=1, dependent=2, relation='nsubj')
        arc2 = DependencyArc(head=1, dependent=2, relation='nsubj')
        
        arc_set = {arc1, arc2}
        self.assertEqual(len(arc_set), 1)
    
    def test_arc_repr(self):
        arc = DependencyArc(head=1, dependent=2, relation='nsubj')
        repr_str = repr(arc)
        self.assertIn('head=1', repr_str)
        self.assertIn('dep=2', repr_str)
        self.assertIn('nsubj', repr_str)


class TestDependencyNode(unittest.TestCase):
    def test_node_creation(self):
        node = DependencyNode(
            id=1,
            form='我',
            lemma='我',
            pos='r',
            head=2,
            deprel='nsubj'
        )
        self.assertEqual(node.id, 1)
        self.assertEqual(node.form, '我')
        self.assertEqual(node.pos, 'r')
        self.assertEqual(node.head, 2)
        self.assertEqual(node.deprel, 'nsubj')
    
    def test_node_is_root(self):
        root_node = DependencyNode(id=1, form='爱', head=0, deprel='root')
        non_root_node = DependencyNode(id=2, form='我', head=1, deprel='nsubj')
        
        self.assertTrue(root_node.is_root())
        self.assertFalse(non_root_node.is_root())
    
    def test_node_to_conllu(self):
        node = DependencyNode(
            id=1,
            form='我',
            lemma='我',
            pos='r',
            cpos='r',
            feats={'Case': 'Nom'},
            head=2,
            deprel='nsubj'
        )
        conllu = node.to_conllu()
        parts = conllu.split('\t')
        
        self.assertEqual(parts[0], '1')
        self.assertEqual(parts[1], '我')
        self.assertEqual(parts[2], '我')
        self.assertEqual(parts[4], 'r')
        self.assertIn('Case=Nom', parts[5])
        self.assertEqual(parts[6], '2')
        self.assertEqual(parts[7], 'nsubj')


class TestDependencyTree(unittest.TestCase):
    def test_tree_creation(self):
        tree = DependencyTree()
        self.assertEqual(len(tree), 0)
        self.assertEqual(len(tree.get_arcs()), 0)
    
    def test_add_node(self):
        tree = DependencyTree()
        node1 = DependencyNode(id=1, form='我', pos='r', head=2, deprel='nsubj')
        node2 = DependencyNode(id=2, form='爱', pos='v', head=0, deprel='root')
        node3 = DependencyNode(id=3, form='中国', pos='ns', head=2, deprel='obj')
        
        tree.add_node(node1)
        tree.add_node(node2)
        tree.add_node(node3)
        
        self.assertEqual(len(tree), 3)
        self.assertEqual(len(tree.get_arcs()), 3)
    
    def test_get_node(self):
        tree = DependencyTree()
        node = DependencyNode(id=1, form='我', pos='r', head=2, deprel='nsubj')
        tree.add_node(node)
        
        retrieved = tree.get_node(1)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.form, '我')
        
        self.assertIsNone(tree.get_node(999))
    
    def test_get_children(self):
        tree = DependencyTree()
        node1 = DependencyNode(id=1, form='我', pos='r', head=2, deprel='nsubj')
        node2 = DependencyNode(id=2, form='爱', pos='v', head=0, deprel='root')
        node3 = DependencyNode(id=3, form='中国', pos='ns', head=2, deprel='obj')
        
        tree.add_node(node1)
        tree.add_node(node2)
        tree.add_node(node3)
        
        children = tree.get_children(2)
        self.assertEqual(len(children), 2)
        
        child_ids = [c[0] for c in children]
        self.assertIn(1, child_ids)
        self.assertIn(3, child_ids)
    
    def test_get_head(self):
        tree = DependencyTree()
        node1 = DependencyNode(id=1, form='我', pos='r', head=2, deprel='nsubj')
        node2 = DependencyNode(id=2, form='爱', pos='v', head=0, deprel='root')
        
        tree.add_node(node1)
        tree.add_node(node2)
        
        head = tree.get_head(1)
        self.assertIsNotNone(head)
        self.assertEqual(head[0], 2)
        self.assertEqual(head[1], 'nsubj')
    
    def test_get_root(self):
        tree = DependencyTree()
        node1 = DependencyNode(id=1, form='我', pos='r', head=2, deprel='nsubj')
        node2 = DependencyNode(id=2, form='爱', pos='v', head=0, deprel='root')
        
        tree.add_node(node1)
        tree.add_node(node2)
        
        root = tree.get_root()
        self.assertIsNotNone(root)
        self.assertEqual(root.id, 2)
    
    def test_is_projective(self):
        tree = DependencyTree()
        node1 = DependencyNode(id=1, form='我', pos='r', head=2, deprel='nsubj')
        node2 = DependencyNode(id=2, form='爱', pos='v', head=0, deprel='root')
        node3 = DependencyNode(id=3, form='中国', pos='ns', head=2, deprel='obj')
        
        tree.add_node(node1)
        tree.add_node(node2)
        tree.add_node(node3)
        
        self.assertTrue(tree.is_projective())
    
    def test_is_tree(self):
        tree = DependencyTree()
        node1 = DependencyNode(id=1, form='我', pos='r', head=2, deprel='nsubj')
        node2 = DependencyNode(id=2, form='爱', pos='v', head=0, deprel='root')
        node3 = DependencyNode(id=3, form='中国', pos='ns', head=2, deprel='obj')
        
        tree.add_node(node1)
        tree.add_node(node2)
        tree.add_node(node3)
        
        self.assertTrue(tree.is_tree())
    
    def test_to_conllu(self):
        tree = DependencyTree()
        node1 = DependencyNode(id=1, form='我', pos='r', head=2, deprel='nsubj')
        node2 = DependencyNode(id=2, form='爱', pos='v', head=0, deprel='root')
        
        tree.add_node(node1)
        tree.add_node(node2)
        
        conllu = tree.to_conllu()
        lines = conllu.strip().split('\n')
        self.assertEqual(len(lines), 2)
    
    def test_from_conllu(self):
        conllu_str = """1\t我\t_\tr\tr\t_\t2\tnsubj\t_\t_
2\t爱\t_\tv\tv\t_\t0\troot\t_\t_
3\t中国\t_\tns\tns\t_\t2\tobj\t_\t_"""
        
        tree = DependencyTree.from_conllu(conllu_str)
        
        self.assertEqual(len(tree), 3)
        self.assertEqual(tree.get_node(1).form, '我')
        self.assertEqual(tree.get_node(2).form, '爱')
        self.assertEqual(tree.get_node(3).form, '中国')


class TestParserState(unittest.TestCase):
    def setUp(self):
        self.nodes = [
            DependencyNode(id=0, form='我', pos='r'),
            DependencyNode(id=1, form='爱', pos='v'),
            DependencyNode(id=2, form='中国', pos='ns'),
        ]
    
    def test_initial_state(self):
        state = ParserState(self.nodes)
        
        self.assertEqual(len(state.stack), 0)
        self.assertEqual(len(state.buffer), 3)
        self.assertEqual(len(state.arcs), 0)
    
    def test_shift(self):
        state = ParserState(self.nodes)
        
        self.assertTrue(state.can_shift())
        self.assertTrue(state.shift())
        
        self.assertEqual(len(state.stack), 1)
        self.assertEqual(len(state.buffer), 2)
        self.assertEqual(state.stack[-1], 0)
    
    def test_left_arc(self):
        state = ParserState(self.nodes)
        state.shift()
        
        self.assertTrue(state.can_left_arc())
        self.assertTrue(state.left_arc('nsubj'))
        
        self.assertEqual(len(state.stack), 0)
        self.assertEqual(len(state.arcs), 1)
        self.assertEqual(state.arcs[0].head, 1)
        self.assertEqual(state.arcs[0].dependent, 0)
    
    def test_right_arc(self):
        state = ParserState(self.nodes)
        state.shift()
        
        self.assertTrue(state.can_right_arc())
        self.assertTrue(state.right_arc('obj'))
        
        self.assertEqual(len(state.stack), 1)
        self.assertEqual(len(state.buffer), 1)
        self.assertEqual(len(state.arcs), 1)
    
    def test_reduce(self):
        state = ParserState(self.nodes)
        state.shift()
        state.right_arc('obj')
        
        self.assertTrue(state.can_reduce())
        self.assertTrue(state.reduce())
        
        self.assertEqual(len(state.stack), 0)
    
    def test_is_terminal(self):
        state = ParserState(self.nodes)
        self.assertFalse(state.is_terminal())
        
        while state.buffer:
            state.shift()
        
        self.assertFalse(state.is_terminal())
        
        while len(state.stack) > 1:
            if state.can_reduce():
                state.reduce()
            else:
                break
        
        state.stack = []
        self.assertTrue(state.is_terminal())
    
    def test_copy(self):
        state = ParserState(self.nodes)
        state.shift()
        
        copied = state.copy()
        
        self.assertEqual(len(copied.stack), len(state.stack))
        self.assertEqual(len(copied.buffer), len(state.buffer))
        
        copied.shift()
        self.assertNotEqual(len(copied.stack), len(state.stack))
    
    def test_to_tree(self):
        state = ParserState(self.nodes)
        state.shift()
        state.left_arc('nsubj')
        state.shift()
        state.right_arc('obj')
        
        tree = state.to_tree()
        
        self.assertEqual(len(tree), 3)


class TestArcEagerOracle(unittest.TestCase):
    def setUp(self):
        self.tree = DependencyTree()
        self.tree.add_node(DependencyNode(id=1, form='我', pos='r', head=2, deprel='nsubj'))
        self.tree.add_node(DependencyNode(id=2, form='爱', pos='v', head=0, deprel='root'))
        self.tree.add_node(DependencyNode(id=3, form='中国', pos='ns', head=2, deprel='obj'))
        
        self.oracle = ArcEagerOracle(self.tree)
    
    def test_get_gold_head(self):
        self.assertEqual(self.oracle.get_gold_head(0), 1)
        self.assertEqual(self.oracle.get_gold_head(1), None)
        self.assertEqual(self.oracle.get_gold_head(2), 1)
    
    def test_get_gold_relation(self):
        self.assertEqual(self.oracle.get_gold_relation(0), 'nsubj')
        self.assertEqual(self.oracle.get_gold_relation(2), 'obj')
    
    def test_get_next_action(self):
        nodes = [
            DependencyNode(id=0, form='我', pos='r'),
            DependencyNode(id=1, form='爱', pos='v'),
            DependencyNode(id=2, form='中国', pos='ns'),
        ]
        state = ParserState(nodes)
        
        action, relation = self.oracle.get_next_action(state)
        self.assertEqual(action, Transition.SHIFT)
        
        state.shift()
        action, relation = self.oracle.get_next_action(state)
        self.assertEqual(action, Transition.LEFT_ARC)
        self.assertEqual(relation, 'nsubj')


class TestDependencyFeatureExtractor(unittest.TestCase):
    def setUp(self):
        self.nodes = [
            DependencyNode(id=0, form='我', pos='r'),
            DependencyNode(id=1, form='爱', pos='v'),
            DependencyNode(id=2, form='中国', pos='ns'),
        ]
        self.extractor = DependencyFeatureExtractor()
    
    def test_extract_features(self):
        state = ParserState(self.nodes)
        state.shift()
        
        features = self.extractor.extract_features(state)
        
        self.assertIsInstance(features, list)
        self.assertTrue(len(features) > 0)
        
        feature_str = ' '.join(features)
        self.assertIn('S0_FORM:我', feature_str)
        self.assertIn('B0_FORM:爱', feature_str)
    
    def test_extract_features_with_pos(self):
        state = ParserState(self.nodes)
        state.shift()
        
        features = self.extractor.extract_features(state)
        feature_str = ' '.join(features)
        
        self.assertIn('S0_POS:r', feature_str)
        self.assertIn('B0_POS:v', feature_str)
    
    def test_extract_features_for_action(self):
        state = ParserState(self.nodes)
        state.shift()
        
        features = self.extractor.extract_features_for_action(
            state, Transition.LEFT_ARC, 'nsubj'
        )
        
        self.assertIsInstance(features, list)
        feature_str = ' '.join(features)
        self.assertIn('ACTION:LEFT_ARC', feature_str)
        self.assertIn('RELATION:nsubj', feature_str)


class TestDependencyParser(unittest.TestCase):
    def setUp(self):
        self.corpus = create_sample_dependency_corpus()
        self.parser = DependencyParser()
    
    def test_parser_creation(self):
        self.assertIsInstance(self.parser, DependencyParser)
        self.assertFalse(self.parser.is_trained())
    
    def test_train(self):
        self.parser.train(self.corpus, max_iter=5, verbose=False)
        
        self.assertTrue(self.parser.is_trained())
    
    def test_parse_without_training(self):
        with self.assertRaises(RuntimeError):
            self.parser.parse(['我', '爱', '中国'])
    
    def test_parse(self):
        self.parser.train(self.corpus, max_iter=10, verbose=False)
        
        tree = self.parser.parse(['我', '爱', '中国'])
        
        self.assertIsInstance(tree, DependencyTree)
        self.assertEqual(len(tree), 3)
    
    def test_parse_with_pos_tags(self):
        self.parser.train(self.corpus, max_iter=10, verbose=False)
        
        tree = self.parser.parse(
            ['我', '爱', '中国'],
            pos_tags=['r', 'v', 'ns']
        )
        
        self.assertIsInstance(tree, DependencyTree)
        self.assertEqual(len(tree), 3)
    
    def test_parse_empty(self):
        self.parser.train(self.corpus, max_iter=5, verbose=False)
        
        tree = self.parser.parse([])
        
        self.assertIsInstance(tree, DependencyTree)
        self.assertEqual(len(tree), 0)
    
    def test_parse_with_confidence(self):
        self.parser.train(self.corpus, max_iter=10, verbose=False)
        
        tree, scores = self.parser.parse_with_confidence(['我', '爱', '中国'])
        
        self.assertIsInstance(tree, DependencyTree)
        self.assertIsInstance(scores, list)
    
    def test_save_and_load_model(self):
        self.parser.train(self.corpus, max_iter=5, verbose=False)
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            temp_path = f.name
        
        try:
            self.parser.save_model(temp_path)
            
            new_parser = DependencyParser()
            new_parser.load_model(temp_path)
            
            self.assertTrue(new_parser.is_trained())
            
            tree = new_parser.parse(['我', '爱', '中国'])
            self.assertEqual(len(tree), 3)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_get_model_info(self):
        self.parser.train(self.corpus, max_iter=5, verbose=False)
        
        info = self.parser.get_model_info()
        
        self.assertTrue(info['trained'])
        self.assertIn('num_relations', info)
        self.assertIn('num_features', info)


class TestCreateSampleDependencyCorpus(unittest.TestCase):
    def test_create_sample_corpus(self):
        corpus = create_sample_dependency_corpus()
        
        self.assertIsInstance(corpus, list)
        self.assertTrue(len(corpus) > 0)
        
        for tree in corpus:
            self.assertIsInstance(tree, DependencyTree)
            self.assertTrue(len(tree) > 0)


class TestDependencyRelations(unittest.TestCase):
    def test_dependency_relations_dict(self):
        self.assertIsInstance(DEPENDENCY_RELATIONS, dict)
        self.assertIn('nsubj', DEPENDENCY_RELATIONS)
        self.assertIn('obj', DEPENDENCY_RELATIONS)
        self.assertIn('root', DEPENDENCY_RELATIONS)
    
    def test_default_relations_list(self):
        self.assertIsInstance(DEFAULT_RELATIONS, list)
        self.assertIn('nsubj', DEFAULT_RELATIONS)
        self.assertIn('root', DEFAULT_RELATIONS)


class TestTransition(unittest.TestCase):
    def test_transition_values(self):
        self.assertEqual(Transition.SHIFT.name, 'SHIFT')
        self.assertEqual(Transition.LEFT_ARC.name, 'LEFT_ARC')
        self.assertEqual(Transition.RIGHT_ARC.name, 'RIGHT_ARC')
        self.assertEqual(Transition.REDUCE.name, 'REDUCE')


if __name__ == '__main__':
    unittest.main()
