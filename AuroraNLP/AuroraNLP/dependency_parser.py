from typing import List, Tuple, Dict, Optional, Iterator
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
import pickle
import os


class Transition(Enum):
    SHIFT = auto()
    LEFT_ARC = auto()
    RIGHT_ARC = auto()
    REDUCE = auto()


DEPENDENCY_RELATIONS = {
    'root': '根节点',
    'nsubj': '名词性主语',
    'obj': '宾语',
    'iobj': '间接宾语',
    'csubj': '从句主语',
    'ccomp': '从句补语',
    'xcomp': '开放性从句补语',
    'obl': '斜修饰语',
    'vocative': '呼格',
    'expl': '形式主语',
    'dislocated': '移位成分',
    'advcl': '状语从句',
    'advmod': '状语修饰语',
    'discourse': '话语标记',
    'aux': '助动词',
    'cop': '系动词',
    'mark': '标记词',
    'nmod': '名词修饰语',
    'appos': '同位语',
    'nummod': '数词修饰语',
    'amod': '形容词修饰语',
    'det': '限定词',
    'clf': '量词',
    'case': '格标记',
    'conj': '并列连接',
    'cc': '并列连词',
    'fixed': '固定搭配',
    'flat': '扁平结构',
    'compound': '复合词',
    'list': '列表',
    'parataxis': '意合',
    'orphan': '孤立项',
    'goeswith': '连写',
    'reparandum': '修正',
    'punct': '标点',
    'dep': '未指定依存',
    'sbj': '主语',
    'vmod': '动词修饰语',
    'attr': '属性',
    'coo': '并列',
    'rad': '附加',
    'wp': '标点符号',
}

DEFAULT_RELATIONS = list(DEPENDENCY_RELATIONS.keys())


@dataclass
class DependencyArc:
    head: int
    dependent: int
    relation: str
    
    def __repr__(self) -> str:
        return f"DependencyArc(head={self.head}, dep={self.dependent}, rel='{self.relation}')"
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, DependencyArc):
            return False
        return (self.head == other.head and 
                self.dependent == other.dependent and 
                self.relation == other.relation)
    
    def __hash__(self) -> int:
        return hash((self.head, self.dependent, self.relation))


@dataclass
class DependencyNode:
    id: int
    form: str
    lemma: str = ''
    pos: str = ''
    cpos: str = ''
    feats: Dict[str, str] = field(default_factory=dict)
    head: int = -1
    deprel: str = ''
    deps: List[Tuple[int, str]] = field(default_factory=list)
    misc: str = ''
    
    def __repr__(self) -> str:
        return f"DependencyNode(id={self.id}, form='{self.form}', pos='{self.pos}', head={self.head}, deprel='{self.deprel}')"
    
    def is_root(self) -> bool:
        return self.head == 0
    
    def to_conllu(self) -> str:
        feats_str = '|'.join(f"{k}={v}" for k, v in self.feats.items()) if self.feats else '_'
        deps_str = '|'.join(f"{h}:{r}" for h, r in self.deps) if self.deps else '_'
        head_str = str(self.head) if self.head >= 0 else '_'
        return "\t".join([str(self.id), self.form, self.lemma or '_', self.cpos or self.pos or '_', self.pos or '_', feats_str, head_str, self.deprel or '_', deps_str, self.misc or '_'])


class DependencyTree:
    def __init__(self, nodes: Optional[List[DependencyNode]] = None):
        self.nodes: List[DependencyNode] = nodes if nodes else []
        self._build_indices()
    
    def _build_indices(self):
        self._head_to_dependents: Dict[int, List[Tuple[int, str]]] = defaultdict(list)
        for node in self.nodes:
            if node.head >= 0:
                self._head_to_dependents[node.head].append((node.id, node.deprel))
    
    def add_node(self, node: DependencyNode):
        self.nodes.append(node)
        if node.head >= 0:
            self._head_to_dependents[node.head].append((node.id, node.deprel))
    
    def get_node(self, node_id: int) -> Optional[DependencyNode]:
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None
    
    def get_children(self, node_id: int) -> List[Tuple[int, str]]:
        return self._head_to_dependents.get(node_id, [])
    
    def get_head(self, node_id: int) -> Optional[Tuple[int, str]]:
        node = self.get_node(node_id)
        if node and node.head >= 0:
            return (node.head, node.deprel)
        return None
    
    def get_root(self) -> Optional[DependencyNode]:
        for node in self.nodes:
            if node.head == 0:
                return node
        return None
    
    def get_arcs(self) -> List[DependencyArc]:
        arcs = []
        for node in self.nodes:
            if node.head >= 0:
                arcs.append(DependencyArc(node.head, node.id, node.deprel))
        return arcs
    
    def is_projective(self) -> bool:
        arcs = self.get_arcs()
        for arc1 in arcs:
            for arc2 in arcs:
                if arc1 == arc2:
                    continue
                
                h1, d1 = arc1.head, arc1.dependent
                h2, d2 = arc2.head, arc2.dependent
                
                if h1 == h2 or d1 == d2:
                    continue
                
                min1, max1 = min(h1, d1), max(h1, d1)
                min2, max2 = min(h2, d2), max(h2, d2)
                
                if min1 < min2 < max1 < max2:
                    return False
                if min2 < min1 < max2 < max1:
                    return False
        
        return True
    
    def is_tree(self) -> bool:
        if not self.nodes:
            return True
        
        roots = [n for n in self.nodes if n.head == 0]
        if len(roots) != 1:
            return False
        
        arcs = self.get_arcs()
        if len(arcs) != len({arc.dependent for arc in arcs}):
            return False
        
        visited = set()
        queue = [roots[0].id]
        
        while queue:
            current = queue.pop(0)
            if current in visited:
                return False
            visited.add(current)
            
            for child_id, _ in self.get_children(current):
                queue.append(child_id)
        
        return len(visited) == len(self.nodes)
    
    def to_conllu(self) -> str:
        lines = []
        for node in sorted(self.nodes, key=lambda n: n.id):
            lines.append(node.to_conllu())
        return '\n'.join(lines)
    
    @classmethod
    def from_conllu(cls, conllu_str: str) -> 'DependencyTree':
        nodes = []
        for line in conllu_str.strip().split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split('\t')
            if len(parts) < 8:
                continue
            
            id_str = parts[0]
            if '-' in id_str or '.' in id_str:
                continue
            
            try:
                node_id = int(id_str)
            except ValueError:
                continue
            
            form = parts[1]
            lemma = parts[2] if parts[2] != '_' else ''
            cpos = parts[3] if parts[3] != '_' else ''
            pos = parts[4] if parts[4] != '_' else ''
            
            feats = {}
            if parts[5] != '_':
                for feat in parts[5].split('|'):
                    if '=' in feat:
                        k, v = feat.split('=', 1)
                        feats[k] = v
            
            head = int(parts[6]) if parts[6] != '_' else -1
            deprel = parts[7] if parts[7] != '_' else ''
            
            deps = []
            if len(parts) > 8 and parts[8] != '_':
                for dep in parts[8].split('|'):
                    if ':' in dep:
                        h, r = dep.split(':', 1)
                        deps.append((int(h), r))
            
            misc = parts[9] if len(parts) > 9 and parts[9] != '_' else ''
            
            node = DependencyNode(
                id=node_id,
                form=form,
                lemma=lemma,
                pos=pos,
                cpos=cpos,
                feats=feats,
                head=head,
                deprel=deprel,
                deps=deps,
                misc=misc
            )
            nodes.append(node)
        
        return cls(nodes)
    
    def __repr__(self) -> str:
        return f"DependencyTree(nodes={len(self.nodes)}, arcs={len(self.get_arcs())})"
    
    def __len__(self) -> int:
        return len(self.nodes)
    
    def __iter__(self) -> Iterator[DependencyNode]:
        return iter(self.nodes)


class ParserState:
    def __init__(self, nodes: List[DependencyNode]):
        self.nodes = nodes
        self.stack: List[int] = []
        self.buffer: List[int] = list(range(len(nodes)))
        self.arcs: List[DependencyArc] = []
        self._heads: Dict[int, int] = {}
    
    def copy(self) -> 'ParserState':
        new_state = ParserState(self.nodes)
        new_state.stack = self.stack.copy()
        new_state.buffer = self.buffer.copy()
        new_state.arcs = self.arcs.copy()
        new_state._heads = self._heads.copy()
        return new_state
    
    def is_terminal(self) -> bool:
        return len(self.buffer) == 0 and len(self.stack) <= 1
    
    def can_shift(self) -> bool:
        return len(self.buffer) > 0
    
    def can_left_arc(self) -> bool:
        return len(self.stack) > 0 and len(self.buffer) > 0 and self.stack[-1] not in self._heads
    
    def can_right_arc(self) -> bool:
        return len(self.stack) > 0 and len(self.buffer) > 0
    
    def can_reduce(self) -> bool:
        return len(self.stack) > 0 and self.stack[-1] in self._heads
    
    def shift(self) -> bool:
        if not self.can_shift():
            return False
        self.stack.append(self.buffer.pop(0))
        return True
    
    def left_arc(self, relation: str) -> bool:
        if not self.can_left_arc():
            return False
        dependent = self.stack.pop()
        head = self.buffer[0]
        self.arcs.append(DependencyArc(head, dependent, relation))
        self._heads[dependent] = head
        return True
    
    def right_arc(self, relation: str) -> bool:
        if not self.can_right_arc():
            return False
        head = self.stack.pop()
        dependent = self.buffer.pop(0)
        self.arcs.append(DependencyArc(head, dependent, relation))
        self._heads[dependent] = head
        self.stack.append(dependent)
        return True
    
    def reduce(self) -> bool:
        if not self.can_reduce():
            return False
        self.stack.pop()
        return True
    
    def get_node(self, idx: int) -> Optional[DependencyNode]:
        if 0 <= idx < len(self.nodes):
            return self.nodes[idx]
        return None
    
    def get_stack_node(self, offset: int = 0) -> Optional[DependencyNode]:
        idx = len(self.stack) - 1 - offset
        if 0 <= idx < len(self.stack):
            return self.get_node(self.stack[idx])
        return None
    
    def get_buffer_node(self, offset: int = 0) -> Optional[DependencyNode]:
        if offset < len(self.buffer):
            return self.get_node(self.buffer[offset])
        return None
    
    def get_head(self, node_id: int) -> Optional[int]:
        return self._heads.get(node_id)
    
    def to_tree(self) -> DependencyTree:
        tree = DependencyTree()
        for node in self.nodes:
            new_node = DependencyNode(
                id=node.id + 1,
                form=node.form,
                lemma=node.lemma,
                pos=node.pos,
                cpos=node.cpos,
                feats=node.feats.copy(),
                head=0,
                deprel='dep',
                deps=[],
                misc=node.misc
            )
            tree.add_node(new_node)
        
        for arc in self.arcs:
            dep_node = tree.get_node(arc.dependent + 1)
            if dep_node:
                dep_node.head = arc.head + 1
                dep_node.deprel = arc.relation
        
        if len(self.stack) > 1:
            root_candidate = self.stack[-1]
            for stack_item in self.stack[:-1]:
                if stack_item not in self._heads:
                    dep_node = tree.get_node(stack_item + 1)
                    if dep_node:
                        dep_node.head = root_candidate + 1
                        dep_node.deprel = 'dep'
        
        return tree


class ArcEagerOracle:
    def __init__(self, gold_tree: DependencyTree):
        self.gold_tree = gold_tree
        self._gold_heads: Dict[int, int] = {}
        self._gold_rels: Dict[int, str] = {}
        
        for node in gold_tree.nodes:
            if node.head > 0:
                self._gold_heads[node.id - 1] = node.head - 1
                self._gold_rels[node.id - 1] = node.deprel
    
    def get_gold_head(self, node_id: int) -> Optional[int]:
        return self._gold_heads.get(node_id)
    
    def get_gold_relation(self, node_id: int) -> str:
        return self._gold_rels.get(node_id, 'dep')
    
    def has_head(self, node_id: int) -> bool:
        return node_id in self._gold_heads
    
    def get_next_action(self, state: ParserState) -> Tuple[Transition, str]:
        if len(state.stack) == 0:
            return (Transition.SHIFT, '')
        
        s0 = state.stack[-1]
        
        if len(state.buffer) > 0:
            b0 = state.buffer[0]
            
            if self.get_gold_head(s0) == b0 and s0 not in state._heads:
                return (Transition.LEFT_ARC, self.get_gold_relation(s0))
            
            if self.get_gold_head(b0) == s0:
                return (Transition.RIGHT_ARC, self.get_gold_relation(b0))
        
        if s0 in state._heads:
            has_unprocessed_children = False
            for node_id in self._gold_heads:
                if self._gold_heads[node_id] == s0 and node_id not in state._heads:
                    has_unprocessed_children = True
                    break
            
            if not has_unprocessed_children:
                return (Transition.REDUCE, '')
        
        return (Transition.SHIFT, '')
    
    def is_valid_action(self, state: ParserState, action: Transition) -> bool:
        if action == Transition.SHIFT:
            return state.can_shift()
        elif action == Transition.LEFT_ARC:
            return state.can_left_arc()
        elif action == Transition.RIGHT_ARC:
            return state.can_right_arc()
        elif action == Transition.REDUCE:
            return state.can_reduce()
        return False


class DependencyFeatureExtractor:
    def __init__(self):
        self.use_pos = True
        self.use_deprel = True
        self.use_distance = True
    
    def extract_features(self, state: ParserState) -> List[str]:
        features = []
        
        s0 = state.get_stack_node(0)
        s1 = state.get_stack_node(1)
        s2 = state.get_stack_node(2)
        
        b0 = state.get_buffer_node(0)
        b1 = state.get_buffer_node(1)
        b2 = state.get_buffer_node(2)
        
        if s0:
            features.append(f"S0_FORM:{s0.form}")
            if self.use_pos and s0.pos:
                features.append(f"S0_POS:{s0.pos}")
        
        if s1:
            features.append(f"S1_FORM:{s1.form}")
            if self.use_pos and s1.pos:
                features.append(f"S1_POS:{s1.pos}")
        
        if s0 and s1:
            features.append(f"S0_S1_FORM:{s0.form}_{s1.form}")
            if self.use_pos and s0.pos and s1.pos:
                features.append(f"S0_S1_POS:{s0.pos}_{s1.pos}")
        
        if b0:
            features.append(f"B0_FORM:{b0.form}")
            if self.use_pos and b0.pos:
                features.append(f"B0_POS:{b0.pos}")
        
        if b1:
            features.append(f"B1_FORM:{b1.form}")
            if self.use_pos and b1.pos:
                features.append(f"B1_POS:{b1.pos}")
        
        if b0 and b1:
            features.append(f"B0_B1_FORM:{b0.form}_{b1.form}")
            if self.use_pos and b0.pos and b1.pos:
                features.append(f"B0_B1_POS:{b0.pos}_{b1.pos}")
        
        if s0 and b0:
            features.append(f"S0_B0_FORM:{s0.form}_{b0.form}")
            if self.use_pos and s0.pos and b0.pos:
                features.append(f"S0_B0_POS:{s0.pos}_{b0.pos}")
        
        if s0:
            s0_head_id = state.get_head(state.stack[-1])
            if s0_head_id is not None:
                s0_head = state.get_node(s0_head_id)
                if s0_head:
                    features.append(f"S0_HEAD_FORM:{s0_head.form}")
                    if self.use_pos and s0_head.pos:
                        features.append(f"S0_HEAD_POS:{s0_head.pos}")
        
        if s0:
            children = state.arcs
            s0_children = [(a.dependent, a.relation) for a in children if a.head == state.stack[-1]]
            if s0_children:
                leftmost = min(s0_children, key=lambda x: x[0])
                rightmost = max(s0_children, key=lambda x: x[0])
                
                leftmost_node = state.get_node(leftmost[0])
                rightmost_node = state.get_node(rightmost[0])
                
                if leftmost_node:
                    features.append(f"S0_LM_FORM:{leftmost_node.form}")
                    if self.use_deprel:
                        features.append(f"S0_LM_DEPREL:{leftmost[1]}")
                
                if rightmost_node:
                    features.append(f"S0_RM_FORM:{rightmost_node.form}")
                    if self.use_deprel:
                        features.append(f"S0_RM_DEPREL:{rightmost[1]}")
        
        if s0:
            features.append(f"S0_LEN:{len(s0.form)}")
            if s0.form:
                features.append(f"S0_FIRST:{s0.form[0]}")
                features.append(f"S0_LAST:{s0.form[-1]}")
        
        if b0:
            features.append(f"B0_LEN:{len(b0.form)}")
            if b0.form:
                features.append(f"B0_FIRST:{b0.form[0]}")
                features.append(f"B0_LAST:{b0.form[-1]}")
        
        if self.use_distance and s0 and b0:
            dist = abs(state.stack[-1] - state.buffer[0])
            features.append(f"DIST:{min(dist, 5)}")
        
        features.append(f"STACK_LEN:{min(len(state.stack), 5)}")
        features.append(f"BUFFER_LEN:{min(len(state.buffer), 5)}")
        
        return features
    
    def extract_features_for_action(
        self, 
        state: ParserState, 
        action: Transition, 
        relation: str = ''
    ) -> List[str]:
        base_features = self.extract_features(state)
        
        action_features = [f"ACTION:{action.name}"]
        if relation:
            action_features.append(f"RELATION:{relation}")
        
        return base_features + action_features


class DependencyParser:
    def __init__(
        self, 
        relations: Optional[List[str]] = None,
        use_pos: bool = True
    ):
        self.relations = relations or DEFAULT_RELATIONS.copy()
        self.use_pos = use_pos
        self.feature_extractor = DependencyFeatureExtractor()
        self.feature_extractor.use_pos = use_pos
        
        self.weights: Dict[str, float] = defaultdict(float)
        self._trained = False
        self._learning_rate = 1.0
        self._max_iter = 20
        self._averaged_weights: Dict[str, float] = defaultdict(float)
        self._weight_updates: Dict[str, int] = defaultdict(int)
        self._total_updates = 0
    
    def _get_score(self, features: List[str]) -> float:
        return sum(self.weights.get(f, 0.0) for f in features)
    
    def _get_possible_actions(self, state: ParserState) -> List[Tuple[Transition, str]]:
        actions = []
        
        if state.can_left_arc():
            for rel in self.relations:
                actions.append((Transition.LEFT_ARC, rel))
        
        if state.can_right_arc():
            for rel in self.relations:
                actions.append((Transition.RIGHT_ARC, rel))
        
        if state.can_reduce():
            actions.append((Transition.REDUCE, ''))
        
        if state.can_shift():
            actions.append((Transition.SHIFT, ''))
        
        return actions
    
    def _get_best_action(
        self, 
        state: ParserState
    ) -> Tuple[Transition, str]:
        actions = self._get_possible_actions(state)
        
        if not actions:
            return (Transition.SHIFT, '')
        
        best_action = actions[0]
        best_score = float('-inf')
        
        for action, relation in actions:
            features = self.feature_extractor.extract_features_for_action(
                state, action, relation
            )
            score = self._get_score(features)
            
            if score > best_score:
                best_score = score
                best_action = (action, relation)
        
        return best_action
    
    def _update_weights(
        self, 
        features: List[str], 
        delta: float
    ):
        for feature in features:
            old_weight = self.weights.get(feature, 0.0)
            self.weights[feature] = old_weight + delta
            
            if feature not in self._averaged_weights:
                self._averaged_weights[feature] = 0.0
            self._averaged_weights[feature] += delta * self._total_updates
            
            self._weight_updates[feature] += 1
        
        self._total_updates += 1
    
    def _finalize_averaged_weights(self):
        for feature in self.weights:
            if self._weight_updates[feature] > 0:
                avg_update = self._averaged_weights.get(feature, 0.0) / self._total_updates
                self._averaged_weights[feature] = self.weights[feature] - avg_update
    
    def train(
        self,
        corpus: List[DependencyTree],
        max_iter: int = 20,
        learning_rate: float = 1.0,
        verbose: bool = True
    ):
        if not corpus:
            raise ValueError("Training corpus cannot be empty")
        
        projective_trees = [t for t in corpus if t.is_projective()]
        if not projective_trees:
            raise ValueError("No projective trees found in corpus. Arc-eager parser requires projective trees.")
        
        if len(projective_trees) < len(corpus) and verbose:
            skipped = len(corpus) - len(projective_trees)
            print(f"Warning: Skipped {skipped} non-projective tree(s)")
        
        self._max_iter = max_iter
        self._learning_rate = learning_rate
        self.weights.clear()
        self._averaged_weights.clear()
        self._weight_updates.clear()
        self._total_updates = 0
        
        for iteration in range(max_iter):
            correct = 0
            total = 0
            
            for tree in projective_trees:
                nodes = [DependencyNode(
                    id=i,
                    form=n.form,
                    lemma=n.lemma,
                    pos=n.pos if self.use_pos else '',
                    cpos=n.cpos,
                    feats=n.feats,
                    head=-1,
                    deprel='',
                    deps=[],
                    misc=n.misc
                ) for i, n in enumerate(sorted(tree.nodes, key=lambda x: x.id))]
                
                state = ParserState(nodes)
                oracle = ArcEagerOracle(tree)
                
                while not state.is_terminal():
                    gold_action, gold_relation = oracle.get_next_action(state)
                    
                    if not oracle.is_valid_action(state, gold_action):
                        break
                    
                    pred_action, pred_relation = self._get_best_action(state)
                    
                    if pred_action != gold_action or pred_relation != gold_relation:
                        gold_features = self.feature_extractor.extract_features_for_action(
                            state, gold_action, gold_relation
                        )
                        pred_features = self.feature_extractor.extract_features_for_action(
                            state, pred_action, pred_relation
                        )
                        
                        self._update_weights(gold_features, learning_rate)
                        self._update_weights(pred_features, -learning_rate)
                    else:
                        correct += 1
                    total += 1
                    
                    if gold_action == Transition.SHIFT:
                        state.shift()
                    elif gold_action == Transition.LEFT_ARC:
                        state.left_arc(gold_relation)
                    elif gold_action == Transition.RIGHT_ARC:
                        state.right_arc(gold_relation)
                    elif gold_action == Transition.REDUCE:
                        state.reduce()
            
            if verbose:
                accuracy = correct / total if total > 0 else 0
                print(f"Iteration {iteration + 1}/{max_iter}, Accuracy: {accuracy:.4f}")
        
        self._finalize_averaged_weights()
        self.weights = defaultdict(float, self._averaged_weights)
        self._trained = True
    
    def parse(
        self, 
        words: List[str], 
        pos_tags: Optional[List[str]] = None
    ) -> DependencyTree:
        if not words:
            return DependencyTree()
        
        if not self._trained:
            raise RuntimeError("Model has not been trained. Call train() first.")
        
        nodes = []
        for i, word in enumerate(words):
            pos = pos_tags[i] if pos_tags and i < len(pos_tags) else ''
            node = DependencyNode(
                id=i,
                form=word,
                lemma='',
                pos=pos,
                cpos='',
                feats={},
                head=-1,
                deprel='',
                deps=[],
                misc=''
            )
            nodes.append(node)
        
        state = ParserState(nodes)
        
        max_steps = len(words) * 4 + 10
        steps = 0
        
        while not state.is_terminal() and steps < max_steps:
            action, relation = self._get_best_action(state)
            
            if action == Transition.SHIFT:
                if not state.shift():
                    break
            elif action == Transition.LEFT_ARC:
                if not state.left_arc(relation):
                    state.shift()
            elif action == Transition.RIGHT_ARC:
                if not state.right_arc(relation):
                    state.shift()
            elif action == Transition.REDUCE:
                if not state.reduce():
                    state.shift()
            
            steps += 1
        
        return state.to_tree()
    
    def parse_with_confidence(
        self, 
        words: List[str], 
        pos_tags: Optional[List[str]] = None
    ) -> Tuple[DependencyTree, List[float]]:
        if not words:
            return DependencyTree(), []
        
        if not self._trained:
            raise RuntimeError("Model has not been trained. Call train() first.")
        
        nodes = []
        for i, word in enumerate(words):
            pos = pos_tags[i] if pos_tags and i < len(pos_tags) else ''
            node = DependencyNode(
                id=i,
                form=word,
                lemma='',
                pos=pos,
                cpos='',
                feats={},
                head=-1,
                deprel='',
                deps=[],
                misc=''
            )
            nodes.append(node)
        
        state = ParserState(nodes)
        scores = []
        
        max_steps = len(words) * 4 + 10
        steps = 0
        
        while not state.is_terminal() and steps < max_steps:
            action, relation = self._get_best_action(state)
            
            features = self.feature_extractor.extract_features_for_action(
                state, action, relation
            )
            score = self._get_score(features)
            scores.append(score)
            
            if action == Transition.SHIFT:
                if not state.shift():
                    break
            elif action == Transition.LEFT_ARC:
                if not state.left_arc(relation):
                    state.shift()
            elif action == Transition.RIGHT_ARC:
                if not state.right_arc(relation):
                    state.shift()
            elif action == Transition.REDUCE:
                if not state.reduce():
                    state.shift()
            
            steps += 1
        
        return state.to_tree(), scores
    
    def save_model(self, filepath: str):
        if not self._trained:
            raise RuntimeError("Model has not been trained. Call train() first before saving.")
        
        model_data = {
            'relations': self.relations,
            'use_pos': self.use_pos,
            'weights': dict(self.weights),
            'learning_rate': self._learning_rate,
            'max_iter': self._max_iter,
            'trained': self._trained
        }
        
        dir_path = os.path.dirname(filepath)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str):
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.relations = model_data['relations']
        self.use_pos = model_data['use_pos']
        self.weights = defaultdict(float, model_data['weights'])
        self._learning_rate = model_data['learning_rate']
        self._max_iter = model_data['max_iter']
        self._trained = model_data['trained']
        
        self.feature_extractor.use_pos = self.use_pos
    
    def is_trained(self) -> bool:
        return self._trained
    
    def get_model_info(self) -> Dict[str, any]:
        if not self._trained:
            return {'trained': False}
        
        return {
            'trained': True,
            'num_relations': len(self.relations),
            'relations': self.relations,
            'num_features': len(self.weights),
            'use_pos': self.use_pos,
            'learning_rate': self._learning_rate,
            'max_iter': self._max_iter
        }


def create_sample_dependency_corpus() -> List[DependencyTree]:
    corpus = []
    
    tree1 = DependencyTree()
    tree1.add_node(DependencyNode(id=1, form='我', pos='r', head=2, deprel='nsubj'))
    tree1.add_node(DependencyNode(id=2, form='爱', pos='v', head=0, deprel='root'))
    tree1.add_node(DependencyNode(id=3, form='中国', pos='ns', head=2, deprel='obj'))
    corpus.append(tree1)
    
    tree2 = DependencyTree()
    tree2.add_node(DependencyNode(id=1, form='他', pos='r', head=2, deprel='nsubj'))
    tree2.add_node(DependencyNode(id=2, form='吃', pos='v', head=0, deprel='root'))
    tree2.add_node(DependencyNode(id=3, form='了', pos='u', head=2, deprel='aux'))
    tree2.add_node(DependencyNode(id=4, form='一个', pos='m', head=5, deprel='nummod'))
    tree2.add_node(DependencyNode(id=5, form='苹果', pos='n', head=2, deprel='obj'))
    corpus.append(tree2)
    
    tree3 = DependencyTree()
    tree3.add_node(DependencyNode(id=1, form='北京', pos='ns', head=2, deprel='nsubj'))
    tree3.add_node(DependencyNode(id=2, form='是', pos='v', head=0, deprel='root'))
    tree3.add_node(DependencyNode(id=3, form='中国', pos='ns', head=4, deprel='nmod'))
    tree3.add_node(DependencyNode(id=4, form='首都', pos='n', head=2, deprel='attr'))
    corpus.append(tree3)
    
    tree4 = DependencyTree()
    tree4.add_node(DependencyNode(id=1, form='小明', pos='nr', head=2, deprel='nsubj'))
    tree4.add_node(DependencyNode(id=2, form='在', pos='p', head=0, deprel='root'))
    tree4.add_node(DependencyNode(id=3, form='学校', pos='n', head=2, deprel='obj'))
    tree4.add_node(DependencyNode(id=4, form='读书', pos='v', head=2, deprel='vmod'))
    corpus.append(tree4)
    
    tree5 = DependencyTree()
    tree5.add_node(DependencyNode(id=1, form='这', pos='r', head=2, deprel='det'))
    tree5.add_node(DependencyNode(id=2, form='本书', pos='n', head=3, deprel='nsubj'))
    tree5.add_node(DependencyNode(id=3, form='很', pos='d', head=0, deprel='root'))
    tree5.add_node(DependencyNode(id=4, form='有趣', pos='a', head=3, deprel='advmod'))
    corpus.append(tree5)
    
    return corpus


def train_dependency_parser_from_file(
    parser: DependencyParser,
    filepath: str,
    encoding: str = 'utf-8'
) -> None:
    corpus = []
    
    with open(filepath, 'r', encoding=encoding) as f:
        content = f.read()
    
    sentences = content.strip().split('\n\n')
    
    for sentence in sentences:
        if sentence.strip():
            tree = DependencyTree.from_conllu(sentence)
            if len(tree) > 0:
                corpus.append(tree)
    
    if corpus:
        parser.train(corpus)


__all__ = [
    'Transition',
    'DEPENDENCY_RELATIONS',
    'DEFAULT_RELATIONS',
    'DependencyArc',
    'DependencyNode',
    'DependencyTree',
    'ParserState',
    'ArcEagerOracle',
    'DependencyFeatureExtractor',
    'DependencyParser',
    'create_sample_dependency_corpus',
    'train_dependency_parser_from_file',
]
