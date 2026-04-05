from typing import List, Tuple, Dict, Optional, Iterator, Set, Union
from collections import defaultdict
from dataclasses import dataclass, field
import math
import pickle
import os
import re


CONSTITUENT_LABELS = {
    'S': '句子',
    'NP': '名词短语',
    'VP': '动词短语',
    'PP': '介词短语',
    'ADJP': '形容词短语',
    'ADVP': '副词短语',
    'QP': '量词短语',
    'CP': '从句',
    'IP': '简单句',
    'DP': '限定词短语',
    'LCP': '方位短语',
    'DNP': '的字短语',
    'DVP': '地字短语',
    'M': '数词短语',
    'PRN': '插入语',
    'INTJ': '感叹词',
    'FRAG': '片段',
    'ROOT': '根节点',
    'TOP': '顶层节点',
}

POS_LABELS = {
    'NN': '普通名词',
    'NR': '专有名词',
    'NT': '时间名词',
    'VV': '动词',
    'VA': '形容词动词',
    'VC': '系动词',
    'VE': '存在动词',
    'AD': '副词',
    'JJ': '形容词',
    'DT': '限定词',
    'CD': '基数词',
    'OD': '序数词',
    'M': '量词',
    'PN': '代词',
    'P': '介词',
    'CC': '并列连词',
    'CS': '从属连词',
    'DEC': '的(补语标记)',
    'DEG': '的(领属标记)',
    'DER': '得(补语标记)',
    'DEV': '地(状语标记)',
    'SP': '句末助词',
    'AS': '动态助词',
    'ETC': '等',
    'MSP': '其他助词',
    'IJ': '感叹词',
    'ON': '拟声词',
    'PU': '标点',
    'FW': '外来词',
    'LB': '被字句标记',
    'SB': '被字句动词',
    'BA': '把字句标记',
}

DEFAULT_NON_TERMINALS = list(CONSTITUENT_LABELS.keys())
DEFAULT_TERMINALS = list(POS_LABELS.keys())


@dataclass
class GrammarRule:
    lhs: str
    rhs: Tuple[str, ...]
    probability: float = 1.0
    count: int = 0
    
    def __repr__(self) -> str:
        rhs_str = ' '.join(self.rhs)
        return f"GrammarRule({self.lhs} -> {rhs_str}, prob={self.probability:.4f})"
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, GrammarRule):
            return False
        return self.lhs == other.lhs and self.rhs == other.rhs
    
    def __hash__(self) -> int:
        return hash((self.lhs, self.rhs))
    
    def is_unary(self) -> bool:
        return len(self.rhs) == 1
    
    def is_binary(self) -> bool:
        return len(self.rhs) == 2
    
    def is_terminal(self) -> bool:
        if len(self.rhs) != 1:
            return False
        symbol = self.rhs[0]
        if symbol.isupper() and symbol.isalpha():
            return False
        return True
    
    def to_string(self) -> str:
        return f"{self.lhs} -> {' '.join(self.rhs)}"
    
    @classmethod
    def from_string(cls, rule_str: str, probability: float = 1.0) -> 'GrammarRule':
        parts = rule_str.split('->')
        if len(parts) != 2:
            raise ValueError(f"Invalid rule format: {rule_str}")
        
        lhs = parts[0].strip()
        rhs = tuple(parts[1].strip().split())
        
        return cls(lhs=lhs, rhs=rhs, probability=probability)


@dataclass
class ConstituentNode:
    label: str
    children: List['ConstituentNode'] = field(default_factory=list)
    word: str = ''
    start: int = 0
    end: int = 0
    
    def __repr__(self) -> str:
        if self.is_terminal():
            return f"ConstituentNode({self.label}, word='{self.word}')"
        return f"ConstituentNode({self.label}, children={len(self.children)})"
    
    def is_terminal(self) -> bool:
        return len(self.children) == 0
    
    def is_preterminal(self) -> bool:
        return len(self.children) == 1 and self.children[0].is_terminal()
    
    def get_words(self) -> List[str]:
        if self.is_terminal():
            return [self.word]
        words = []
        for child in self.children:
            words.extend(child.get_words())
        return words
    
    def get_text(self) -> str:
        return ''.join(self.get_words())
    
    def get_yield(self) -> List['ConstituentNode']:
        if self.is_terminal():
            return [self]
        result = []
        for child in self.children:
            result.extend(child.get_yield())
        return result
    
    def get_preterminals(self) -> List['ConstituentNode']:
        if self.is_preterminal():
            return [self]
        result = []
        for child in self.children:
            result.extend(child.get_preterminals())
        return result
    
    def get_height(self) -> int:
        if self.is_terminal():
            return 0
        return 1 + max(child.get_height() for child in self.children)
    
    def get_span(self) -> Tuple[int, int]:
        return (self.start, self.end)
    
    def to_penn_treebank(self, indent: int = 0) -> str:
        if self.is_terminal():
            return ' ' * indent + self.word
        
        if self.is_preterminal():
            return ' ' * indent + f"({self.label} {self.children[0].word})"
        
        lines = [' ' * indent + f"({self.label}"]
        for child in self.children:
            child_str = child.to_penn_treebank(indent + 2)
            if child_str.strip():
                lines.append(child_str)
        lines.append(' ' * indent + ')')
        
        return '\n'.join(lines)
    
    def to_lisp_string(self) -> str:
        if self.is_terminal():
            return self.word
        if self.is_preterminal():
            return f"({self.label} {self.children[0].word})"
        children_str = ' '.join(child.to_lisp_string() for child in self.children)
        return f"({self.label} {children_str})"
    
    def get_subtrees(self, label: Optional[str] = None) -> List['ConstituentNode']:
        result = []
        if label is None or self.label == label:
            result.append(self)
        for child in self.children:
            result.extend(child.get_subtrees(label))
        return result
    
    def get_phrases(self, phrase_type: str) -> List['ConstituentNode']:
        return self.get_subtrees(phrase_type)
    
    def get_noun_phrases(self) -> List['ConstituentNode']:
        return self.get_phrases('NP')
    
    def get_verb_phrases(self) -> List['ConstituentNode']:
        return self.get_phrases('VP')
    
    def get_prep_phrases(self) -> List['ConstituentNode']:
        return self.get_phrases('PP')
    
    def find_node_at(self, start: int, end: int) -> Optional['ConstituentNode']:
        if self.start == start and self.end == end:
            return self
        for child in self.children:
            if child.start <= start and end <= child.end:
                result = child.find_node_at(start, end)
                if result:
                    return result
        return None
    
    def to_dict(self) -> Dict:
        result = {
            'label': self.label,
            'start': self.start,
            'end': self.end,
        }
        if self.is_terminal():
            result['word'] = self.word
        else:
            result['children'] = [child.to_dict() for child in self.children]
        return result
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ConstituentNode':
        node = cls(
            label=data['label'],
            word=data.get('word', ''),
            start=data['start'],
            end=data['end']
        )
        if 'children' in data:
            node.children = [cls.from_dict(c) for c in data['children']]
        return node


class ConstituentTree:
    def __init__(self, root: Optional[ConstituentNode] = None):
        self.root = root
    
    def __repr__(self) -> str:
        if self.root:
            return f"ConstituentTree(root={self.root.label}, height={self.get_height()})"
        return "ConstituentTree(empty)"
    
    def __len__(self) -> int:
        return len(self.get_words()) if self.root else 0
    
    def __iter__(self) -> Iterator[ConstituentNode]:
        if self.root:
            yield from self._traverse(self.root)
    
    def _traverse(self, node: ConstituentNode) -> Iterator[ConstituentNode]:
        yield node
        for child in node.children:
            yield from self._traverse(child)
    
    def get_words(self) -> List[str]:
        return self.root.get_words() if self.root else []
    
    def get_text(self) -> str:
        return self.root.get_text() if self.root else ''
    
    def get_height(self) -> int:
        return self.root.get_height() if self.root else 0
    
    def get_yield(self) -> List[ConstituentNode]:
        return self.root.get_yield() if self.root else []
    
    def get_preterminals(self) -> List[ConstituentNode]:
        return self.root.get_preterminals() if self.root else []
    
    def get_pos_tags(self) -> List[Tuple[str, str]]:
        preterminals = self.get_preterminals()
        return [(node.label, node.children[0].word if node.children else '') for node in preterminals]
    
    def to_penn_treebank(self) -> str:
        return self.root.to_penn_treebank() if self.root else ''
    
    def to_lisp_string(self) -> str:
        return self.root.to_lisp_string() if self.root else ''
    
    def get_phrases(self, phrase_type: str) -> List[ConstituentNode]:
        return self.root.get_phrases(phrase_type) if self.root else []
    
    def get_noun_phrases(self) -> List[ConstituentNode]:
        return self.root.get_noun_phrases() if self.root else []
    
    def get_verb_phrases(self) -> List[ConstituentNode]:
        return self.root.get_verb_phrases() if self.root else []
    
    def get_prep_phrases(self) -> List[ConstituentNode]:
        return self.root.get_prep_phrases() if self.root else []
    
    def to_dict(self) -> Dict:
        return self.root.to_dict() if self.root else {}
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ConstituentTree':
        return cls(ConstituentNode.from_dict(data))
    
    @classmethod
    def from_penn_treebank(cls, tree_str: str) -> 'ConstituentTree':
        tree_str = tree_str.strip()
        if not tree_str:
            return cls()
        
        tokens = cls._tokenize(tree_str)
        root, _ = cls._parse_tokens(tokens, 0)
        return cls(root)
    
    @staticmethod
    def _tokenize(tree_str: str) -> List[str]:
        tokens = []
        i = 0
        while i < len(tree_str):
            c = tree_str[i]
            if c == '(':
                tokens.append('(')
                i += 1
            elif c == ')':
                tokens.append(')')
                i += 1
            elif c.isspace():
                i += 1
            else:
                j = i
                while j < len(tree_str) and tree_str[j] not in '() \t\n\r':
                    j += 1
                tokens.append(tree_str[i:j])
                i = j
        return tokens
    
    @staticmethod
    def _parse_tokens(tokens: List[str], pos: int) -> Tuple[Optional[ConstituentNode], int]:
        if pos >= len(tokens) or tokens[pos] != '(':
            return None, pos
        
        pos += 1
        if pos >= len(tokens):
            return None, pos
        
        label = tokens[pos]
        pos += 1
        
        children = []
        start = 0
        end = 0
        
        while pos < len(tokens) and tokens[pos] != ')':
            if tokens[pos] == '(':
                child, pos = ConstituentTree._parse_tokens(tokens, pos)
                if child:
                    children.append(child)
            else:
                word = tokens[pos]
                pos += 1
                word_node = ConstituentNode(label=word, word=word, start=end, end=end + 1)
                children.append(word_node)
                end += 1
        
        if pos < len(tokens) and tokens[pos] == ')':
            pos += 1
        
        if children:
            start = children[0].start
            end = children[-1].end
        
        node = ConstituentNode(label=label, children=children, start=start, end=end)
        return node, pos
    
    def extract_rules(self) -> List[GrammarRule]:
        rules = []
        if self.root:
            self._extract_rules_recursive(self.root, rules)
        return rules
    
    def _extract_rules_recursive(self, node: ConstituentNode, rules: List[GrammarRule]):
        if node.is_terminal():
            return
        
        if node.is_preterminal():
            rule = GrammarRule(
                lhs=node.label,
                rhs=(node.children[0].word,),
                probability=1.0,
                count=1
            )
            rules.append(rule)
        else:
            rhs = tuple(child.label for child in node.children)
            rule = GrammarRule(
                lhs=node.label,
                rhs=rhs,
                probability=1.0,
                count=1
            )
            rules.append(rule)
            
            for child in node.children:
                self._extract_rules_recursive(child, rules)


class PCFG:
    def __init__(self):
        self.rules: Dict[str, List[GrammarRule]] = defaultdict(list)
        self.non_terminals: Set[str] = set()
        self.terminals: Set[str] = set()
        self.start_symbol: str = 'S'
        self._rule_index: Dict[Tuple[str, Tuple[str, ...]], int] = {}
        self._trained = False
    
    def add_rule(self, rule: GrammarRule):
        key = (rule.lhs, rule.rhs)
        if key in self._rule_index:
            idx = self._rule_index[key]
            self.rules[rule.lhs][idx].count += rule.count
        else:
            self._rule_index[key] = len(self.rules[rule.lhs])
            self.rules[rule.lhs].append(rule)
        
        self.non_terminals.add(rule.lhs)
        for symbol in rule.rhs:
            if len(symbol) == 1 or symbol.isupper():
                self.non_terminals.add(symbol)
            else:
                self.terminals.add(symbol)
    
    def get_rules(self, lhs: str) -> List[GrammarRule]:
        return self.rules.get(lhs, [])
    
    def get_rule(self, lhs: str, rhs: Tuple[str, ...]) -> Optional[GrammarRule]:
        key = (lhs, rhs)
        if key in self._rule_index:
            idx = self._rule_index[key]
            return self.rules[lhs][idx]
        return None
    
    def get_binary_rules(self, lhs: str) -> List[GrammarRule]:
        return [r for r in self.get_rules(lhs) if r.is_binary()]
    
    def get_unary_rules(self, lhs: str) -> List[GrammarRule]:
        return [r for r in self.get_rules(lhs) if r.is_unary()]
    
    def train(self, trees: List[ConstituentTree], smooth: float = 0.0):
        self.rules.clear()
        self.non_terminals.clear()
        self.terminals.clear()
        self._rule_index.clear()
        
        rule_counts: Dict[Tuple[str, Tuple[str, ...]], int] = defaultdict(int)
        lhs_counts: Dict[str, int] = defaultdict(int)
        
        for tree in trees:
            rules = tree.extract_rules()
            for rule in rules:
                key = (rule.lhs, rule.rhs)
                rule_counts[key] += rule.count
                lhs_counts[rule.lhs] += rule.count
        
        for (lhs, rhs), count in rule_counts.items():
            total = lhs_counts[lhs]
            prob = (count + smooth) / (total + smooth * len([k for k in rule_counts if k[0] == lhs]))
            
            rule = GrammarRule(
                lhs=lhs,
                rhs=rhs,
                probability=prob,
                count=count
            )
            self.add_rule(rule)
        
        self._trained = True
    
    def compute_probability(self, tree: ConstituentTree) -> float:
        if not tree.root:
            return 0.0
        
        return self._compute_prob_recursive(tree.root)
    
    def _compute_prob_recursive(self, node: ConstituentNode) -> float:
        if node.is_terminal():
            return 1.0
        
        if node.is_preterminal():
            rule = self.get_rule(node.label, (node.children[0].word,))
            if rule:
                return rule.probability
            return 0.0
        
        rhs = tuple(child.label for child in node.children)
        rule = self.get_rule(node.label, rhs)
        
        if not rule:
            return 0.0
        
        prob = rule.probability
        for child in node.children:
            prob *= self._compute_prob_recursive(child)
        
        return prob
    
    def is_non_terminal(self, symbol: str) -> bool:
        return symbol in self.non_terminals
    
    def is_terminal(self, symbol: str) -> bool:
        return symbol in self.terminals
    
    def get_vocabulary(self) -> Set[str]:
        return self.terminals.union(self.non_terminals)
    
    def save(self, filepath: str):
        model_data = {
            'rules': {lhs: [(r.rhs, r.probability, r.count) for r in rules] 
                     for lhs, rules in self.rules.items()},
            'non_terminals': list(self.non_terminals),
            'terminals': list(self.terminals),
            'start_symbol': self.start_symbol,
            'trained': self._trained
        }
        
        dir_path = os.path.dirname(filepath)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load(self, filepath: str):
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.rules.clear()
        self._rule_index.clear()
        
        for lhs, rule_data in model_data['rules'].items():
            for rhs, prob, count in rule_data:
                rule = GrammarRule(lhs=lhs, rhs=rhs, probability=prob, count=count)
                self.add_rule(rule)
        
        self.non_terminals = set(model_data['non_terminals'])
        self.terminals = set(model_data['terminals'])
        self.start_symbol = model_data['start_symbol']
        self._trained = model_data['trained']
    
    def get_model_info(self) -> Dict:
        return {
            'trained': self._trained,
            'num_rules': sum(len(rules) for rules in self.rules.values()),
            'num_non_terminals': len(self.non_terminals),
            'num_terminals': len(self.terminals),
            'start_symbol': self.start_symbol,
            'non_terminals': sorted(list(self.non_terminals)),
            'terminals': sorted(list(self.terminals))[:100]
        }


class CKYParser:
    def __init__(self, grammar: PCFG):
        self.grammar = grammar
    
    def parse(self, words: List[str]) -> Optional[ConstituentTree]:
        if not words:
            return None
        
        n = len(words)
        
        chart: Dict[Tuple[int, int], Dict[str, Tuple[float, Optional[Tuple]]]] = defaultdict(dict)
        
        for i, word in enumerate(words):
            for lhs, rules in self.grammar.rules.items():
                for rule in rules:
                    if rule.is_unary() and rule.rhs[0] == word:
                        if lhs not in chart[(i, i + 1)]:
                            chart[(i, i + 1)][lhs] = (math.log(rule.probability), (word,))
                        else:
                            current_prob = chart[(i, i + 1)][lhs][0]
                            new_prob = math.log(rule.probability)
                            if new_prob > current_prob:
                                chart[(i, i + 1)][lhs] = (new_prob, (word,))
        
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length
                
                for k in range(i + 1, j):
                    for lhs, rules in self.grammar.rules.items():
                        for rule in rules:
                            if not rule.is_binary():
                                continue
                            
                            left_sym, right_sym = rule.rhs
                            
                            if (i, k) in chart and left_sym in chart[(i, k)]:
                                if (k, j) in chart and right_sym in chart[(k, j)]:
                                    left_prob = chart[(i, k)][left_sym][0]
                                    right_prob = chart[(k, j)][right_sym][0]
                                    new_prob = left_prob + right_prob + math.log(rule.probability)
                                    
                                    if lhs not in chart[(i, j)] or new_prob > chart[(i, j)][lhs][0]:
                                        chart[(i, j)][lhs] = (new_prob, (k, left_sym, right_sym))
        
        start = self.grammar.start_symbol
        if (0, n) in chart and start in chart[(0, n)]:
            root = self._build_tree(chart, 0, n, start, words)
            return ConstituentTree(root)
        
        return None
    
    def _build_tree(
        self,
        chart: Dict,
        start: int,
        end: int,
        symbol: str,
        words: List[str]
    ) -> Optional[ConstituentNode]:
        if (start, end) not in chart or symbol not in chart[(start, end)]:
            return None
        
        prob, backpointer = chart[(start, end)][symbol]
        
        if isinstance(backpointer[0], str) and len(backpointer) == 1:
            word = backpointer[0]
            word_node = ConstituentNode(label=word, word=word, start=start, end=end)
            return ConstituentNode(label=symbol, children=[word_node], start=start, end=end)
        
        if len(backpointer) == 3:
            split, left_sym, right_sym = backpointer
            
            left_child = self._build_tree(chart, start, split, left_sym, words)
            right_child = self._build_tree(chart, split, end, right_sym, words)
            
            if left_child and right_child:
                return ConstituentNode(
                    label=symbol,
                    children=[left_child, right_child],
                    start=start,
                    end=end
                )
        
        return None
    
    def parse_k_best(self, words: List[str], k: int = 5) -> List[Tuple[ConstituentTree, float]]:
        if not words:
            return []
        
        n = len(words)
        
        chart: Dict[Tuple[int, int], Dict[str, List[Tuple[float, Optional[Tuple]]]]] = defaultdict(dict)
        
        for i, word in enumerate(words):
            for lhs, rules in self.grammar.rules.items():
                for rule in rules:
                    if rule.is_unary() and rule.rhs[0] == word:
                        prob = math.log(rule.probability)
                        if lhs not in chart[(i, i + 1)]:
                            chart[(i, i + 1)][lhs] = []
                        chart[(i, i + 1)][lhs].append((prob, (word,)))
        
        for key in chart:
            for sym in chart[key]:
                chart[key][sym].sort(reverse=True, key=lambda x: x[0])
                chart[key][sym] = chart[key][sym][:k]
        
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length
                
                candidates: Dict[str, List[Tuple[float, Optional[Tuple]]]] = defaultdict(list)
                
                for split in range(i + 1, j):
                    for lhs, rules in self.grammar.rules.items():
                        for rule in rules:
                            if not rule.is_binary():
                                continue
                            
                            left_sym, right_sym = rule.rhs
                            
                            if (i, split) in chart and left_sym in chart[(i, split)]:
                                if (split, j) in chart and right_sym in chart[(split, j)]:
                                    for left_prob, _ in chart[(i, split)][left_sym]:
                                        for right_prob, _ in chart[(split, j)][right_sym]:
                                            new_prob = left_prob + right_prob + math.log(rule.probability)
                                            candidates[lhs].append((new_prob, (split, left_sym, right_sym)))
                
                for sym, entries in candidates.items():
                    entries.sort(reverse=True, key=lambda x: x[0])
                    chart[(i, j)][sym] = entries[:k]
        
        start = self.grammar.start_symbol
        results = []
        
        if (0, n) in chart and start in chart[(0, n)]:
            for prob, _ in chart[(0, n)][start]:
                root = self._build_tree_k_best(chart, 0, n, start, words, 0)
                if root:
                    tree = ConstituentTree(root)
                    results.append((tree, prob))
        
        return results[:k]
    
    def _build_tree_k_best(
        self,
        chart: Dict,
        start: int,
        end: int,
        symbol: str,
        words: List[str],
        rank: int
    ) -> Optional[ConstituentNode]:
        if (start, end) not in chart or symbol not in chart[(start, end)]:
            return None
        
        if rank >= len(chart[(start, end)][symbol]):
            return None
        
        prob, backpointer = chart[(start, end)][symbol][rank]
        
        if isinstance(backpointer[0], str) and len(backpointer) == 1:
            word = backpointer[0]
            word_node = ConstituentNode(label=word, word=word, start=start, end=end)
            return ConstituentNode(label=symbol, children=[word_node], start=start, end=end)
        
        if len(backpointer) == 3:
            split, left_sym, right_sym = backpointer
            
            left_child = self._build_tree_k_best(chart, start, split, left_sym, words, 0)
            right_child = self._build_tree_k_best(chart, split, end, right_sym, words, 0)
            
            if left_child and right_child:
                return ConstituentNode(
                    label=symbol,
                    children=[left_child, right_child],
                    start=start,
                    end=end
                )
        
        return None


class ConstituentParser:
    def __init__(
        self,
        start_symbol: str = 'S',
        use_pos: bool = True
    ):
        self.grammar = PCFG()
        self.grammar.start_symbol = start_symbol
        self.cky_parser = CKYParser(self.grammar)
        self.use_pos = use_pos
        self._trained = False
        self._pos_tagger = None
    
    def train(self, trees: List[ConstituentTree], smooth: float = 0.1):
        if not trees:
            raise ValueError("Training corpus cannot be empty")
        
        self.grammar.train(trees, smooth)
        self._trained = True
    
    def train_from_file(self, filepath: str, encoding: str = 'utf-8'):
        trees = []
        
        with open(filepath, 'r', encoding=encoding) as f:
            content = f.read()
        
        tree_strings = content.strip().split('\n\n')
        
        for tree_str in tree_strings:
            if tree_str.strip():
                tree = ConstituentTree.from_penn_treebank(tree_str)
                if tree.root:
                    trees.append(tree)
        
        if trees:
            self.train(trees)
    
    def parse(self, words: List[str]) -> Optional[ConstituentTree]:
        if not words:
            return None
        
        if not self._trained:
            raise RuntimeError("Model has not been trained. Call train() first.")
        
        return self.cky_parser.parse(words)
    
    def parse_k_best(self, words: List[str], k: int = 5) -> List[Tuple[ConstituentTree, float]]:
        if not words:
            return []
        
        if not self._trained:
            raise RuntimeError("Model has not been trained. Call train() first.")
        
        return self.cky_parser.parse_k_best(words, k)
    
    def parse_sentence(self, text: str, segmentor=None) -> Optional[ConstituentTree]:
        if segmentor:
            words = segmentor.cut(text)
        else:
            words = list(text)
        
        return self.parse(words)
    
    def get_parse_probability(self, tree: ConstituentTree) -> float:
        if not self._trained:
            raise RuntimeError("Model has not been trained. Call train() first.")
        
        return self.grammar.compute_probability(tree)
    
    def save_model(self, filepath: str):
        if not self._trained:
            raise RuntimeError("Model has not been trained. Call train() first before saving.")
        
        model_data = {
            'grammar': {
                'rules': {lhs: [(r.rhs, r.probability, r.count) for r in rules]
                         for lhs, rules in self.grammar.rules.items()},
                'non_terminals': list(self.grammar.non_terminals),
                'terminals': list(self.grammar.terminals),
                'start_symbol': self.grammar.start_symbol,
            },
            'use_pos': self.use_pos,
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
        
        self.grammar = PCFG()
        grammar_data = model_data['grammar']
        
        for lhs, rule_data in grammar_data['rules'].items():
            for rhs, prob, count in rule_data:
                rule = GrammarRule(lhs=lhs, rhs=rhs, probability=prob, count=count)
                self.grammar.add_rule(rule)
        
        self.grammar.non_terminals = set(grammar_data['non_terminals'])
        self.grammar.terminals = set(grammar_data['terminals'])
        self.grammar.start_symbol = grammar_data['start_symbol']
        
        self.cky_parser = CKYParser(self.grammar)
        self.use_pos = model_data['use_pos']
        self._trained = model_data['trained']
    
    def is_trained(self) -> bool:
        return self._trained
    
    def get_model_info(self) -> Dict:
        if not self._trained:
            return {'trained': False}
        
        return {
            'trained': True,
            'grammar': self.grammar.get_model_info(),
            'use_pos': self.use_pos
        }


def create_sample_constituent_trees() -> List[ConstituentTree]:
    trees = []
    
    tree1_str = """
    (S
        (NP (NR 我))
        (VP (VV 爱)
            (NP (NR 中国))))
    """
    tree1 = ConstituentTree.from_penn_treebank(tree1_str)
    trees.append(tree1)
    
    tree2_str = """
    (S
        (NP (NR 他))
        (VP (VV 吃)
            (AS 了)
            (NP (QP (CD 一个))
                (NP (NN 苹果)))))
    """
    tree2 = ConstituentTree.from_penn_treebank(tree2_str)
    trees.append(tree2)
    
    tree3_str = """
    (S
        (NP (NR 北京))
        (VP (VC 是)
            (NP (NP (NR 中国))
                (NP (NN 首都)))))
    """
    tree3 = ConstituentTree.from_penn_treebank(tree3_str)
    trees.append(tree3)
    
    tree4_str = """
    (S
        (NP (NR 小明))
        (VP (PP (P 在)
                (NP (NN 学校)))
            (VP (VV 读书))))
    """
    tree4 = ConstituentTree.from_penn_treebank(tree4_str)
    trees.append(tree4)
    
    tree5_str = """
    (S
        (NP (DP (DT 这))
            (NP (NN 本书)))
        (VP (ADVP (AD 很))
            (VP (VA 有趣))))
    """
    tree5 = ConstituentTree.from_penn_treebank(tree5_str)
    trees.append(tree5)
    
    tree6_str = """
    (S
        (NP (NR 她))
        (VP (VV 买)
            (AS 了)
            (NP (QP (CD 三))
                (NP (M 本)
                    (NP (NN 书))))))
    """
    tree6 = ConstituentTree.from_penn_treebank(tree6_str)
    trees.append(tree6)
    
    tree7_str = """
    (S
        (NP (NR 我们))
        (VP (VV 学习)
            (NP (NN 中文))))
    """
    tree7 = ConstituentTree.from_penn_treebank(tree7_str)
    trees.append(tree7)
    
    tree8_str = """
    (S
        (NP (NR 老师))
        (VP (VV 教)
            (NP (NR 学生))
            (IP (NP (NR 他们))
                (VP (VV 学习)
                    (NP (NN 数学))))))
    """
    tree8 = ConstituentTree.from_penn_treebank(tree8_str)
    trees.append(tree8)
    
    return trees


def train_constituent_parser_from_file(
    parser: ConstituentParser,
    filepath: str,
    encoding: str = 'utf-8'
) -> None:
    parser.train_from_file(filepath, encoding)


__all__ = [
    'CONSTITUENT_LABELS',
    'POS_LABELS',
    'DEFAULT_NON_TERMINALS',
    'DEFAULT_TERMINALS',
    'GrammarRule',
    'ConstituentNode',
    'ConstituentTree',
    'PCFG',
    'CKYParser',
    'ConstituentParser',
    'create_sample_constituent_trees',
    'train_constituent_parser_from_file',
]
