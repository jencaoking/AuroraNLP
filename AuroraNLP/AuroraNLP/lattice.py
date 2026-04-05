from typing import List, Tuple, Optional, Dict, Set, TYPE_CHECKING
from dataclasses import dataclass, field
from collections import defaultdict
import heapq
import math

if TYPE_CHECKING:
    from .dictionary import Dictionary
    from .ngram import NGramModel, BigramModel


@dataclass
class LatticeEdge:
    word: str
    start: int
    end: int
    pos_tag: Optional[str] = None
    weight: float = 0.0
    freq: int = 0
    
    def __hash__(self):
        return hash((self.start, self.end, self.word))
    
    def __eq__(self, other):
        if not isinstance(other, LatticeEdge):
            return False
        return self.start == other.start and self.end == other.end and self.word == other.word
    
    def __lt__(self, other):
        return self.weight < other.weight
    
    def __repr__(self):
        return f"LatticeEdge('{self.word}', [{self.start}->{self.end}], weight={self.weight:.4f})"


@dataclass
class LatticeNode:
    position: int
    incoming_edges: List[LatticeEdge] = field(default_factory=list)
    outgoing_edges: List[LatticeEdge] = field(default_factory=list)
    
    def add_incoming(self, edge: LatticeEdge) -> None:
        if edge not in self.incoming_edges:
            self.incoming_edges.append(edge)
    
    def add_outgoing(self, edge: LatticeEdge) -> None:
        if edge not in self.outgoing_edges:
            self.outgoing_edges.append(edge)
    
    def __repr__(self):
        return f"LatticeNode(pos={self.position}, in={len(self.incoming_edges)}, out={len(self.outgoing_edges)})"


class Lattice:
    def __init__(self, text: str):
        self.text = text
        self.length = len(text)
        self.nodes: Dict[int, LatticeNode] = {}
        self.edges: List[LatticeEdge] = []
        
        for i in range(self.length + 1):
            self.nodes[i] = LatticeNode(position=i)
    
    def add_edge(self, word: str, start: int, end: int, 
                 pos_tag: Optional[str] = None, 
                 weight: float = 0.0, 
                 freq: int = 0) -> LatticeEdge:
        if start < 0 or end > self.length or start >= end:
            raise ValueError(f"Invalid edge range: [{start}, {end}) for text length {self.length}")
        
        edge = LatticeEdge(
            word=word,
            start=start,
            end=end,
            pos_tag=pos_tag,
            weight=weight,
            freq=freq
        )
        
        self.edges.append(edge)
        self.nodes[start].add_outgoing(edge)
        self.nodes[end].add_incoming(edge)
        
        return edge
    
    def get_node(self, position: int) -> Optional[LatticeNode]:
        return self.nodes.get(position)
    
    def get_outgoing_edges(self, position: int) -> List[LatticeEdge]:
        node = self.get_node(position)
        return node.outgoing_edges if node else []
    
    def get_incoming_edges(self, position: int) -> List[LatticeEdge]:
        node = self.get_node(position)
        return node.incoming_edges if node else []
    
    def get_all_paths(self, max_paths: int = 100) -> List[List[LatticeEdge]]:
        paths: List[List[LatticeEdge]] = []
        self._dfs_paths(0, [], paths, max_paths)
        return paths
    
    def _dfs_paths(self, current_pos: int, current_path: List[LatticeEdge], 
                   all_paths: List[List[LatticeEdge]], max_paths: int) -> None:
        if len(all_paths) >= max_paths:
            return
        
        if current_pos == self.length:
            all_paths.append(current_path.copy())
            return
        
        outgoing = self.get_outgoing_edges(current_pos)
        
        if not outgoing:
            return
        
        for edge in outgoing:
            current_path.append(edge)
            self._dfs_paths(edge.end, current_path, all_paths, max_paths)
            current_path.pop()
    
    def get_path_words(self, path: List[LatticeEdge]) -> List[str]:
        return [edge.word for edge in path]
    
    def get_path_with_pos(self, path: List[LatticeEdge]) -> List[Tuple[str, Optional[str]]]:
        return [(edge.word, edge.pos_tag) for edge in path]
    
    def has_path(self) -> bool:
        return len(self.get_outgoing_edges(0)) > 0
    
    def is_fully_connected(self) -> bool:
        visited = set()
        stack = [0]
        
        while stack:
            pos = stack.pop()
            if pos in visited:
                continue
            visited.add(pos)
            
            for edge in self.get_outgoing_edges(pos):
                if edge.end not in visited:
                    stack.append(edge.end)
        
        return self.length in visited
    
    def get_statistics(self) -> Dict:
        total_edges = len(self.edges)
        avg_branching = 0.0
        max_branching = 0
        
        if self.length > 0:
            branching_factors = [len(self.get_outgoing_edges(i)) for i in range(self.length)]
            non_zero = [b for b in branching_factors if b > 0]
            avg_branching = sum(non_zero) / len(non_zero) if non_zero else 0.0
            max_branching = max(branching_factors) if branching_factors else 0
        
        return {
            'text_length': self.length,
            'total_edges': total_edges,
            'total_nodes': len(self.nodes),
            'avg_branching_factor': avg_branching,
            'max_branching_factor': max_branching,
            'has_path': self.has_path(),
            'is_fully_connected': self.is_fully_connected()
        }
    
    def visualize(self) -> str:
        lines = [f"Lattice for: '{self.text}'", "=" * 50]
        
        for i in range(self.length + 1):
            node = self.get_node(i)
            if node:
                lines.append(f"\nPosition {i}:")
                if node.outgoing_edges:
                    lines.append("  Outgoing:")
                    for edge in node.outgoing_edges:
                        lines.append(f"    -> [{edge.end}] '{edge.word}' (weight={edge.weight:.4f})")
                if node.incoming_edges:
                    lines.append("  Incoming:")
                    for edge in node.incoming_edges:
                        lines.append(f"    <- [{edge.start}] '{edge.word}'")
        
        return "\n".join(lines)


class LatticeBuilder:
    def __init__(self, dictionary: 'Dictionary', max_word_len: int = 15):
        self.dictionary = dictionary
        self.max_word_len = max_word_len
    
    def build(self, text: str, include_unknown: bool = True) -> Lattice:
        lattice = Lattice(text)
        
        for start in range(len(text)):
            max_len = min(self.max_word_len, len(text) - start)
            
            found_words = []
            for length in range(1, max_len + 1):
                word = text[start:start + length]
                if self.dictionary.search_in_dict(word):
                    _, pos_tag = self.dictionary.search_with_pos(word)
                    found_words.append((word, length, pos_tag))
            
            for word, length, pos_tag in found_words:
                lattice.add_edge(word, start, start + length, pos_tag=pos_tag)
            
            if not found_words and include_unknown:
                lattice.add_edge(text[start], start, start + 1, pos_tag='x')
        
        return lattice
    
    def build_with_freq(self, text: str, word_freq: Dict[str, int], 
                        include_unknown: bool = True) -> Lattice:
        lattice = Lattice(text)
        
        for start in range(len(text)):
            max_len = min(self.max_word_len, len(text) - start)
            
            found_words = []
            for length in range(1, max_len + 1):
                word = text[start:start + length]
                if self.dictionary.search_in_dict(word):
                    _, pos_tag = self.dictionary.search_with_pos(word)
                    freq = word_freq.get(word, 0)
                    found_words.append((word, length, pos_tag, freq))
            
            for word, length, pos_tag, freq in found_words:
                lattice.add_edge(word, start, start + length, pos_tag=pos_tag, freq=freq)
            
            if not found_words and include_unknown:
                lattice.add_edge(text[start], start, start + 1, pos_tag='x')
        
        return lattice


class PathScorer:
    @staticmethod
    def score_by_length(path: List[LatticeEdge]) -> float:
        return -len(path)
    
    @staticmethod
    def score_by_word_length(path: List[LatticeEdge]) -> float:
        total = sum(len(edge.word) for edge in path)
        avg_word_len = total / len(path) if path else 0
        return avg_word_len
    
    @staticmethod
    def score_by_frequency(path: List[LatticeEdge]) -> float:
        return sum(edge.freq for edge in path)
    
    @staticmethod
    def score_by_weight(path: List[LatticeEdge]) -> float:
        return sum(edge.weight for edge in path)
    
    @staticmethod
    def score_by_ngram(path: List[LatticeEdge], ngram_model: 'NGramModel') -> float:
        words = [edge.word for edge in path]
        return ngram_model.sentence_probability(words)


class LatticeSegmentor:
    def __init__(self, dictionary: 'Dictionary', max_word_len: int = 15):
        self.dictionary = dictionary
        self.max_word_len = max_word_len
        self.builder = LatticeBuilder(dictionary, max_word_len)
        self.ngram_model: Optional['NGramModel'] = None
        self.word_freq: Dict[str, int] = {}
        self._scoring_method = 'shortest'
    
    def set_ngram_model(self, model: 'NGramModel') -> None:
        self.ngram_model = model
        self._scoring_method = 'ngram'
    
    def set_word_frequency(self, freq_dict: Dict[str, int]) -> None:
        self.word_freq = freq_dict
    
    def set_scoring_method(self, method: str) -> None:
        valid_methods = ['shortest', 'longest_word', 'frequency', 'weight', 'ngram']
        if method not in valid_methods:
            raise ValueError(f"Invalid scoring method: {method}. Valid methods: {valid_methods}")
        self._scoring_method = method
    
    def build_lattice(self, text: str) -> Lattice:
        if self.word_freq:
            return self.builder.build_with_freq(text, self.word_freq)
        return self.builder.build(text)
    
    def shortest_path(self, lattice: Lattice) -> List[LatticeEdge]:
        dist = {i: float('inf') for i in range(lattice.length + 1)}
        dist[0] = 0
        prev: Dict[int, LatticeEdge] = {}
        
        pq = [(0, 0)]
        visited = set()
        
        while pq:
            d, pos = heapq.heappop(pq)
            
            if pos in visited:
                continue
            visited.add(pos)
            
            if pos == lattice.length:
                break
            
            for edge in lattice.get_outgoing_edges(pos):
                new_dist = dist[pos] + 1
                
                if new_dist < dist[edge.end]:
                    dist[edge.end] = new_dist
                    prev[edge.end] = edge
                    heapq.heappush(pq, (new_dist, edge.end))
        
        path = []
        pos = lattice.length
        while pos > 0 and pos in prev:
            edge = prev[pos]
            path.append(edge)
            pos = edge.start
        
        path.reverse()
        return path
    
    def best_path_dijkstra(self, lattice: Lattice, 
                           weight_func=None) -> List[LatticeEdge]:
        if weight_func is None:
            weight_func = lambda e: 1.0
        
        dist = {i: float('inf') for i in range(lattice.length + 1)}
        dist[0] = 0
        prev: Dict[int, LatticeEdge] = {}
        
        pq = [(0, 0)]
        visited = set()
        
        while pq:
            d, pos = heapq.heappop(pq)
            
            if pos in visited:
                continue
            visited.add(pos)
            
            if pos == lattice.length:
                break
            
            for edge in lattice.get_outgoing_edges(pos):
                weight = weight_func(edge)
                new_dist = dist[pos] + weight
                
                if new_dist < dist[edge.end]:
                    dist[edge.end] = new_dist
                    prev[edge.end] = edge
                    heapq.heappush(pq, (new_dist, edge.end))
        
        path = []
        pos = lattice.length
        while pos > 0 and pos in prev:
            edge = prev[pos]
            path.append(edge)
            pos = edge.start
        
        path.reverse()
        return path
    
    def best_path_ngram(self, lattice: Lattice) -> List[LatticeEdge]:
        if self.ngram_model is None:
            return self.shortest_path(lattice)
        
        ngram = self.ngram_model
        
        best_score = {i: float('-inf') for i in range(lattice.length + 1)}
        best_score[0] = 0.0
        prev: Dict[int, Tuple[LatticeEdge, List[str]]] = {}
        best_path_words = {i: [] for i in range(lattice.length + 1)}
        
        for pos in range(lattice.length + 1):
            if best_score[pos] == float('-inf') and pos > 0:
                continue
            
            for edge in lattice.get_outgoing_edges(pos):
                context = best_path_words[pos][-(ngram.n - 1):] if ngram.n > 1 else []
                
                log_prob = ngram.log_probability(edge.word, context)
                new_score = best_score[pos] + log_prob
                
                if new_score > best_score[edge.end]:
                    best_score[edge.end] = new_score
                    prev[edge.end] = edge
                    best_path_words[edge.end] = best_path_words[pos] + [edge.word]
        
        path = []
        pos = lattice.length
        while pos > 0 and pos in prev:
            edge = prev[pos]
            path.append(edge)
            pos = edge.start
        
        path.reverse()
        return path
    
    def best_path_frequency(self, lattice: Lattice) -> List[LatticeEdge]:
        def freq_weight(edge: LatticeEdge) -> float:
            if edge.freq > 0:
                return -math.log(edge.freq + 1)
            return 10.0
        
        return self.best_path_dijkstra(lattice, freq_weight)
    
    def find_k_best_paths(self, lattice: Lattice, k: int = 5) -> List[List[LatticeEdge]]:
        paths: List[List[LatticeEdge]] = []
        scores: List[float] = []
        
        all_paths = lattice.get_all_paths(max_paths=k * 10)
        
        for path in all_paths:
            score = self._score_path(path)
            paths.append(path)
            scores.append(score)
        
        sorted_pairs = sorted(zip(scores, paths), key=lambda x: x[0], reverse=True)
        
        return [path for _, path in sorted_pairs[:k]]
    
    def _score_path(self, path: List[LatticeEdge]) -> float:
        if self._scoring_method == 'shortest':
            return PathScorer.score_by_length(path)
        elif self._scoring_method == 'longest_word':
            return PathScorer.score_by_word_length(path)
        elif self._scoring_method == 'frequency':
            return PathScorer.score_by_frequency(path)
        elif self._scoring_method == 'weight':
            return PathScorer.score_by_weight(path)
        elif self._scoring_method == 'ngram' and self.ngram_model:
            return PathScorer.score_by_ngram(path, self.ngram_model)
        else:
            return PathScorer.score_by_length(path)
    
    def segment(self, text: str) -> List[str]:
        lattice = self.build_lattice(text)
        
        if not lattice.has_path():
            return list(text)
        
        if self._scoring_method == 'ngram' and self.ngram_model:
            path = self.best_path_ngram(lattice)
        elif self._scoring_method == 'frequency' and self.word_freq:
            path = self.best_path_frequency(lattice)
        else:
            path = self.shortest_path(lattice)
        
        return lattice.get_path_words(path)
    
    def segment_with_pos(self, text: str) -> List[Tuple[str, Optional[str]]]:
        lattice = self.build_lattice(text)
        
        if not lattice.has_path():
            return [(char, 'x') for char in text]
        
        if self._scoring_method == 'ngram' and self.ngram_model:
            path = self.best_path_ngram(lattice)
        elif self._scoring_method == 'frequency' and self.word_freq:
            path = self.best_path_frequency(lattice)
        else:
            path = self.shortest_path(lattice)
        
        return lattice.get_path_with_pos(path)
    
    def segment_with_lattice(self, text: str) -> Tuple[List[str], Lattice]:
        lattice = self.build_lattice(text)
        
        if not lattice.has_path():
            return list(text), lattice
        
        if self._scoring_method == 'ngram' and self.ngram_model:
            path = self.best_path_ngram(lattice)
        elif self._scoring_method == 'frequency' and self.word_freq:
            path = self.best_path_frequency(lattice)
        else:
            path = self.shortest_path(lattice)
        
        return lattice.get_path_words(path), lattice
    
    def detect_ambiguity(self, text: str) -> List[Dict]:
        lattice = self.build_lattice(text)
        ambiguities = []
        
        for pos in range(lattice.length):
            outgoing = lattice.get_outgoing_edges(pos)
            if len(outgoing) > 1:
                cross_ambiguity = []
                for edge in outgoing:
                    for other_edge in outgoing:
                        if edge != other_edge:
                            if edge.end > other_edge.start and edge.start < other_edge.end:
                                cross_ambiguity.append({
                                    'type': 'cross',
                                    'position': pos,
                                    'words': [edge.word, other_edge.word],
                                    'ranges': [(edge.start, edge.end), (other_edge.start, other_edge.end)]
                                })
                                break
                
                if cross_ambiguity:
                    ambiguities.extend(cross_ambiguity)
                else:
                    ambiguities.append({
                        'type': 'combination',
                        'position': pos,
                        'words': [e.word for e in outgoing],
                        'count': len(outgoing)
                    })
        
        return ambiguities
    
    def get_all_segmentations(self, text: str, max_results: int = 10) -> List[List[str]]:
        lattice = self.build_lattice(text)
        paths = lattice.get_all_paths(max_paths=max_results)
        return [lattice.get_path_words(path) for path in paths]


__all__ = [
    'LatticeEdge',
    'LatticeNode',
    'Lattice',
    'LatticeBuilder',
    'PathScorer',
    'LatticeSegmentor'
]
