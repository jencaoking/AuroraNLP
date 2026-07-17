from typing import List, Tuple, Dict, Optional, Any
from collections import defaultdict
import math
import pickle
import os

POS_TAGS = {
    'n': '名词',
    'nr': '人名',
    'ns': '地名',
    'nt': '机构团体',
    'nz': '其他专名',
    'v': '动词',
    'vd': '副动词',
    'vn': '名动词',
    'a': '形容词',
    'ad': '副形词',
    'an': '名形词',
    'd': '副词',
    'm': '数词',
    'q': '量词',
    'r': '代词',
    'p': '介词',
    'c': '连词',
    'u': '助词',
    'xc': '其他功能词',
    'w': '标点符号',
    'f': '方位词',
    's': '处所词',
    't': '时间词',
    'b': '区别词',
    'z': '状态词',
    'e': '叹词',
    'y': '语气词',
    'o': '拟声词',
    'l': '习用语',
    'i': '成语',
    'j': '简称',
    'h': '前缀',
    'k': '后缀',
    'g': '语素',
    'x': '非语素字',
}

DEFAULT_TAGS = list(POS_TAGS.keys())


class HMMPOSTagger:
    def __init__(self, tags: Optional[List[str]] = None):
        self.tags = tags or DEFAULT_TAGS.copy()
        
        self.init_prob: Dict[str, float] = {}
        self.trans_prob: Dict[str, Dict[str, float]] = {}
        self.emit_prob: Dict[str, Dict[str, float]] = {}
        
        self.init_count: Dict[str, int] = defaultdict(int)
        self.trans_count: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.emit_count: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        self.tag_count: Dict[str, int] = defaultdict(int)
        self.total_tags = 0
        self.word_count = 0
        
        self._trained = False
        self._smooth = 1.0
    
    def train(self, corpus: List[Tuple[List[str], List[str]]], smooth: float = 1.0):
        self.init_count.clear()
        self.trans_count.clear()
        self.emit_count.clear()
        self.tag_count.clear()
        self.total_tags = 0
        self.word_count = 0
        
        all_tags = set()
        for words, tags in corpus:
            if len(words) != len(tags):
                raise ValueError(f"Words and tags length mismatch: {len(words)} vs {len(tags)}")
            all_tags.update(tags)
        
        self.tags = sorted(list(all_tags))
        
        for words, tags in corpus:
            if not words:
                continue
            
            self.init_count[tags[0]] += 1
            
            for i, (word, tag) in enumerate(zip(words, tags)):
                self.emit_count[tag][word] += 1
                self.tag_count[tag] += 1
                self.total_tags += 1
                self.word_count += 1
                
                if i > 0:
                    prev_tag = tags[i - 1]
                    self.trans_count[prev_tag][tag] += 1
        
        self._calculate_probabilities(smooth)
        self._trained = True
    
    def _calculate_probabilities(self, smooth: float):
        total_init = sum(self.init_count.values())
        for tag in self.tags:
            if total_init > 0:
                self.init_prob[tag] = math.log(
                    (self.init_count[tag] + smooth) / (total_init + smooth * len(self.tags))
                )
            else:
                self.init_prob[tag] = math.log(1.0 / len(self.tags))
        
        for prev_tag in self.tags:
            total_trans = sum(self.trans_count[prev_tag].values())
            self.trans_prob[prev_tag] = {}
            for curr_tag in self.tags:
                if total_trans > 0:
                    self.trans_prob[prev_tag][curr_tag] = math.log(
                        (self.trans_count[prev_tag][curr_tag] + smooth) / 
                        (total_trans + smooth * len(self.tags))
                    )
                else:
                    self.trans_prob[prev_tag][curr_tag] = math.log(1.0 / len(self.tags))
        
        for tag in self.tags:
            total_emit = sum(self.emit_count[tag].values())
            self.emit_prob[tag] = {}
            for word, count in self.emit_count[tag].items():
                if total_emit > 0:
                    self.emit_prob[tag][word] = math.log((count + smooth) / (total_emit + smooth))
        
        self._smooth = smooth
    
    def _get_emit_prob(self, tag: str, word: str) -> float:
        if word in self.emit_prob[tag]:
            return self.emit_prob[tag][word]
        
        total_emit = sum(self.emit_count[tag].values())
        vocab_size = len(self.emit_count[tag])
        
        if total_emit > 0 and vocab_size > 0:
            return math.log(self._smooth / (total_emit + self._smooth * vocab_size))
        else:
            return math.log(1.0 / len(self.tags))
    
    def viterbi(self, words: List[str]) -> List[str]:
        if not words:
            return []
        
        if not self._trained:
            raise RuntimeError("Model has not been trained. Call train() first.")
        
        length = len(words)
        
        V = [{} for _ in range(length)]
        path = {}
        
        first_word = words[0]
        for tag in self.tags:
            V[0][tag] = self.init_prob[tag] + self._get_emit_prob(tag, first_word)
            path[tag] = [tag]
        
        for t in range(1, length):
            word = words[t]
            new_path = {}
            
            for curr_tag in self.tags:
                emit_prob = self._get_emit_prob(curr_tag, word)
                
                best_prob = float('-inf')
                best_prev_tag = None
                
                for prev_tag in self.tags:
                    prob = V[t - 1][prev_tag] + self.trans_prob[prev_tag][curr_tag] + emit_prob
                    if prob > best_prob:
                        best_prob = prob
                        best_prev_tag = prev_tag
                
                V[t][curr_tag] = best_prob
                new_path[curr_tag] = path[best_prev_tag] + [curr_tag]
            
            path = new_path
        
        best_final_prob = float('-inf')
        best_final_tag = None
        for tag in self.tags:
            if V[length - 1][tag] > best_final_prob:
                best_final_prob = V[length - 1][tag]
                best_final_tag = tag
        
        if best_final_tag is None:
            return ['x'] * length
        
        return path[best_final_tag]
    
    def tag(self, words: List[str]) -> List[str]:
        return self.viterbi(words)
    
    def tag_sentence(self, words: List[str]) -> List[Tuple[str, str]]:
        tags = self.viterbi(words)
        return list(zip(words, tags))
    
    def save_model(self, filepath: str):
        if not self._trained:
            raise RuntimeError("Model has not been trained. Call train() first before saving.")
        
        model_data = {
            'tags': self.tags,
            'init_prob': self.init_prob,
            'trans_prob': self.trans_prob,
            'emit_prob': self.emit_prob,
            'init_count': dict(self.init_count),
            'trans_count': {k: dict(v) for k, v in self.trans_count.items()},
            'emit_count': {k: dict(v) for k, v in self.emit_count.items()},
            'tag_count': dict(self.tag_count),
            'total_tags': self.total_tags,
            'word_count': self.word_count,
            'smooth': self._smooth,
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
        
        self.tags = model_data['tags']
        self.init_prob = model_data['init_prob']
        self.trans_prob = model_data['trans_prob']
        self.emit_prob = model_data['emit_prob']
        self.init_count = defaultdict(int, model_data['init_count'])
        self.trans_count = defaultdict(lambda: defaultdict(int))
        for k, v in model_data['trans_count'].items():
            self.trans_count[k] = defaultdict(int, v)
        self.emit_count = defaultdict(lambda: defaultdict(int))
        for k, v in model_data['emit_count'].items():
            self.emit_count[k] = defaultdict(int, v)
        self.tag_count = defaultdict(int, model_data['tag_count'])
        self.total_tags = model_data['total_tags']
        self.word_count = model_data['word_count']
        self._smooth = model_data['smooth']
        self._trained = model_data['trained']
    
    def is_trained(self) -> bool:
        return self._trained
    
    def get_model_info(self) -> Dict[str, Any]:
        if not self._trained:
            return {'trained': False}
        
        vocab_sizes = {tag: len(self.emit_count[tag]) for tag in self.tags}
        
        return {
            'trained': True,
            'num_tags': len(self.tags),
            'tags': self.tags,
            'total_tags': self.total_tags,
            'word_count': self.word_count,
            'tag_counts': dict(self.tag_count),
            'vocabulary_sizes': vocab_sizes,
            'smooth': self._smooth
        }


class CRFPOSTagger:
    def __init__(self, tags: Optional[List[str]] = None):
        self.tags = tags or DEFAULT_TAGS.copy()
        self.weights: Dict[str, float] = defaultdict(float)
        self._trained = False
        self._learning_rate = 0.1
        self._l2_reg = 0.01
        self._max_iter = 100
        self._epsilon = 1e-6
    
    def _extract_features(self, words: List[str], pos: int, prev_tag: str, curr_tag: str) -> List[str]:
        features = []
        
        word = words[pos]
        features.append(f"WORD:{word}|{curr_tag}")
        
        if prev_tag:
            features.append(f"TRANS:{prev_tag}->{curr_tag}")
        
        if pos == 0:
            features.append(f"START:{curr_tag}")
        
        if pos == len(words) - 1:
            features.append(f"END:{curr_tag}")
        
        if pos > 0:
            prev_word = words[pos - 1]
            features.append(f"PREV_WORD:{prev_word}|{curr_tag}")
            features.append(f"BIGRAM:{prev_word}_{word}|{curr_tag}")
        
        if pos < len(words) - 1:
            next_word = words[pos + 1]
            features.append(f"NEXT_WORD:{next_word}|{curr_tag}")
        
        if len(word) == 1:
            features.append(f"LEN:1|{curr_tag}")
        elif len(word) == 2:
            features.append(f"LEN:2|{curr_tag}")
        else:
            features.append(f"LEN:3+|{curr_tag}")
        
        if len(word) >= 1:
            features.append(f"FIRST_CHAR:{word[0]}|{curr_tag}")
            features.append(f"LAST_CHAR:{word[-1]}|{curr_tag}")
        
        if len(word) >= 2:
            features.append(f"PREFIX1:{word[0]}|{curr_tag}")
            features.append(f"SUFFIX1:{word[-1]}|{curr_tag}")
        
        if len(word) >= 3:
            features.append(f"PREFIX2:{word[:2]}|{curr_tag}")
            features.append(f"SUFFIX2:{word[-2:]}|{curr_tag}")
        
        if word.isdigit():
            features.append(f"IS_DIGIT|{curr_tag}")
        elif any(c.isdigit() for c in word):
            features.append(f"HAS_DIGIT|{curr_tag}")
        
        if word.isalpha():
            features.append(f"IS_ALPHA|{curr_tag}")
        
        if any('\u4e00' <= c <= '\u9fff' for c in word):
            features.append(f"HAS_CHINESE|{curr_tag}")
        
        return features
    
    def train(
        self,
        corpus: List[Tuple[List[str], List[str]]],
        learning_rate: float = 0.1,
        l2_reg: float = 0.01,
        max_iter: int = 100,
        epsilon: float = 1e-6,
        verbose: bool = True
    ):
        if not corpus:
            raise ValueError("Training corpus cannot be empty")
        
        all_tags = set()
        for words, tags in corpus:
            if len(words) != len(tags):
                raise ValueError(f"Words and tags length mismatch: {len(words)} vs {len(tags)}")
            all_tags.update(tags)
        
        self.tags = sorted(list(all_tags))
        
        self._learning_rate = learning_rate
        self._l2_reg = l2_reg
        self._max_iter = max_iter
        self._epsilon = epsilon
        
        prev_loss = float('inf')
        
        for iteration in range(max_iter):
            total_loss = 0.0
            gradient: Dict[str, float] = defaultdict(float)
            
            for words, tags in corpus:
                alpha = self._forward(words)
                beta = self._backward(words)
                
                partition = self._log_sum_exp(list(alpha[-1].values()))
                
                for pos in range(len(words)):
                    for tag in self.tags:
                        expected = math.exp(alpha[pos][tag] + beta[pos][tag] - partition)
                        
                        prev_tag = tags[pos - 1] if pos > 0 else ''
                        features = self._extract_features(words, pos, prev_tag, tag)
                        
                        for feature in features:
                            gradient[feature] -= expected
                
                for pos in range(len(words)):
                    curr_tag = tags[pos]
                    prev_tag = tags[pos - 1] if pos > 0 else ''
                    
                    features = self._extract_features(words, pos, prev_tag, curr_tag)
                    
                    for feature in features:
                        gradient[feature] += 1.0
                
                total_loss += partition
            
            for feature, grad in gradient.items():
                self.weights[feature] += learning_rate * (grad - l2_reg * self.weights.get(feature, 0))
            
            for feature in list(self.weights.keys()):
                if abs(self.weights[feature]) < epsilon:
                    del self.weights[feature]
            
            if verbose and (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{max_iter}, Loss: {total_loss:.4f}")
            
            if abs(prev_loss - total_loss) < epsilon:
                if verbose:
                    print(f"Converged at iteration {iteration + 1}")
                break
            
            prev_loss = total_loss
        
        self._trained = True
    
    def _compute_score(self, features: List[str]) -> float:
        score = 0.0
        for feature in features:
            score += self.weights.get(feature, 0.0)
        return score
    
    def _forward(self, words: List[str]) -> List[Dict[str, float]]:
        length = len(words)
        alpha = [{} for _ in range(length)]
        
        for tag in self.tags:
            features = self._extract_features(words, 0, '', tag)
            alpha[0][tag] = self._compute_score(features)
        
        for pos in range(1, length):
            for curr_tag in self.tags:
                scores = []
                
                for prev_tag in self.tags:
                    features = self._extract_features(words, pos, prev_tag, curr_tag)
                    trans_score = self._compute_score(features)
                    scores.append(alpha[pos - 1][prev_tag] + trans_score)
                
                alpha[pos][curr_tag] = self._log_sum_exp(scores)
        
        return alpha
    
    def _backward(self, words: List[str]) -> List[Dict[str, float]]:
        length = len(words)
        beta = [{} for _ in range(length)]
        
        for tag in self.tags:
            beta[length - 1][tag] = 0.0
        
        for pos in range(length - 2, -1, -1):
            for prev_tag in self.tags:
                scores = []
                
                for curr_tag in self.tags:
                    features = self._extract_features(words, pos + 1, prev_tag, curr_tag)
                    trans_score = self._compute_score(features)
                    scores.append(beta[pos + 1][curr_tag] + trans_score)
                
                beta[pos][prev_tag] = self._log_sum_exp(scores)
        
        return beta
    
    def _log_sum_exp(self, values: List[float]) -> float:
        if not values:
            return float('-inf')
        
        max_val = max(values)
        if max_val == float('-inf'):
            return float('-inf')
        
        sum_exp = sum(math.exp(v - max_val) for v in values)
        return max_val + math.log(sum_exp)
    
    def viterbi(self, words: List[str]) -> List[str]:
        if not words:
            return []
        
        if not self._trained:
            raise RuntimeError("Model has not been trained. Call train() first.")
        
        length = len(words)
        
        viterbi_scores = [{} for _ in range(length)]
        backpointers = [{} for _ in range(length)]
        
        for tag in self.tags:
            features = self._extract_features(words, 0, '', tag)
            viterbi_scores[0][tag] = self._compute_score(features)
            backpointers[0][tag] = None
        
        for pos in range(1, length):
            for curr_tag in self.tags:
                best_score = float('-inf')
                best_prev_tag = None
                
                for prev_tag in self.tags:
                    features = self._extract_features(words, pos, prev_tag, curr_tag)
                    trans_score = self._compute_score(features)
                    score = viterbi_scores[pos - 1][prev_tag] + trans_score
                    
                    if score > best_score:
                        best_score = score
                        best_prev_tag = prev_tag
                
                viterbi_scores[pos][curr_tag] = best_score
                backpointers[pos][curr_tag] = best_prev_tag
        
        best_final_score = float('-inf')
        best_final_tag = None
        
        for tag in self.tags:
            if viterbi_scores[length - 1][tag] > best_final_score:
                best_final_score = viterbi_scores[length - 1][tag]
                best_final_tag = tag
        
        if best_final_tag is None:
            return ['x'] * length
        
        path = [best_final_tag]
        for pos in range(length - 1, 0, -1):
            path.append(backpointers[pos][path[-1]])
        
        path.reverse()
        return path
    
    def tag(self, words: List[str]) -> List[str]:
        return self.viterbi(words)
    
    def tag_sentence(self, words: List[str]) -> List[Tuple[str, str]]:
        tags = self.viterbi(words)
        return list(zip(words, tags))
    
    def save_model(self, filepath: str):
        if not self._trained:
            raise RuntimeError("Model has not been trained. Call train() first before saving.")
        
        model_data = {
            'tags': self.tags,
            'weights': dict(self.weights),
            'learning_rate': self._learning_rate,
            'l2_reg': self._l2_reg,
            'max_iter': self._max_iter,
            'epsilon': self._epsilon,
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
        
        self.tags = model_data['tags']
        self.weights = defaultdict(float, model_data['weights'])
        self._learning_rate = model_data['learning_rate']
        self._l2_reg = model_data['l2_reg']
        self._max_iter = model_data['max_iter']
        self._epsilon = model_data['epsilon']
        self._trained = model_data['trained']
    
    def is_trained(self) -> bool:
        return self._trained
    
    def get_model_info(self) -> Dict[str, Any]:
        if not self._trained:
            return {'trained': False}
        
        return {
            'trained': True,
            'num_tags': len(self.tags),
            'tags': self.tags,
            'num_features': len(self.weights),
            'learning_rate': self._learning_rate,
            'l2_reg': self._l2_reg,
            'max_iter': self._max_iter
        }


def train_pos_from_file(
    tagger,
    filepath: str,
    encoding: str = 'utf-8',
    delimiter: str = '/'
) -> None:
    corpus = []
    
    with open(filepath, 'r', encoding=encoding) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            tokens = line.split()
            words = []
            tags = []
            
            for token in tokens:
                if delimiter in token:
                    parts = token.rsplit(delimiter, 1)
                    if len(parts) == 2:
                        words.append(parts[0])
                        tags.append(parts[1])
                else:
                    words.append(token)
                    tags.append('n')
            
            if words and tags and len(words) == len(tags):
                corpus.append((words, tags))
    
    if corpus:
        tagger.train(corpus)


__all__ = ['POS_TAGS', 'DEFAULT_TAGS', 'HMMPOSTagger', 'CRFPOSTagger', 'train_pos_from_file']
