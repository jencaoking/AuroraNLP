from typing import List, Tuple, Dict, Optional, Callable, Any
from collections import defaultdict
import math
import pickle
import os


class CRFFeatureTemplate:
    def __init__(self):
        self.feature_functions: List[Callable] = []
        self.feature_names: List[str] = []
    
    def add_unigram_feature(self, name: str, position: int):
        def feature_func(tokens: List[str], pos: int, prev_tag: str, curr_tag: str) -> Optional[str]:
            idx = pos + position
            if 0 <= idx < len(tokens):
                token = tokens[idx]
                return f"{name}:{token}|{curr_tag}"
            return None
        
        self.feature_functions.append(feature_func)
        self.feature_names.append(f"unigram_{name}_{position}")
    
    def add_bigram_feature(self, name: str, position: int):
        def feature_func(tokens: List[str], pos: int, prev_tag: str, curr_tag: str) -> Optional[str]:
            idx = pos + position
            if 0 <= idx < len(tokens) - 1:
                token1 = tokens[idx]
                token2 = tokens[idx + 1]
                return f"{name}:{token1}_{token2}|{curr_tag}"
            return None
        
        self.feature_functions.append(feature_func)
        self.feature_names.append(f"bigram_{name}_{position}")
    
    def add_transition_feature(self):
        def feature_func(tokens: List[str], pos: int, prev_tag: str, curr_tag: str) -> Optional[str]:
            if prev_tag:
                return f"TRANS:{prev_tag}->{curr_tag}"
            return None
        
        self.feature_functions.append(feature_func)
        self.feature_names.append("transition")
    
    def add_start_feature(self):
        def feature_func(tokens: List[str], pos: int, prev_tag: str, curr_tag: str) -> Optional[str]:
            if pos == 0:
                return f"START:{curr_tag}"
            return None
        
        self.feature_functions.append(feature_func)
        self.feature_names.append("start")
    
    def add_end_feature(self):
        def feature_func(tokens: List[str], pos: int, prev_tag: str, curr_tag: str) -> Optional[str]:
            if pos == len(tokens) - 1:
                return f"END:{curr_tag}"
            return None
        
        self.feature_functions.append(feature_func)
        self.feature_names.append("end")
    
    def add_char_shape_feature(self, position: int):
        def get_shape(token: str) -> str:
            shape = []
            for char in token:
                if char.isupper():
                    shape.append('X')
                elif char.islower():
                    shape.append('x')
                elif char.isdigit():
                    shape.append('d')
                else:
                    shape.append(char)
            return ''.join(shape)
        
        def feature_func(tokens: List[str], pos: int, prev_tag: str, curr_tag: str) -> Optional[str]:
            idx = pos + position
            if 0 <= idx < len(tokens):
                token = tokens[idx]
                shape = get_shape(token)
                return f"SHAPE:{shape}|{curr_tag}"
            return None
        
        self.feature_functions.append(feature_func)
        self.feature_names.append(f"char_shape_{position}")
    
    def add_length_feature(self, position: int):
        def feature_func(tokens: List[str], pos: int, prev_tag: str, curr_tag: str) -> Optional[str]:
            idx = pos + position
            if 0 <= idx < len(tokens):
                token = tokens[idx]
                length = min(len(token), 10)
                return f"LEN:{length}|{curr_tag}"
            return None
        
        self.feature_functions.append(feature_func)
        self.feature_names.append(f"length_{position}")
    
    def add_prefix_feature(self, prefix_len: int, position: int):
        def feature_func(tokens: List[str], pos: int, prev_tag: str, curr_tag: str) -> Optional[str]:
            idx = pos + position
            if 0 <= idx < len(tokens):
                token = tokens[idx]
                prefix = token[:prefix_len] if len(token) >= prefix_len else token
                return f"PREFIX{prefix_len}:{prefix}|{curr_tag}"
            return None
        
        self.feature_functions.append(feature_func)
        self.feature_names.append(f"prefix{prefix_len}_{position}")
    
    def add_suffix_feature(self, suffix_len: int, position: int):
        def feature_func(tokens: List[str], pos: int, prev_tag: str, curr_tag: str) -> Optional[str]:
            idx = pos + position
            if 0 <= idx < len(tokens):
                token = tokens[idx]
                suffix = token[-suffix_len:] if len(token) >= suffix_len else token
                return f"SUFFIX{suffix_len}:{suffix}|{curr_tag}"
            return None
        
        self.feature_functions.append(feature_func)
        self.feature_names.append(f"suffix{suffix_len}_{position}")
    
    def extract_features(
        self,
        tokens: List[str],
        pos: int,
        prev_tag: str,
        curr_tag: str
    ) -> List[str]:
        features = []
        for feature_func in self.feature_functions:
            feature = feature_func(tokens, pos, prev_tag, curr_tag)
            if feature:
                features.append(feature)
        return features
    
    def get_feature_names(self) -> List[str]:
        return self.feature_names.copy()


class CRFModel:
    def __init__(self, tags: Optional[List[str]] = None):
        self.tags = tags or []
        self.weights: Dict[str, float] = defaultdict(float)
        self.feature_template = CRFFeatureTemplate()
        self._trained = False
        self._learning_rate = 0.1
        self._l2_reg = 0.01
        self._max_iter = 100
        self._epsilon = 1e-6
    
    def _setup_default_features(self):
        self.feature_template.add_start_feature()
        self.feature_template.add_end_feature()
        self.feature_template.add_transition_feature()
        
        for i in range(-2, 3):
            self.feature_template.add_unigram_feature(f"word_{i}", i)
        
        for i in range(-1, 1):
            self.feature_template.add_bigram_feature(f"bigram_{i}", i)
        
        for i in range(-1, 2):
            self.feature_template.add_char_shape_feature(i)
            self.feature_template.add_length_feature(i)
        
        for prefix_len in [1, 2, 3]:
            self.feature_template.add_prefix_feature(prefix_len, 0)
        
        for suffix_len in [1, 2, 3]:
            self.feature_template.add_suffix_feature(suffix_len, 0)
    
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
        for tokens, tags in corpus:
            if len(tokens) != len(tags):
                raise ValueError(f"Tokens and tags length mismatch: {len(tokens)} vs {len(tags)}")
            all_tags.update(tags)
        
        self.tags = sorted(list(all_tags))
        
        if not self.feature_template.feature_functions:
            self._setup_default_features()
        
        self._learning_rate = learning_rate
        self._l2_reg = l2_reg
        self._max_iter = max_iter
        self._epsilon = epsilon
        
        prev_loss = float('inf')
        
        for iteration in range(max_iter):
            total_loss = 0.0
            gradient: Dict[str, float] = defaultdict(float)
            
            for tokens, tags in corpus:
                seq_features = self._extract_sequence_features(tokens)
                
                log_probs = self._forward(tokens, seq_features)
                backward = self._backward(tokens, seq_features)
                
                partition = self._log_sum_exp(log_probs[-1].values())
                
                for pos in range(len(tokens)):
                    for tag in self.tags:
                        expected = math.exp(log_probs[pos][tag] + backward[pos][tag] - partition)
                        
                        for feature in seq_features[pos][tag]:
                            gradient[feature] -= expected
                
                for pos in range(len(tokens)):
                    curr_tag = tags[pos]
                    prev_tag = tags[pos - 1] if pos > 0 else None
                    
                    features = self.feature_template.extract_features(
                        tokens, pos, prev_tag or '', curr_tag
                    )
                    
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
    
    def _extract_sequence_features(self, tokens: List[str]) -> List[Dict[str, List[str]]]:
        seq_features = []
        
        for pos in range(len(tokens)):
            pos_features: Dict[str, List[str]] = {}
            
            for tag in self.tags:
                prev_tag = ''
                features = self.feature_template.extract_features(tokens, pos, prev_tag, tag)
                pos_features[tag] = features
            
            seq_features.append(pos_features)
        
        return seq_features
    
    def _compute_score(self, features: List[str]) -> float:
        score = 0.0
        for feature in features:
            score += self.weights.get(feature, 0.0)
        return score
    
    def _forward(self, tokens: List[str], seq_features: List[Dict[str, List[str]]]) -> List[Dict[str, float]]:
        length = len(tokens)
        alpha = [{} for _ in range(length)]
        
        for tag in self.tags:
            features = seq_features[0][tag]
            alpha[0][tag] = self._compute_score(features)
        
        for pos in range(1, length):
            for curr_tag in self.tags:
                scores = []
                
                for prev_tag in self.tags:
                    trans_features = self.feature_template.extract_features(
                        tokens, pos, prev_tag, curr_tag
                    )
                    trans_score = self._compute_score(trans_features)
                    scores.append(alpha[pos - 1][prev_tag] + trans_score)
                
                alpha[pos][curr_tag] = self._log_sum_exp(scores)
        
        return alpha
    
    def _backward(self, tokens: List[str], seq_features: List[Dict[str, List[str]]]) -> List[Dict[str, float]]:
        length = len(tokens)
        beta = [{} for _ in range(length)]
        
        for tag in self.tags:
            beta[length - 1][tag] = 0.0
        
        for pos in range(length - 2, -1, -1):
            for prev_tag in self.tags:
                scores = []
                
                for curr_tag in self.tags:
                    trans_features = self.feature_template.extract_features(
                        tokens, pos + 1, prev_tag, curr_tag
                    )
                    trans_score = self._compute_score(trans_features)
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
    
    def viterbi(self, tokens: List[str]) -> List[str]:
        if not tokens:
            return []
        
        if not self._trained:
            raise RuntimeError("Model has not been trained. Call train() first.")
        
        length = len(tokens)
        seq_features = self._extract_sequence_features(tokens)
        
        viterbi_scores = [{} for _ in range(length)]
        backpointers = [{} for _ in range(length)]
        
        for tag in self.tags:
            features = seq_features[0][tag]
            viterbi_scores[0][tag] = self._compute_score(features)
            backpointers[0][tag] = None
        
        for pos in range(1, length):
            for curr_tag in self.tags:
                best_score = float('-inf')
                best_prev_tag = None
                
                for prev_tag in self.tags:
                    trans_features = self.feature_template.extract_features(
                        tokens, pos, prev_tag, curr_tag
                    )
                    trans_score = self._compute_score(trans_features)
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
        
        path = [best_final_tag]
        for pos in range(length - 1, 0, -1):
            path.append(backpointers[pos][path[-1]])
        
        path.reverse()
        return path
    
    def predict(self, tokens: List[str]) -> List[str]:
        return self.viterbi(tokens)
    
    def predict_with_scores(self, tokens: List[str]) -> Tuple[List[str], List[Dict[str, float]]]:
        if not tokens:
            return [], []
        
        tags = self.viterbi(tokens)
        
        seq_features = self._extract_sequence_features(tokens)
        scores = []
        
        for pos, tag in enumerate(tags):
            score = self._compute_score(seq_features[pos][tag])
            scores.append({tag: score})
        
        return tags, scores
    
    def save_model(self, filepath: str):
        if not self._trained:
            raise RuntimeError("Model has not been trained. Call train() first before saving.")
        
        model_data = {
            'tags': self.tags,
            'weights': dict(self.weights),
            'feature_names': self.feature_template.get_feature_names(),
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
        
        if not self.feature_template.feature_functions:
            self._setup_default_features()
    
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
            'num_feature_templates': len(self.feature_template.feature_functions),
            'learning_rate': self._learning_rate,
            'l2_reg': self._l2_reg,
            'max_iter': self._max_iter
        }


class CRFSegmentor:
    STATE_B = 'B'
    STATE_M = 'M'
    STATE_E = 'E'
    STATE_S = 'S'
    
    STATES = [STATE_B, STATE_M, STATE_E, STATE_S]
    
    def __init__(self):
        self.model = CRFModel(tags=self.STATES)
        self._trained = False
    
    def _get_state_sequence(self, word: str) -> List[str]:
        length = len(word)
        if length == 1:
            return [self.STATE_S]
        elif length == 2:
            return [self.STATE_B, self.STATE_E]
        else:
            return [self.STATE_B] + [self.STATE_M] * (length - 2) + [self.STATE_E]
    
    def _tokens_to_chars(self, tokens: List[str]) -> Tuple[List[str], List[str]]:
        chars = []
        states = []
        
        for token in tokens:
            if not token:
                continue
            token_states = self._get_state_sequence(token)
            chars.extend(list(token))
            states.extend(token_states)
        
        return chars, states
    
    def train(
        self,
        corpus: List[List[str]],
        learning_rate: float = 0.1,
        l2_reg: float = 0.01,
        max_iter: int = 100,
        epsilon: float = 1e-6,
        verbose: bool = True
    ):
        char_corpus = []
        
        for tokens in corpus:
            if not tokens:
                continue
            chars, states = self._tokens_to_chars(tokens)
            char_corpus.append((chars, states))
        
        if not char_corpus:
            raise ValueError("No valid training data")
        
        self.model.train(
            char_corpus,
            learning_rate=learning_rate,
            l2_reg=l2_reg,
            max_iter=max_iter,
            epsilon=epsilon,
            verbose=verbose
        )
        self._trained = True
    
    def segment(self, text: str) -> List[str]:
        if not text:
            return []
        
        if not self._trained:
            raise RuntimeError("Model has not been trained. Call train() first.")
        
        chars = list(text)
        states = self.model.predict(chars)
        
        if not states:
            return list(text)
        
        words = []
        word_start = 0
        
        for i, state in enumerate(states):
            if state == self.STATE_S:
                words.append(text[i])
                word_start = i + 1
            elif state == self.STATE_E:
                words.append(text[word_start:i + 1])
                word_start = i + 1
            elif state == self.STATE_B:
                word_start = i
        
        if word_start < len(text) and states and states[-1] not in (self.STATE_E, self.STATE_S):
            words.append(text[word_start:])
        
        return words
    
    def segment_with_states(self, text: str) -> List[Tuple[str, str]]:
        if not text:
            return []
        
        chars = list(text)
        states = self.model.predict(chars)
        
        return list(zip(chars, states))
    
    def save_model(self, filepath: str):
        if not self._trained:
            raise RuntimeError("Model has not been trained. Call train() first before saving.")
        self.model.save_model(filepath)
    
    def load_model(self, filepath: str):
        self.model.load_model(filepath)
        self._trained = True
    
    def is_trained(self) -> bool:
        return self._trained
    
    def get_model_info(self) -> Dict[str, Any]:
        return self.model.get_model_info()


__all__ = ['CRFModel', 'CRFSegmentor', 'CRFFeatureTemplate']
