from typing import List, Tuple, Dict, Optional, Any, Callable
from collections import defaultdict
import pickle
import os
import math


class PerceptronFeatureTemplate:
    
    def __init__(self):
        self.feature_functions: List[Callable] = []
        self.feature_names: List[str] = []
    
    def add_unigram_feature(self, name: str, position: int):
        def feature_func(chars: List[str], pos: int, prev_tag: str, curr_tag: str) -> Optional[str]:
            idx = pos + position
            if 0 <= idx < len(chars):
                char = chars[idx]
                return f"{name}:{char}|{curr_tag}"
            return None
        
        self.feature_functions.append(feature_func)
        self.feature_names.append(f"unigram_{name}_{position}")
    
    def add_bigram_feature(self, name: str, position: int):
        def feature_func(chars: List[str], pos: int, prev_tag: str, curr_tag: str) -> Optional[str]:
            idx = pos + position
            if 0 <= idx < len(chars) - 1:
                char1 = chars[idx]
                char2 = chars[idx + 1]
                return f"{name}:{char1}{char2}|{curr_tag}"
            return None
        
        self.feature_functions.append(feature_func)
        self.feature_names.append(f"bigram_{name}_{position}")
    
    def add_transition_feature(self):
        def feature_func(chars: List[str], pos: int, prev_tag: str, curr_tag: str) -> Optional[str]:
            if prev_tag:
                return f"T:{prev_tag}->{curr_tag}"
            return None
        
        self.feature_functions.append(feature_func)
        self.feature_names.append("transition")
    
    def add_start_feature(self):
        def feature_func(chars: List[str], pos: int, prev_tag: str, curr_tag: str) -> Optional[str]:
            if pos == 0:
                return f"START:{curr_tag}"
            return None
        
        self.feature_functions.append(feature_func)
        self.feature_names.append("start")
    
    def add_end_feature(self):
        def feature_func(chars: List[str], pos: int, prev_tag: str, curr_tag: str) -> Optional[str]:
            if pos == len(chars) - 1:
                return f"END:{curr_tag}"
            return None
        
        self.feature_functions.append(feature_func)
        self.feature_names.append("end")
    
    def add_char_type_feature(self, position: int):
        def get_char_type(char: str) -> str:
            if char.isdigit():
                return 'D'
            elif char.isalpha():
                if char.isupper():
                    return 'U'
                else:
                    return 'L'
            elif '\u4e00' <= char <= '\u9fff':
                return 'C'
            else:
                return 'O'
        
        def feature_func(chars: List[str], pos: int, prev_tag: str, curr_tag: str) -> Optional[str]:
            idx = pos + position
            if 0 <= idx < len(chars):
                char_type = get_char_type(chars[idx])
                return f"TYPE:{char_type}|{curr_tag}"
            return None
        
        self.feature_functions.append(feature_func)
        self.feature_names.append(f"char_type_{position}")
    
    def add_length_context_feature(self, window: int):
        def feature_func(chars: List[str], pos: int, prev_tag: str, curr_tag: str) -> Optional[str]:
            start = max(0, pos - window)
            end = min(len(chars), pos + window + 1)
            context_len = end - start
            return f"CTXLEN:{context_len}|{curr_tag}"
        
        self.feature_functions.append(feature_func)
        self.feature_names.append(f"length_context_{window}")
    
    def extract_features(
        self,
        chars: List[str],
        pos: int,
        prev_tag: str,
        curr_tag: str
    ) -> List[str]:
        features = []
        for feature_func in self.feature_functions:
            feature = feature_func(chars, pos, prev_tag, curr_tag)
            if feature:
                features.append(feature)
        return features
    
    def get_feature_names(self) -> List[str]:
        return self.feature_names.copy()


class StructuredPerceptron:
    
    def __init__(self, tags: Optional[List[str]] = None):
        self.tags = tags or []
        self.weights: Dict[str, float] = defaultdict(float)
        self._accumulated_weights: Dict[str, float] = defaultdict(float)
        self.feature_template = PerceptronFeatureTemplate()
        self._trained = False
        self._learning_rate = 1.0
        self._max_iter = 10
        self._num_updates = 0
        self._averaged = True
    
    def _setup_default_features(self):
        self.feature_template.add_start_feature()
        self.feature_template.add_end_feature()
        self.feature_template.add_transition_feature()
        
        for i in range(-2, 3):
            self.feature_template.add_unigram_feature(f"c{i}", i)
        
        for i in range(-1, 2):
            self.feature_template.add_bigram_feature(f"cc{i}", i)
        
        for i in range(-1, 2):
            self.feature_template.add_char_type_feature(i)
    
    def _get_state_sequence(self, word: str) -> List[str]:
        length = len(word)
        if length == 1:
            return ['S']
        elif length == 2:
            return ['B', 'E']
        else:
            return ['B'] + ['M'] * (length - 2) + ['E']
    
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
    
    def _compute_score(self, features: List[str]) -> float:
        score = 0.0
        for feature in features:
            score += self.weights.get(feature, 0.0)
        return score
    
    def viterbi(self, chars: List[str]) -> List[str]:
        if not chars:
            return []
        
        length = len(chars)
        viterbi_scores = [{} for _ in range(length)]
        backpointers = [{} for _ in range(length)]
        
        for tag in self.tags:
            features = self.feature_template.extract_features(chars, 0, '', tag)
            viterbi_scores[0][tag] = self._compute_score(features)
            backpointers[0][tag] = None
        
        for pos in range(1, length):
            for curr_tag in self.tags:
                best_score = float('-inf')
                best_prev_tag = None
                
                for prev_tag in self.tags:
                    features = self.feature_template.extract_features(
                        chars, pos, prev_tag, curr_tag
                    )
                    score = viterbi_scores[pos - 1][prev_tag] + self._compute_score(features)
                    
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
            return ['S'] * length
        
        path = [best_final_tag]
        for pos in range(length - 1, 0, -1):
            path.append(backpointers[pos][path[-1]])
        
        path.reverse()
        return path
    
    def _update_weights(
        self,
        chars: List[str],
        gold_tags: List[str],
        pred_tags: List[str]
    ):
        for pos in range(len(chars)):
            gold_tag = gold_tags[pos]
            pred_tag = pred_tags[pos]
            
            if gold_tag != pred_tag:
                prev_gold = gold_tags[pos - 1] if pos > 0 else ''
                prev_pred = pred_tags[pos - 1] if pos > 0 else ''
                
                gold_features = self.feature_template.extract_features(
                    chars, pos, prev_gold, gold_tag
                )
                pred_features = self.feature_template.extract_features(
                    chars, pos, prev_pred, pred_tag
                )
                
                for feature in gold_features:
                    self.weights[feature] += self._learning_rate
                    self._accumulated_weights[feature] += self._num_updates * self._learning_rate
                
                for feature in pred_features:
                    self.weights[feature] -= self._learning_rate
                    self._accumulated_weights[feature] -= self._num_updates * self._learning_rate
                
                self._num_updates += 1
    
    def _apply_averaging(self):
        if not self._averaged or self._num_updates == 0:
            return
        
        for feature in self.weights:
            self.weights[feature] -= self._accumulated_weights[feature] / self._num_updates
    
    def train(
        self,
        corpus: List[List[str]],
        learning_rate: float = 1.0,
        max_iter: int = 10,
        averaged: bool = True,
        verbose: bool = True
    ):
        if not corpus:
            raise ValueError("Training corpus cannot be empty")
        
        self.tags = ['B', 'M', 'E', 'S']
        
        if not self.feature_template.feature_functions:
            self._setup_default_features()
        
        self._learning_rate = learning_rate
        self._max_iter = max_iter
        self._averaged = averaged
        
        for iteration in range(max_iter):
            correct = 0
            total = 0
            
            for tokens in corpus:
                if not tokens:
                    continue
                
                chars, gold_tags = self._tokens_to_chars(tokens)
                pred_tags = self.viterbi(chars)
                
                if gold_tags != pred_tags:
                    self._update_weights(chars, gold_tags, pred_tags)
                else:
                    correct += 1
                total += 1
            
            if verbose:
                accuracy = correct / total if total > 0 else 0
                print(f"Iteration {iteration + 1}/{max_iter}, Accuracy: {accuracy:.4f}")
        
        if averaged:
            self._apply_averaging()
        
        self._trained = True
    
    def train_online(
        self,
        tokens: List[str],
        update_weights: bool = True
    ) -> Tuple[bool, float]:
        if not tokens:
            return True, 1.0
        
        if not self.feature_template.feature_functions:
            self._setup_default_features()
        
        chars, gold_tags = self._tokens_to_chars(tokens)
        pred_tags = self.viterbi(chars)
        
        is_correct = gold_tags == pred_tags
        
        if update_weights and not is_correct:
            self._update_weights(chars, gold_tags, pred_tags)
        
        correct_count = sum(1 for g, p in zip(gold_tags, pred_tags) if g == p)
        accuracy = correct_count / len(gold_tags) if gold_tags else 1.0
        
        return is_correct, accuracy
    
    def partial_fit(
        self,
        corpus: List[List[str]],
        learning_rate: Optional[float] = None,
        verbose: bool = False
    ):
        if not self.tags:
            self.tags = ['B', 'M', 'E', 'S']
        
        if not self.feature_template.feature_functions:
            self._setup_default_features()
        
        if learning_rate is not None:
            self._learning_rate = learning_rate
        
        correct = 0
        total = 0
        
        for tokens in corpus:
            if not tokens:
                continue
            
            is_correct, accuracy = self.train_online(tokens, update_weights=True)
            if is_correct:
                correct += 1
            total += 1
        
        if verbose:
            print(f"Partial fit completed. Accuracy: {correct / total if total > 0 else 0:.4f}")
        
        self._trained = True
    
    def predict(self, chars: List[str]) -> List[str]:
        return self.viterbi(chars)
    
    def save_model(self, filepath: str):
        if not self._trained:
            raise RuntimeError("Model has not been trained. Call train() first before saving.")
        
        model_data = {
            'tags': self.tags,
            'weights': dict(self.weights),
            'feature_names': self.feature_template.get_feature_names(),
            'learning_rate': self._learning_rate,
            'max_iter': self._max_iter,
            'num_updates': self._num_updates,
            'averaged': self._averaged,
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
        self._max_iter = model_data['max_iter']
        self._num_updates = model_data.get('num_updates', 0)
        self._averaged = model_data.get('averaged', True)
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
            'max_iter': self._max_iter,
            'num_updates': self._num_updates,
            'averaged': self._averaged
        }


class PerceptronSegmentor:
    STATE_B = 'B'
    STATE_M = 'M'
    STATE_E = 'E'
    STATE_S = 'S'
    
    STATES = [STATE_B, STATE_M, STATE_E, STATE_S]
    
    def __init__(self):
        self.model = StructuredPerceptron(tags=self.STATES)
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
        learning_rate: float = 1.0,
        max_iter: int = 10,
        averaged: bool = True,
        verbose: bool = True
    ):
        self.model.train(
            corpus,
            learning_rate=learning_rate,
            max_iter=max_iter,
            averaged=averaged,
            verbose=verbose
        )
        self._trained = True
    
    def train_online(self, tokens: List[str], update_weights: bool = True) -> Tuple[bool, float]:
        is_correct, accuracy = self.model.train_online(tokens, update_weights)
        if update_weights:
            self._trained = True
        return is_correct, accuracy
    
    def partial_fit(
        self,
        corpus: List[List[str]],
        learning_rate: Optional[float] = None,
        verbose: bool = False
    ):
        self.model.partial_fit(corpus, learning_rate, verbose)
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


def train_from_file(model: PerceptronSegmentor, filepath: str, encoding: str = 'utf-8') -> None:
    corpus = []
    
    with open(filepath, 'r', encoding=encoding) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            words = line.split()
            if words:
                corpus.append(words)
    
    model.train(corpus)


__all__ = ['StructuredPerceptron', 'PerceptronSegmentor', 'PerceptronFeatureTemplate', 'train_from_file']
