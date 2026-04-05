from typing import List, Tuple, Dict, Optional
import pickle
import math
from collections import defaultdict
import os
import hashlib
import hmac

HMM_MODEL_SIGNATURE = b'AuroraNLP_HMM_MODEL_v1'


class HMMSegmentor:
    STATE_B = 'B'
    STATE_M = 'M'
    STATE_E = 'E'
    STATE_S = 'S'
    
    STATES = [STATE_B, STATE_M, STATE_E, STATE_S]
    
    def __init__(self):
        self.init_prob: Dict[str, float] = {}
        self.trans_prob: Dict[str, Dict[str, float]] = {}
        self.emit_prob: Dict[str, Dict[str, float]] = {}
        
        self.init_count: Dict[str, int] = defaultdict(int)
        self.trans_count: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.emit_count: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        self.state_count: Dict[str, int] = defaultdict(int)
        self.total_states = 0
        
        self._trained = False
        self._smooth = 1.0
    
    def _get_state_sequence(self, word: str) -> List[str]:
        length = len(word)
        if length == 1:
            return [self.STATE_S]
        elif length == 2:
            return [self.STATE_B, self.STATE_E]
        else:
            return [self.STATE_B] + [self.STATE_M] * (length - 2) + [self.STATE_E]
    
    def train(self, corpus: List[List[str]], smooth: float = 1.0):
        self.init_count.clear()
        self.trans_count.clear()
        self.emit_count.clear()
        self.state_count.clear()
        self.total_states = 0
        
        for sentence in corpus:
            if not sentence:
                continue
            
            char_state_pairs = []
            for word in sentence:
                if not word:
                    continue
                states = self._get_state_sequence(word)
                for char, state in zip(word, states):
                    char_state_pairs.append((char, state))
            
            if not char_state_pairs:
                continue
            
            first_char, first_state = char_state_pairs[0]
            self.init_count[first_state] += 1
            
            for i, (char, state) in enumerate(char_state_pairs):
                self.emit_count[state][char] += 1
                self.state_count[state] += 1
                self.total_states += 1
                
                if i > 0:
                    prev_state = char_state_pairs[i - 1][1]
                    self.trans_count[prev_state][state] += 1
        
        self._calculate_probabilities(smooth)
        self._trained = True
    
    def _calculate_probabilities(self, smooth: float):
        total_init = sum(self.init_count.values())
        for state in self.STATES:
            if total_init > 0:
                self.init_prob[state] = math.log(
                    (self.init_count[state] + smooth) / (total_init + smooth * len(self.STATES))
                )
            else:
                self.init_prob[state] = math.log(1.0 / len(self.STATES))
        
        for prev_state in self.STATES:
            total_trans = sum(self.trans_count[prev_state].values())
            self.trans_prob[prev_state] = {}
            for curr_state in self.STATES:
                if total_trans > 0:
                    self.trans_prob[prev_state][curr_state] = math.log(
                        (self.trans_count[prev_state][curr_state] + smooth) / 
                        (total_trans + smooth * len(self.STATES))
                    )
                else:
                    self.trans_prob[prev_state][curr_state] = math.log(1.0 / len(self.STATES))
        
        for state in self.STATES:
            total_emit = sum(self.emit_count[state].values())
            self.emit_prob[state] = {}
            for char, count in self.emit_count[state].items():
                if total_emit > 0:
                    self.emit_prob[state][char] = math.log((count + smooth) / (total_emit + smooth))
        
        self._smooth = smooth
    
    def _get_emit_prob(self, state: str, char: str) -> float:
        if char in self.emit_prob[state]:
            return self.emit_prob[state][char]
        
        total_emit = sum(self.emit_count[state].values())
        vocab_size = len(self.emit_count[state])
        
        if total_emit > 0 and vocab_size > 0:
            return math.log(self._smooth / (total_emit + self._smooth * vocab_size))
        else:
            return math.log(1.0 / len(self.STATES))
    
    def viterbi(self, text: str) -> List[str]:
        if not text:
            return []
        
        if not self._trained:
            raise RuntimeError("Model has not been trained. Call train() first.")
        
        length = len(text)
        
        V = [{} for _ in range(length)]
        path = {}
        
        first_char = text[0]
        for state in self.STATES:
            V[0][state] = self.init_prob[state] + self._get_emit_prob(state, first_char)
            path[state] = [state]
        
        for t in range(1, length):
            char = text[t]
            new_path = {}
            
            for curr_state in self.STATES:
                emit_prob = self._get_emit_prob(curr_state, char)
                
                best_prob = float('-inf')
                best_prev_state = None
                
                for prev_state in self.STATES:
                    prob = V[t - 1][prev_state] + self.trans_prob[prev_state][curr_state] + emit_prob
                    if prob > best_prob:
                        best_prob = prob
                        best_prev_state = prev_state
                
                V[t][curr_state] = best_prob
                new_path[curr_state] = path[best_prev_state] + [curr_state]
            
            path = new_path
        
        best_final_prob = float('-inf')
        best_final_state = None
        for state in self.STATES:
            if V[length - 1][state] > best_final_prob:
                best_final_prob = V[length - 1][state]
                best_final_state = state
        
        if best_final_state is None:
            return [self.STATE_S] * length
        
        return path[best_final_state]
    
    def segment(self, text: str) -> List[str]:
        if not text:
            return []
        
        states = self.viterbi(text)
        
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
        
        states = self.viterbi(text)
        
        result = []
        for char, state in zip(text, states):
            result.append((char, state))
        
        return result
    
    def save_model(self, filepath: str, key: Optional[str] = None):
        if not self._trained:
            raise RuntimeError("Model has not been trained. Call train() first before saving.")
        
        model_data = {
            'init_prob': self.init_prob,
            'trans_prob': self.trans_prob,
            'emit_prob': self.emit_prob,
            'init_count': dict(self.init_count),
            'trans_count': {k: dict(v) for k, v in self.trans_count.items()},
            'emit_count': {k: dict(v) for k, v in self.emit_count.items()},
            'state_count': dict(self.state_count),
            'total_states': self.total_states,
            'smooth': self._smooth,
            'trained': self._trained
        }
        
        serialized = pickle.dumps(model_data)
        
        dir_path = os.path.dirname(filepath)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            f.write(HMM_MODEL_SIGNATURE)
            f.write(len(serialized).to_bytes(8, 'big'))
            f.write(serialized)
            
            if key:
                signature = hmac.new(
                    key.encode('utf-8'),
                    HMM_MODEL_SIGNATURE + serialized,
                    hashlib.sha256
                ).digest()
                f.write(signature)
    
    def load_model(self, filepath: str, key: Optional[str] = None, verify: bool = True):
        with open(filepath, 'rb') as f:
            signature = f.read(len(HMM_MODEL_SIGNATURE))
            if signature != HMM_MODEL_SIGNATURE:
                raise ValueError("Invalid model file format or corrupted file")
            
            size_bytes = f.read(8)
            size = int.from_bytes(size_bytes, 'big')
            serialized = f.read(size)
            
            if verify and key:
                stored_signature = f.read(32)
                expected_signature = hmac.new(
                    key.encode('utf-8'),
                    HMM_MODEL_SIGNATURE + serialized,
                    hashlib.sha256
                ).digest()
                if not hmac.compare_digest(stored_signature, expected_signature):
                    raise ValueError("Model signature verification failed. File may be tampered.")
        
        model_data = pickle.loads(serialized)
        
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
        self.state_count = defaultdict(int, model_data['state_count'])
        self.total_states = model_data['total_states']
        self._smooth = model_data['smooth']
        self._trained = model_data['trained']
    
    def is_trained(self) -> bool:
        return self._trained
    
    def get_model_info(self) -> Dict[str, any]:
        if not self._trained:
            return {'trained': False}
        
        vocab_sizes = {state: len(self.emit_count[state]) for state in self.STATES}
        
        return {
            'trained': True,
            'total_states': self.total_states,
            'state_counts': dict(self.state_count),
            'vocabulary_sizes': vocab_sizes,
            'smooth': self._smooth
        }


def train_from_file(model: HMMSegmentor, filepath: str, encoding: str = 'utf-8') -> None:
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


__all__ = ['HMMSegmentor', 'train_from_file']
