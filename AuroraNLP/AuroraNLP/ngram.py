from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import math
import pickle
import os


class NGramModel:
    START_TOKEN = '<s>'
    END_TOKEN = '</s>'
    UNK_TOKEN = '<UNK>'
    
    def __init__(self, n: int = 2):
        if n < 1:
            raise ValueError("n must be at least 1")
        self.n = n
        self.ngram_counts: Dict[int, Dict[Tuple[str, ...], int]] = defaultdict(lambda: defaultdict(int))
        self.context_counts: Dict[int, Dict[Tuple[str, ...], int]] = defaultdict(lambda: defaultdict(int))
        self.vocabulary: Dict[str, int] = defaultdict(int)
        self.total_tokens = 0
        self._trained = False
        self._smoothing = 'laplace'
        self._alpha = 1.0
    
    def _tokenize_sentence(self, sentence: List[str]) -> List[str]:
        tokens = [self.START_TOKEN] * (self.n - 1) + sentence + [self.END_TOKEN]
        return tokens
    
    def train(self, corpus: List[List[str]], smoothing: str = 'laplace', alpha: float = 1.0):
        self.ngram_counts.clear()
        self.context_counts.clear()
        self.vocabulary.clear()
        self.total_tokens = 0
        
        self._smoothing = smoothing
        self._alpha = alpha
        
        for sentence in corpus:
            if not sentence:
                continue
            
            tokens = self._tokenize_sentence(sentence)
            
            for token in tokens:
                if token not in (self.START_TOKEN, self.END_TOKEN):
                    self.vocabulary[token] += 1
                    self.total_tokens += 1
            
            for i in range(len(tokens) - self.n + 1):
                for j in range(1, self.n + 1):
                    ngram = tuple(tokens[i:i + j])
                    self.ngram_counts[j][ngram] += 1
                    
                    if j > 1:
                        context = tuple(tokens[i:i + j - 1])
                        self.context_counts[j - 1][context] += 1
        
        self.vocabulary[self.UNK_TOKEN] = 0
        self._trained = True
    
    def _get_word_count(self, word: str) -> int:
        return self.vocabulary.get(word, self.vocabulary.get(self.UNK_TOKEN, 0))
    
    def _get_ngram_count(self, ngram: Tuple[str, ...]) -> int:
        return self.ngram_counts[len(ngram)].get(ngram, 0)
    
    def _get_context_count(self, context: Tuple[str, ...]) -> int:
        return self.context_counts[len(context)].get(context, 0)
    
    def _apply_smoothing(
        self,
        ngram_count: int,
        context_count: int,
        vocab_size: int
    ) -> float:
        if self._smoothing == 'laplace':
            return (ngram_count + self._alpha) / (context_count + self._alpha * vocab_size)
        elif self._smoothing == 'none':
            if context_count == 0:
                return 0.0
            return ngram_count / context_count
        else:
            if context_count == 0:
                return 1.0 / vocab_size
            return (ngram_count + self._alpha) / (context_count + self._alpha * vocab_size)
    
    def probability(self, word: str, context: List[str]) -> float:
        if not self._trained:
            raise RuntimeError("Model has not been trained. Call train() first.")
        
        if len(context) < self.n - 1:
            context = [self.START_TOKEN] * (self.n - 1 - len(context)) + context
        
        context = context[-(self.n - 1):]
        
        ngram = tuple(context + [word])
        ngram_count = self._get_ngram_count(ngram)
        context_count = self._get_context_count(tuple(context))
        
        vocab_size = len(self.vocabulary)
        
        prob = self._apply_smoothing(ngram_count, context_count, vocab_size)
        
        return prob
    
    def log_probability(self, word: str, context: List[str]) -> float:
        prob = self.probability(word, context)
        if prob == 0:
            return float('-inf')
        return math.log(prob)
    
    def sentence_probability(self, sentence: List[str]) -> float:
        if not sentence:
            return 0.0
        
        tokens = self._tokenize_sentence(sentence)
        
        log_prob = 0.0
        for i in range(self.n - 1, len(tokens) - 1):
            word = tokens[i]
            context = tokens[i - self.n + 1:i]
            log_prob += self.log_probability(word, context)
        
        return log_prob
    
    def perplexity(self, sentences: List[List[str]]) -> float:
        if not sentences:
            return float('inf')
        
        total_log_prob = 0.0
        total_words = 0
        
        for sentence in sentences:
            if not sentence:
                continue
            
            tokens = self._tokenize_sentence(sentence)
            total_words += len(sentence)
            
            for i in range(self.n - 1, len(tokens) - 1):
                word = tokens[i]
                context = tokens[i - self.n + 1:i]
                total_log_prob += self.log_probability(word, context)
        
        if total_words == 0:
            return float('inf')
        
        avg_log_prob = total_log_prob / total_words
        perplexity = math.exp(-avg_log_prob)
        
        return perplexity
    
    def predict_next_word(self, context: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        if not self._trained:
            raise RuntimeError("Model has not been trained. Call train() first.")
        
        if len(context) < self.n - 1:
            context = [self.START_TOKEN] * (self.n - 1 - len(context)) + context
        
        context = context[-(self.n - 1):]
        
        candidates = []
        for word in self.vocabulary:
            if word in (self.UNK_TOKEN,):
                continue
            
            prob = self.probability(word, context)
            candidates.append((word, prob))
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        return candidates[:top_k]
    
    def disambiguate(
        self,
        candidates: List[List[str]],
        context_before: List[str] = None,
        context_after: List[str] = None
    ) -> Tuple[List[str], float]:
        if not candidates:
            return [], float('-inf')
        
        if not self._trained:
            return candidates[0], 0.0
        
        if context_before is None:
            context_before = []
        if context_after is None:
            context_after = []
        
        best_candidate = None
        best_score = float('-inf')
        
        for candidate in candidates:
            full_context = context_before + candidate + context_after
            score = self.sentence_probability(full_context)
            
            if score > best_score:
                best_score = score
                best_candidate = candidate
        
        return best_candidate, best_score
    
    def save_model(self, filepath: str):
        if not self._trained:
            raise RuntimeError("Model has not been trained. Call train() first before saving.")
        
        model_data = {
            'n': self.n,
            'ngram_counts': {k: dict(v) for k, v in self.ngram_counts.items()},
            'context_counts': {k: dict(v) for k, v in self.context_counts.items()},
            'vocabulary': dict(self.vocabulary),
            'total_tokens': self.total_tokens,
            'smoothing': self._smoothing,
            'alpha': self._alpha,
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
        
        self.n = model_data['n']
        self.ngram_counts = defaultdict(lambda: defaultdict(int), 
                                        {k: defaultdict(int, v) for k, v in model_data['ngram_counts'].items()})
        self.context_counts = defaultdict(lambda: defaultdict(int),
                                          {k: defaultdict(int, v) for k, v in model_data['context_counts'].items()})
        self.vocabulary = defaultdict(int, model_data['vocabulary'])
        self.total_tokens = model_data['total_tokens']
        self._smoothing = model_data['smoothing']
        self._alpha = model_data['alpha']
        self._trained = model_data['trained']
    
    def is_trained(self) -> bool:
        return self._trained
    
    def get_model_info(self) -> Dict[str, any]:
        if not self._trained:
            return {'trained': False}
        
        return {
            'trained': True,
            'n': self.n,
            'vocabulary_size': len(self.vocabulary),
            'total_tokens': self.total_tokens,
            'smoothing': self._smoothing,
            'alpha': self._alpha,
            'ngram_types': {k: len(v) for k, v in self.ngram_counts.items()}
        }


class BigramModel(NGramModel):
    def __init__(self):
        super().__init__(n=2)


class TrigramModel(NGramModel):
    def __init__(self):
        super().__init__(n=3)


__all__ = ['NGramModel', 'BigramModel', 'TrigramModel']
