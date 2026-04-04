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
        self.ngram_counts = defaultdict(lambda: defaultdict(int))
        for k, v in model_data['ngram_counts'].items():
            self.ngram_counts[k] = defaultdict(int, v)
        self.context_counts = defaultdict(lambda: defaultdict(int))
        for k, v in model_data['context_counts'].items():
            self.context_counts[k] = defaultdict(int, v)
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
        self._bigram_freq: Dict[Tuple[str, str], int] = defaultdict(int)
        self._unigram_freq: Dict[str, int] = defaultdict(int)
        self._total_bigrams = 0
    
    def train(self, corpus: List[List[str]], smoothing: str = 'laplace', alpha: float = 1.0):
        super().train(corpus, smoothing, alpha)
        
        self._bigram_freq.clear()
        self._unigram_freq.clear()
        self._total_bigrams = 0
        
        for sentence in corpus:
            if not sentence:
                continue
            
            for i, word in enumerate(sentence):
                self._unigram_freq[word] += 1
                
                if i > 0:
                    bigram = (sentence[i - 1], word)
                    self._bigram_freq[bigram] += 1
                    self._total_bigrams += 1
    
    def get_bigram_count(self, word1: str, word2: str) -> int:
        return self._bigram_freq.get((word1, word2), 0)
    
    def get_unigram_count(self, word: str) -> int:
        return self._unigram_freq.get(word, 0)
    
    def get_bigram_frequency(self, word1: str, word2: str) -> float:
        if self._total_bigrams == 0:
            return 0.0
        return self._bigram_freq.get((word1, word2), 0) / self._total_bigrams
    
    def conditional_probability(self, word1: str, word2: str) -> float:
        count_w1 = self.get_unigram_count(word1)
        if count_w1 == 0:
            return 0.0
        
        bigram_count = self.get_bigram_count(word1, word2)
        return bigram_count / count_w1
    
    def joint_probability(self, word1: str, word2: str) -> float:
        if self._total_bigrams == 0:
            return 0.0
        return self.get_bigram_count(word1, word2) / self._total_bigrams
    
    def pmi(self, word1: str, word2: str) -> float:
        p_xy = self.joint_probability(word1, word2)
        if p_xy == 0:
            return float('-inf')
        
        total_unigrams = sum(self._unigram_freq.values())
        if total_unigrams == 0:
            return float('-inf')
        
        p_x = self.get_unigram_count(word1) / total_unigrams
        p_y = self.get_unigram_count(word2) / total_unigrams
        
        if p_x == 0 or p_y == 0:
            return float('-inf')
        
        return math.log(p_xy / (p_x * p_y))
    
    def pmi_normalized(self, word1: str, word2: str) -> float:
        p_xy = self.joint_probability(word1, word2)
        if p_xy == 0:
            return -1.0
        
        if p_xy >= 1.0:
            return 1.0
        
        total_unigrams = sum(self._unigram_freq.values())
        if total_unigrams == 0:
            return -1.0
        
        p_x = self.get_unigram_count(word1) / total_unigrams
        p_y = self.get_unigram_count(word2) / total_unigrams
        
        if p_x == 0 or p_y == 0:
            return -1.0
        
        pmi = math.log(p_xy / (p_x * p_y))
        denominator = -math.log(p_xy)
        
        if denominator == 0:
            return 1.0 if pmi > 0 else -1.0
        
        npmi = pmi / denominator
        return max(-1.0, min(1.0, npmi))
    
    def extract_collocations(
        self,
        min_freq: int = 2,
        min_pmi: float = 0.0,
        top_k: int = 20
    ) -> List[Tuple[str, str, int, float]]:
        if not self._trained:
            raise RuntimeError("Model has not been trained. Call train() first.")
        
        collocations = []
        
        for (word1, word2), count in self._bigram_freq.items():
            if count < min_freq:
                continue
            
            pmi_score = self.pmi(word1, word2)
            
            if pmi_score >= min_pmi:
                collocations.append((word1, word2, count, pmi_score))
        
        collocations.sort(key=lambda x: x[3], reverse=True)
        
        return collocations[:top_k]
    
    def get_frequent_bigrams(
        self,
        min_freq: int = 2,
        top_k: int = 20
    ) -> List[Tuple[str, str, int]]:
        bigrams = [
            (w1, w2, count)
            for (w1, w2), count in self._bigram_freq.items()
            if count >= min_freq
        ]
        
        bigrams.sort(key=lambda x: x[2], reverse=True)
        
        return bigrams[:top_k]
    
    def get_word_collocations(
        self,
        word: str,
        position: str = 'after',
        min_freq: int = 1,
        top_k: int = 10
    ) -> List[Tuple[str, int, float]]:
        if position not in ('before', 'after'):
            raise ValueError("position must be 'before' or 'after'")
        
        collocations = []
        
        if position == 'after':
            for (w1, w2), count in self._bigram_freq.items():
                if w1 == word and count >= min_freq:
                    pmi_score = self.pmi(w1, w2)
                    collocations.append((w2, count, pmi_score))
        else:
            for (w1, w2), count in self._bigram_freq.items():
                if w2 == word and count >= min_freq:
                    pmi_score = self.pmi(w1, w2)
                    collocations.append((w1, count, pmi_score))
        
        collocations.sort(key=lambda x: x[2], reverse=True)
        
        return collocations[:top_k]
    
    def save_model(self, filepath: str):
        super().save_model(filepath)
        
        model_data = {
            'bigram_freq': dict(self._bigram_freq),
            'unigram_freq': dict(self._unigram_freq),
            'total_bigrams': self._total_bigrams
        }
        
        bigram_filepath = filepath.replace('.pkl', '_bigram.pkl')
        with open(bigram_filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str):
        super().load_model(filepath)
        
        bigram_filepath = filepath.replace('.pkl', '_bigram.pkl')
        if os.path.exists(bigram_filepath):
            with open(bigram_filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self._bigram_freq = defaultdict(int, model_data['bigram_freq'])
            self._unigram_freq = defaultdict(int, model_data['unigram_freq'])
            self._total_bigrams = model_data['total_bigrams']
    
    def get_model_info(self) -> Dict[str, any]:
        info = super().get_model_info()
        if self._trained:
            info['total_bigrams'] = self._total_bigrams
            info['unique_bigrams'] = len(self._bigram_freq)
            info['vocabulary_size'] = len(self._unigram_freq)
        return info


class TrigramModel(NGramModel):
    def __init__(self):
        super().__init__(n=3)


__all__ = ['NGramModel', 'BigramModel', 'TrigramModel']
