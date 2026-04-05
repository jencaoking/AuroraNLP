from typing import List, Tuple, Dict, Optional, Any, Set
from collections import defaultdict
import math
import pickle
import os
import re


NER_ENTITY_TYPES = {
    'PER': '人名',
    'LOC': '地名',
    'ORG': '机构名',
    'TIME': '时间',
    'NUM': '数值',
    'MISC': '其他实体',
}

NER_TAGS = ['O']
for entity_type in NER_ENTITY_TYPES.keys():
    NER_TAGS.extend([
        f'B-{entity_type}',
        f'I-{entity_type}',
        f'E-{entity_type}',
        f'S-{entity_type}',
    ])

DEFAULT_NER_TAGS = NER_TAGS.copy()


class Entity:
    def __init__(
        self,
        text: str,
        entity_type: str,
        start: int,
        end: int,
        confidence: float = 1.0
    ):
        self.text = text
        self.entity_type = entity_type
        self.start = start
        self.end = end
        self.confidence = confidence
    
    @property
    def length(self) -> int:
        return self.end - self.start
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'text': self.text,
            'type': self.entity_type,
            'type_name': NER_ENTITY_TYPES.get(self.entity_type, '未知'),
            'start': self.start,
            'end': self.end,
            'length': self.length,
            'confidence': self.confidence
        }
    
    def __repr__(self) -> str:
        return f"Entity('{self.text}', {self.entity_type}, [{self.start}:{self.end}])"
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Entity):
            return False
        return (
            self.text == other.text and
            self.entity_type == other.entity_type and
            self.start == other.start and
            self.end == other.end
        )


class NERFeatureTemplate:
    def __init__(self):
        self.feature_functions = []
        self.feature_names = []
    
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
                return f"TRANS:{prev_tag}->{curr_tag}"
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
            if '\u4e00' <= char <= '\u9fff':
                return 'H'
            elif char.isupper():
                return 'U'
            elif char.islower():
                return 'L'
            elif char.isdigit():
                return 'D'
            elif char.isspace():
                return 'S'
            else:
                return 'P'
        
        def feature_func(chars: List[str], pos: int, prev_tag: str, curr_tag: str) -> Optional[str]:
            idx = pos + position
            if 0 <= idx < len(chars):
                char_type = get_char_type(chars[idx])
                return f"CHARTYPE:{char_type}|{curr_tag}"
            return None
        
        self.feature_functions.append(feature_func)
        self.feature_names.append(f"char_type_{position}")
    
    def add_char_shape_feature(self, window: int = 2):
        def feature_func(chars: List[str], pos: int, prev_tag: str, curr_tag: str) -> Optional[str]:
            start = max(0, pos - window)
            end = min(len(chars), pos + window + 1)
            shape = []
            for i in range(start, end):
                char = chars[i]
                if '\u4e00' <= char <= '\u9fff':
                    shape.append('H')
                elif char.isupper():
                    shape.append('U')
                elif char.islower():
                    shape.append('L')
                elif char.isdigit():
                    shape.append('D')
                else:
                    shape.append('P')
            return f"SHAPE:{''.join(shape)}|{curr_tag}"
        
        self.feature_functions.append(feature_func)
        self.feature_names.append(f"char_shape_{window}")
    
    def add_is_chinese_feature(self, position: int):
        def feature_func(chars: List[str], pos: int, prev_tag: str, curr_tag: str) -> Optional[str]:
            idx = pos + position
            if 0 <= idx < len(chars):
                char = chars[idx]
                is_chinese = '\u4e00' <= char <= '\u9fff'
                return f"IS_CHINESE:{is_chinese}|{curr_tag}"
            return None
        
        self.feature_functions.append(feature_func)
        self.feature_names.append(f"is_chinese_{position}")
    
    def add_is_digit_feature(self, position: int):
        def feature_func(chars: List[str], pos: int, prev_tag: str, curr_tag: str) -> Optional[str]:
            idx = pos + position
            if 0 <= idx < len(chars):
                char = chars[idx]
                is_digit = char.isdigit()
                return f"IS_DIGIT:{is_digit}|{curr_tag}"
            return None
        
        self.feature_functions.append(feature_func)
        self.feature_names.append(f"is_digit_{position}")
    
    def add_is_punctuation_feature(self, position: int):
        def feature_func(chars: List[str], pos: int, prev_tag: str, curr_tag: str) -> Optional[str]:
            idx = pos + position
            if 0 <= idx < len(chars):
                char = chars[idx]
                is_punct = not (char.isalnum() or char.isspace())
                return f"IS_PUNCT:{is_punct}|{curr_tag}"
            return None
        
        self.feature_functions.append(feature_func)
        self.feature_names.append(f"is_punct_{position}")
    
    def add_surname_feature(self, position: int):
        COMMON_SURNAMES = set('王李张刘陈杨赵黄周吴徐孙胡朱高林何郭马罗梁宋郑谢韩唐冯于董萧程曹袁邓许傅沈曾彭吕苏卢蒋蔡贾丁魏薛叶阎余潘杜戴夏钟汪田任姜范方石姚谭廖邹熊金陆郝孔白崔康毛邱秦江史顾侯邵孟龙万段雷钱汤尹黎易常武乔贺赖龚文')
        
        def feature_func(chars: List[str], pos: int, prev_tag: str, curr_tag: str) -> Optional[str]:
            idx = pos + position
            if 0 <= idx < len(chars):
                char = chars[idx]
                is_surname = char in COMMON_SURNAMES
                return f"IS_SURNAME:{is_surname}|{curr_tag}"
            return None
        
        self.feature_functions.append(feature_func)
        self.feature_names.append(f"is_surname_{position}")
    
    def add_place_suffix_feature(self, position: int):
        PLACE_SUFFIXES = set('省市县区镇乡村路街道州岛山江河湖海港湾城关省市区县镇乡村路街道')
        
        def feature_func(chars: List[str], pos: int, prev_tag: str, curr_tag: str) -> Optional[str]:
            idx = pos + position
            if 0 <= idx < len(chars):
                char = chars[idx]
                is_place_suffix = char in PLACE_SUFFIXES
                return f"IS_PLACE_SUFFIX:{is_place_suffix}|{curr_tag}"
            return None
        
        self.feature_functions.append(feature_func)
        self.feature_names.append(f"is_place_suffix_{position}")
    
    def add_org_suffix_feature(self, position: int):
        ORG_SUFFIXES = set('公司集团银行大学学院医院研究所中心局部会社团队厂店院校')
        
        def feature_func(chars: List[str], pos: int, prev_tag: str, curr_tag: str) -> Optional[str]:
            idx = pos + position
            if 0 <= idx < len(chars):
                char = chars[idx]
                is_org_suffix = char in ORG_SUFFIXES
                return f"IS_ORG_SUFFIX:{is_org_suffix}|{curr_tag}"
            return None
        
        self.feature_functions.append(feature_func)
        self.feature_names.append(f"is_org_suffix_{position}")
    
    def add_time_word_feature(self, position: int):
        TIME_WORDS = set('年月日时分秒周天今昨明前后上午下午晚上早晨傍晚')
        
        def feature_func(chars: List[str], pos: int, prev_tag: str, curr_tag: str) -> Optional[str]:
            idx = pos + position
            if 0 <= idx < len(chars):
                char = chars[idx]
                is_time_word = char in TIME_WORDS
                return f"IS_TIME_WORD:{is_time_word}|{curr_tag}"
            return None
        
        self.feature_functions.append(feature_func)
        self.feature_names.append(f"is_time_word_{position}")
    
    def add_number_pattern_feature(self):
        def feature_func(chars: List[str], pos: int, prev_tag: str, curr_tag: str) -> Optional[str]:
            if 0 <= pos < len(chars):
                char = chars[pos]
                if char.isdigit():
                    num_len = 1
                    i = pos + 1
                    while i < len(chars) and chars[i].isdigit():
                        num_len += 1
                        i += 1
                    return f"NUM_LEN:{num_len}|{curr_tag}"
            return None
        
        self.feature_functions.append(feature_func)
        self.feature_names.append("number_pattern")
    
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


class CRFNERModel:
    def __init__(self, tags: Optional[List[str]] = None, entity_types: Optional[Set[str]] = None):
        self.entity_types = entity_types or set(NER_ENTITY_TYPES.keys())
        self.tags = tags or self._generate_tags()
        self.weights: Dict[str, float] = defaultdict(float)
        self.feature_template = NERFeatureTemplate()
        self._trained = False
        self._learning_rate = 0.1
        self._l2_reg = 0.01
        self._max_iter = 100
        self._epsilon = 1e-6
    
    def _generate_tags(self) -> List[str]:
        tags = ['O']
        for entity_type in sorted(self.entity_types):
            tags.extend([
                f'B-{entity_type}',
                f'I-{entity_type}',
                f'E-{entity_type}',
                f'S-{entity_type}',
            ])
        return tags
    
    def _setup_default_features(self):
        self.feature_template.add_start_feature()
        self.feature_template.add_end_feature()
        self.feature_template.add_transition_feature()
        
        for i in range(-2, 3):
            self.feature_template.add_unigram_feature(f"char_{i}", i)
        
        for i in range(-1, 2):
            self.feature_template.add_bigram_feature(f"bigram_{i}", i)
        
        for i in range(-2, 3):
            self.feature_template.add_char_type_feature(i)
        
        self.feature_template.add_char_shape_feature(window=2)
        
        for i in range(-1, 2):
            self.feature_template.add_is_chinese_feature(i)
            self.feature_template.add_is_digit_feature(i)
            self.feature_template.add_is_punctuation_feature(i)
        
        self.feature_template.add_surname_feature(0)
        self.feature_template.add_surname_feature(-1)
        self.feature_template.add_place_suffix_feature(0)
        self.feature_template.add_place_suffix_feature(1)
        self.feature_template.add_org_suffix_feature(0)
        self.feature_template.add_org_suffix_feature(1)
        self.feature_template.add_time_word_feature(0)
        self.feature_template.add_number_pattern_feature()
    
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
        for chars, tags in corpus:
            if len(chars) != len(tags):
                raise ValueError(f"Chars and tags length mismatch: {len(chars)} vs {len(tags)}")
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
            
            for chars, tags in corpus:
                seq_features = self._extract_sequence_features(chars)
                
                log_probs = self._forward(chars, seq_features)
                backward = self._backward(chars, seq_features)
                
                partition = self._log_sum_exp(list(log_probs[-1].values()))
                
                for pos in range(len(chars)):
                    for tag in self.tags:
                        expected = math.exp(log_probs[pos][tag] + backward[pos][tag] - partition)
                        
                        for feature in seq_features[pos][tag]:
                            gradient[feature] -= expected
                
                for pos in range(len(chars)):
                    curr_tag = tags[pos]
                    prev_tag = tags[pos - 1] if pos > 0 else None
                    
                    features = self.feature_template.extract_features(
                        chars, pos, prev_tag or '', curr_tag
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
    
    def _extract_sequence_features(self, chars: List[str]) -> List[Dict[str, List[str]]]:
        seq_features = []
        
        for pos in range(len(chars)):
            pos_features: Dict[str, List[str]] = {}
            
            for tag in self.tags:
                prev_tag = ''
                features = self.feature_template.extract_features(chars, pos, prev_tag, tag)
                pos_features[tag] = features
            
            seq_features.append(pos_features)
        
        return seq_features
    
    def _compute_score(self, features: List[str]) -> float:
        score = 0.0
        for feature in features:
            score += self.weights.get(feature, 0.0)
        return score
    
    def _forward(self, chars: List[str], seq_features: List[Dict[str, List[str]]]) -> List[Dict[str, float]]:
        length = len(chars)
        alpha = [{} for _ in range(length)]
        
        for tag in self.tags:
            features = seq_features[0][tag]
            alpha[0][tag] = self._compute_score(features)
        
        for pos in range(1, length):
            for curr_tag in self.tags:
                scores = []
                
                for prev_tag in self.tags:
                    trans_features = self.feature_template.extract_features(
                        chars, pos, prev_tag, curr_tag
                    )
                    trans_score = self._compute_score(trans_features)
                    scores.append(alpha[pos - 1][prev_tag] + trans_score)
                
                alpha[pos][curr_tag] = self._log_sum_exp(scores)
        
        return alpha
    
    def _backward(self, chars: List[str], seq_features: List[Dict[str, List[str]]]) -> List[Dict[str, float]]:
        length = len(chars)
        beta = [{} for _ in range(length)]
        
        for tag in self.tags:
            beta[length - 1][tag] = 0.0
        
        for pos in range(length - 2, -1, -1):
            for prev_tag in self.tags:
                scores = []
                
                for curr_tag in self.tags:
                    trans_features = self.feature_template.extract_features(
                        chars, pos + 1, prev_tag, curr_tag
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
    
    def viterbi(self, chars: List[str]) -> List[str]:
        if not chars:
            return []
        
        if not self._trained:
            raise RuntimeError("Model has not been trained. Call train() first.")
        
        length = len(chars)
        seq_features = self._extract_sequence_features(chars)
        
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
                        chars, pos, prev_tag, curr_tag
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
    
    def predict(self, text: str) -> List[str]:
        chars = list(text)
        return self.viterbi(chars)
    
    def predict_entities(self, text: str) -> List[Entity]:
        chars = list(text)
        tags = self.viterbi(chars)
        
        return self._extract_entities(chars, tags)
    
    def _extract_entities(self, chars: List[str], tags: List[str]) -> List[Entity]:
        entities = []
        i = 0
        
        while i < len(tags):
            tag = tags[i]
            
            if tag == 'O':
                i += 1
                continue
            
            if tag.startswith('S-'):
                entity_type = tag[2:]
                entities.append(Entity(
                    text=chars[i],
                    entity_type=entity_type,
                    start=i,
                    end=i + 1
                ))
                i += 1
            elif tag.startswith('B-'):
                entity_type = tag[2:]
                start = i
                i += 1
                
                while i < len(tags):
                    curr_tag = tags[i]
                    if curr_tag == f'I-{entity_type}':
                        i += 1
                    elif curr_tag == f'E-{entity_type}':
                        i += 1
                        break
                    else:
                        break
                
                entity_text = ''.join(chars[start:i])
                entities.append(Entity(
                    text=entity_text,
                    entity_type=entity_type,
                    start=start,
                    end=i
                ))
            else:
                i += 1
        
        return entities
    
    def save_model(self, filepath: str):
        if not self._trained:
            raise RuntimeError("Model has not been trained. Call train() first before saving.")
        
        model_data = {
            'tags': self.tags,
            'entity_types': list(self.entity_types),
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
        self.entity_types = set(model_data['entity_types'])
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
            'entity_types': list(self.entity_types),
            'num_features': len(self.weights),
            'num_feature_templates': len(self.feature_template.feature_functions),
            'learning_rate': self._learning_rate,
            'l2_reg': self._l2_reg,
            'max_iter': self._max_iter
        }


class NERRecognizer:
    def __init__(self, model: Optional[CRFNERModel] = None):
        self.model = model or CRFNERModel()
        self._trained = False
    
    def train(
        self,
        corpus: List[Tuple[str, List[Entity]]],
        learning_rate: float = 0.1,
        l2_reg: float = 0.01,
        max_iter: int = 100,
        epsilon: float = 1e-6,
        verbose: bool = True
    ):
        char_corpus = []
        
        for text, entities in corpus:
            chars = list(text)
            tags = self._entities_to_tags(chars, entities)
            char_corpus.append((chars, tags))
        
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
    
    def _entities_to_tags(self, chars: List[str], entities: List[Entity]) -> List[str]:
        tags = ['O'] * len(chars)
        
        for entity in entities:
            start = entity.start
            end = entity.end
            entity_type = entity.entity_type
            length = end - start
            
            if length == 1:
                tags[start] = f'S-{entity_type}'
            elif length == 2:
                tags[start] = f'B-{entity_type}'
                tags[start + 1] = f'E-{entity_type}'
            else:
                tags[start] = f'B-{entity_type}'
                for i in range(start + 1, end - 1):
                    tags[i] = f'I-{entity_type}'
                tags[end - 1] = f'E-{entity_type}'
        
        return tags
    
    def recognize(self, text: str) -> List[Entity]:
        if not text:
            return []
        
        if not self._trained:
            raise RuntimeError("Model has not been trained. Call train() first.")
        
        return self.model.predict_entities(text)
    
    def recognize_with_tags(self, text: str) -> Tuple[List[str], List[str]]:
        chars = list(text)
        tags = self.model.viterbi(chars)
        return chars, tags
    
    def batch_recognize(self, texts: List[str]) -> List[List[Entity]]:
        return [self.recognize(text) for text in texts]
    
    def get_entities_by_type(self, text: str, entity_type: str) -> List[Entity]:
        entities = self.recognize(text)
        return [e for e in entities if e.entity_type == entity_type]
    
    def get_persons(self, text: str) -> List[Entity]:
        return self.get_entities_by_type(text, 'PER')
    
    def get_locations(self, text: str) -> List[Entity]:
        return self.get_entities_by_type(text, 'LOC')
    
    def get_organizations(self, text: str) -> List[Entity]:
        return self.get_entities_by_type(text, 'ORG')
    
    def get_times(self, text: str) -> List[Entity]:
        return self.get_entities_by_type(text, 'TIME')
    
    def get_numbers(self, text: str) -> List[Entity]:
        return self.get_entities_by_type(text, 'NUM')
    
    def annotate_text(self, text: str) -> str:
        entities = self.recognize(text)
        
        if not entities:
            return text
        
        entities = sorted(entities, key=lambda e: e.start)
        
        result = []
        last_end = 0
        
        for entity in entities:
            result.append(text[last_end:entity.start])
            result.append(f'[{entity.text}/{entity.entity_type}]')
            last_end = entity.end
        
        result.append(text[last_end:])
        
        return ''.join(result)
    
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


def create_sample_ner_corpus() -> List[Tuple[str, List[Entity]]]:
    corpus = [
        (
            "张三在北京清华大学学习",
            [
                Entity("张三", "PER", 0, 2),
                Entity("北京", "LOC", 3, 5),
                Entity("清华大学", "ORG", 5, 9),
            ]
        ),
        (
            "李四在上海交通大学工作",
            [
                Entity("李四", "PER", 0, 2),
                Entity("上海", "LOC", 3, 5),
                Entity("交通大学", "ORG", 5, 9),
            ]
        ),
        (
            "王五于2023年加入阿里巴巴集团",
            [
                Entity("王五", "PER", 0, 2),
                Entity("2023年", "TIME", 3, 8),
                Entity("阿里巴巴集团", "ORG", 10, 16),
            ]
        ),
        (
            "小明在广州的腾讯公司上班",
            [
                Entity("小明", "PER", 0, 2),
                Entity("广州", "LOC", 3, 5),
                Entity("腾讯公司", "ORG", 6, 10),
            ]
        ),
        (
            "刘医生在北京协和医院工作",
            [
                Entity("刘医生", "PER", 0, 3),
                Entity("北京", "LOC", 4, 6),
                Entity("协和医院", "ORG", 6, 10),
            ]
        ),
    ]
    return corpus


def train_ner_from_file(
    recognizer: NERRecognizer,
    filepath: str,
    encoding: str = 'utf-8'
) -> None:
    corpus = []
    
    with open(filepath, 'r', encoding=encoding) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('|||')
            if len(parts) >= 2:
                text = parts[0]
                entities = []
                
                entity_strs = parts[1].split()
                for entity_str in entity_strs:
                    entity_parts = entity_str.split('/')
                    if len(entity_parts) == 3:
                        entity_type = entity_parts[0]
                        start = int(entity_parts[1])
                        end = int(entity_parts[2])
                        entity_text = text[start:end]
                        entities.append(Entity(entity_text, entity_type, start, end))
                
                corpus.append((text, entities))
    
    if corpus:
        recognizer.train(corpus)


__all__ = [
    'NER_ENTITY_TYPES',
    'NER_TAGS',
    'DEFAULT_NER_TAGS',
    'Entity',
    'NERFeatureTemplate',
    'CRFNERModel',
    'NERRecognizer',
    'create_sample_ner_corpus',
    'train_ner_from_file',
]
