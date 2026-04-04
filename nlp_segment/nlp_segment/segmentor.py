from typing import List, Tuple, Optional
from .tokenizer import (
    forward_max_match,
    backward_max_match,
    bidirectional_max_match,
    forward_max_match_with_pos,
    backward_max_match_with_pos,
    bidirectional_max_match_with_pos
)
from .dictionary import Dictionary


POS_TAG_NAMES = {
    'n': '名词',
    'v': '动词',
    'a': '形容词',
    'd': '副词',
    't': '时间词',
    'r': '代词',
    'p': '介词',
    'c': '连词',
    'm': '数词',
    'q': '量词',
    'u': '助词',
    'w': '标点',
    'nr': '人名',
    'ns': '地名',
    'nt': '机构名',
    'nz': '其他专名',
    'vn': '动名词',
    'x': '未知',
}

NER_TAG_MAP = {
    'nr': 'PERSON',
    'ns': 'LOCATION',
    'nt': 'ORGANIZATION',
}


class Segmentor:
    def __init__(self, dictionary: Optional[Dictionary] = None, load_default_dict: bool = True):
        if dictionary is not None:
            self.dictionary = dictionary
        else:
            self.dictionary = Dictionary(load_default=load_default_dict)
        self.mode: str = 'bidirectional'

    def segment(self, text: str, mode: Optional[str] = None) -> List[str]:
        if mode is None:
            mode = self.mode

        if mode == 'forward':
            return forward_max_match(text, self.dictionary)
        elif mode == 'backward':
            return backward_max_match(text, self.dictionary)
        elif mode == 'bidirectional':
            return bidirectional_max_match(text, self.dictionary)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def segment_with_pos(self, text: str, mode: Optional[str] = None) -> List[Tuple[str, str]]:
        if mode is None:
            mode = self.mode

        if mode == 'forward':
            return forward_max_match_with_pos(text, self.dictionary)
        elif mode == 'backward':
            return backward_max_match_with_pos(text, self.dictionary)
        elif mode == 'bidirectional':
            return bidirectional_max_match_with_pos(text, self.dictionary)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def recognize_entities(self, text: str, mode: Optional[str] = None) -> List[Tuple[str, str]]:
        pos_result = self.segment_with_pos(text, mode)
        entities = []

        for word, pos_tag in pos_result:
            if pos_tag in NER_TAG_MAP:
                entities.append((word, NER_TAG_MAP[pos_tag]))

        return entities

    def segment_with_entities(self, text: str, mode: Optional[str] = None) -> List[Tuple[str, str, str]]:
        pos_result = self.segment_with_pos(text, mode)
        result = []

        for word, pos_tag in pos_result:
            entity_type = NER_TAG_MAP.get(pos_tag, 'O')
            result.append((word, pos_tag, entity_type))

        return result

    def set_mode(self, mode: str) -> None:
        if mode not in ('forward', 'backward', 'bidirectional'):
            raise ValueError(f"Unknown mode: {mode}")
        self.mode = mode

    def load_dictionary(self, path: str) -> None:
        self.dictionary.load_dictionary(path)

    def save_dictionary(self, path: str) -> None:
        self.dictionary.save_dictionary(path)

    def add_word(self, word: str, pos_tag: Optional[str] = None) -> None:
        self.dictionary.add_word(word, pos_tag)

    def remove_word(self, word: str) -> bool:
        return self.dictionary.remove_word(word)

    def get_dictionary_size(self) -> int:
        return len(self.dictionary)

    def get_pos_tag_name(self, pos_tag: str) -> str:
        return POS_TAG_NAMES.get(pos_tag, '未知')

    def get_pos_tags(self) -> dict:
        return POS_TAG_NAMES.copy()
