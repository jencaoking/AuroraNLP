from typing import List, Optional
from .tokenizer import forward_max_match, backward_max_match, bidirectional_max_match
from .dictionary import Dictionary


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

    def set_mode(self, mode: str) -> None:
        if mode not in ('forward', 'backward', 'bidirectional'):
            raise ValueError(f"Unknown mode: {mode}")
        self.mode = mode

    def load_dictionary(self, path: str) -> None:
        self.dictionary.load_dictionary(path)

    def add_word(self, word: str) -> None:
        self.dictionary.add_word(word)

    def remove_word(self, word: str) -> bool:
        return self.dictionary.remove_word(word)

    def get_dictionary_size(self) -> int:
        return len(self.dictionary)
