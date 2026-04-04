from .tokenizer import forward_max_match, backward_max_match, bidirectional_max_match
from .dictionary import Dictionary

class Segmentor:
    def __init__(self, dictionary=None):
        self.dictionary = dictionary or Dictionary()
        self.mode = 'bidirectional'

    def segment(self, text, mode=None):
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

    def set_mode(self, mode):
        if mode not in ('forward', 'backward', 'bidirectional'):
            raise ValueError(f"Unknown mode: {mode}")
        self.mode = mode

    def load_dictionary(self, path):
        self.dictionary.load_dictionary(path)