import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lp_segment.tokenizer import (
    forward_max_match,
    backward_max_match,
    bidirectional_max_match,
    choose_best_result
)