from typing import List, Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .dictionary import Dictionary


def forward_max_match(text: str, dictionary: 'Dictionary', max_len: int = 15) -> List[str]:
    result: List[str] = []
    i = 0
    text_len = len(text)

    while i < text_len:
        match_len = dictionary.get_max_match_length(text, i, max_len)

        if match_len > 0:
            result.append(text[i:i + match_len])
            i += match_len
        else:
            result.append(text[i])
            i += 1

    return result


def backward_max_match(text: str, dictionary: 'Dictionary', max_len: int = 15) -> List[str]:
    result: List[str] = []
    i = len(text)

    while i > 0:
        matched = False
        for j in range(min(max_len, i), 0, -1):
            word = text[i - j:i]
            if dictionary.search_in_dict(word):
                result.append(word)
                i -= j
                matched = True
                break
        if not matched:
            result.append(text[i - 1])
            i -= 1

    result.reverse()
    return result


def choose_best_result(forward_result: List[str], backward_result: List[str]) -> List[str]:
    forward_single = sum(1 for word in forward_result if len(word) == 1)
    backward_single = sum(1 for word in backward_result if len(word) == 1)

    if forward_single < backward_single:
        return forward_result
    elif backward_single < forward_single:
        return backward_result
    else:
        if len(forward_result) <= len(backward_result):
            return forward_result
        else:
            return backward_result


def bidirectional_max_match(text: str, dictionary: 'Dictionary', max_len: int = 15) -> List[str]:
    forward_result = forward_max_match(text, dictionary, max_len)
    backward_result = backward_max_match(text, dictionary, max_len)

    if forward_result == backward_result:
        return forward_result

    return choose_best_result(forward_result, backward_result)


def forward_max_match_with_pos(text: str, dictionary: 'Dictionary', max_len: int = 15) -> List[Tuple[str, str]]:
    result: List[Tuple[str, str]] = []
    i = 0
    text_len = len(text)

    while i < text_len:
        match_len, pos_tag = dictionary.get_max_match_with_pos(text, i, max_len)

        if match_len > 0:
            word = text[i:i + match_len]
            result.append((word, pos_tag or 'x'))
            i += match_len
        else:
            result.append((text[i], 'x'))
            i += 1

    return result


def backward_max_match_with_pos(text: str, dictionary: 'Dictionary', max_len: int = 15) -> List[Tuple[str, str]]:
    result: List[Tuple[str, str]] = []
    i = len(text)

    while i > 0:
        matched = False
        for j in range(min(max_len, i), 0, -1):
            word = text[i - j:i]
            found, pos_tag = dictionary.search_with_pos(word)
            if found:
                result.append((word, pos_tag or 'x'))
                i -= j
                matched = True
                break
        if not matched:
            result.append((text[i - 1], 'x'))
            i -= 1

    result.reverse()
    return result


def choose_best_result_with_pos(
    forward_result: List[Tuple[str, str]],
    backward_result: List[Tuple[str, str]]
) -> List[Tuple[str, str]]:
    forward_single = sum(1 for word, _ in forward_result if len(word) == 1)
    backward_single = sum(1 for word, _ in backward_result if len(word) == 1)

    if forward_single < backward_single:
        return forward_result
    elif backward_single < forward_single:
        return backward_result
    else:
        if len(forward_result) <= len(backward_result):
            return forward_result
        else:
            return backward_result


def bidirectional_max_match_with_pos(text: str, dictionary: 'Dictionary', max_len: int = 15) -> List[Tuple[str, str]]:
    forward_result = forward_max_match_with_pos(text, dictionary, max_len)
    backward_result = backward_max_match_with_pos(text, dictionary, max_len)

    if forward_result == backward_result:
        return forward_result

    return choose_best_result_with_pos(forward_result, backward_result)


__all__ = [
    'forward_max_match',
    'backward_max_match',
    'bidirectional_max_match',
    'choose_best_result',
    'forward_max_match_with_pos',
    'backward_max_match_with_pos',
    'bidirectional_max_match_with_pos',
    'choose_best_result_with_pos'
]
