from typing import Dict, Optional, List, Tuple


class TrieNode:
    __slots__ = ['children', 'is_word', 'pos_tag', 'weight', 'priority']

    def __init__(self):
        self.children: Dict[str, 'TrieNode'] = {}
        self.is_word: bool = False
        self.pos_tag: Optional[str] = None
        self.weight: float = 1.0
        self.priority: int = 0


class Trie:
    def __init__(self):
        self.root = TrieNode()
        self._word_count: int = 0

    def insert(
        self,
        word: str,
        pos_tag: Optional[str] = None,
        weight: float = 1.0,
        priority: int = 0
    ) -> None:
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        if not node.is_word:
            node.is_word = True
            self._word_count += 1
        node.pos_tag = pos_tag
        node.weight = weight
        node.priority = priority

    def search(self, word: str) -> bool:
        node = self._find_node(word)
        return node is not None and node.is_word

    def search_with_pos(self, word: str) -> Tuple[bool, Optional[str]]:
        node = self._find_node(word)
        if node is not None and node.is_word:
            return True, node.pos_tag
        return False, None

    def search_with_info(self, word: str) -> Tuple[bool, Optional[str], float, int]:
        node = self._find_node(word)
        if node is not None and node.is_word:
            return True, node.pos_tag, node.weight, node.priority
        return False, None, 1.0, 0

    def get_weight(self, word: str) -> float:
        node = self._find_node(word)
        if node is not None and node.is_word:
            return node.weight
        return 1.0

    def get_priority(self, word: str) -> int:
        node = self._find_node(word)
        if node is not None and node.is_word:
            return node.priority
        return 0

    def set_weight(self, word: str, weight: float) -> bool:
        node = self._find_node(word)
        if node is not None and node.is_word:
            node.weight = weight
            return True
        return False

    def set_priority(self, word: str, priority: int) -> bool:
        node = self._find_node(word)
        if node is not None and node.is_word:
            node.priority = priority
            return True
        return False

    def starts_with(self, prefix: str) -> bool:
        return self._find_node(prefix) is not None

    def _find_node(self, prefix: str) -> Optional[TrieNode]:
        node = self.root
        for char in prefix:
            if char not in node.children:
                return None
            node = node.children[char]
        return node

    def get_max_match_length(self, text: str, start: int = 0, max_len: int = 15) -> int:
        node = self.root
        last_match = 0
        end = min(start + max_len, len(text))

        for i in range(start, end):
            char = text[i]
            if char not in node.children:
                break
            node = node.children[char]
            if node.is_word:
                last_match = i - start + 1

        return last_match

    def get_max_match_with_pos(self, text: str, start: int = 0, max_len: int = 15) -> Tuple[int, Optional[str]]:
        node = self.root
        last_match = 0
        last_pos = None
        end = min(start + max_len, len(text))

        for i in range(start, end):
            char = text[i]
            if char not in node.children:
                break
            node = node.children[char]
            if node.is_word:
                last_match = i - start + 1
                last_pos = node.pos_tag

        return last_match, last_pos

    def get_max_match_with_info(
        self,
        text: str,
        start: int = 0,
        max_len: int = 15
    ) -> Tuple[int, Optional[str], float, int]:
        node = self.root
        last_match = 0
        last_pos = None
        last_weight = 1.0
        last_priority = 0
        end = min(start + max_len, len(text))

        for i in range(start, end):
            char = text[i]
            if char not in node.children:
                break
            node = node.children[char]
            if node.is_word:
                last_match = i - start + 1
                last_pos = node.pos_tag
                last_weight = node.weight
                last_priority = node.priority

        return last_match, last_pos, last_weight, last_priority

    def get_all_matches_with_info(
        self,
        text: str,
        start: int = 0,
        max_len: int = 15
    ) -> List[Tuple[int, str, Optional[str], float, int]]:
        node = self.root
        matches = []
        end = min(start + max_len, len(text))

        for i in range(start, end):
            char = text[i]
            if char not in node.children:
                break
            node = node.children[char]
            if node.is_word:
                word_len = i - start + 1
                word = text[start:start + word_len]
                matches.append((word_len, word, node.pos_tag, node.weight, node.priority))

        return matches

    def remove(self, word: str) -> bool:
        if not word:
            return False

        path: List[TrieNode] = [self.root]
        node = self.root

        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
            path.append(node)

        if not path[-1].is_word:
            return False

        path[-1].is_word = False
        path[-1].pos_tag = None
        path[-1].weight = 1.0
        path[-1].priority = 0
        self._word_count -= 1

        for i in range(len(path) - 1, 0, -1):
            node = path[i]
            parent = path[i - 1]
            char = word[i - 1]

            if node.children or node.is_word:
                break

            del parent.children[char]

        return True

    def __len__(self) -> int:
        return self._word_count

    def __contains__(self, word: str) -> bool:
        return self.search(word)
