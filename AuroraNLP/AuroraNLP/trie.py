from typing import Dict, Optional, List, Tuple


class TrieNode:
    __slots__ = ['children', 'is_word', 'pos_tag']

    def __init__(self):
        self.children: Dict[str, 'TrieNode'] = {}
        self.is_word: bool = False
        self.pos_tag: Optional[str] = None


class Trie:
    def __init__(self):
        self.root = TrieNode()
        self._word_count: int = 0

    def insert(self, word: str, pos_tag: Optional[str] = None) -> None:
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        if not node.is_word:
            node.is_word = True
            self._word_count += 1
        node.pos_tag = pos_tag

    def search(self, word: str) -> bool:
        node = self._find_node(word)
        return node is not None and node.is_word

    def search_with_pos(self, word: str) -> Tuple[bool, Optional[str]]:
        node = self._find_node(word)
        if node is not None and node.is_word:
            return True, node.pos_tag
        return False, None

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
