import struct
import os
from typing import Dict, List, Tuple, Optional, BinaryIO
from dataclasses import dataclass, field


@dataclass
class ScelWord:
    word: str
    pinyin: str
    frequency: int = 0
    pos_tag: Optional[str] = None


@dataclass
class ScelMetadata:
    name: str = ""
    category: str = ""
    description: str = ""
    example: str = ""
    word_count: int = 0


class ScelParser:
    MAGIC_NUMBER = b'\x40\x15\x00\x00\x44\x43\x53\x01'
    HEADER_SIZE = 0x1540
    PINYIN_TABLE_OFFSET = 0x1540
    CHINESE_TABLE_OFFSET = 0x2628

    def __init__(self):
        self._pinyin_table: Dict[int, str] = {}
        self._words: List[ScelWord] = []
        self._metadata: ScelMetadata = ScelMetadata()

    @property
    def words(self) -> List[ScelWord]:
        return self._words

    @property
    def metadata(self) -> ScelMetadata:
        return self._metadata

    def parse(self, file_path: str) -> List[ScelWord]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"SCEL文件不存在: {file_path}")

        with open(file_path, 'rb') as f:
            data = f.read()

        self._validate_file(data)
        self._parse_metadata(data)
        self._parse_pinyin_table(data)
        self._parse_words(data)

        return self._words

    def parse_from_bytes(self, data: bytes) -> List[ScelWord]:
        self._validate_file(data)
        self._parse_metadata(data)
        self._parse_pinyin_table(data)
        self._parse_words(data)
        return self._words

    def _validate_file(self, data: bytes) -> None:
        if len(data) < 8:
            raise ValueError("文件太小，不是有效的SCEL文件")

        if data[:8] != self.MAGIC_NUMBER:
            alt_magic = b'\x40\x15\x00\x00\x44\x43\x53\x01'
            if data[:8] != alt_magic:
                raise ValueError("无效的SCEL文件格式：魔数不匹配")

    def _parse_metadata(self, data: bytes) -> None:
        try:
            self._metadata.name = self._bytes_to_str(data[0x130:0x338])
            self._metadata.category = self._bytes_to_str(data[0x338:0x540])
            self._metadata.description = self._bytes_to_str(data[0x540:0xD40])
            self._metadata.example = self._bytes_to_str(data[0xD40:self.PINYIN_TABLE_OFFSET])
        except Exception:
            pass

    def _bytes_to_str(self, data: bytes) -> str:
        result = []
        pos = 0
        while pos < len(data) - 1:
            char_code = struct.unpack('<H', data[pos:pos + 2])[0]
            if char_code == 0:
                break
            result.append(chr(char_code))
            pos += 2
        return ''.join(result)

    def _parse_pinyin_table(self, data: bytes) -> None:
        self._pinyin_table = {}

        pinyin_start = self.PINYIN_TABLE_OFFSET
        if pinyin_start >= len(data):
            return

        pinyin_data = data[pinyin_start:]
        pos = 4

        while pos < len(pinyin_data) - 4:
            try:
                index = struct.unpack('<H', pinyin_data[pos:pos + 2])[0]
                pos += 2
                length = struct.unpack('<H', pinyin_data[pos:pos + 2])[0]
                pos += 2

                if length == 0 or pos + length > len(pinyin_data):
                    break

                pinyin = self._bytes_to_str(pinyin_data[pos:pos + length])
                self._pinyin_table[index] = pinyin
                pos += length

                if len(self._pinyin_table) > 500:
                    break
            except struct.error:
                break

    def _parse_words(self, data: bytes) -> None:
        self._words = []

        chinese_start = self.CHINESE_TABLE_OFFSET
        if chinese_start >= len(data):
            return

        chinese_data = data[chinese_start:]
        pos = 0

        while pos < len(chinese_data) - 6:
            try:
                same_count = struct.unpack('<H', chinese_data[pos:pos + 2])[0]
                if same_count == 0 or same_count > 1000:
                    break
                pos += 2

                pinyin_length = struct.unpack('<H', chinese_data[pos:pos + 2])[0]
                pos += 2

                if pinyin_length == 0 or pos + pinyin_length > len(chinese_data):
                    break

                pinyin = self._get_word_pinyin(chinese_data[pos:pos + pinyin_length])
                pos += pinyin_length

                for _ in range(same_count):
                    if pos + 2 > len(chinese_data):
                        break

                    word_length = struct.unpack('<H', chinese_data[pos:pos + 2])[0]
                    pos += 2

                    if word_length == 0 or pos + word_length > len(chinese_data):
                        break

                    word = self._bytes_to_str(chinese_data[pos:pos + word_length])
                    pos += word_length

                    if pos + 2 > len(chinese_data):
                        break

                    ext_length = struct.unpack('<H', chinese_data[pos:pos + 2])[0]
                    pos += 2

                    if pos + 2 > len(chinese_data):
                        break

                    frequency = struct.unpack('<H', chinese_data[pos:pos + 2])[0]
                    pos += ext_length

                    if word:
                        self._words.append(ScelWord(
                            word=word,
                            pinyin=pinyin,
                            frequency=frequency
                        ))

                if len(self._words) > 1000000:
                    break

            except struct.error:
                break

        self._metadata.word_count = len(self._words)

    def _get_word_pinyin(self, data: bytes) -> str:
        pinyins = []
        pos = 0
        while pos < len(data) - 1:
            index = struct.unpack('<H', data[pos:pos + 2])[0]
            if index in self._pinyin_table:
                pinyins.append(self._pinyin_table[index])
            pos += 2
        return "'".join(pinyins)

    def get_words_as_dict(self) -> List[Dict]:
        return [
            {
                'word': w.word,
                'pinyin': w.pinyin,
                'frequency': w.frequency
            }
            for w in self._words
        ]


class ScelBatchParser:
    def __init__(self):
        self._parsers: List[ScelParser] = []
        self._all_words: List[ScelWord] = []

    def parse_directory(self, directory: str, recursive: bool = False) -> List[ScelWord]:
        if not os.path.isdir(directory):
            raise NotADirectoryError(f"目录不存在: {directory}")

        scel_files = self._find_scel_files(directory, recursive)

        for file_path in scel_files:
            try:
                parser = ScelParser()
                parser.parse(file_path)
                self._parsers.append(parser)
                self._all_words.extend(parser.words)
            except Exception as e:
                print(f"解析文件 {file_path} 失败: {e}")

        return self._all_words

    def _find_scel_files(self, directory: str, recursive: bool) -> List[str]:
        scel_files = []

        if recursive:
            for root, _, files in os.walk(directory):
                for file in files:
                    if file.lower().endswith('.scel'):
                        scel_files.append(os.path.join(root, file))
        else:
            for file in os.listdir(directory):
                if file.lower().endswith('.scel'):
                    scel_files.append(os.path.join(directory, file))

        return scel_files

    @property
    def total_words(self) -> int:
        return len(self._all_words)

    @property
    def parsed_files(self) -> int:
        return len(self._parsers)

    def get_all_words(self) -> List[ScelWord]:
        return self._all_words

    def get_unique_words(self) -> List[ScelWord]:
        seen = set()
        unique_words = []
        for w in self._all_words:
            if w.word not in seen:
                seen.add(w.word)
                unique_words.append(w)
        return unique_words
