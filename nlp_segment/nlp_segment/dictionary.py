class Dictionary:
    def __init__(self):
        self.words = set()

    def load_dictionary(self, path):
        """从文件加载词典，每行一个词语"""
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip()
                if word:
                    self.words.add(word)

    def get_words(self):
        """返回词典词语集合"""
        return self.words

    def add_word(self, word):
        """添加新词到词典"""
        self.words.add(word)

    def search_in_dict(self, word):
        """查询词是否存在于词典中"""
        return word in self.words
