import jieba
import collections
import pandas as pd

import random
random.seed(666)

class EDA():
    
    def __init__(self):
        self.syno_list = self.load_syno_lib()
        self.common_words = self.load_common_words()

    def load_syno_lib(self, file_path='/nfs/users/zhanghaipeng/transformers/src/transformers/data/processors/eigen_eda/syno_from_baidu_hanyu.txt'):
        syno_list = []
        with open(file_path) as reader:
            for line in reader:
                syno_list.append(line.strip().split())
        return syno_list

    def load_common_words(self, file_path='/nfs/users/zhanghaipeng/data/kuaishou/bucket/train.csv'):
        data = pd.read_csv(file_path, header=None)
        words_list = []
        for raw_str in data[1].tolist():
            raw_words = list(jieba.cut(raw_str))
            words_list.extend(raw_words)
        word_counter = collections.Counter(words_list)
        common_num = 100
        common_words = []
        for item in word_counter.most_common(common_num):
            common_words.append(item[0])
        return common_words
    
    def origin(self, raw_words):
        return raw_words
    
    def synonym_replacement(self, raw_words):
        sr_words = list(raw_words)
        sample_num = 1
        if len(sr_words) == 0:
            return sr_words
        random_idx = random.sample(range(len(sr_words)), sample_num)[0]
        random_word = sr_words[random_idx]
        for syno in self.syno_list:
            if random_word in syno:
                [first_word, second_word] = syno
                if random_word == first_word:
                    sr_words[random_idx] = second_word
                else:
                    sr_words[random_idx] = first_word
                return sr_words
        return sr_words

    def random_insertion(self, raw_words):
        ri_words = list(raw_words)
        sample_num = 1
        if len(ri_words) == 0:
            return ri_words
        random_idx = random.sample(range(len(ri_words)), sample_num)[0]
        common_word_idx = random.sample(range(len(self.common_words)), sample_num)[0]
        ri_words.insert(random_idx, self.common_words[common_word_idx])
        return ri_words
    
    def random_deletion(self, raw_words):
        rd_words = list(raw_words)
        sample_num = 1
        if len(rd_words) == 0:
            return rd_words
        random_idx = random.sample(range(len(rd_words)), sample_num)[0]
        del rd_words[random_idx]
        return rd_words
    
    def random_swap(self, raw_words):
        rs_words = list(raw_words)
        sample_num = 2
        if len(rs_words) < 2:
            return rs_words
        [start, end] = random.sample(range(len(rs_words)), sample_num)
        rs_words[end], rs_words[start] = rs_words[start], rs_words[end]
        return rs_words

if __name__ == '__main__':
    raw_str = '不要巴结坏人。'
    raw_words = list(jieba.cut(raw_str))
    eda = EDA()
    og_words = eda.origin(raw_words)
    sr_words = eda.synonym_replacement(raw_words)
    ri_words = eda.random_insertion(raw_words)
    rd_words = eda.random_deletion(raw_words)
    rs_words = eda.random_swap(raw_words)
