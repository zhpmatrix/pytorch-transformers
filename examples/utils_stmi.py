import torch
from transformers import *
from collections import Counter
from tqdm import tqdm

def construct_marker_vocab(data_dir='/data/share/zhanghaipeng/data/yelp/style_transfer/',data_path='data.tsv', pos_save_path='pos_vocab.txt', neg_save_path='neg_vocab.txt'):
    model = 'roberta-base'
    tokenizer = RobertaTokenizer.from_pretrained(model)
    with open(data_dir+data_path, 'r') as reader:
        textlines = reader.readlines()
    tokens_pos = []
    tokens_neg = []
    for line in tqdm(textlines):
        line_ = line.lower().strip().split('\t')[0]
        label_ = line.lower().strip().split('\t')[1]
        tokenized_line = tokenizer.tokenize(line_)
        if label_ == '0':
            tokens_neg.extend(tokenized_line)
        else:
            tokens_pos.extend(tokenized_line)
    vocab_pos = Counter(tokens_pos)
    vocab_neg = Counter(tokens_neg)

    pos_list = []
    neg_list = []
    smooth_ratio = 0.1
    pos_thresh = 60
    neg_thresh = 60

    pos_salience = {}
    neg_salience = {}
    for key, value in vocab_pos.items():
       salience = ( value + smooth_ratio)/(vocab_neg.get(key,0)+smooth_ratio)
       pos_salience[key] = salience
       if salience > pos_thresh:
           pos_list.append(key)
    for key, value in vocab_neg.items():
       salience = ( value + smooth_ratio)/(vocab_pos.get(key,0)+smooth_ratio)
       neg_salience[key] = salience
       if salience > neg_thresh:
           neg_list.append(key)
    with open(data_dir+pos_save_path, 'w') as pos_writer:
        for word in pos_list:
            pos_writer.write(word+'\n') 
    with open(data_dir+neg_save_path, 'w') as neg_writer:
        for word in neg_list:
            neg_writer.write(word+'\n')
if __name__ == '__main__':
    construct_marker_vocab()
