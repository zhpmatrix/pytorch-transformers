import re
import os
import pandas as pd
from pprint import pprint
from simplex_sdk import SimplexClient

class Utils(object):
    
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.model = self.load_online_model()
    
    def load_online_model(self):
        return SimplexClient('bert-ner-api-v1',namespace='production')

    def get_ner_filenames(self)->list:
        filenames = []
        for root, dirs, files in os.walk(self.root_dir):
            for filename in files:
                if filename.endswith('.name'):
                    filenames.append(os.path.join(root,filename))
        return filenames
    
    def get_split_dot(self, filenames):
        def is_chinese(ch):
            if '\u4e00' <= ch <= '\u9fff':
                return True
            return False
        dot = {}
        #频次最高的结束符
        topk = 11
        for i, filename in enumerate(filenames):
            print(i)
            with open(filename, 'r') as reader:
                for line in reader:
                    ch = line.strip()[-1]
                    if not is_chinese(ch):
                        dot[ch] = dot.get(ch, 0) + 1
        dot_sorted = sorted(dot.items(), key=lambda x: x[1], reverse=True)
        return dot_sorted[:topk]
    
    def load_format_data_lines(self, read_dir, filename):
        data_path = os.path.join(read_dir, filename+'.name')
        lines = []
        with open(data_path, 'r') as reader:
            line_str = []
            for line in reader:
                if line != '\n':
                    [ch, label] = line.strip().split('\t')
                    assert(len(ch) == 1)
                    line_str.append(ch)
                else:
                    lines.append(''.join(line_str))
                    line_str = []
        return lines

    def load_format_data_conll(self, read_dir, filename):
        data_path = os.path.join(read_dir, filename+'.name')
        lines = []
        with open(data_path, 'r') as reader:
            for line in reader:
                if line != '\n':
                    lines.append(line.strip().split())
                else:
                    lines.append(['\n','O'])
        return lines
     
    def write_online_predictions_to_file(self, input_data, read_dir, save_name):
        results = []
        batch_size = 100
        epoch = len(input_data) // batch_size
        for i in range(epoch):
            tmp_input = input_data[batch_size * i : batch_size * (i+1)]
            if i == epoch - 1:
                tmp_input = input_data[batch_size * i:]
            tmp_results = self.model.predict(tmp_input)
            results.extend(tmp_results)
        lines = []
        for result, data in zip(results, input_data):
            tmp_lines = list(data)
            tmp_labels = ['O'] * len(tmp_lines)
            ne_list = result['ne']
            for ne in ne_list:
                offset = ne['offset']
                tag = ne['tag']
                length = ne['length']
                tmp_labels[offset] = 'B-'+tag
                tmp_labels[offset + 1: offset + length] = ['I-'+tag] * (length - 1)
            tmp_lines.append('\n')
            tmp_labels.append('O')
            lines.extend([[line, label] for line, label in zip(tmp_lines, tmp_labels)])
        save_path = os.path.join(read_dir, save_name+'.name')
        with open(save_path, 'a') as writer:
            for line in lines:
                [ch,_] = line
                if ch != '\n':
                    writer.write('\t'.join(line)+'\n')
                else:
                    writer.write(ch)
        return lines

    def get_brat_data(self, raw_data, start, end, save_dir, save_name):
        ann_path = os.path.join(save_dir, save_name+'.ann')
        txt_path = os.path.join(save_dir, save_name+'.txt')
        ann_writer = open(ann_path, 'a')
        txt_writer = open(txt_path, 'a')
        ann_count = 0
        raw_data = raw_data[start:end]
        loc_list = ''.join([label.split('-')[0] for [ch, label] in raw_data])
        entity_locs = [item for item in re.finditer('BI*', loc_list)]
        for item in entity_locs:
            (start, end) = item.span()
            ann_count += 1
            entity_label = raw_data[start][1].split('-')[-1]
            entity_str = ''.join([line[0] for line in raw_data[start:end]]) 
            ann_str = 'T'+str(ann_count)+'\t'+entity_label+' '+str(start)+' '+str(end)+'\t'+''.join(entity_str)+'\n'
            ann_writer.write(ann_str)
        txt_str = ''.join([item[0] for item in raw_data])
        txt_writer.write(txt_str)
        ann_writer.close()
        txt_writer.close()
    
    def get_data_descs(self, data_path):
        lens = []
        with open(data_path, 'r') as reader:
            line_str = []
            for line in reader:
                if line != '\n':
                    [ch, label] = line.strip().split('\t')
                    assert(len(ch) == 1)
                    line_str.append(ch)
                else:
                    lens.append(len(''.join(line_str)))
                    line_str = []
        report = pd.Series(lens).describe()
        print(report)
        return lens

if __name__ == '__main__':
    root_dir = '/data/share/ontonotes-release-5.0/data/files/data/chinese/annotations/'
    utils = Utils(root_dir)
    
    input_data = ['百团大战是八路军在抗战期间发动的规模最大的一次战役。',
                    '它由主碑，副碑，一座大型圆雕和烽火台，长城等组成。',
                    ]
    #model = utils.load_online_model()
    #results = model.predict(input_data)
    #pprint(results)
    #exit()

    online = False
    start, end = 0, 200
    save_dir = '/nfs/users/zhanghaipeng/general_ner/brat-master/data/ontonotes/V1'
    if not online:
        read_dir = '/nfs/users/zhanghaipeng/general_ner/data/models/1/checkpoint-200/'
        read_name = 'predictions'
        save_name = 'offline'
    else:
        read_dir = '/nfs/users/zhanghaipeng/general_ner/data/chinese/'
        test_name = 'input'
        read_name = 'prediction'
        save_name = 'online' 
        #input_data = utils.load_format_data_lines(read_dir, test_name)
        #utils.write_online_predictions_to_file(input_data, read_dir, read_name)
    raw_data = utils.load_format_data_conll(read_dir, read_name)
    utils.get_brat_data(raw_data, start, end, save_dir, save_name)
    
    data_path = '/nfs/users/zhanghaipeng/general_ner/data/chinese'
    data_name = 'train.name'
    #utils.get_data_descs(os.path.join(data_path, data_name))
