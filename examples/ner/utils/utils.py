import re
import os
import six
import pandas as pd
from pprint import pprint
from general_utils import GeneralUtils
from simplex_sdk import SimplexClient

class Utils(object):
    
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.general_utils = GeneralUtils()
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
        dot = {}
        #频次最高的结束符
        topk = 11
        for i, filename in enumerate(filenames):
            print(i)
            with open(filename, 'r') as reader:
                for line in reader:
                    ch = line.strip()[-1]
                    if not self.general_utils.is_chinese(ch):
                        dot[ch] = dot.get(ch, 0) + 1
        dot_sorted = sorted(dot.items(), key=lambda x: x[1], reverse=True)
        return dot_sorted[:topk]
    
    def filter_data(self, read_name, save_name):
        #特殊编码的字符统计
        ch_dict = dict()
        with open(os.path.join(self.root_dir, read_name)) as reader:
            for line in reader:
                if line != '\n':
                    [ch,label] = line.strip().split('\t')
                    if not self.general_utils.is_chinese(ch):
                        ch_dict[ch] = ch_dict.get(ch,0) + 1
        sorted_ch_dict = sorted(ch_dict.items(),key=lambda x:x[1], reverse=True)
        pprint(sorted_ch_dict)
        import pdb;pdb.set_trace()
        return ch_dict

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
    
    def add_boundary_tag_to_data(self, raw_data, save_dir, save_name):
        save_path = os.path.join(save_dir, save_name+'.name')
        loc_list = ''.join([label.split('-')[0] for [ch, label] in raw_data])
        entity_locs = [item for item in re.finditer('BI*', loc_list)]
        data_with_boundary = []
        for line in raw_data:
            line.append('O')
            data_with_boundary.append(line)
        for item in entity_locs:
            (start, end) = item.span()
            data_with_boundary[start][2] = 'B'
            data_with_boundary[end-1][2] = 'E'
        with open(save_path, 'a') as save_writer:
            for line in data_with_boundary:
                [ch, label0, label1] = line
                if ch != '\n':
                    save_writer.write('\t'.join(line)+'\n')
                else:
                    save_writer.write('\n')
    
    
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
    
    root_dir = '/nfs/users/zhanghaipeng/general_ner/data/chinese'
    utils = Utils(root_dir)
    #utils.filter_data('dev.name','filter_data/dev.name')
    #exit()

    root_dir = '/data/share/ontonotes-release-5.0/data/files/data/chinese/annotations/'
    utils = Utils(root_dir)
    
    line0 = '阿里巴巴的使命是“让天下没有难做的生意”'
    line1 = '特斯拉今年销售了50万辆车。'
    line2 = '直通达沃斯|WTO总干事：WTO不可或缺，如果没有了就需要再“发明”一个出来　　当地时间1月22日下午，世贸组织总干事阿泽维多（RobertoAzevedo）在世界经济论坛年会现场召开小型记者会。'
    line3 = '当地时间1月22日下午，世贸组织总干事阿泽维多（RobertoAzevedo）在世界经济论坛年会现场召开小型记者会。'
    line = line3
    #model = utils.load_online_model()
    #results = model.predict([line])
    #pprint(results)
    #import pdb;pdb.set_trace()

    online = False
    start, end = 0, 200
    save_dir = '/nfs/users/zhanghaipeng/general_ner/brat-master/data/ontonotes/V1'
    if not online:
        ckpt=1300
        read_dir = '/nfs/users/zhanghaipeng/general_ner/data/models/1/checkpoint-'+str(ckpt)
        read_name = 'hand_prediction'
        save_name = 'offline'
    else:
        read_dir = '/nfs/users/zhanghaipeng/general_ner/data/chinese/'
        test_name = 'input'
        read_name = 'prediction'
        save_name = 'online' 
        #input_data = utils.load_format_data_lines(read_dir, test_name)
        #utils.write_online_predictions_to_file(input_data, read_dir, read_name)
    #raw_data = utils.load_format_data_conll(read_dir, read_name)
    #utils.get_brat_data(raw_data, start, end, save_dir, save_name)
    
    data_path = '/nfs/users/zhanghaipeng/general_ner/data/chinese'
    data_name = 'train.name'
    #utils.get_data_descs(os.path.join(data_path, data_name))

    read_dir = '/nfs/users/zhanghaipeng/general_ner/data/chinese/ontonotes'
    read_name = 'dev'
    save_dir = '/nfs/users/zhanghaipeng/general_ner/data/chinese/boundary_ontonotes'
    raw_data = utils.load_format_data_conll(read_dir, read_name)
    utils.add_boundary_tag_to_data(raw_data, save_dir, read_name)
