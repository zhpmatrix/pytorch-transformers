import re
import os
from simplex_sdk import SimplexClient

class Utils(object):
    
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.model = self.load_online_model()
    
    def load_online_model(self):
        return SimplexClient('bert-ner-api-v1',namespace='production')

    def get_ner_filenames(self, root_dir)->list:
        filenames = []
        for root, dirs, files in os.walk(root_dir):
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

    def load_format_data_conll(self, filename, read_dir):
        data_path = os.path.join(read_dir, filename+'.name')
        lines = []
        with open(data_path, 'r') as reader:
            for line in reader:
                if line != '\n':
                    lines.append(line.strip().split())
                else:
                    lines.append(['\n','O'])
        return lines
    
    def load_format_data_online(self, input_data):
        results = self.model.predict(input_data)
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
        return lines

    def get_brat_data(self, raw_data, save_dir, save_name):
        ann_path = os.path.join(save_dir, save_name+'.ann')
        txt_path = os.path.join(save_dir, save_name+'.txt')
        ann_writer = open(ann_path, 'a')
        txt_writer = open(txt_path, 'a')
        ann_count = 0
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
    

if __name__ == '__main__':
    root_dir = '/data/share/ontonotes-release-5.0/data/files/data/chinese/annotations/'
    test_root_dir = '/data/jh/notebooks/fanxiaokun/code/general_ner/data/ontonotes_data/ontonotes_raw_chinese/'
    utils = Utils(root_dir)
    filenames = utils.get_ner_filenames(root_dir)
    test_filenames = utils.get_ner_filenames(test_root_dir)
    import pdb;pdb.set_trace()

    read_dir = '/data/jh/notebooks/fanxiaokun/code/general_ner/data/ontonotes_data/ontonotes_raw_chinese/'
    file_name = 'ctv_0078'
    save_dir = '/nfs/users/zhanghaipeng/general_ner/brat-master/data/ontonotes/V1'
    raw_data = utils.load_format_data_conll(file_name, read_dir)
    #utils.get_brat_data(raw_data, save_dir, file_name)

    input_data = ["毛泽东  是国家主席,他生于湖南长沙，没去过美国。","新华社北京6月14日电6月14日，“2019·中国西藏发展论坛”在西藏拉萨举行。国家主席习近平发来贺信，向论坛开幕表示祝贺。","曹斌是机器学习和自然语言处理专家，香港科技大学博士。曾任职于微软研究院、Bing 搜索，担任 Cortana 首席算法科学家"]
    raw_data = utils.load_format_data_online(input_data)
    file_name = 'online'
    utils.get_brat_data(raw_data, save_dir, file_name)
