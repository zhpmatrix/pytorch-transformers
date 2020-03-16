import os
class Utils(object):
    def __init__(self, root_dir):
        self.root_dir = root_dir
    
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

    def get_brat_data(self, filename, read_dir, save_dir):
        data_path = os.path.join(read_dir, filename+'.name')
        ann_path = os.path.join(save_dir, filename+'.ann')
        txt_path = os.path.join(save_dir, filename+'.txt')
        ann_writer = open(ann_path, 'a')
        txt_writer = open(txt_path, 'a')
        lines = []
        with open(data_path, 'r') as reader:
            for line in reader:
                if line != '\n':
                    lines.append(line.strip().split())
        ann_count = 0
        str_list = []
        start, end = 0, 0
        tag = None
        for i,item in enumerate(lines):
            ch, label = item[0], item[1]
            if label != 'O':
                [loc, tmp_tag] = label.split('-')
                if loc == 'B':
                    start = i
                    end = i + 1
                    tag = tmp_tag
                    ann_count += 1
                    str_list.append(ch)
                else:
                    end += 1
                    str_list.append(ch)
            else:
                if len(str_list) > 0:
                    ann_writer.write('T'+str(ann_count)+'\t'+tag+' '+str(start)+' '+str(end)+'\t'+''.join(str_list)+'\n')
                    str_list = []
                    start, end = 0, 0
                    tag = None
        output_str = ''.join([item[0] for item in lines])
        txt_writer.write(output_str)
        ann_writer.close()
        txt_writer.close()

if __name__ == '__main__':
    root_dir = '/data/share/ontonotes-release-5.0/data/files/data/chinese/annotations/'
    utils = Utils(root_dir)
    filenames = utils.get_ner_filenames()
    
    read_dir = '/data/jh/notebooks/fanxiaokun/code/general_ner/data/ontonotes_data/ontonotes_raw_chinese/'
    file_name = 'vom_0325'
    save_dir = '/nfs/users/zhanghaipeng/general_ner/brat-master/data/ontonotes/V1'
    utils.get_brat_data(file_name, read_dir, save_dir)
