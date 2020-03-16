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

    def get_ner_data(self, filenames):

if __name__ == '__main__':
    root_dir = '/data/share/ontonotes-release-5.0/data/files/data/chinese/annotations/'
    utils = Utils(root_dir)
    filenames = utils.get_ner_filenames()
    #utils.get_split_dot(filenames)
