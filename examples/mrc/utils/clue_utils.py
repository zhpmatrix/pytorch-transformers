import os
import json

class CLUEUtils():
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.locs = ['scene', 'address']
        self.orgs = ['government','organization','company']
        self.names = ['name']
        self.works = ['movie']

    def load_data(self, filename, savename):

        lines = []
        writer = open(savepath, 'a')
        with open(os.path.join(self.root_dir, filename),'r') as reader:
            for line in reader:
                jsonline = json.loads(line.strip())
                text = jsonline['text']
                label_dict = jsonline['label']
                labels = ['O'] * len(text)
                for key in label_dict:
                    if key in self.locs:
                        label = 'LOC'
                    elif key in self.orgs:
                        label = 'ORG'
                    elif key in self.names:
                        label = 'PER'
                    elif key in self.works:
                        label = 'WORK'
                    else:
                        continue
                    entities = label_dict[key]
                    for chars, locations in entities.items():
                        for loc in locations:
                            [start, end] = loc
                            labels[start] = 'B-'+label
                            for i in range(start + 1, end+1):
                                labels[i] = 'I-'+label
                for ch, label in zip(list(text), labels):
                    writer.write('\t'.join([ch, label])+'\n')
                writer.write('\n')
        writer.close()

if __name__ == '__main__':
    root_dir = '/nfs/users/zhanghaipeng/general_ner/data/data_backup/cluener_public'
    clue = CLUEUtils(root_dir)
    savepath = '/nfs/users/zhanghaipeng/general_ner/data/chinese/cluener/train.json'
    clue.load_data('train.json',savepath)
