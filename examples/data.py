import os
import pandas as pd

class DataSplitter():
    
    def __init__(self, root_dir, save_path: list):
        self.root_dir = root_dir
        self.save_path = save_path
        self.data = self.get_raw_data()

    def get_raw_data(self):
        raw_data = pd.read_csv(os.path.join(self.root_dir,save_path[0]))
        start = 0
        end = 3000
        return raw_data[start: end].sample(frac=1).reset_index(drop=True)

    def get_train_dev_test(self):
        return None

class RandomSplitter(DataSplitter):
    
    def __init__(self, root_dir, save_path):
        super().__init__(root_dir, save_path)
        self.type_dir = 'random'

    def get_train_dev_test(self, ratio: list):
        row_num = self.data.shape[0]
        train_end = int(row_num * ratio[0])
        dev_end = int(row_num * sum(ratio[:2]))
        assert(dev_end > train_end)

        train   = self.data[                :   train_end]
        dev     = self.data[    train_end   :   dev_end]
        test    = self.data[    dev_end     :   ]
        for name, each_data in zip(self.save_path[1:], [train, dev, test]):
            each_data.to_csv(os.path.join(os.path.join(self.root_dir,self.type_dir), name), index=False, header=False)

    def get_label(self, raw_label_path, save_path):
         labels = pd.read_csv(os.path.join(self.root_dir, raw_label_path))
         labels.iloc[:,2].to_csv(os.path.join(os.path.join(self.root_dir, self.type_dir), save_path),index=False,header=None)

class BucketSplitter(DataSplitter):
    
    def __init__(self, root_dir, save_path):
        super().__init__(root_dir, save_path)
        self.type_dir = 'bucket'

    def get_train_dev_test(self, ratio: list):
        train_list = []
        dev_list = []
        test_list = []
        counter = 0
        for name, group in self.data.groupby(self.data['mark_result']):
            counter += 1
            print(counter)

            row_num = group.shape[0]
            train_end = int(row_num * ratio[0])
            dev_end = int(row_num * sum(ratio[:2]))
            test_end = int(row_num * sum(ratio))
            assert(dev_end >= train_end)
            train_list.extend(group[: train_end].values.tolist())
            dev_list.extend(group[train_end : dev_end].values.tolist())
            test_list.extend(group[dev_end :].values.tolist())

        for name, each_data in zip(self.save_path[1:], [train_list, dev_list, test_list]):
            df_each_data = pd.DataFrame(each_data)
            df_each_data.to_csv(os.path.join(self.root_dir, self.type_dir, name), index=False, header=False)


    def get_label(self, raw_label_path, save_path):
         labels = pd.read_csv(os.path.join(self.root_dir, raw_label_path))
         labels['question_and_id'].to_csv(os.path.join(self.root_dir, self.type_dir, save_path),index=False,header=None)
         self.labels = labels['question_and_id'].values.tolist()

    def get_real_label(self, save_path):
        real_labels = self.data['mark_result'].unique().tolist()
        with open(os.path.join(self.root_dir, self.type_dir, save_path),'w') as writer:
            for label in real_labels:
                writer.write(label+'\n')

if __name__ == '__main__':
    
    root_dir = '/nfs/users/zhanghaipeng/data/kuaishou/'
    save_path = ['data.csv', 'train.csv', 'dev.csv', 'test.csv']
    
    raw_label_path = 'raw_label.csv'
    save_label_path = 'label.csv'
    ratio = [0.8, 0.1, 0.1]
    
    random_splitter = RandomSplitter(root_dir, save_path)
    bucket_splitter = BucketSplitter(root_dir, save_path)
    
    splitter = bucket_splitter
    #splitter.get_label(raw_label_path, save_label_path)
    splitter.get_real_label(save_label_path)
    splitter.get_train_dev_test(ratio)
