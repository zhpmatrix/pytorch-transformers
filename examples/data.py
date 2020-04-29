import json
import os
import pandas as pd
from tqdm import tqdm

class DataSplitter():
    
    def __init__(self, root_dir, save_path: list):
        self.root_dir = root_dir
        self.save_path = save_path
        self.data = self.get_raw_data()

    def get_raw_data(self):
        raw_data = pd.read_csv(os.path.join(self.root_dir,save_path[0]))
        start = 0
        end = 3000
        return raw_data[start: end].sample(frac=1, random_state=666).reset_index(drop=True)

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
        real_labels = sorted(real_labels, key = lambda x: int(x.split('_')[0]))
        with open(os.path.join(self.root_dir, self.type_dir, save_path),'w') as writer:
            for label in real_labels:
                writer.write(label+'\n')

class PredictHelper(DataSplitter):
    def __init__(self, root_dir, save_path):
        super().__init__(root_dir, save_path)
        self.type_dir = 'bucket'
        self.data = self.get_raw_data()
    
    def get_raw_data(self):
        raw_data = pd.read_csv(os.path.join(self.root_dir,save_path[0]))
        start = 4000
        end = 8000
        return raw_data[start: end]
        #return raw_data[start: end].sample(frac=1, random_state=666).reset_index(drop=True)
    
    def get_predict_data(self, predict_save_path): 
        self.data['mark_result'] = '0_无匹配'
        self.data.to_csv(os.path.join(self.root_dir, self.type_dir, predict_save_path), index=False, header=False)

class SemiSupervisedHelper(DataSplitter):
    def __init__(self, root_dir, save_path):
        super().__init__(root_dir, save_path)
        self.type_dir = 'bucket'
        self.data = self.get_raw_data()
        self.model = self.load_model()
        self.labels = self.load_label()

    def load_label(self):
        label_path = '/nfs/users/zhanghaipeng/data/kuaishou/bucket/label.csv'
        labels = []
        with open(label_path, 'r') as reader:
            for line in reader:
                labels.append(line.strip())
        return labels

    def load_model(self):
        from simplex_sdk import SimplexClient
        return SimplexClient('ks-bot',namespace='dev')

    def get_raw_data(self):
        raw_data = pd.read_csv(os.path.join(self.root_dir,save_path[0]))
        start = 4000
        end = 8000
        return raw_data[start: end]
        #return raw_data[start: end].sample(frac=1, random_state=666).reset_index(drop=True)
    
    def get_predict_data(self, predict_save_path): 
        self.data['mark_result'] = '无标签'
        for i in tqdm(range(self.data.shape[0])):
            query = self.data.iloc[i]['query']
            pred_results = self.model.predict([query])
            pred_label = pred_results[0]['results_list'][0]['title']
            pred_prob = pred_results[0]['results_list'][0]['prob']
            thresh = 0.8
            if float(pred_prob) > thresh:
                for label in self.labels:
                    if pred_label == label.split('_')[1]:
                        self.data.iloc[i] = ['',query,'',label, '']
        self.data = self.data[self.data['mark_result'] != '无标签']
        self.data.to_csv(os.path.join(self.root_dir, self.type_dir, predict_save_path), index=False, header=False)

class SoftLabelHelper(DataSplitter):
    def __init__(self, root_dir, save_path):
        super().__init__(root_dir, save_path)
        self.type_dir = 'bucket'
        self.labels = self.load_label()
        self.data = self.get_raw_data()

    def load_label(self):
        label_path = '/nfs/users/zhanghaipeng/data/kuaishou/bucket/label.csv'
        labels = []
        with open(label_path, 'r') as reader:
            for line in reader:
                labels.append(line.strip())
        return labels

    def load_model(self):
        from simplex_sdk import SimplexClient
        return SimplexClient('ks-bot',namespace='dev')

    def get_raw_data(self):
        raw_data = pd.read_csv(os.path.join(self.root_dir,"bucket/dev.csv"), header=None)
        return raw_data

    def get_predict_data(self, predict_save_path): 
        writer = open(predict_save_path, 'w')
        class_num = 85
        for i in tqdm(range(self.data.shape[0])):
            query = self.data.iloc[i][1]
            label = self.data.iloc[i][3]
            hard_label = self.labels.index(label)
            writer.write(json.dumps({'input':query, 'soft_label':[0.0]*class_num, 'hard_label':[hard_label]}, ensure_ascii=False)+'\n')
        writer.close()

if __name__ == '__main__':
    
    root_dir = '/nfs/users/zhanghaipeng/data/kuaishou/'
    save_path = ['data.csv', 'train.csv', 'dev.csv', 'test.csv']
    
    raw_label_path = 'raw_label.csv'
    save_label_path = 'label.csv'
    ratio = [0.8, 0.1, 0.1]
    
    predict_save_path = 'predict.csv'
    pred = PredictHelper(root_dir, [save_path[0]])
    pred.get_predict_data(predict_save_path)
    exit()

    soft = SoftLabelHelper(root_dir, ['bucket/dev.csv'])
    soft.get_predict_data('dev_with_soft_label.csv')
    exit()

    semi_supervised_save_path = 'semi_supervised_0.8.csv'
    semi = SemiSupervisedHelper(root_dir, [save_path[0]])
    semi.get_predict_data(semi_supervised_save_path)
    exit()
    

    random_splitter = RandomSplitter(root_dir, save_path)
    bucket_splitter = BucketSplitter(root_dir, save_path)
    
    splitter = bucket_splitter
    splitter.get_predict_data(predict_save_path)
    #splitter.get_label(raw_label_path, save_label_path)
    #splitter.get_real_label(save_label_path)
    #splitter.get_train_dev_test(ratio)
