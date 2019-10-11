import pandas as pd
from pyltp import Segmentor

def get_model(model_path = '/data/share/zhanghaipeng/data/ltp_model/cws.model'):
    segmentor = Segmentor()
    segmentor.load(model_path)
    return segmentor

def data_split(data_dir='/data/share/zhanghaipeng/data/lanben_headline/', data_name='data.txt'):
    train_path = data_dir + 'train.csv'
    eval_path = data_dir  + 'dev.csv'
    data = pd.read_csv(data_dir + data_name, sep='\t')
    # 删除Unnamed列
    data = data.iloc[:,:2]
    # train/eval中负例的比例
    ratio = 0.1
    neg_data = data[data['是否情报'] == '否']
    neg_num = neg_data.shape[0]
    train_neg = neg_data[int(ratio * neg_num):]
    eval_neg = neg_data[:int(ratio * neg_num)] 
    
    pos_data = data[data['是否情报'] == '是']
    pos_num = pos_data.shape[0]
    train_pos = pos_data[int(ratio * pos_num):]
    eval_pos = pos_data[:int(ratio * pos_num)] 
    

    train = pd.concat([train_neg,train_pos]).sample(frac=1).reset_index(drop=True)
    eval = pd.concat([eval_neg,eval_pos]).sample(frac=1).reset_index(drop=True)
    train.to_csv(train_path, sep='\t', index=False)
    eval.to_csv(eval_path, sep='\t', index=False)

def data_split_fastext(data_dir='/data/share/zhanghaipeng/data/lanben_headline/raw_data/', data_name='第一批标题整理0930.xlsx'):
    save_dir = '/data/share/zhanghaipeng/lanben_cls/fastText/examples/main/data/data_v2/'
    train_path = save_dir + 'train.txt'
    dev_path = save_dir  + 'dev.txt'
    test_path = save_dir  + 'test.txt'
    train_writer = open(train_path, 'w')
    dev_writer = open(dev_path, 'w')
    test_writer = open(test_path, 'w')
    def relabel(label):
        if label.find('是') != -1:
            return '是'
        elif label.find('待定') == -1:
            return '否'
    data = pd.read_excel(data_dir+data_name)
    data = data.iloc[:,1:3]
    data.dropna(axis=0, how='any', inplace=True)
    data.columns = ['标题','是否情报']
    data['是否情报'] = data['是否情报'].apply(relabel)
    # train/eval中负例的比例
    ratio = 0.1
    neg_data = data[data['是否情报'] == '否']
    neg_num = neg_data.shape[0]
    train_neg = neg_data[int(ratio * neg_num):]
    dev_neg = neg_data[:int(ratio * neg_num)] 
    
    pos_data = data[data['是否情报'] == '是']
    pos_num = pos_data.shape[0]
    train_pos = pos_data[int(ratio * pos_num):]
    dev_pos = pos_data[:int(ratio * pos_num)] 
    

    train = pd.concat([train_neg,train_pos]).sample(frac=1).reset_index(drop=True)
    dev = pd.concat([dev_neg,dev_pos]).sample(frac=1).reset_index(drop=True)
    model =  get_model()
    train_num = train.shape[0]
    dev_num = dev.shape[0]
    for i in range(train_num):
        title = train.iloc[i]['标题']
        label = train.iloc[i]['是否情报']
        try:
            example = '__label__' + label + ' ' + ' '.join(model.segment(title))
        except:# title为空时，title=nan，过滤
            continue
        train_writer.write(example+'\n')
    
    for i in range(dev_num):
        title = dev.iloc[i]['标题']
        label = dev.iloc[i]['是否情报']
        example = '__label__' + label + ' ' + ' '.join(model.segment(title))
        dev_writer.write(example+'\n')

if __name__ == '__main__':
    #data_split_fastext()
    #data_split()
