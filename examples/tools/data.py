import pandas as pd
def data_split(data_dir='/data/share/zhanghaipeng/data/lanben_headline/', data_name='data.txt'):
    train_path = data_dir + 'train.csv'
    eval_path = data_dir  + 'eval.csv'
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
    train.to_csv(train_path, index=False)
    eval.to_csv(eval_path, index=False)

if __name__ == '__main__':
    data_split()
