import random
import os
import pandas as pd
random.seed(666)

def get_data(data_dir, data_name, save_dir): 
    data_path = os.path.join(data_dir, data_name)
    data = pd.read_csv(data_path, header=None)
    samples = []
    class_dict = {}
    for name, group in data.groupby(data[3]):
        if name not in ['-1_不是问题','0_无匹配']:
            class_dict[name] = group[1].tolist()
         
    for q0_class, q0_values in class_dict.items():
        for q1_class, q1_values in class_dict.items():
            if q0_class != q1_class:
                for q0 in q0_values:
                        samples.append([q0, q1_class.split('_')[1], '0'])
            else:
                for q0 in q0_values:
                    samples.append([q0, q1_class.split('_')[1], '1'])
    
    simi_samples = [ sample for sample in samples if sample[2] == '1']
    unsimi_samples = [sample for sample in samples if sample[2] == '0']
    sampled_unsimi_samples = random.sample(unsimi_samples, len(simi_samples))
    save_samples = simi_samples + sampled_unsimi_samples
    save_samples = random.sample(save_samples, len(save_samples))
    with open(os.path.join(os.path.join(data_dir, save_dir), data_name), 'w') as writer:
        for sample in save_samples:
            writer.write('\t'.join(sample)+'\n')

if __name__ == '__main__':
    data_dir = '/nfs/users/zhanghaipeng/data/kuaishou/simi'
    data_name = 'train.csv'
    save_dir = 'data_with_label_balanced/'
    get_data(data_dir, data_name, save_dir)
