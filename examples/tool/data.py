import os

def data_checker(data_dir='/data/share/zhanghaipeng/data/cars_article/tongyinzi_xingjinzi_merge/', filename='train.txt'):
    number = 0
    with open(data_dir+filename, 'r') as reader:
        for i, line in enumerate( reader ):
            fields = line.split('\t')
            text, label = fields[0], fields[1]
            if len(text) != len(label):
                number += 1
                print(text+'\n'+label+'\n')
                print(number, i)
def del_ckpt(dir_,ckpt_num):
    for dirname in os.listdir(dir_):
        if dirname.startswith('checkpoint'):
            cur_ckpt_idx = dirname.find('-')
            cur_ckpt_num = int(dirname[cur_ckpt_idx + 1:])
            if cur_ckpt_num < ckpt_num:
                os.system('rm -rf '+dir_+dirname)

if __name__ == '__main__':
    #data_checker()
    del_ckpt('../train/5/',659000)
