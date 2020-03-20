import os
from pprint import pprint
class Data():
    def __init__(self, save_dir, save_name, read_dir):
        self.save_dir = save_dir
        self.save_name = save_name
        self.read_dir = read_dir

    def convert_data_to_predict_format(self, lines):
        with open(os.path.join(self.save_dir, self.save_name+'.name'), 'a') as writer:
            for line in lines:
                for ch in line:
                    writer.write('\t'.join([ch,'O'])+'\n')
                writer.write('\n')
    def get_predictions(self, expr, ckpt, read_name):
        with open(os.path.join(read_dir, str(expr), 'checkpoint-'+str(ckpt), read_name+'.name'),'r') as reader:
                lines = reader.readlines()
        for line in lines:
            print(line.strip())


if __name__ == '__main__':
    save_dir = '/nfs/users/zhanghaipeng/general_ner/data/chinese'
    save_name = 'hand_input'
    read_dir = '/nfs/users/zhanghaipeng/general_ner/data/models'
    expr = 1
    ckpt = 1300
    read_name = 'hand_prediction'
    
    data = Data(save_dir, save_name, read_dir)
    line0 = "直通达沃斯|WTO总干事：WTO不可或缺，如果没有了就需要再“发明”一个出来　　当地时间1月22日下午，世贸组织总干事阿泽维多（RobertoAzevedo）在世界经济论坛年会现场召开小型记者会。"
    line = line0
    
    #转化为CoNLL格式，模型读取格式
    #data.convert_data_to_predict_format([line])
    
    #注释掉上一行代码，执行下一行代码(注意清Cache)
    #模型预测: sh test.sh $ckpt;

    
    #预测结果显示
    data.get_predictions(expr, ckpt, read_name)
