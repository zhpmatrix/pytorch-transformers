import fasttext
from fasttext import train_supervised
import json
from pyltp import Segmentor

DATA_DIR = '/data/share/zhanghaipeng/data/chuangtouribao/event/fasttext/invest/'
CWS_MODEL_PATH = '/data/share/zhanghaipeng/data/ltp_model/cws.model'
MODEL_SAVE_PATH = '/data/share/zhanghaipeng/data/chuangtouribao/event/train/invest/'

def load_segmentor():
    segmentor = Segmentor()
    segmentor.load(CWS_MODEL_PATH)
    return segmentor

def get_data_fasttext(data_dir=DATA_DIR, save_dir= DATA_DIR+'data/', train_path='train.json',test_path='test.json'):
    train_list = []
    test_list = []
    segmentor = load_segmentor()
    with open(data_dir+train_path, 'r') as reader:
        for line in reader:
            example = json.loads(line)
            train_list.append([' '.join(segmentor.segment(example['title'])),'__label__'+str(example['is_vc_text'])])
    with open(data_dir + test_path, 'r') as reader:
        for line in reader:
            example = json.loads(line)
            test_list.append([' '.join(segmentor.segment(example['title'])),'__label__'+str(example['is_vc_text'])])
    with open(save_dir+test_path, 'a') as writer:
        for example in test_list:
            writer.write(' '.join(example)+'\n')
    with open(save_dir+train_path, 'a') as writer:
        for example in train_list:
            writer.write(' '.join(example)+'\n')

def print_results(N, p, r):
    print("N\t" + str(N))
    print("P@{}\t{:.3f}".format(1, p))
    print("R@{}\t{:.3f}".format(1, r))

def train(train_data, valid_data):
    model = train_supervised(input=train_data, epoch=100, lr=0.01, wordNgrams=2, verbose=2, minCount=2,label='__label__', loss='softmax', dim=100)
    print_results(*model.test(valid_data))
    model.save_model(MODEL_SAVE_PATH+"/model.bin")

def predict(test_data):
    seq_list = []
    test_num = 10
    label = '__label__'
    with open(test_data) as reader:
        for line in reader:
            seq_list.append(line[:line.find(label)])
            if len(seq_list) == test_num:
                break
    model = fasttext.load_model(MODEL_SAVE_PATH+"/model.bin")
    result = model.predict(seq_list)
    print(result)

if __name__ == "__main__":
    #get_data_fasttext()
    #data_dir = '/data/share/zhanghaipeng/data/chuangtouribao/event/fasttext/invest/data/'
    data_dir = '/data/share/zhanghaipeng/data/chuangtouribao/event/fasttext_anno/'
    train_data = data_dir + 'train.json'
    valid_data = data_dir + 'dev.json'
    test_data = data_dir + 'test.json'
    
    valid_data = data_dir + 'test_a.json'
    test_data = data_dir + 'test_b.json'
    
    train(train_data, valid_data)
    #predict(test_data)
    
    # model = train_supervised(
    #     input=train_data, epoch=25, lr=0.4, wordNgrams=3, verbose=2, minCount=1,
    #     loss="hs"
    # )
    # print_results(*model.test(valid_data))
    # model.save_model("cooking.bin")
    
    # model.quantize(input=train_data, qnorm=True, retrain=True, cutoff=100000)
    # print_results(*model.test(valid_data))
    # model.save_model("cooking.ftz")
