import random
import copy
import json
import itertools
from collections import Counter
from tqdm import tqdm

random.seed(666)
DATA_DIR = '/data/share/zhanghaipeng/data/chuangtouribao/event/'
TARGET_LABELS = ['融资主体','投资机构','投融资金额','融资轮次']

def get_target_fields(annotation):
    return_info = 0
    return_anno = {}
    old_label = annotation['label'][0]
    if old_label.find(TARGET_LABELS[1]) != -1:
        new_label = TARGET_LABELS[1]
    else:
        new_label = old_label[old_label.find('-')+1:]
    
    return_anno['label'] = {'label':new_label}
    return_anno['position'] = annotation['position']
    
    if new_label in TARGET_LABELS:
        return_info = 1
    return return_info, return_anno

def get_data(data_dir = DATA_DIR ,data_path='raw_data.json',save_path = 'data_with_label_position.json'):
    """
        从标注数据中提取待处理的数据
    """
    writer = open(data_dir+save_path, 'a')
    with open(data_dir+data_path, 'r') as reader:
        textlines = reader.readlines()
        total_num = len(textlines)
        for i,line in enumerate(textlines):
            print(i,total_num)
            article = json.loads(json.loads(line)['input'])
            article_dict = {}
            article_dict['id'] = i
            article_dict['event_list'] = []
            for para in article:
                if 'annotation' in para.keys():
                    annotations = para['annotation']
                    data = para['data']
                    group_id = Counter([anno['group_id'] for anno in annotations])
                    for key in group_id.keys():
                        event_dict = {}
                        event_dict['text'] = data
                        event_dict['anno_list'] = []
                        event = [anno for anno in annotations if anno['group_id'] == key]
                        for anno in event:
                            info, new_anno = get_target_fields(anno)
                            if info:
                                event_dict['anno_list'].append([new_anno['label'], new_anno['position']])
                        article_dict['event_list'].append(event_dict)
            writer.write(json.dumps(article_dict,ensure_ascii=False)+'\n')
    writer.close()
                        
def construct_data(data_dir = DATA_DIR, data_path='data_with_label_position.json',save_path = 'data.json'):
    """
        数据构建
    """
    writer = open(data_dir+save_path, 'a')
    with open(data_dir+data_path, 'r') as reader:
        textlines = reader.readlines()
        total_num = len(textlines)
        for i,line in enumerate(textlines):
            event_list = json.loads(line)['event_list']
            text_dict = Counter([event['text'] for event in event_list])
            for text, cnt in text_dict.items():
                if cnt > 1:# 相同段落中有多个事件
                    subject_bucket, org_invest_bucket, money_bucket, round_bucket = get_buckets(text, event_list)
                    real_event, construct_event = get_events(round_bucket, money_bucket, subject_bucket, org_invest_bucket, text, event_list)
                    examples = get_examples(text, real_event, construct_event) 
                    new_examples = balance_examples(examples)
                    for example in new_examples:
                        #print(example['cls_label'],example['anno_label'])
                        writer.write(json.dumps(example,ensure_ascii=False)+'\n')
    writer.close()

def balance_examples(examples):
    """
        正负样本不平衡采样
        局部平衡：单个句子中的正负样本平衡
    """
    new_examples = []
    cls_counter = Counter([example['cls_label'] for example in examples])
    cls_label_pos = [idx for idx, example in enumerate(examples) if example['cls_label'] == 1]
    cls_label_neg = [idx for idx, example in enumerate(examples) if example['cls_label'] == 0]
    if len(cls_counter.keys()) > 1:
        num = cls_counter[0] - cls_counter[1]
        if num > 0:
            cls_label_neg_ = random.sample(cls_label_neg,cls_counter[1])
            new_examples.extend([examples[i] for i in cls_label_pos])
            new_examples.extend([examples[i] for i in cls_label_neg_])
        else:
            cls_label_pos_ = random.sample(cls_label_pos,cls_counter[0])
            new_examples.extend([examples[i] for i in cls_label_pos_])
            new_examples.extend([examples[i] for i in cls_label_neg])
    return new_examples

def get_examples(input_str, real_event, construct_event):
    """
        融资主体-投资机构-投融资金额-融资轮次
        用'#'表示字段为空
    """
    examples = []
    
    target_num = len(TARGET_LABELS)
    real_list = []
    construct_list = []

    for event in real_event:
        real = ['#'] * target_num
        anno_list = event['anno_list']
        
        #支持多个投资机构
        org_invest_idx = [idx for idx, anno in enumerate(anno_list) if anno[0]['label'] == TARGET_LABELS[1]]
        org_invest_num = len(org_invest_idx)
        if org_invest_num > 1:   
            anno_list_ = []
            for idx in org_invest_idx:
                tmp_anno_list = copy.deepcopy(anno_list)
                del(tmp_anno_list[idx])
                anno_list_.append(tmp_anno_list)
        else:
            anno_list_ = [anno_list]
        for anno_item in anno_list_:
            for anno in anno_item:
                label = anno[0]['label']
                text = anno[1]['text']
                for i in range(len(TARGET_LABELS)):
                    if label == TARGET_LABELS[i]:
                        real[i] = text
            real_list.append('-'.join(real))
    
    #支持子事件
    real_all_list = []
    for real_ in real_list:
        text_split = real_.split('-')
        sub_bucket = [[],[],[],[]]
        for i in range(len(TARGET_LABELS)):
            sub_bucket[i].append('#')
        for i in range(len(TARGET_LABELS)):
            sub_bucket[i].append(text_split[i])
        sub_event = list( itertools.product(*sub_bucket) )
        real_all_list.extend(sub_event)
    
    #子事件去重，筛选(至少两个字段确定一个事件)
    field_num = 2
    real_filter_list = ['-'.join(real) for real in list(set(real_all_list)) if Counter(real)['#'] <= field_num]
    
    for event in construct_event:
        example = {}
        cls_label = 0
        construct = ['#'] * 4
        for item in event:
            label = item[0]['label']
            text = item[1]['text']
            for i in range(len(TARGET_LABELS)):
                if label == TARGET_LABELS[i]:
                    construct[i] = text
        construct_str = '-'.join(construct)
        #合成事件筛选(至少两个字段确定一个事件) 
        if Counter(construct_str)['#'] > field_num:
            continue
        construct_list.append(construct_str)
        if construct_str in real_filter_list: #支持事件和子事件
            cls_label = 1
        example['text'] = input_str
        example['cls_label'] = cls_label
        example['anno_label'] = construct_str
        example['anno_list'] = [item for item in event]
        examples.append(example)
    #数据验证
    if len(set(real_list) & set(construct_list)) > len(set(real_list)):
        import pdb;pdb.set_trace()
    print(real_filter_list)
    print(construct_list)
    return examples

def get_events(round_bucket, money_bucket, subject_bucket, org_invest_bucket, text, event_list):
    
    ground_truth = []
    for event in event_list:
        if event['text'] == text:
            ground_truth.append(event)
   
    print('输入文本: ', text)
    print('\n')

    print('真实事件:')
    for ground in ground_truth:
        anno_list = ground['anno_list']
        print([anno[0]['label'] for anno in anno_list])
        print([anno[1]['text'] for anno in anno_list])
    print('\n') 

    buckets = [subject_bucket, org_invest_bucket, money_bucket, round_bucket]
    
    subject_list = [subject[1]['text'] for subject in subject_bucket]
    org_invest_list = [org_invest[1]['text'] for org_invest in org_invest_bucket]
    money_list = [money[1]['text'] for money in money_bucket]
    round_list = [round_[1]['text'] for round_ in round_bucket]
    
    print('桶:')
    print(subject_list)
    print(org_invest_list)
    print(money_list)
    print(round_list)
    print('\n')
    
    
    print('桶合成:')
    buckets_none = [subject_list, org_invest_list, money_list, round_list]
    feature_none_list = list( itertools.product(*buckets_none) )
    for feature_none in feature_none_list:
        print(feature_none)
    
    print('合成事件:')
    buckets_not_none = [bucket for bucket in buckets if len(bucket) > 0]
    feature_list = list( itertools.product(*buckets_not_none) )
    
    assert( len(feature_none_list) == len(feature_list) )
    
    for feature in feature_list:
        print([f[0]['label'] for f in feature])
        print([f[1]['text'] for f in feature])
    print('*'*20)
    return ground_truth, feature_list

def init_buckets():
    subject_bucket = []
    org_invest_bucket = []
    round_bucket = [] 
    money_bucket = []
    
    #添加空字段#
    placeholder = {}
    pos_dict = {}
    pos_dict['endOffset'] = 10000
    pos_dict['startOffset'] = 10000
    pos_dict['paraOffset'] = 10000
    placeholder['pos'] = pos_dict
    placeholder['text'] = '#'

    label_dict = {}
    label_dict['label'] = TARGET_LABELS[0]
    subject_bucket.append([label_dict, placeholder])
    
    label_dict = {}
    label_dict['label'] = TARGET_LABELS[1]
    org_invest_bucket.append([label_dict, placeholder])
    
    label_dict = {}
    label_dict['label'] = TARGET_LABELS[2]
    money_bucket.append([label_dict, placeholder])
    
    label_dict = {}
    label_dict['label'] = TARGET_LABELS[3]
    round_bucket.append([label_dict, placeholder])
    return subject_bucket, org_invest_bucket, money_bucket, round_bucket

def get_buckets(text, event_list):
    subject_bucket, org_invest_bucket, money_bucket, round_bucket = init_buckets()

    for event in event_list:
        if event['text'] == text:
            anno_list = event['anno_list']
            for anno in anno_list:
                if anno[0]['label'] == TARGET_LABELS[0]:
                    subject_bucket.append(anno)
                if anno[0]['label'] == TARGET_LABELS[1]:
                    org_invest_bucket.append(anno)
                if anno[0]['label'] == TARGET_LABELS[2]:
                    money_bucket.append(anno)
                if anno[0]['label'] == TARGET_LABELS[3]:
                    round_bucket.append(anno)
    
    return subject_bucket, org_invest_bucket, money_bucket, round_bucket

if __name__ == '__main__':
    #get_data()
    construct_data()
