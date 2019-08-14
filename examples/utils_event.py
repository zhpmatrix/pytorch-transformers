import copy
import json
import itertools
from collections import Counter
from tqdm import tqdm

def get_target_fields(annotation):
    return_info = 0
    return_anno = {}
    target_labels = ['融资主体','投资机构','投融资金额','融资轮次']
    old_label = annotation['label'][0]
    if old_label.find(target_labels[1]) != -1:
        new_label = target_labels[1]
    else:
        new_label = old_label[old_label.find('-')+1:]
    
    return_anno['label'] = {'label':new_label}
    return_anno['position'] = annotation['position']
    
    if new_label in target_labels:
        return_info = 1
    return return_info, return_anno

def get_data(data_dir = '/data/share/zhanghaipeng/data/chuangtouribao/event/',data_path='raw_data.json',save_path = 'data_with_label_position.json'):
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
                        
def construct_data(data_dir = '/data/share/zhanghaipeng/data/chuangtouribao/event/',data_path='data_with_label_position.json',save_path = 'data.json'):
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
                    round_bucket, money_bucket, subject_bucket, org_invest_bucket = get_buckets(text, event_list)
                    real_event, construct_event = get_events(round_bucket, money_bucket, subject_bucket, org_invest_bucket, text, event_list)
                    examples = get_examples(text, real_event, construct_event) 
                    for example in examples:
                        if example['cls_label'] == 1:
                            print(len(example['anno_list']))
                            print(example['anno_list'])

                        writer.write(json.dumps(example,ensure_ascii=False)+'\n')
    writer.close()

def get_examples(input_str, real_event, construct_event):
    """
        融资主体-投资机构-投融资金额-融资轮次
        如果字段=空，用'#'表示
    """
    examples = []
    
    target_labels = ['融资主体','投资机构','投融资金额','融资轮次']
    
    target_num = len(target_labels)
    real_list = []
    construct_list = []

    for event in real_event:
        real = ['#'] * target_num
        anno_list = event['anno_list']
        
        #支持多个投资机构
        org_invest_idx = [idx for idx, anno in enumerate(anno_list) if anno[0]['label'] == target_labels[1]]
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
                for i in range(len(target_labels)):
                    if label == target_labels[i]:
                        real[i] = text
            real_list.append('-'.join(real))
    for event in construct_event:
        example = {}
        cls_label = 0
        construct = ['#'] * 4
        for item in event:
            label = item[0]['label']
            text = item[1]['text']
            for i in range(len(target_labels)):
                if label == target_labels[i]:
                    construct[i] = text
        construct_str = '-'.join(construct)
        construct_list.append(construct_str)
        
        if construct_str in real_list:
            cls_label = 1
        example['text'] = input_str
        example['cls_label'] = cls_label
        example['anno_list'] = [item for item in event]
        examples.append(example)
    #数据验证
    if len(set(real_list) & set(construct_list)) > len(set(real_list)):
        import pdb;pdb.set_trace()
    print(real_list)
    print(construct_list)
    return examples

def get_events(round_bucket, money_bucket, subject_bucket, org_invest_bucket, text, event_list):
    target_labels = ['融资主体','投资机构','投融资金额','融资轮次']
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

    buckets = [round_bucket, money_bucket, subject_bucket, org_invest_bucket]
    
    round_list = [round_[1]['text'] for round_ in round_bucket]
    money_list = [money[1]['text'] for money in money_bucket]
    subject_list = [subject[1]['text'] for subject in subject_bucket]
    org_invest_list = [org_invest[1]['text'] for org_invest in org_invest_bucket]
    
    print('桶:')
    if len(round_list) > 0:
        print(round_list)
    if len(money_list) > 0:
        print(money_list)
    if len(subject_list) > 0:
        print(subject_list)
    if len(org_invest_list) > 0:
        print(org_invest_list)
    print('\n')

    print('合成事件:')
    buckets_not_none = [bucket for bucket in buckets if len(bucket) > 0]
    feature_list = list( itertools.product(*buckets_not_none) )
    #允许一个slot为空
    slot_none_list = []
    for feature in feature_list:
        feature_list_ = [item for item in feature]
        for i in range(len(feature_list_)):
            feature_ = copy.deepcopy(feature_list_)
            if feature_[i][0]['label'] != target_labels[0]:
                del(feature_[i])
                slot_none_list.append(tuple(feature_))
    #允许多个slot为空
    #TODO
    for feature in feature_list+slot_none_list:
        print([f[0]['label'] for f in feature])
        print([f[1]['text'] for f in feature])
    print('*'*20)
    return ground_truth, feature_list+slot_none_list


def get_buckets(text, event_list):
    target_labels = ['融资主体','投资机构','投融资金额','融资轮次']
    round_bucket = [] 
    money_bucket = []
    subject_bucket = []
    org_invest_bucket = []
    for event in event_list:
        if event['text'] == text:
            anno_list = event['anno_list']
            for anno in anno_list:
                if anno[0]['label'] == target_labels[0]:
                    subject_bucket.append(anno)
                if anno[0]['label'] == target_labels[1]:
                    org_invest_bucket.append(anno)
                if anno[0]['label'] == target_labels[2]:
                    money_bucket.append(anno)
                if anno[0]['label'] == target_labels[3]:
                    round_bucket.append(anno)
    return round_bucket,money_bucket,subject_bucket,org_invest_bucket

if __name__ == '__main__':
    #get_data()
    construct_data()
