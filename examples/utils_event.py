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
                    examples = get_examples(round_bucket, money_bucket, subject_bucket, org_invest_bucket, text, event_list)
                                    
    writer.close()

def get_examples(round_bucket, money_bucket, subject_bucket, org_invest_bucket, text, event_list):
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
    for feature in feature_list:
        print([f[0]['label'] for f in feature])
        print([f[1]['text'] for f in feature])
    print('*'*20)
    #import pdb;pdb.set_trace()
    return None


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
