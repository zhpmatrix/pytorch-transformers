import json
import torch
import numpy as np
import pandas as pd
from pprint import pprint

def load_labels(label_path = '/nfs/users/zhanghaipeng/data/kuaishou/label.csv'):
    label_list = []
    with open(label_path, 'r') as reader:
        for line in reader:
            label_list.append(line.strip())
    return label_list

def load_question_map(keywords_path = '/nfs/users/zhanghaipeng/data/kuaishou/bucket/m2u_man_question_kw.csv'):
    kw = pd.read_csv(keywords_path)
    question_map = {}
    for i in range(kw.shape[0]):
        key = str(kw.iloc[i]['id']) + '_' + kw.iloc[i]['question']
        value_list = kw.iloc[i]['keywords'].split(',')
        question_map[key] = value_list
    return question_map

def load_answer_map(answer_path = '/nfs/users/zhanghaipeng/data/kuaishou/bucket/m2u_man_answer.csv'):
    answer = pd.read_csv(answer_path)
    answer_map = {}
    for i in range(answer.shape[0]):
        qid = answer.iloc[i]['question_id']
        answer_value = answer.iloc[i]['answer']
        if qid not in answer_map:
            answer_map[qid] = []
            answer_map[qid].append(answer_value)
        else:
            answer_map[qid].append(answer_value)
    return answer_map

def get_answers(questions, answer_map):
    answer_list = []
    for qus in questions:
        qid = int(qus.split('_')[0])
        if qid == -1:
            answer_list.extend(['不是问题'])
        elif qid == 0:
            answer_list.extend(['无匹配'])
        else:
            answer_list.extend(answer_map[qid])
    return answer_list

def topk_test(examples, raw_preds, real_labels, tokenizer, args, processor):
    eigen_hit_num = 0.0
    kuaishou_hit_num = 0.0
    test_num = 0.00000005
    label_list = load_labels()
    
    label_map = {}
    for i, label in enumerate(label_list):
        label_map[i] = label
   
    #是否去掉"不是问题"和"无匹配"
    label_used = [i for i in range(len(label_list))]
    remove_labels = [0,1]
    #remove_labels = []
    for label in remove_labels:
        label_used.remove(label)

    #问题->关键词列表
    question_map = load_question_map()
    question_map_used = {}
    for i, (key, value_list) in enumerate(question_map.items()):
        if i in label_used:
            question_map_used[key] = value_list
    
    #问题->答案列表
    answer_map = load_answer_map()
    answer_map_used = {}
    for key, value_list in answer_map.items():
        if key in label_used:
            answer_map_used[key] = value_list
    pred_probs = torch.softmax(torch.tensor(raw_preds), axis=1).numpy()
    pred_labels = np.argmax(raw_preds, axis=1) 
    for example, pred_prob, pred_label, real_label in zip(examples, pred_probs, pred_labels, real_labels):
        tokens = tokenizer.convert_ids_to_tokens(example)
        query_valid_example_start = tokens.index('[CLS]')
        query_valid_example_end = tokens.index('[SEP]')
        query = ''.join(tokens[query_valid_example_start+1 : query_valid_example_end])
        
        tokens = tokens[query_valid_example_end + 1:]
        question_valid_example_end = tokens.index('[SEP]')
        question = ''.join(tokens[query_valid_example_start : question_valid_example_end])
        print('\t'.join([query, question, str(pred_label), str(round(pred_prob[pred_label], 4)), str(real_label)]))
        import pdb;pdb.set_trace()
        continue
        topk_questions = [label_map[index] for index in topk_pred]
        #快手：关键词匹配
        kuaishou_all_questions = set()
        for key, value_list in question_map_used.items():
            for value in value_list:
                if value in input_example:
                    kuaishou_all_questions.add(key)
        kuaishou_questions = list(kuaishou_all_questions)
        
        #print(input_example)
        #print(topk_questions)
        #print(kuaishou_questions)
                
        #eigen_answers = get_answers(topk_questions, answer_map)
        #kuaishou_answers = get_answers(kuaishou_questions, answer_map)
        #print('\n')
          
        if real_label in label_used:
            test_num += 1
            real_question = label_map[real_label]
            if real_question in topk_questions:
                eigen_hit_num += 1
            if real_question in kuaishou_questions:
                kuaishou_hit_num += 1
    
    results = {'topk': topk, 'test_num': test_num, 'eigen_hit_ratio': eigen_hit_num / test_num, 'eigen_hit_num': eigen_hit_num, 'kuaishou_hit_ratio':kuaishou_hit_num / test_num, 'kuaishou_hit_num':kuaishou_hit_num}
    pprint(results)

if __name__ == '__main__':
    pass
