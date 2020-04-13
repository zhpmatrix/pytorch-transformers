import numpy as np
import pandas as pd
from pprint import pprint

    
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

def topk_test(examples, raw_preds, real_labels, topk, tokenizer, args, processor):
    hit_ratio = 0.0
    eigen_hit_num = 0.0
    kuaishou_hit_num = 0.0
    test_num = 0.00000005
    label_list = processor.get_labels(args.label_path)
    
    label_map = {}
    for i, label in enumerate(label_list):
        label_map[i] = label
    
    label_used = [i for i in range(len(label_list))]
    remove_labels = [0,1]
    #remove_labels = []
    for label in remove_labels:
        label_used.remove(label)
    
    question_map = load_question_map()
    question_map_used = {}
    for i, (key, value_list) in enumerate(question_map.items()):
        if i in label_used:
            question_map_used[key] = value_list
    
    answer_map = load_answer_map()
    answer_map_used = {}
    for key, value_list in answer_map.items():
        if key in label_used:
            answer_map_used[key] = value_list

    topk_preds = np.argsort(-1*raw_preds, axis=1)[:,:topk]
    
    for example, topk_pred, real_label in zip(examples, topk_preds, real_labels):
        tokens = tokenizer.convert_ids_to_tokens(example)
        valid_example_start = tokens.index('[CLS]')
        valid_example_end = tokens.index('[SEP]')
        input_example = ''.join(tokens[valid_example_start+1:valid_example_end])
        topk_questions = [label_map[index] for index in topk_pred]    
         
        kuaishou_all_questions = set()
        for key, value_list in question_map_used.items():
            for value in value_list:
                if value in input_example:
                    kuaishou_all_questions.add(key)
        kuaishou_questions = list(kuaishou_all_questions)
        
        print(input_example)
        print(topk_questions)
        print(kuaishou_questions)
                
        eigen_answers = get_answers(topk_questions, answer_map)
        kuaishou_answers = get_answers(kuaishou_questions, answer_map)
        print('\n')
          
        if real_label in label_used:
            import pdb;pdb.set_trace()
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
