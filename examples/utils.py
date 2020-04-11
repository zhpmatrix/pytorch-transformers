import numpy as np
from pprint import pprint

def topk_test(examples, raw_preds, real_labels, topk, tokenizer, args, processor):
    hit_ratio = 0.0
    hit_num = 0.0
    test_num = 0.00000005
    label_list = processor.get_labels(args.label_path)
    
    label_map = {}
    for i, label in enumerate(label_list):
        label_map[i] = label
    
    label_used = [i for i in range(len(label_list))]
    remove_labels = [0,1]
    remove_labels = []
    for label in remove_labels:
        label_used.remove(label)
    #preds = np.argmax(raw_preds, axis=1)
    topk_preds = np.argsort(-1*raw_preds, axis=1)[:,:topk]
    
    for example, topk_pred, real_label in zip(examples, topk_preds, real_labels):
        tokens = tokenizer.convert_ids_to_tokens(example)
        valid_example_start = tokens.index('[CLS]')
        valid_example_end = tokens.index('[SEP]')
        input_example = ''.join(tokens[valid_example_start+1:valid_example_end])
        topk_questions = [label_map[index] for index in topk_pred]
        
        print(input_example)
        print(topk_questions)
        print('\n')
        
        if real_label in label_used:
            test_num += 1
            real_question = label_map[real_label]
            if real_question in topk_questions:
                hit_num += 1
    #results = {'topk': topk, 'hit_ratio': hit_num / test_num, 'hit_num':hit_num, 'test_num':test_num}
    #pprint(results)


if __name__ == '__main__':
    pass
