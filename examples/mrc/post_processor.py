import re
from collections import Counter
from utils_ner import get_query_map

class PostProcessor():
    def __init__(self, input_list, preds_list, bd_preds_list, out_label_list=None, out_bd_label_list=None):
        self.input_list = input_list
        self.preds_list = preds_list
        self.bd_preds_list = bd_preds_list
        self.out_label_list = out_label_list
        self.out_bd_label_list = out_bd_label_list
    
    def batch_processor(self):
        pass

    def each_processor(self):
        pass

class AlignProcessor(PostProcessor):
    
    def __init__(self, input_list, preds_list, bd_preds_list, out_label_list):
        super().__init__(input_list, preds_list, bd_preds_list, out_label_list)
    
    def batch_processor(self):
        new_preds_list = []
        for i, (preds, bd_preds) in enumerate(zip(self.preds_list, self.bd_preds_list)):
            new_preds = self.each_processor(preds, bd_preds)
            new_preds_list.append(new_preds)
        return new_preds_list
    
    def each_processor(self, preds, bd_preds):
        new_preds = []
        B_num = len(list(filter(lambda x: x == 'B', bd_preds)))
        E_num = len(list(filter(lambda x: x == 'E', bd_preds)))
        if not self.has_boundary_tags(B_num, E_num):
            return preds
        else:
            bd_locs = self.get_boundary(bd_preds)
            if len(bd_locs) > 0:
                new_preds = self.normalize_tags(preds, bd_locs)
            else:
                return preds
        return new_preds
    
    def has_boundary_tags(self, B_num, E_num):
        """
            边界预测序列:
                - 只有B
                - 只有E
                - 全O，即没有B，也没有E
        """
        return False if B_num == 0 or E_num == 0 else True

    def get_boundary(self, bd_preds):
        """
            B和E应该成对出现，否则不处理
        """
        bd_locs = []
        tag_dict = Counter(bd_preds)
        if tag_dict.get('B') == tag_dict.get('E'):
            bd_locs = [item.span() for item in re.finditer('BO*E', ''.join(bd_preds))]
        return bd_locs
    
    def normalize_tags(self, preds, bd_locs):
        """
            标签修正：根据边界预测和实体类型预测
        """
        new_preds = ['O'] * len(preds)
        for (start, end) in bd_locs:
            entity_tag_list = preds[start:end]
            if any([ tag != 'O' for tag in entity_tag_list]):
                new_entity_tag_list = self.change_tag(entity_tag_list, self.use_max_tag)
                new_preds[start:end] = new_entity_tag_list
        return new_preds
    
    def change_tag(self,entity_tag_list, use_tag):
        """
            标签修正的模板函数
        """
        filter_tag_list = [tag for tag in entity_tag_list if tag.split('-')[-1] != 'O']
        fix_tag = use_tag(filter_tag_list)
        new_entity_tag_list = []
        for i, tag in enumerate(entity_tag_list):
            if i == 0:
                new_entity_tag_list.append('B-'+fix_tag)
            else:
                new_entity_tag_list.append('I-'+fix_tag)
        return new_entity_tag_list
    
    def use_first_tag(self, entity_tag_list):
        """
            使用第一个实体的标签
        """
        return entity_tag_list[0].split('-')[-1]
    
    def use_last_tag(self, entity_tag_list):
        """
            使用最后一个实体的标签
        """
        return entity_tag_list[-1].split('-')[-1]

    def use_max_tag(self, entity_tag_list):
        """
            使用出现次数最多的标签
        """
        tag_dict = Counter([i.split('-')[-1] for i in entity_tag_list])
        (tag,_) = sorted(tag_dict.items(), key=lambda x: x[1], reverse=True)[0]
        return tag

class MRCProcessor(PostProcessor):
    
    def __init__(self, input_list, query_list, bd_preds_list, out_label_list):
        super().__init__(input_list, None, bd_preds_list, out_label_list = out_label_list)
        self.query_list = query_list
        _, self.query_to_tag = get_query_map()
        self.query_context_split_chars = ['。','？']
    
    def batch_processor(self):
        new_preds = []
        for input_text, bd_preds in zip(self.input_list, self.bd_preds_list):
            new_preds.append(self.each_processor(''.join(input_text), bd_preds))
        return new_preds
    
    def batch_processor_merge(self):
        examples = self.get_each_example()
        new_preds = []
        real_preds = []
        for input_str, label_list in examples.items():
            preds, reals = self.change_be_to_tag(input_str, label_list)
            new_preds.append(preds)
            real_preds.append(reals)
        return new_preds, real_preds

    def change_be_to_tag(self, input_str, label_list):
        """
            特别提示：统计冲突位置和原因分析
        """
        new_preds = ['O'] * len(input_str)
        new_reals = ['O'] * len(input_str)
        for (tag, bd_preds, real_preds) in label_list:
            #处理边界标签
            bd_locs = [item.span() for item in re.finditer('BO*E', ''.join(bd_preds))]
            for (start, end) in bd_locs:
                for i in range(start, end):
                    if i == start:
                        new_preds[i] = 'B-'+tag
                    else:
                        new_preds[i] = 'I-'+tag
            #处理真实标签
            for i,label in enumerate(real_preds):
                if label != 'O':
                    new_reals[i] = label
        return new_preds, new_reals
    
    def get_each_example(self):
        """
            特别说明：假定测试样本不重复
        """
        examples = {}
        labels = []
        for query, input_text, bd_preds, real_out_label in zip(self.query_list, self.input_list, self.bd_preds_list, self.out_label_list):
            input_str = ''.join(input_text)
            tag = self.query_to_tag[query]
            if input_str not in examples:
                examples[input_str] = []
                examples[input_str].append((tag,bd_preds, real_out_label))
            else:
                examples[input_str].append((tag,bd_preds, real_out_label))
        return examples

    def each_processor(self, input_text, bd_preds):
        """
            不处理的情况：
                只有一个B
                只有一个E
            待测试：
                B和E的个数相等，但是顺序不对 
        """
        query = re.split('|'.join(self.query_context_split_chars), input_text)[0]        
        tag = self.query_to_tag[query]
        new_preds = ['O'] * len(bd_preds)
        bd_locs = [item.span() for item in re.finditer('BO*E', ''.join(bd_preds))]
        for (start, end) in bd_locs:
            for i in range(start, end):
                if i == start:
                    new_preds[i] = 'B-'+tag
                else:
                    new_preds[i] = 'I-'+tag
        return new_preds

if __name__== '__main__':


    bd_preds0 = ['B','E','O']
    bd_preds1 = ['B','B','O']
    bd_preds2 = ['E','E','O']
    bd_preds3 = ['O','O','O']
    bd_preds4 = ['B','O','O','E','B','E','O','B','E'] 
    bd_preds5 = ['B','O','O','E','O','E','B','E','O','B','E','O','B'] 
    bd_preds6 = ['O','B','O','O','E','O','B','O','O','E'] 
    bd_preds7 = ['O','B','O','E']
    bd_preds8 = ['O','B','O','E','O','B','E']
    bd_preds9 = ['O','B','E','O','O','O']
    preds0 = ['O','B-PER','I-PER','B-GPE','I-PER','O','B-ORG','I-ORG','I-GPE','I-GPE']
    preds1 = ['O','B-PER','I-GPE','O','O','O']
    preds2 = ['O','O','B-GPE','O']
    preds3 = ['O','B-GPE','I-GPE','I-GPE','I-GPE','I-GPE','I-GPE']
    bd_preds = bd_preds6
    preds = preds0
    
    input_list = None
    preds_list = None
    bd_preds_list = None
    out_label_list = None
    
    mrc = MRCProcessor(input_list, bd_preds_list, out_label_list)
    input_text = '找出艺术作品。敬请收看走遍中国特别节目，'
    bd_preds = ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B', 'O', 'O', 'E', 'O', 'O', 'O', 'O', 'O']
    bd_preds = ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B', 'O', 'O', 'B', 'O', 'O', 'E', 'B', 'E', 'O', 'O', 'O']
    bd_preds = ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B', 'O', 'O', 'B', 'O', 'O', 'E', 'B', 'E', 'E', 'O', 'O']
    new_preds = mrc.each_processor(input_text, bd_preds)
    print(new_preds)
    exit()
    

    aligner = AlignProcessor(input_list, preds_list, bd_preds_list, out_label_list)
    #new_preds_list = aligner.batch_processor()
    
    bd_locs = aligner.get_boundary(bd_preds)
    #new_preds = aligner.normalize_tags(preds, bd_locs)
    new_preds = aligner.each_processor(preds, bd_preds)
    print(new_preds)

