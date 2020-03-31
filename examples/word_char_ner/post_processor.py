import re
from collections import Counter

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
    aligner = AlignProcessor(input_list, preds_list, bd_preds_list, out_label_list)
    #new_preds_list = aligner.batch_processor()
    
    bd_locs = aligner.get_boundary(bd_preds)
    #new_preds = aligner.normalize_tags(preds, bd_locs)
    new_preds = aligner.each_processor(preds, bd_preds)
    print(new_preds)

