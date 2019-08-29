# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" BERT classification fine-tuning: utilities to work with GLUE tasks """

from __future__ import absolute_import, division, print_function

import csv
import logging
import os
import sys
import json
import copy
from io import open

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score, classification_report

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, apos=None, aarc=None, text_b=None, bpos=None, barc=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.apos = apos
        self.aarc = aarc
        self.text_b = text_b
        self.bpos = bpos
        self.barc = barc
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, pos_ids, pos_mask, pos_segment_ids, arc_rel_ids, arc_rel_mask, arc_rel_segment_ids, arc_idx_ids, arc_idx_mask, arc_idx_segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.pos_ids = pos_ids
        self.pos_mask = pos_mask
        self.pos_segment_ids = pos_segment_ids
        self.arc_rel_ids = arc_rel_ids
        self.arc_rel_mask = arc_rel_mask
        self.arc_rel_segment_ids = arc_rel_segment_ids
        self.arc_idx_ids = arc_idx_ids
        self.arc_idx_mask = arc_idx_mask
        self.arc_idx_segment_ids = arc_idx_segment_ids
        self.label_id = label_id
    


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            text_b = line[4]
            label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")),
            "dev_matched")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[8]
            text_b = line[9]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliMismatchedProcessor(MnliProcessor):
    """Processor for the MultiNLI Mismatched data set (GLUE version)."""

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_mismatched.tsv")),
            "dev_matched")


class ColaProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class Sst2Processor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class StsbProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return [None]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[7]
            text_b = line[8]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QqpProcessor(DataProcessor):
    """Processor for the QQP data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            try:
                text_a = line[3]
                text_b = line[4]
                label = line[5]
            except IndexError:
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QnliProcessor(DataProcessor):
    """Processor for the QNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), 
            "dev_matched")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class RteProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class WnliProcessor(DataProcessor):
    """Processor for the WNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class EventProcessor(DataProcessor):
    """多事件判别数据预处理"""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.json")), "dev")
    
    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.json")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]
    
    def anno_to_nl(self,text):
        try:
            subject_, org_invest_, money_, round_ = text.split('-')
        except:
            #Pre-A单独处理
            special_str = 'Pre-A'
            start_idx = text.find(special_str)
            try:
                subject_, org_invest_, money_ = text[:start_idx - 1].split('-')
                round_ = text[start_idx:]
            except:
                # 处理投资机构中含有'-'的情况
                import re
                delimiter_idx =[i.start() for i in re.finditer('-', text)]
                subject_ = text[:delimiter_idx[0]] 
                org_invest_ = text[delimiter_idx[-3] + 1: delimiter_idx[-2]]
                money_ = text[delimiter_idx[-2] + 1: delimiter_idx[-1]]
                round_ = text[delimiter_idx[-1] + 1:]
        return 1, subject_ + '完成' + money_ + round_ + '融资，投资方为' + org_invest_
    
    def anno_to_nl_(self,text):
        """将四个字段转化为自然语言"""
        target_str = ''
        tag = 0
        try:
            subject_, org_invest_, money_, round_ = text.split('-')
        except:
            #Pre-A单独处理
            special_str = 'Pre-A'
            start_idx = text.find(special_str)
            try:
                subject_, org_invest_, money_ = text[:start_idx - 1].split('-')
                round_ = text[start_idx:]
            except:
                # 处理投资机构中含有'-'的情况
                import re
                delimiter_idx =[i.start() for i in re.finditer('-', text)]
                subject_ = text[:delimiter_idx[0]] 
                org_invest_ = text[delimiter_idx[-3] + 1: delimiter_idx[-2]]
                money_ = text[delimiter_idx[-2] + 1: delimiter_idx[-1]]
                round_ = text[delimiter_idx[-1] + 1:]

        tag_subject = 1 if subject_ != '#' else 0
        tag_org_invest = 1 if org_invest_ != '#' else 0
        tag_money = 1 if money_ != '#' else 0
        tag_round = 1 if round_ != '#' else 0
        if tag_subject == 1 and tag_org_invest == 0 and tag_money == 1 and tag_round == 1:
            tag = 1
            target_str = subject_ + '在' + round_ + '融资' + money_ + '。'
        if tag_subject == 1 and tag_org_invest == 1 and tag_money == 1 and tag_round == 0:
            tag = 1
            target_str = subject_ + '获' + org_invest_ + money_ + '投资。'
        if tag_subject == 1 and tag_org_invest == 1 and tag_money == 0 and tag_round == 1:
            tag = 1
            target_str = subject_ + '在' + round_ + '的投资机构是' + org_invest_ + '。'
        if tag_subject == 1 and tag_org_invest == 1 and tag_money == 1 and tag_round == 1:
            tag = 1
            target_str = subject_ + '在' + round_ + '融资' + money_ + '，' + '由' + org_invest_ + '投资' + '。'
        return tag, target_str
    
    def get_all_entity(self,text):
        """获取所有实体"""
        entity_num = 4
        target_list = [''] * entity_num
        tag = 0
        try:
            subject_, org_invest_, money_, round_ = text.split('-')
        except:
            #Pre-A单独处理
            special_str = 'Pre-A'
            start_idx = text.find(special_str)
            try:
                subject_, org_invest_, money_ = text[:start_idx - 1].split('-')
                round_ = text[start_idx:]
            except:
                # 处理投资机构中含有'-'的情况
                import re
                delimiter_idx =[i.start() for i in re.finditer('-', text)]
                subject_ = text[:delimiter_idx[0]] 
                org_invest_ = text[delimiter_idx[-3] + 1: delimiter_idx[-2]]
                money_ = text[delimiter_idx[-2] + 1: delimiter_idx[-1]]
                round_ = text[delimiter_idx[-1] + 1:]

        tag_subject = 1 if subject_ != '#' else 0
        tag_org_invest = 1 if org_invest_ != '#' else 0
        tag_money = 1 if money_ != '#' else 0
        tag_round = 1 if round_ != '#' else 0
        if tag_subject == 1 and tag_org_invest == 0 and tag_money == 1 and tag_round == 1:
            tag = 1
            target_list[0] = subject_
            target_list[1] = round_
            target_list[2] = money_
        if tag_subject == 1 and tag_org_invest == 1 and tag_money == 1 and tag_round == 0:
            tag = 1
            target_str = subject_ + '获' + org_invest_ + money_ + '投资。'
            target_list[0] = subject_
            target_list[2] = money_
            target_list[3] = org_invest_
        if tag_subject == 1 and tag_org_invest == 1 and tag_money == 0 and tag_round == 1:
            tag = 1
            target_str = subject_ + '在' + round_ + '的投资机构是' + org_invest_ + '。'
            target_list[0] = subject_
            target_list[1] = round_
            target_list[3] = org_invest_
        if tag_subject == 1 and tag_org_invest == 1 and tag_money == 1 and tag_round == 1:
            tag = 1
            target_str = subject_ + '在' + round_ + '融资' + money_ + '，' + '由' + org_invest_ + '投资' + '。'
            target_list[0] = subject_
            target_list[1] = round_
            target_list[2] = money_
            target_list[3] = org_invest_
        return tag, target_list
    
    def get_model(self):
        from simplex_sdk import SimplexClient
        from pyltp import Segmentor
        from pyltp import Postagger
        from pyltp import Parser
        LTP_MODEL_PATH = '/data/share/zhanghaipeng/data/ltp_model/'
        EULER_SERVICE_NAME = 'general-pos-ideal'
        model = SimplexClient(EULER_SERVICE_NAME)
        segmentor = Segmentor()
        postagger = Postagger()
        parser = Parser()
        segmentor.load(LTP_MODEL_PATH+'cws.model')
        postagger.load(LTP_MODEL_PATH+'pos.model')
        parser.load(LTP_MODEL_PATH+'parser.model')
        return segmentor, postagger, parser
    
    def get_pos_list_based_euler(self, model, text):   
        text_pos_list = []
        pos = model.predict([text])[0]['pos']
        for pos_ in pos:
            tag, offset, length = pos_['tag'], pos_['offset'], pos_['length']
            for i in range(offset, offset + length):
                text_pos_list.append(tag)
        return pos,text_pos_list
    
    def get_pos_list_based_ltp(self, segmentor, postagger, parser, text):   
        text_pos_list = []
        text_arc_list = []
        words = segmentor.segment(text)
        poses  = postagger.postag(words)
        for word, pos in zip(words, poses):
            text_pos_list.extend(['['+pos+']']*len(word))
        assert(len(text) == len(text_pos_list))
        arcs = parser.parse(words, poses)
        for arc in arcs:
            text_arc_list.append((arc.head, '['+arc.relation+']'))
        return text_pos_list, text_arc_list
    
    def text_filter(self,text):
        text = text.strip()
        text = text.replace(' ','')
        return text
    
    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        label_dict = {}
        segmentor, postagger, parser = self.get_model()
        for (i, line) in enumerate(lines):
            example_ = json.loads(line[0])
            text_b = example_['text']
            anno_label = example_['anno_label']
            label = str(example_['cls_label'])
            guid = "%s-%s-%s-%s" % (set_type, text_b, anno_label, label)
            tag, text_a = self.anno_to_nl_(anno_label)
            #tag, text_a = self.get_all_entity(anno_label)
            text_a = self.text_filter(text_a)
            text_b = self.text_filter(text_b)
            apos, aarc = self.get_pos_list_based_ltp(segmentor, postagger, parser, text_a)
            bpos, barc = self.get_pos_list_based_ltp(segmentor, postagger, parser, text_b)
            if tag == 0:#过滤不符合模版的example
                continue
            if label not in label_dict.keys():
                label_dict[label] = 1
            else:
                label_dict[label] += 1
            examples.append(
                InputExample(guid=guid, text_a=text_a, apos = apos, aarc=aarc, text_b = text_b, bpos = bpos, barc=barc, label=label))
        return examples

class ParserTokenizer:
    def __init__(self,vocab_path = '/data/share/zhanghaipeng/data/pt_bert_models/bert-base-chinese/parser_vocab.txt'):
        self.vocab = self.load_vocab(vocab_path)
    def load_vocab(self, vocab_path):
        parser_vocab = {}
        with open(vocab_path, 'r') as reader:
            for i, id_ in enumerate(reader):
                parser_vocab[i] = id_.strip()
        return parser_vocab
    def convert_tokens_to_ids(self, tokens):
        id_to_tokens = self.vocab
        tokens_to_ids = {j:i for i,j in self.vocab.items()}
        return [tokens_to_ids[token] for token in tokens]
    def convert_ids_to_tokens(self, ids):
        return [self.vocab[id_] for id_ in ids]
    
def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, parser_tokenizer, output_mode,
                                 cls_token_at_end=False, pad_on_left=False,
                                 cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                 cls_token_segment_id=1, pad_token_segment_id=0,
                                 mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        tokens_a = tokenizer.tokenize(example.text_a)
        #tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
            _truncate_pos_pair(example.apos, example.bpos, max_seq_length - 3)
            _truncate_parser_pair(example.aarc, example.barc, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = tokens_a + [sep_token]
        poses = example.apos + [sep_token]
        aarc_idx = [str(arc[0]) for arc in example.aarc]
        aarc_rel = [arc[1] for arc in example.aarc]
        barc_idx = [str(arc[0]) for arc in example.barc]
        barc_rel = [arc[1] for arc in example.barc]
        arc_rel = aarc_rel + [sep_token]
        arc_idx = aarc_idx + [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)
        pos_segment_ids = [sequence_a_segment_id] * len(poses)
        arc_rel_segment_ids = [sequence_a_segment_id] * len(arc_rel)
        arc_idx_segment_ids = [sequence_a_segment_id] * len(arc_idx)
        
        if tokens_b:
            tokens += tokens_b + [sep_token]
            poses += example.bpos + [sep_token]
            arc_rel += barc_rel + [sep_token]
            arc_idx += barc_idx + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)
            pos_segment_ids += [sequence_b_segment_id] * (len(example.bpos) + 1)
            arc_rel_segment_ids += [sequence_b_segment_id] * (len(barc_rel) + 1)
            arc_idx_segment_ids += [sequence_b_segment_id] * (len(barc_idx) + 1)
            
        if cls_token_at_end:
            tokens = tokens + [cls_token]
            poses = poses + [cls_token]
            arc_rel  = arc_rel + [cls_token]
            arc_idx  = arc_idx + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
            pos_segment_ids = pos_segment_ids + [cls_token_segment_id]
            arc_rel_segment_ids = arc_rel_segment_ids + [cls_token_segment_id]
            arc_idx_segment_ids = arc_idx_segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            poses = [cls_token] + poses
            arc_rel = [cls_token] + arc_rel
            arc_idx = [cls_token] + arc_idx
            segment_ids = [cls_token_segment_id] + segment_ids
            pos_segment_ids = [cls_token_segment_id] + pos_segment_ids
            arc_rel_segment_ids = [cls_token_segment_id] + arc_rel_segment_ids
            arc_idx_segment_ids = [cls_token_segment_id] + arc_idx_segment_ids
        

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        pos_ids = tokenizer.convert_tokens_to_ids(poses)
        arc_rel_ids = tokenizer.convert_tokens_to_ids(arc_rel)
        arc_idx_ids = parser_tokenizer.convert_tokens_to_ids(arc_idx)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        pos_mask = [1 if mask_padding_with_zero else 0] * len(pos_ids)
        arc_rel_mask = [1 if mask_padding_with_zero else 0] * len(arc_rel_ids)
        arc_idx_mask = [1 if mask_padding_with_zero else 0] * len(arc_idx_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        pos_padding_length = max_seq_length - len(pos_ids)
        arc_rel_padding_length = max_seq_length - len(arc_rel_ids)
        arc_idx_padding_length = max_seq_length - len(arc_idx_ids)
        
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            pos_ids = ([pad_token] * pos_padding_length) + pos_ids
            arc_rel_ids = ([pad_token] * arc_rel_padding_length) + arc_rel_ids
            arc_idx_ids = ([pad_token] * arc_idx_padding_length) + arc_idx_ids

            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            pos_mask = ([0 if mask_padding_with_zero else 1] * pos_padding_length) + pos_mask
            arc_rel_mask = ([0 if mask_padding_with_zero else 1] * arc_rel_padding_length) + arc_rel_mask
            arc_idx_mask = ([0 if mask_padding_with_zero else 1] * arc_idx_padding_length) + arc_idx_mask
            
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            pos_segment_ids = ([pad_token_segment_id] * pos_padding_length) + pos_segment_ids
            arc_rel_segment_ids = ([pad_token_segment_id] * arc_rel_padding_length) + arc_rel_segment_ids
            arc_idx_segment_ids = ([pad_token_segment_id] * arc_idx_padding_length) + arc_idx_segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            pos_ids = pos_ids + ([pad_token] * pos_padding_length)
            arc_rel_ids = arc_rel_ids + ([pad_token] * arc_rel_padding_length)
            arc_idx_ids = arc_idx_ids + ([pad_token] * arc_idx_padding_length)
            
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            pos_mask = pos_mask + ([0 if mask_padding_with_zero else 1] * pos_padding_length)
            arc_rel_mask = arc_rel_mask + ([0 if mask_padding_with_zero else 1] * arc_rel_padding_length)
            arc_idx_mask = arc_idx_mask + ([0 if mask_padding_with_zero else 1] * arc_idx_padding_length)
            
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)
            pos_segment_ids = pos_segment_ids + ([pad_token_segment_id] * pos_padding_length)
            arc_rel_segment_ids = arc_rel_segment_ids + ([pad_token_segment_id] * arc_rel_padding_length)
            arc_idx_segment_ids = arc_idx_segment_ids + ([pad_token_segment_id] * arc_idx_padding_length)
        
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        
        assert len(pos_ids) == max_seq_length
        assert len(pos_mask) == max_seq_length
        assert len(pos_segment_ids) == max_seq_length
        
        assert len(arc_rel_ids) == max_seq_length
        assert len(arc_rel_mask) == max_seq_length
        assert len(arc_rel_segment_ids) == max_seq_length
        
        assert len(arc_idx_ids) == max_seq_length
        assert len(arc_idx_mask) == max_seq_length
        assert len(arc_idx_segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 100000:# 输出所有样本(MAX_NUM=100000)
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("pos_ids: %s" % " ".join([str(x) for x in pos_ids]))
            logger.info("pos_mask: %s" % " ".join([str(x) for x in pos_mask]))
            logger.info("pos_segment_ids: %s" % " ".join([str(x) for x in pos_segment_ids]))
            logger.info("arc_rel_ids: %s" % " ".join([str(x) for x in arc_rel_ids]))
            logger.info("arc_rel_mask: %s" % " ".join([str(x) for x in arc_rel_mask]))
            logger.info("arc_rel_segment_ids: %s" % " ".join([str(x) for x in arc_rel_segment_ids]))
            logger.info("arc_idx_ids: %s" % " ".join([str(x) for x in arc_idx_ids]))
            logger.info("arc_idx_mask: %s" % " ".join([str(x) for x in arc_idx_mask]))
            logger.info("arc_idx_segment_ids: %s" % " ".join([str(x) for x in arc_idx_segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))
        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              pos_ids = pos_ids,
                              pos_mask = pos_mask,
                              pos_segment_ids = pos_segment_ids,
                              arc_rel_ids = arc_rel_ids,
                              arc_rel_mask = arc_rel_mask,
                              arc_rel_segment_ids = arc_rel_segment_ids,
                              arc_idx_ids = arc_idx_ids,
                              arc_idx_mask = arc_idx_mask,
                              arc_idx_segment_ids = arc_idx_segment_ids,
                              label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def _truncate_pos_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def _truncate_parser_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    report = classification_report(labels, preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
        "report":report
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }

def init_logger(log_file):
    logger = logging.getLogger('PREDICTION')
    logger.setLevel(level = logging.INFO)
    handler = logging.FileHandler(log_file)
    logger.addHandler(handler)
    return logger

def event_extraction(event_list, pred_labels):
    """
        事件聚合: 根据构造事件 和 预测(真实)标签
    """
    anno_list = [event[0] for event in event_list]
    real_labels = [int(event[1]) for event in event_list]
    real_event_list = event_extraction_(anno_list, real_labels)
    pred_event_list = event_extraction_(anno_list, pred_labels.tolist())
    return real_event_list, pred_event_list

def event_extraction_(anno_list, labels):
    # 找到类别=1的所有事件
    same_event_list = []
    # 找到类别=1的所有事件的subject，忽略'#'
    subject_set = set()
    for anno, label in zip(anno_list, labels):
        subject = anno.split('-')[0]
        if int(label) == 1:
            same_event_list.append(anno)
            if subject != '#':
                subject_set.add(subject)
    # 多事件slot填充
    event_list = []
    for subject in subject_set:
        tmp_event = {}
        tmp_event['subject'] = subject
        tmp_event['org_invest'] = []
        tmp_event['money'] = ''
        tmp_event['round'] = ''
        for anno_label in same_event_list:
            try:
                subject_, org_invest_, money_, round_ = anno_label.split('-')
            except:
                #Pre-A单独处理
                special_str = 'Pre-A'
                start_idx = anno_label.find(special_str)
                subject_, org_invest_, money_ = anno_label[:start_idx - 1].split('-')
                round_ = anno_label[start_idx:]
            if subject_ != '#' and subject_ == subject:
                if org_invest_ != '#':
                    tmp_event['org_invest'].append(org_invest_)
                if money_ != '#':
                    tmp_event['money'] = money_
                if round_ != '#':
                    tmp_event['round'] = round_
        event_list.append(tmp_event)
    return event_list

def get_predictions(pred_labels, real_labels, examples, output_file):
    header = ['预测标签','真实标签','事件']
    logger = init_logger(output_file)
    text_set = {}
    for example in examples:
        text_start = example.text_a.find('。')
        text = example.text_a[text_start+1:]
        anno_label = example.text_a[:text_start]
        label = example.label
        if text in text_set.keys():
            text_set[text].append([anno_label,label])
        else:
            text_set[text] = []
            text_set[text].append([anno_label,label])
    
    text_range_dict = {}
    example_num = 0
    for k, v in text_set.items():
        text_range_dict[k] = [example_num,example_num+len(v)]
        example_num += len(v)
    assert(example_num == len(pred_labels))
    assert(example_num == len(real_labels))
    
    for text,range_ in text_range_dict.items():
        start, end = range_[0], range_[1]
        construct_event_list = text_set[text]
        real_list = real_labels[start:end]
        pred_list = pred_labels[start:end]
        
        real_event_list, pred_event_list = event_extraction(construct_event_list, pred_list) 
        logger.info(text)
        logger.info('\n')
        
        logger.info('真实事件:\n')
        for real_event in real_event_list:
            subject = real_event['subject']
            if subject == '':
                subject = '#'
            org_invest = real_event['org_invest']
            money = real_event['money']
            if money == '':
                money = '#'
            round_ = real_event['round']
            if round_ == '':
                round_ = '#'
            if len(org_invest) > 0:
                for org in org_invest:
                    event_str = '-'.join([subject,org,money,round_])
                    logger.info(event_str)
            else:
                event_str = '-'.join([subject,'#',money,round_])
                logger.info(event_str)
        logger.info('\n')
        
        logger.info('预测事件:\n')
        for pred_event in pred_event_list:
            subject = pred_event['subject']
            if subject == '':
                subject = '#'
            org_invest = pred_event['org_invest']
            money = pred_event['money']
            if money == '':
                money = '#'
            round_ = pred_event['round']
            if round_ == '':
                round_ = '#'
            if len(org_invest) > 0:
                for org in org_invest:
                    event_str = '-'.join([subject,org,money,round_])
                    logger.info(event_str)
            else:
                event_str = '-'.join([subject,'#',money,round_])
                logger.info(event_str)
        logger.info('\n')

        logger.info('\t\t'.join(header))
        for i in range( len(construct_event_list) ):
            logger.info( '\t\t'.join([str(pred_list[i]),str(real_list[i]),construct_event_list[i][0]]) )
        logger.info('*' * 30)
    return {'predictions': None}


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "cola":
        return {"mcc": matthews_corrcoef(labels, preds)}
    elif task_name == "sst-2":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mrpc":
        return acc_and_f1(preds, labels)
    elif task_name == "event":
        return acc_and_f1(preds, labels)
    elif task_name == "sts-b":
        return pearson_and_spearman(preds, labels)
    elif task_name == "qqp":
        return acc_and_f1(preds, labels)
    elif task_name == "mnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mnli-mm":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "qnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "rte":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "wnli":
        return {"acc": simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)

processors = {
    "event": EventProcessor,
    "cola": ColaProcessor,
    "mnli": MnliProcessor,
    "mnli-mm": MnliMismatchedProcessor,
    "mrpc": MrpcProcessor,
    "sst-2": Sst2Processor,
    "sts-b": StsbProcessor,
    "qqp": QqpProcessor,
    "qnli": QnliProcessor,
    "rte": RteProcessor,
    "wnli": WnliProcessor,
}

output_modes = {
    "event": "classification",
    "cola": "classification",
    "mnli": "classification",
    "mnli-mm": "classification",
    "mrpc": "classification",
    "sst-2": "classification",
    "sts-b": "regression",
    "qqp": "classification",
    "qnli": "classification",
    "rte": "classification",
    "wnli": "classification",
}

GLUE_TASKS_NUM_LABELS = {
    "event": 2,
    "cola": 2,
    "mnli": 3,
    "mrpc": 2,
    "sst-2": 2,
    "sts-b": 1,
    "qqp": 2,
    "qnli": 2,
    "rte": 2,
    "wnli": 2,
}
