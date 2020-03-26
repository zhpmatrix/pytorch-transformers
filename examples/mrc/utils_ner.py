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
""" Named entity recognition fine-tuning: utilities to work with CoNLL-2003 task. """


import logging
import os
import copy
from itertools import chain
from utils.general_utils import GeneralUtils
from sklearn.metrics import f1_score, classification_report
from collections import Counter
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid, words, labels):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            words: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.words = words
        self.labels = labels


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids

general_utils = GeneralUtils()

def make_questions_by_tag(tag):
    questions = {
            'PER':'找出真实和虚构的人名。',
            'NORP':'找出政治，宗教团体。',
            'LOC':'找出国家，城市，山川等抽象或具体的地点。',
            'ORG':'找出公司，商业机构，社会组织等组织机构。',
            'GPE':'找出国家，城市和州的描述。',
            'DEV':'找出设备和产品。',
            'EVENT':'找出事件。',
            'WORK':'找出艺术作品。',
            'LAW':'找出关于法律的描述。',
            'LAN':'找出关于语言的描述。',
            'TIME':'句子中描述时间的词有哪些？',
            'PERCENT':'找出关于百分比的描述。',
            'CUR':'找出关于金钱的描述。',
            'QUANTITY':'数量词有哪些？',
            'ORDINAL':'找出序数词。',
            'CARDINAL':'找出数词。'
            }
    return questions[tag]

def make_examples(mode, guid_index, words, labels, bd_labels, examples):
    tag_set = set()
    for label in labels:
        tag_set.add(label.split('-')[-1])
    if 'O' in tag_set:
        tag_set.remove('O')
    for tag in tag_set:
        mask_labels = list( map(lambda x: x.split('-')[-1] == tag, labels) )
        tag_labels = [bd_labels[i] if mask == True else 'O' for i,mask in enumerate(mask_labels)]
        question = make_questions_by_tag(tag)
        new_words = list(question) + words
        new_labels = ['O'] * len(question) + tag_labels
        examples.append(InputExample(guid="{}-{}".format(mode, guid_index), words=new_words, labels=new_labels))
    return examples

def read_examples_from_file(data_dir, mode):
    file_path = os.path.join(data_dir, "{}.name".format(mode))
    guid_index = 1
    examples = []
    with open(file_path, encoding="utf-8") as f:
        words = []
        labels = []
        bd_labels = []
        for line in f:
            if line == "\n":
                if words:
                    examples = make_examples(mode, guid_index, words, labels, bd_labels, examples)
                    guid_index += 1
                    words = []
                    labels = []
                    bd_labels = []
            else:
                splits = line.strip().split("\t")
                ch = splits[0]
                #英文字母转小写
                if general_utils.is_alphabet(ch):                
                    words.append(ch.lower())
                else:
                    words.append(ch)
                if len(splits) > 1:
                    labels.append(splits[1])
                    bd_labels.append(splits[2])
                else:
                    # Examples could have no label for mode = "test"
                    labels.append("O")
                    bd_labels.append("O")
        if words:
            examples = make_examples(mode, guid_index, words, labels, bd_labels, examples)
    return examples


def convert_examples_to_features(
    examples,
    label_list,
    bd_label_list,
    max_seq_length,
    tokenizer,
    cls_token_at_end=False,
    cls_token="[CLS]",
    cls_token_segment_id=1,
    sep_token="[SEP]",
    sep_token_extra=False,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    pad_token_label_id=-100,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True,
):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    #label_map = {label: i for i, label in enumerate(label_list)}
    label_map = {label: i for i, label in enumerate(bd_label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        tokens = []
        label_ids = []
        bd_label_ids = []
        for word, label in zip(example.words, example.labels):
            #word_tokens = tokenizer.tokenize(word)
            word_tokens = word
            tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = 3 if sep_token_extra else 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            label_ids = label_ids[: (max_seq_length - special_tokens_count)]

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
        tokens += [sep_token]
        label_ids += [pad_token_label_id]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
            label_ids += [pad_token_label_id]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            label_ids += [pad_token_label_id]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            label_ids = [pad_token_label_id] + label_ids
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            label_ids = ([pad_token_label_id] * padding_length) + label_ids
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            label_ids += [pad_token_label_id] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))
        features.append(
            InputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, label_ids=label_ids)
        )
    return features


def get_labels(path):
    if path:
        with open(path, "r") as f:
            labels = f.read().splitlines()
        if "O" not in labels:
            labels = ["O"] + labels
        return labels
    else:
        return ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]

def compute_metrics(in_real_labels, in_pred_labels, labels):
    real_labels = list(chain(*in_real_labels)) 
    pred_labels = list(chain(*in_pred_labels)) 
    metric_labels = copy.deepcopy(labels)
    metric_labels.remove('O')
    report = classification_report(real_labels, pred_labels, labels=metric_labels)
    micro_f1 = f1_score(real_labels, pred_labels, labels=metric_labels, average='micro')
    macro_f1 = f1_score(real_labels, pred_labels, labels=metric_labels, average='macro')
    
    return {'report':report,
            'micro_f1':micro_f1,
            'macro_f1':macro_f1}
