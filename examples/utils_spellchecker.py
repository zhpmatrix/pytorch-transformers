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
import pandas as pd
from io import open

from collections import OrderedDict
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score, classification_report
from tqdm import tqdm
import numpy as np
logger = logging.getLogger(__name__)

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
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
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, input_length):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.input_length = input_length


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


class SpellCheckerProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.txt")),"train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.txt")),"dev")
    
    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.txt")),"test")

    def get_labels(self):
        """See base class."""
        VOCAB_PATH = '/data/share/zhanghaipeng/data/pt_bert_models/bert-base-chinese/vocab.txt'
        label_list = []
        with open(VOCAB_PATH, 'r') as reader:
            for vocab in reader:
                label_list.append(vocab.strip())
        return label_list

    def _create_examples(self, lines_label,set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        if set_type == 'train':
            test_number = 100000
        if set_type == 'dev':
            test_number = 2000
        if set_type == 'test':
            test_number = 10000
        for (i, line) in enumerate(lines_label[:test_number]):
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            if len(text_a) != len(label):
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a,label=label))
        return examples

def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode,
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
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        tokens_a = [char.lower() for char in example.text_a]
        labels = [char.lower() for char in example.label]
        input_length = len(tokens_a) 
        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]
                labels = labels[:(max_seq_length - 2)]
                
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
        labels_ = labels + [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            labels_ = [cls_token]+labels_
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        if output_mode == "classification":
            label_ids = tokenizer.convert_tokens_to_ids(labels_)
        else:
            raise KeyError(output_mode)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)
            label_ids = label_ids + ([pad_token] * padding_length)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("labels: %s" % " ".join([str(x) for x in labels_]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))
            
        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_ids,
                              input_length=input_length))
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


def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def custom_metrics(preds, srcs, tgts, lengths, max_seq_length):
    srcs = np.array([srcs[i:i+max_seq_length] for i in range(0,len(srcs),max_seq_length)])
    example_num = srcs.shape[0]
    s2_p1_t2 = 0
    s2_p2_t2 = 0
    s0_p0_t2 = 0
    s0_p1_t2 = 0
    s0_p2_t2 = 0
    for i in range(example_num):
        real_seq_len    = lengths[i]
        if real_seq_len <= max_seq_length - 2:
            src             = srcs[i].tolist()[1 : 1 + real_seq_len]
            tgt             = tgts[i].tolist()[1 : 1 + real_seq_len]
            pred            = preds[i].tolist()[1: 1 + real_seq_len]
        else:
            src             = srcs[i].tolist()[1 : -1]
            tgt             = tgts[i].tolist()[1 : -1]
            pred            = preds[i].tolist()[1: -1]
        for j in range(len(pred)):
            if (src[j] == 0) and (pred[j] == 0) and (tgt[j] == 0):
                break
            else:
                if (src[j] == tgt[j]):
                    if (pred[j] == src[j]):
                        s2_p2_t2 += 1
                    else:
                        s2_p1_t2 += 1
                else:
                    if (pred[j] == src[j]):
                        s0_p0_t2 += 1
                    elif (pred[j] == tgt[j]):
                        s0_p2_t2 += 1
                    else:
                        s0_p1_t2 += 1

    print('s2_p1_t2:{}'.format(s2_p1_t2))
    print('s2_p2_t2:{}'.format(s2_p2_t2))
    print('s0_p0_t2:{}'.format(s0_p0_t2))
    print('s0_p1_t2:{}'.format(s0_p1_t2))
    print('s0_p2_t2:{}'.format(s0_p2_t2))
    
    total_count = s2_p1_t2 + s2_p2_t2 + s0_p0_t2 + s0_p1_t2 + s0_p2_t2
    prediction_accuracy = (s2_p2_t2 + s0_p2_t2) *1.0/(total_count)
    detection_accuracy = (s0_p1_t2 + s0_p2_t2) *1.0/(s0_p0_t2 + s0_p1_t2 + s0_p2_t2)
    correction_accuracy = (s0_p2_t2) *1.0/(s0_p0_t2 + s0_p1_t2 + s0_p2_t2)
    detection_error = (s2_p1_t2) *1.0/(total_count)
    
    return {'prediction accuracy': prediction_accuracy,
            'detection accuracy': detection_accuracy,
            'correction accuracy': correction_accuracy,
            'detection error rate': detection_error}

def acc_and_f1(preds, srcs, labels, max_seq_length):
    labels_list = labels.reshape(-1).tolist()
    preds_list = preds.reshape(-1).tolist()
    report = classification_report(labels_list, preds_list)
    return {
        "report": report
    }


def compute_metrics(task_name, preds, srcs, labels, lengths, max_seq_length):
    assert len(preds) == len(labels)
    if task_name == "spellchecker":
        return custom_metrics(preds, srcs, labels, lengths, max_seq_length)
        
    else:
        raise KeyError(task_name)

processors = {
    "spellchecker": SpellCheckerProcessor,
}

output_modes = {
    "spellchecker": "classification",
}

if __name__ == '__main__':
    pass
