#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
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
"""
Fine-tuning the library models for masked language modeling (BERT, ALBERT, RoBERTa...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=fill-mask

source: https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_mlm.py
"""
# You can also adapt this script on your own masked language modeling task. Pointers for this are left as comments.

import logging
import math
import os
import pdb
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional
import datasets
import evaluate
from datasets import load_dataset
from collections import defaultdict
from nltk import ngrams
import xxhash
import numpy as np
from scipy.stats import entropy
from datasets import Dataset

import datasets
import evaluate
from datasets import load_dataset
import argparse

from utils import count_ngrams, CMS

logger = logging.getLogger(__name__)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.


    parser = argparse.ArgumentParser(description='Example argument parser')

    # Add positional argument
    parser.add_argument('--sampled_data_path', help='path for the sampled data. It should be saved using a Dataset object from datasets .save_to_disk() method and the the sentences should be stored with the "text" key')

    # Add flag argument
    parser.add_argument('--seed', '-s', help='seed', default=42 , type=int)

    args = parser.parse_args()
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    task_to_keys = {
        "cola": ("sentence", None),
        "mnli": ("premise", "hypothesis"),
        "mrpc": ("sentence1", "sentence2"),
        "qnli": ("question", "sentence"),
        "qqp": ("question1", "question2"),
        "rte": ("sentence1", "sentence2"),
        "sst2": ("sentence", None),
        "stsb": ("sentence1", "sentence2"),
        "wnli": ("sentence1", "sentence2"),
    }


    the_pile = load_dataset('the_pile', split='train', streaming=True)
    the_pile_sample = the_pile.shuffle(seed=args.seed).take(100000)
    #the_pile_sample_2 = the_pile.shuffle(seed=17).take(100000)
    #sampled_data = []
    #for sample in the_pile_sample_2:
    #    sampled_data.append(sample['text'])
    #turn sampled_dataset into a dataset object
    #sampled_dataset = Dataset.from_dict({'text': sampled_data})
    #sampled_dataset.save_to_disk('/Users/muhammed/Desktop/Projects/CS543/CS543-final-project/data/the_pile_sample_2')
    # sampled_data = datasets.load_from_disk(data_files=args.sampled_data_path)
    sampled_data = datasets.load_dataset(args.sampled_data_path)
    sampled_data = sampled_data.shuffle(seed=args.seed)
    if len(sampled_data) > 100000:
        sampled_data = sampled_data.select(range(100000))

    sentences = []
    for taskname in task_to_keys.keys():
        raw_datasets = load_dataset('glue', taskname, split='train')
        for key in task_to_keys[taskname]:
            if key is not None:
                sentences.extend(raw_datasets[key])
    #from sentences filter sentences that are shorter than 2 words
    sentences = [sent for sent in sentences if len(sent.split()) > 2]
    #sample 100k sentences from sentences
    sentences = np.random.choice(sentences, 100000, replace=False)

    logger.info("Calculating glue features ...")
    print("Calculating glue features ...")
    counted_ngrams = count_ngrams(sentences, 2)
    cms = CMS(10000, 1, seed=args.seed, independence_k=2)
    cms.update(counted_ngrams)
    text_vector = cms.counter

    logger.info("Calculating Pile sample features ...")
    print("Calculating Pile sample features ...")
    cms.reset()
    counted_ngrams = count_ngrams(the_pile_sample, 2)
    cms.update(counted_ngrams)
    data_vector = cms.counter

    logger.info("Calculating filtered dataset features ...")
    print("Calculating filtered dataset features ...")
    cms.reset()
    counted_ngrams = count_ngrams(sampled_data, 2)
    cms.update(counted_ngrams)
    sampled_vector = cms.counter

    logger.info("Calculating KL reduction ...")
    print("Calculating KL reduction ...")
    #calculate KL divergence between text_vector an data_vector
    target_to_random = entropy(text_vector, data_vector, axis=1)
    target_to_sampled = entropy(text_vector, sampled_vector, axis=1)
    logger.info(f'KL divergence between text_vector and data_vector: {target_to_random}')
    logger.info(f'KL divergence between text_vector and sampled_vector: {target_to_sampled}')
    logger.info(f'KL reduction is {target_to_random - target_to_sampled}')


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()

"""
## example training script
python run_quick_eval.py \
    --sampled_data_path /Users/muhammed/Desktop/Projects/CS543/CS543-final-project/data/the_pile_sample_2/
    --seed 42 \
"""