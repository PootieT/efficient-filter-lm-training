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

    class CMS():
        def __init__(self, k, t, seed=42, independence_k =2):
            # number of buckets
            self.k = k
            # number of repeats
            self.t = t
            # the k wise independence of the hash functions. Default is pairwise independence
            self.independence_k = independence_k
            # seed for the random number generator
            self.seed = seed
            self._init()
        
        def _init(self):
            self._init_hash()
            self._init_counter()
        
        def _init_hash(self):
            # hash functions
            self.p=2**31-1
            #select (t, independence_k) random numbers from 0 to p-1. The first column is the constant term and should be non-zero. 
            self.hash_weights = np.random.randint(0, self.p, size=(self.t, self.independence_k))
            self.hash_weights[:,0] = self.hash_weights[:,0] + 1

        def _init_counter(self):
            self.counter = np.zeros((self.t, self.k), dtype=np.int32)
        
        def _hash(self, x):
            all_hashes = []
            for i in range(self.t):
                #hash = sum([self.hash_weights[i, j] * x**j for j in range(self.independence_k)]) % self.p % self.k
                hash = x % self.k
                all_hashes.append(hash)
            return all_hashes

        def _update(self, x, value):
            h = self._hash(x)
            try:
                self.counter[np.arange(self.t), h] += int(value)
            except:
                print(x, value, h)
        
        def _query(self, x):
            h = self._hash(x)
            return np.min(self.counter[np.arange(self.t), h])
        
        def update(self, X):
            for x, value in X.items():
                str_hasher = xxhash.xxh64(seed=self.seed)
                str_hasher.update(x)
                int_x = int(str_hasher.intdigest())
                self._update(int_x, value)
        
        def _get_sum(self):
            return np.sum(self.counter)

        def query(self, X):
            #return np.array([self._query(x)/self._get_sum() for x in X])
            return np.array([self._query(x) for x in X])

        def reset(self):
            self._init_counter()

    def generate_ngrams(sentence, n):
        tokens = sentence.split()
        all_ngrams = []
        for i in range(1,n+1):
            all_ngrams.extend(['-'.join(a) for a in list(ngrams(tokens, i))])
        return all_ngrams

    #generate a dictionary of unigram and bigrams that occur in a dataset and their total counts
    def count_ngrams(dataset, n):
        ngram_counter = {}
        for sent in dataset:
            if type(sent) == dict:
                sent = sent['text']
            generated_ngrams = generate_ngrams(sent,n)
            for ngram in generated_ngrams:    
                if ngram in ngram_counter:
                    ngram_counter[ngram] += 1
                else:
                    ngram_counter[ngram] = 1
        return ngram_counter

    the_pile = load_dataset('the_pile', split='train', streaming=True)
    the_pile_sample = the_pile.shuffle(seed=args.seed).take(100000)
    #the_pile_sample_2 = the_pile.shuffle(seed=17).take(100000)
    #sampled_data = []
    #for sample in the_pile_sample_2:
    #    sampled_data.append(sample['text'])
    #turn sampled_dataset into a dataset object
    #sampled_dataset = Dataset.from_dict({'text': sampled_data})
    #sampled_dataset.save_to_disk('/Users/muhammed/Desktop/Projects/CS543/CS543-final-project/data/the_pile_sample_2')
    sampled_data = datasets.load_from_disk(data_files=args.sampled_data_path)

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
    
    counted_ngrams = count_ngrams(sentences, 2)
    cms = CMS(10000, 1, seed=args.seed, independence_k=2)
    cms.update(counted_ngrams)
    text_vector = cms.counter
    
    cms.reset()
    counted_ngrams = count_ngrams(the_pile_sample, 2)
    cms.update(counted_ngrams)
    data_vector = cms.counter

    cms.reset()
    counted_ngrams = count_ngrams(sampled_data, 2)
    cms.update(counted_ngrams)
    sampled_vector = cms.counter

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
python run_mlm.py \
    --sampled_data_path /Users/muhammed/Desktop/Projects/CS543/CS543-final-project/data/the_pile_sample_2/
    --seed 42 \
"""