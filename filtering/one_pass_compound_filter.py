import numpy as np
import spacy
from scipy.spatial.distance import jensenshannon
from collections import Counter

class CompoundCounter():
    def __init__(self, n_counters=10000):
        self.compounds = Counter()
        self.n_counters = n_counters

    #add reset method
    def _reset(self):
        self.compounds = Counter()
    
    @staticmethod
    def traverse_tree(node, branch=""):
        if not branch:
            branch = node.pos_
        else:
            branch += f"_{node.dep_}_{node.pos_}"
            
        if not list(node.children):
            return [branch]
        
        branches = []
        for child in node.children:
            try:
                branches.extend(CompoundCounter.traverse_tree(child, branch))
            except RecursionError:
                print("Recursion Error")
                return ["Empty"]
        return branches

    def count_compounds(self, doc):
        branches = []
        for token in doc:
            if token.dep_ == "ROOT":
                branches.extend(CompoundCounter.traverse_tree(token))
        for branch in branches:
            self.compounds[branch] = self.compounds.get(branch,0) + 1
        if len(self.compounds) > self.n_counters:
            #remove the least frequent compounds
            self.compounds = Counter(dict(self.compounds.most_common(self.n_counters)))
        return True


class CompoundDistanceFilter():
    def __init__(self, CompoundCounter, Distance, cold_start = 1000, filter_size = 100000, n_counters = 10000,select_threshold = 1):
        self.counter = CompoundCounter(n_counters = n_counters)
        self.cold_start = cold_start
        self.cold = True
        self.pos_tagger = spacy.load("en_core_web_lg",disable=["ner"])
        self.indx = 0
        self.n_counters = n_counters
        self.sentences = []
        self.Distance = Distance
        self.filter_size = filter_size
        self.n_counters = n_counters
        self.threshold = select_threshold
        self.verbose = True

    def _reset(self):
        self.counter._reset()
        self.cold = True
        self.indx = 0

    def select_sentence(self, doc):
        #select a sentence if the TVD is decreased with a uniform distribution by adding this sentence into the corpus. Else reject the sentence.
        #return True or False
        base_tvd = self.Distance(self.counter.compounds, n_counters = self.n_counters)
        cash_counter = self.counter.compounds.copy()
        self.counter.count_compounds(doc)
        new_tvd = self.Distance(self.counter.compounds, n_counters = self.n_counters)
        if new_tvd < base_tvd:
            #select a uniform ranodom number between 0 and 1 and if it is smaller than 0.9 then add the sentence to the corpus
            if np.random.uniform() < 0.9:
                self.counter.compounds = cash_counter
                return True
            else:
                self.counter.compounds = cash_counter
                return False
        else:
            #select a uniform ranodom number between 0 and 1 and if it is smaller than 0.1 then add the sentence to the corpus
            if np.random.uniform() < 0.1:
                self.counter.compounds = cash_counter
                return True
            else:
                self.counter.compounds = cash_counter
                return False

    def filter(self, stream):
        for document in stream:
            if type(document) == dict:
                document = document["text"]
            sentences = document.split(".")
            #remove sentences with less than 5 words
            sentences = [sentence for sentence in sentences if len(sentence.split()) > 5]
            if len(sentences) > 100:
                continue
            self.indx +=1
            sentence_selected = []
            for sentence in sentences:
                doc = self.pos_tagger(sentence)
                if self.cold:
                    self.counter.count_compounds(doc)
                    if self.indx == self.cold_start:
                        self.cold = False
                else:
                    sentence_selected.append(self.select_sentence(doc))
            
            if len(sentence_selected)>0:
                if sum(sentence_selected) > self.threshold:
                    self.sentences.append(document)
                    for sentence in sentences:
                        self.counter.count_compounds(self.pos_tagger(sentence))
            if self.verbose:
                print(len(self.sentences))
            if len(self.sentences) > self.filter_size:
                return True


class CompoundRuleFilter():
    def __init__(self, CompoundCounter, cold_start = 1000, filter_size = 100000, n_counters = 10000,limit = 1024, limit_counter = 256):
        self.counter = CompoundCounter(n_counters = n_counters)
        self.cold_start = cold_start
        self.cold = True
        self.pos_tagger = spacy.load("en_core_web_lg",disable=["ner"])
        self.indx = 0
        self.n_counters = n_counters
        self.sentences = []
        self.filter_size = filter_size
        self.n_counters = n_counters
        self.verbose = True
        self.limit = limit
        self.limit_counter = limit_counter

    def _reset(self):
        self.counter._reset()
        self.cold = True
        self.indx = 0

    def select_sentence(self, doc):
        cash_counter = self.counter.compounds.copy()
        self.counter.count_compounds(doc)
        # if max count is higher than limit reject the sentence
        if max(self.counter.compounds.values()) > self.limit:
            if sum([1 for count in self.counter.compounds.values() if count >= self.limit]) >= self.limit_counter:
                if self.limit_counter > 32:
                    self.limit *= 2
                    self.limit_counter /= 2
            self.counter.compounds = cash_counter
            return False
        else:
            self.counter.compounds = cash_counter
            return True
            

    def filter(self, stream):
        for document in stream:
            if type(document) == dict:
                document = document["text"]
            sentences = document.split(".")
            #remove sentences with less than 5 words
            sentences = [sentence for sentence in sentences if len(sentence.split()) > 5]
            if len(sentences) > 100:
                continue
            self.indx +=1
            sentence_selected = []
            for sentence in sentences:
                doc = self.pos_tagger(sentence)
                if self.cold:
                    self.counter.count_compounds(doc)
                    if self.indx == self.cold_start:
                        self.cold = False
                else:
                    sentence_selected.append(self.select_sentence(doc))
            
            if len(sentence_selected)>0:
                if sum(sentence_selected) > 1:
                    self.sentences.append(document)
                    for sentence in sentences:
                        self.counter.count_compounds(self.pos_tagger(sentence))
            if self.verbose:
                print(len(self.sentences))
            if len(self.sentences) > self.filter_size:
                return True


def tdv_distance(counter,n_counters=10000):
    #return the total variation distance between the empirical distribution of compounds and the uniform distribution
    #return a float
    uniform = [1/n_counters]*n_counters
    norm_values = np.array(list(counter.values()))/sum(counter.values())
    difference = 0
    for i in range(len(norm_values)):
        if norm_values[i] > uniform[i]:
            difference += norm_values[i] - uniform[i]
    return difference

def wasserstein_distance(counter,n_counters=10000):
    #return the total variation distance between the empirical distribution of compounds and the uniform distribution
    #return a float
    uniform = [1/n_counters]*n_counters
    norm_values = np.array(list(counter.values()))/sum(counter.values())
    #if norm_values is shorther than n_counters append zeros
    if len(norm_values) < n_counters:
        norm_values = np.append(norm_values, np.zeros(n_counters - len(norm_values)))
    difference = 0
    for i in range(len(norm_values)):
        difference += abs(norm_values[i] - uniform[i])
    return difference

def jsd_distance(counter,n_counters=10000):
    #return the total variation distance between the empirical distribution of compounds and the uniform distribution
    #return a float
    uniform = [1/n_counters]*n_counters
    norm_values = np.array(list(counter.values()))/sum(counter.values())
    #if norm_values is shorther than n_counters append zeros
    if len(norm_values) < n_counters:
        norm_values = np.append(norm_values, np.zeros(n_counters - len(norm_values)))
    return jensenshannon(norm_values, uniform)


