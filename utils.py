import numpy as np
import xxhash
from nltk import ngrams


class CMS():
    def __init__(self, k, t, seed=42, independence_k=2):
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
        self.p = 2 ** 31 - 1
        # select (t, independence_k) random numbers from 0 to p-1. The first column is the constant term and should be non-zero.
        self.hash_weights = np.random.randint(0, self.p, size=(self.t, self.independence_k))
        self.hash_weights[:, 0] = self.hash_weights[:, 0] + 1

    def _init_counter(self):
        self.counter = np.zeros((self.t, self.k), dtype=np.int32)

    def _hash(self, x):
        all_hashes = []
        for i in range(self.t):
            # hash = sum([self.hash_weights[i, j] * x**j for j in range(self.independence_k)]) % self.p % self.k
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
        # return np.array([self._query(x)/self._get_sum() for x in X])
        return np.array([self._query(x) for x in X])

    def reset(self):
        self._init_counter()


def generate_ngrams(sentence, n):
    tokens = sentence.split()
    all_ngrams = []
    for i in range(1, n + 1):
        all_ngrams.extend(['-'.join(a) for a in list(ngrams(tokens, i))])
    return all_ngrams


# generate a dictionary of unigram and bigrams that occur in a dataset and their total counts
def count_ngrams(dataset, n):
    ngram_counter = {}
    for sent in dataset:
        if type(sent) == dict:
            sent = sent['text']
        generated_ngrams = generate_ngrams(sent, n)
        for ngram in generated_ngrams:
            if ngram in ngram_counter:
                ngram_counter[ngram] += 1
            else:
                ngram_counter[ngram] = 1
    return ngram_counter