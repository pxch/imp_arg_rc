from collections import Counter
from copy import deepcopy

import torch
from gensim.models import KeyedVectors
from torchtext.vocab import Vocab

from utils import log

default_specials = ['<pad>']
miss_specials = ['MISS-SUBJ', 'MISS-OBJ', 'MISS-PREP']
target_specials = ['TARGET-SUBJ', 'TARGET-OBJ', 'TARGET_PREP']


def load_vocab(fname, fvocab=None, binary=True, normalize=True,
               use_target_specials=True, use_miss_specials=True):
    word2vec = KeyedVectors.load_word2vec_format(
        fname=fname, fvocab=fvocab, binary=binary)
    if normalize:
        word2vec.init_sims(replace=True)

    counter = Counter(
        {word: vocab.count for word, vocab in word2vec.vocab.items()})

    stoi = {word: vocab.index for word, vocab in word2vec.vocab.items()}
    vectors = [
        torch.from_numpy(word2vec.vectors[idx])
        for idx in range(len(word2vec.vocab))]
    dim = word2vec.vector_size

    specials = deepcopy(default_specials)
    if use_target_specials:
        specials.extend(target_specials)
    if use_miss_specials:
        specials.extend(miss_specials)

    log.info('Building Vocab with {} words and specials = {}'.format(
        len(counter), specials))
    vocab = Vocab(counter, specials=specials)
    log.info('Setting vectors with word2vec vectors')
    vocab.set_vectors(stoi, vectors, dim)

    return vocab
