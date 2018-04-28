from collections import Counter

import torch
from gensim.models import KeyedVectors
from torchtext.vocab import Vocab

from utils import log

default_specials = ['<pad>', 'MISS-SUBJ', 'MISS-OBJ', 'MISS-PREP']


def load_vocab(fname, fvocab=None, binary=True, normalize=True,
               specials=default_specials):
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

    log.info('Building Vocab with {} words and specials = {}'.format(
        len(counter), specials))
    vocab = Vocab(counter, specials=specials)
    log.info('Setting vectors with word2vec vectors')
    vocab.set_vectors(stoi, vectors, dim)

    return vocab
