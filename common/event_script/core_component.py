import abc

from utils import consts
from torchtext.vocab import Vocab


class CoreComponent(object):
    @abc.abstractmethod
    def get_index(self, vocab: Vocab, **kwargs):
        pass


class CorePredicate(CoreComponent):
    def __init__(self, word, neg=False, prt=''):
        self.word = word
        self.neg = neg
        self.prt = prt
        # list of candidates to lookup the Word2Vec vocabulary
        self.candidates = []

    def get_candidates(self):
        if not self.candidates:
            # add just the verb to the list
            self.candidates = [self.word]
            # add verb_prt to the list if the particle exists
            if self.prt:
                self.candidates.append(self.word + '_' + self.prt)
            # add not_verb and not_verb_prt to the list if there exists
            # a negation relation to the predicate
            if self.neg:
                for idx in range(len(self.candidates)):
                    self.candidates.append('not_' + self.candidates[idx])
            # reverse the list, so now the order of candidates becomes:
            # 1) not_verb_prt (if both negation and particle exists)
            # 2) not_verb (if negation exists)
            # 3) verb_prt (if particle exists)
            # 4) verb
            self.candidates.reverse()
            # append the UNK token the list of candidates in case none of
            # the above can be found in the vocabulary
            # candidates.append('UNK')
        return self.candidates

    def get_index(self, vocab: Vocab, include_type=True, use_unk=True):
        # TODO: add logic to process stop predicates

        candidates = self.get_candidates()
        # add UNK to the candidates if use_unk is set to True
        if use_unk:
            candidates.append('UNK')

        # TODO: down sample most frequent predicates?

        if include_type:
            candidates = [candidate + '-PRED' for candidate in candidates]
        index = -1
        for text in candidates:
            index = vocab.stoi[text]
            if index != -1:
                break
        return index


class CoreArgument(CoreComponent):
    def __init__(self, word, pos, ner):
        self._word = word
        self._pos = pos
        assert ner in consts.valid_ner_tags or ner == '', \
            'unrecognized NER tag: ' + ner
        self._ner = ner

    @property
    def word(self):
        return self._word

    @property
    def pos(self):
        return self._pos

    @property
    def ner(self):
        return self._ner

    def __eq__(self, other):
        return self.word == other.word and self.pos == other.pos \
               and self.ner == other.ner

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return '{} // {} // {}'.format(self.word, self.pos, self.ner)

    @staticmethod
    def get_candidates_by_arg_type(text, arg_type):
        candidates = []
        if arg_type != '':
            candidates.append(text + '-' + arg_type)
            # back off to remove preposition
            if arg_type.startswith('PREP'):
                candidates.append(text + '-PREP')
        else:
            candidates.append(text)
        return candidates

    def get_index(self, vocab: Vocab, arg_type='', use_unk=True):
        # add candidates from self.word
        candidates = CoreArgument.get_candidates_by_arg_type(
            self.word, arg_type)
        # add candidates from self.ner if self.ner is not empty string
        if self.ner != '':
            candidates.extend(
                CoreArgument.get_candidates_by_arg_type(self.ner, arg_type))
        # add UNK to the candidates if use_unk is set to True
        if use_unk:
            candidates.extend(
                CoreArgument.get_candidates_by_arg_type('UNK', arg_type))
        index = -1
        # iterate through all candidates, and return the first one that exists
        # in the vocabulary of the Word2Vec model, otherwise return -1
        for text in candidates:
            index = vocab.stoi[text]
            if index != -1:
                break
        return index
