import re

from common import document
from utils import escape, unescape
from .token import Token


class Predicate(Token):
    def __init__(self, word, lemma, pos, sentnum=-1, wordnum=-1,
                 neg=False, prt=''):
        super(Predicate, self).__init__(word, lemma, pos, sentnum, wordnum)

        # whether the predicate is negated, default is False
        assert type(neg) == bool, 'neg must be a boolean value'
        self._neg = neg

        # particle attached to the predicate, default is empty string
        self._prt = prt

    @property
    def neg(self):
        return self._neg

    @property
    def prt(self):
        return self._prt

    def __eq__(self, other):
        if not isinstance(other, Predicate):
            return False
        else:
            return self.word == other.word and self.lemma == other.lemma \
                   and self.pos == other.pos and self.neg == other.neg \
                   and self.prt == other.prt

    def __ne__(self, other):
        return not self.__eq__(other)

    def get_full_representation(self, use_lemma=True):
        result = super(Predicate, self).get_representation(use_lemma=use_lemma)
        if self.prt:
            result += '_' + self.prt
        if self.neg:
            result = 'not_' + result
        return result

    def to_text(self):
        text = super(Predicate, self).to_text()
        if self.neg:
            text = 'not//' + text
        if self.prt != '':
            text += '//' + escape(self.prt)
        return '{}-{}-{}'.format(self.sentnum, self.wordnum, text)

    pred_re = re.compile(
        r'^((?P<neg>not)(?://))?(?P<word>[^/]*)/(?P<lemma>[^/]*)/(?P<pos>[^/]*)'
        r'((?://)(?P<prt>[^/]*))?$')

    @classmethod
    def from_text(cls, text):
        parts = text.split('-')
        if len(parts) == 3:
            sentnum = int(parts[0])
            wordnum = int(parts[1])
            content_text = parts[2]
        else:
            assert len(parts) == 1
            sentnum = -1
            wordnum = -1
            content_text = parts[0]

        match = cls.pred_re.match(content_text)
        assert match, 'cannot parse Predicate from {}'.format(text)
        groups = match.groupdict()

        word = unescape(groups['word'])
        lemma = unescape(groups['lemma'])
        pos = unescape(groups['pos'])
        neg = True if groups['neg'] is not None else False
        prt = unescape(groups['prt']) if groups['prt'] is not None else ''

        return cls(word, lemma, pos, neg=neg, prt=prt,
                   sentnum=sentnum, wordnum=wordnum)

    @classmethod
    def from_token(cls, token: document.Token, **kwargs):
        word = token.word
        lemma = token.lemma
        pos = token.pos
        assert pos.startswith('VB'), \
            'Predicate cannot be created from a {} token'.format(token.pos)

        assert 'neg' in kwargs, 'neg must be provided when creating Predicate'
        neg = kwargs['neg']

        assert 'prt' in kwargs, 'prt must be provided when creating Predicate'
        prt = kwargs['prt']

        sentnum = token.sent_idx
        wordnum = token.token_idx

        return cls(word, lemma, pos, sentnum, wordnum, neg, prt)
