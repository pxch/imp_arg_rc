import re

from common import document
from utils import escape, unescape


class Token(object):
    def __init__(self, word, lemma, pos, sentnum=-1, wordnum=-1):
        self._word = word
        self._lemma = lemma
        self._pos = pos

        self.sentnum = sentnum
        self.wordnum = wordnum

    @property
    def word(self):
        return self._word

    @property
    def lemma(self):
        return self._lemma

    @property
    def pos(self):
        return self._pos

    def __eq__(self, other):
        if not isinstance(other, Token):
            return False
        else:
            return self.word == other.word and self.lemma == other.lemma \
                   and self.pos == other.pos

    def __ne__(self, other):
        return not self.__eq__(other)

    def get_representation(self, use_lemma=True):
        # NOBUG: do not return empty string when token is not noun or verb
        # if self.pos[:2] not in ['NN', 'VB']: return ''
        if use_lemma:
            return self.lemma.lower()
        else:
            return self.word.lower()

    def to_text(self):
        return '{}/{}/{}'.format(
            escape(self.word), escape(self.lemma), escape(self.pos))

    token_re = re.compile(
        r'^(?P<word>[^/]*)/(?P<lemma>[^/]*)/(?P<pos>[^/]*)$')

    @classmethod
    def from_text(cls, text):
        match = cls.token_re.match(text)
        assert match, 'cannot parse Token from {}'.format(text)
        groups = match.groupdict()

        word = unescape(groups['word'])
        lemma = unescape(groups['lemma'])
        pos = unescape(groups['pos'])

        return cls(word, lemma, pos)

    @classmethod
    def from_token(cls, token: document.Token):
        word = token.word
        lemma = token.lemma
        pos = token.pos

        sentnum = token.sent_idx
        wordnum = token.token_idx

        return cls(word, lemma, pos, sentnum, wordnum)
