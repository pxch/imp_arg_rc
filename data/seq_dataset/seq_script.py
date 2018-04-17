import abc
from typing import List

from torchtext.vocab import Vocab

from common.event_script import Argument, Event, Predicate, Script
from .seq_example import SeqExample


def _get_token_id(candidates, vocab: Vocab, ensure_found=True):
    unk_id = vocab.stoi.default_factory()
    for text in candidates:
        token_id = vocab.stoi[text]
        if token_id != unk_id:
            return token_id
    if ensure_found:
        raise IndexError('Out of vocabulary: {}'.format(candidates))
    else:
        return unk_id


class SeqEventComponent(object):
    def __init__(self, token_id: int, component_id: int):
        self.token_id = token_id
        self.component_id = component_id

    @abc.abstractmethod
    def to_tuple(self):
        pass

    def __eq__(self, other):
        return self.to_tuple() == other.to_tuple()

    def __ne__(self, other):
        return not self.__eq__(other)


class SeqPredicate(SeqEventComponent):
    def __init__(self, token_id: int):
        super().__init__(token_id, component_id=0)

    @classmethod
    def build(cls, vocab: Vocab, predicate: Predicate, use_lemma=True,
              include_type=True, use_unk=True):
        word = predicate.get_representation(use_lemma=use_lemma)
        candidates = [word]
        if predicate.prt:
            candidates.append(word + '_' + predicate.prt)
        if predicate.neg:
            candidates.extend(['not_' + cand for cand in candidates])
        candidates.reverse()
        if use_unk:
            candidates.append('UNK')

        if include_type:
            candidates = [cand + '-PRED' for cand in candidates]

        token_id = _get_token_id(candidates, vocab, ensure_found=use_unk)

        return cls(token_id)

    def to_tuple(self):
        return self.token_id, self.component_id

    @classmethod
    def from_tuple(cls, tup):
        assert len(tup) == 2 and tup[1] == 0
        return cls(tup[0])

    def to_text(self):
        return str(self.to_tuple())

    def __str__(self):
        return self.to_text()

    @classmethod
    def from_text(cls, text):
        return cls.from_tuple(eval(text))


class SeqArgument(SeqEventComponent):
    def __init__(self, token_id: int, component_id: int, entity_id: int,
                 mention_id: int):
        super().__init__(token_id, component_id)
        self.entity_id = entity_id
        self.mention_id = mention_id

    @staticmethod
    def get_candidates(text, arg_type):
        candidates = []
        if arg_type != '':
            candidates.append(text + '-' + arg_type)
            # back off to remove preposition
            if arg_type.startswith('PREP'):
                candidates.append(text + '-PREP')
        else:
            candidates.append(text)
        return candidates

    @classmethod
    def build(cls, vocab: Vocab, argument: Argument, use_lemma=True,
              arg_type='', use_unk=True):
        word = argument.get_representation(use_lemma=use_lemma)
        candidates = SeqArgument.get_candidates(word, arg_type)
        if argument.ner != '':
            candidates.extend(
                SeqArgument.get_candidates(argument.ner, arg_type))
        if use_unk:
            candidates.extend(
                SeqArgument.get_candidates('UNK', arg_type))

        token_id = _get_token_id(candidates, vocab, ensure_found=use_unk)

        if arg_type == 'SUBJ':
            component_id = 1
        elif arg_type == 'OBJ':
            component_id = 2
        elif arg_type.startswith('PREP'):
            component_id = 3
        else:
            raise NotImplementedError

        return cls(token_id, component_id, argument.entity_idx,
                   argument.mention_idx)

    def to_tuple(self):
        return self.token_id, self.component_id, self.entity_id, \
               self.mention_id

    @classmethod
    def from_tuple(cls, tup):
        assert len(tup) == 4 and tup[1] in [1, 2, 3]
        return cls(*tup)

    def to_text(self):
        return str(self.to_tuple())

    def __str__(self):
        return self.to_text()

    @classmethod
    def from_text(cls, text):
        return cls.from_tuple(eval(text))


class SeqEvent(object):
    def __init__(self, seq_pred: SeqPredicate, seq_arg_list: List[SeqArgument]):
        self.seq_pred = seq_pred
        self.seq_arg_list = seq_arg_list

    def __eq__(self, other):
        return self.to_list() == other.to_list()

    def __ne__(self, other):
        return not self.__eq__(other)

    @classmethod
    def build(cls, vocab: Vocab, event: Event, use_lemma=True, use_unk=True):
        seq_pred = SeqPredicate.build(
            vocab=vocab,
            predicate=event.pred,
            use_lemma=use_lemma,
            include_type=True,
            use_unk=use_unk
        )

        seq_arg_list = []
        if event.subj is not None:
            seq_arg_list.append(SeqArgument.build(
                vocab=vocab,
                argument=event.subj,
                use_lemma=use_lemma,
                arg_type='SUBJ',
                use_unk=use_unk
            ))
        if event.dobj is not None:
            seq_arg_list.append(SeqArgument.build(
                vocab=vocab,
                argument=event.dobj,
                use_lemma=use_lemma,
                arg_type='OBJ',
                use_unk=use_unk
            ))
        for prep, pobj in event.pobj_list:
            if prep != '':
                arg_type = 'PREP_' + prep
            else:
                arg_type = 'PREP'
            seq_arg_list.append(SeqArgument.build(
                vocab=vocab,
                argument=pobj,
                use_lemma=use_lemma,
                arg_type=arg_type,
                use_unk=use_unk
            ))

        return cls(seq_pred, seq_arg_list)

    def to_list(self):
        return [self.seq_pred.to_tuple()] + \
               [seq_arg.to_tuple() for seq_arg in self.seq_arg_list]

    @classmethod
    def from_list(cls, lst):
        seq_pred = SeqPredicate.from_tuple(lst[0])
        seq_arg_list = \
            [SeqArgument.from_tuple(t) for t in lst[1:]]
        return cls(seq_pred, seq_arg_list)

    def to_text(self):
        return str(self.to_list())

    def __str__(self):
        return self.to_text()

    @classmethod
    def from_text(cls, text):
        return cls.from_list(eval(text))

    def token_id_list(self):
        return [self.seq_pred.token_id] + \
               [seq_arg.token_id for seq_arg in self.seq_arg_list]

    def entity_id_list(self):
        return [-1] + [seq_arg.entity_id for seq_arg in self.seq_arg_list]


class SeqScript(object):
    def __init__(self, seq_event_list: List[SeqEvent]):
        self.seq_event_list = seq_event_list
        self.entity_id_mapping = {}
        self.entity_id_mapping_rev = {}
        self.singleton_processed = False

    def __eq__(self, other):
        return self.to_list() == other.to_list()

    def __ne__(self, other):
        return not self.__eq__(other)

    def process_singletons(self):
        if not self.singleton_processed:
            # mapping from original entity_id to new entity_id for entities
            entity_id_mapping = {}
            # number of singletons already processed
            singleton_count = 0
            # number of entities already processed
            entity_count = 0
            for seq_event in self.seq_event_list:
                for seq_arg in seq_event.seq_arg_list:
                    # if the argument is an entity
                    if seq_arg.entity_id != -1:
                        entity_id = seq_arg.entity_id
                        # if the entity_id has not been processed before
                        if entity_id not in entity_id_mapping:
                            # assign the new entity_id
                            entity_id_mapping[entity_id] = \
                                entity_count + singleton_count
                            # increase entity_count
                            entity_count += 1
                        seq_arg.entity_id = entity_id_mapping[entity_id]
                    # if the argument is a singleton
                    else:
                        # assign the pseudo entity_id
                        seq_arg.entity_id = \
                            len(entity_id_mapping) + singleton_count
                        # assign the pseudo mention_id (0)
                        seq_arg.mention_id = 0
                        # increase singleton_count
                        singleton_count += 1

            self.singleton_processed = True
            self.entity_id_mapping = entity_id_mapping
            # store the reverse mapping for convenient restoration.
            self.entity_id_mapping_rev = \
                {val: key for key, val in entity_id_mapping.items()}

    def restore_singletons(self):
        if self.singleton_processed:
            for seq_event in self.seq_event_list:
                for seq_arg in seq_event.seq_arg_list:
                    if seq_arg.entity_id in self.entity_id_mapping_rev:
                        seq_arg.entity_id = \
                            self.entity_id_mapping_rev[seq_arg.entity_id]
                    else:
                        seq_arg.entity_id = -1
                        seq_arg.mention_id = -1
            self.singleton_processed = False

    @classmethod
    def build(cls, vocab: Vocab, script: Script, use_lemma=True, use_unk=True):
        seq_event_list = []
        for event in script.events:
            seq_event_list.append(SeqEvent.build(
                vocab=vocab,
                event=event,
                use_lemma=use_lemma,
                use_unk=use_unk
            ))
        return cls(seq_event_list)

    def to_list(self):
        return [seq_event.to_list() for seq_event in self.seq_event_list]

    @classmethod
    def from_list(cls, lst):
        return cls([SeqEvent.from_list(l) for l in lst])

    def to_text(self):
        return str(self.to_list())

    def __str__(self):
        return self.to_text()

    @classmethod
    def from_text(cls, text):
        return cls.from_list(eval(text))

    def get_all_examples(self):
        all_examples = []
        for query_idx in range(1, len(self.seq_event_list)):
            all_examples.extend(SeqExample.build(
                doc_event_list=self.seq_event_list[:query_idx],
                query_event=self.seq_event_list[query_idx]))
        return all_examples
