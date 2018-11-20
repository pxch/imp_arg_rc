import abc
import math
import random
from typing import List

from torchtext.vocab import Vocab

from common.event_script import Argument, Event, Predicate, Script
from utils import consts
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
    def __init__(self, token_id: int, component_id: int, wordnum: int):
        self.token_id = token_id
        self.component_id = component_id
        self.wordnum = wordnum

    @abc.abstractmethod
    def to_tuple(self):
        pass

    def __eq__(self, other):
        return self.to_tuple() == other.to_tuple()

    def __ne__(self, other):
        return not self.__eq__(other)


class SeqPredicate(SeqEventComponent):
    def __init__(self, token_id: int, wordnum: int, drop: bool = False):
        super().__init__(token_id, 0, wordnum=wordnum)
        self.drop = drop

    @classmethod
    def build(cls, vocab: Vocab, predicate: Predicate, use_lemma=True,
              include_type=True, use_unk=True, pred_vocab_count=None):
        word = predicate.get_representation(use_lemma=use_lemma)
        candidates = [word]
        if predicate.prt:
            candidates.append(word + '_' + predicate.prt)
        if predicate.neg:
            candidates.extend(['not_' + cand for cand in candidates])
        candidates.reverse()
        if use_unk:
            candidates.append('UNK')

        # random down-sample predicates with count over 100,000, and mark it
        # with drop = True
        drop = False
        if candidates and pred_vocab_count:
            pred_count = pred_vocab_count.get(candidates[0], 0)
            if pred_count > consts.pred_count_thres:
                if random.random() < 1.0 - math.sqrt(
                        float(consts.pred_count_thres) / pred_count):
                    drop = True

        if include_type:
            candidates = [cand + '-PRED' for cand in candidates]

        token_id = _get_token_id(candidates, vocab, ensure_found=use_unk)

        return cls(token_id=token_id, wordnum=predicate.wordnum, drop=drop)

    def to_tuple(self):
        return self.token_id, self.component_id, self.wordnum

    @classmethod
    def from_tuple(cls, tup):
        assert len(tup) == 3 and tup[1] == 0
        return cls(token_id=tup[0], wordnum=tup[2], drop=False)

    def to_text(self):
        return str(self.to_tuple())

    def __str__(self):
        return self.to_text()

    @classmethod
    def from_text(cls, text):
        return cls.from_tuple(eval(text))


class SeqArgument(SeqEventComponent):
    def __init__(self, token_id: int, component_id: int, wordnum: int,
                 entity_id: int, mention_id: int, mention_type: int,
                 additional_entity_id: int, additional_mention_id: int):
        super().__init__(token_id, component_id, wordnum)
        self.entity_id = entity_id
        self.mention_id = mention_id
        # mention_type must be one of 0 (other), 1 (named), 2 (nominal),
        # or 3 (pronominal), depending on the POS / NER of the token.
        assert mention_type in [0, 1, 2, 3]
        self.mention_type = mention_type

        self.additional_entity_id = additional_entity_id
        self.additional_mention_id = additional_mention_id

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

        # named mention
        if argument.ner is not '':
            mention_type = 1
        # nominal mention
        elif argument.pos.startswith('NN'):
            mention_type = 2
        # pronominal mention
        elif argument.pos.startswith('PRP'):
            mention_type = 3
        # other mention
        else:
            mention_type = 0

        return cls(token_id=token_id,
                   component_id=component_id,
                   wordnum=argument.wordnum,
                   entity_id=argument.entity_idx,
                   mention_id=argument.mention_idx,
                   mention_type=mention_type,
                   additional_entity_id=argument.additional_entity_idx,
                   additional_mention_id=argument.additional_mention_idx)

    def to_tuple(self):
        return self.token_id, self.component_id, self.wordnum, \
               self.entity_id, self.mention_id, self.mention_type

    @classmethod
    def from_tuple(cls, tup):
        assert len(tup) == 6 and tup[1] in [1, 2, 3] and tup[5] in [0, 1, 2, 3]
        return cls(*tup)

    def to_text(self):
        return str(self.to_tuple())

    def __str__(self):
        return self.to_text()

    @classmethod
    def from_text(cls, text):
        return cls.from_tuple(eval(text))


class SeqEvent(object):
    def __init__(self, seq_pred: SeqPredicate, seq_arg_list: List[SeqArgument],
                 sentnum: int):
        self.seq_pred = seq_pred
        self.seq_arg_list = seq_arg_list
        self.sentnum = sentnum

    def __eq__(self, other):
        return self.to_list() == other.to_list()

    def __ne__(self, other):
        return not self.__eq__(other)

    @classmethod
    def build(cls, vocab: Vocab, event: Event, use_lemma=True, use_unk=True,
              pred_vocab_count=None, prep_vocab_list=None,
              filter_repetitive_prep=False):
        seq_pred = SeqPredicate.build(
            vocab=vocab,
            predicate=event.pred,
            use_lemma=use_lemma,
            include_type=True,
            use_unk=use_unk,
            pred_vocab_count=pred_vocab_count
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
        prep_list = []
        for prep, pobj in event.pobj_list:
            if prep_vocab_list and prep not in prep_vocab_list:
                prep = ''

            if filter_repetitive_prep:
                if prep in prep_list:
                    continue
                else:
                    prep_list.append(prep)

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

        sentnum = event.pred.sentnum
        return cls(seq_pred, seq_arg_list, sentnum)

    def to_list(self):
        # noinspection PyTypeChecker
        return [self.sentnum, self.seq_pred.to_tuple()] + \
               [seq_arg.to_tuple() for seq_arg in self.seq_arg_list]

    @classmethod
    def from_list(cls, lst):
        sentnum = lst[0]
        seq_pred = SeqPredicate.from_tuple(lst[1])
        seq_arg_list = \
            [SeqArgument.from_tuple(t) for t in lst[2:]]
        return cls(seq_pred, seq_arg_list, sentnum)

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

    def entity_id_list(self, include_pred=True):
        result = [seq_arg.entity_id for seq_arg in self.seq_arg_list]
        if include_pred:
            result = [-1] + result
        return result

    def additional_entity_id_list(self, include_pred=True):
        result = [seq_arg.additional_entity_id for seq_arg in self.seq_arg_list]
        if include_pred:
            result = [-1] + result
        return result

    def has_shared_arg(self, other):
        return not set(self.entity_id_list(include_pred=False)).isdisjoint(
            set(other.entity_id_list(include_pred=False)))


class SeqScript(object):
    def __init__(self, seq_event_list: List[SeqEvent]):
        self.seq_event_list = seq_event_list
        self.entity_id_mapping = {}
        self.entity_id_mapping_rev = {}
        self.singleton_processed = False

        self.additional_entity_id_mapping = {}
        self.additional_entity_id_mapping_rev = {}
        self.singleton_processed_additional = False

    def __eq__(self, other):
        return self.to_list() == other.to_list()

    def __ne__(self, other):
        return not self.__eq__(other)

    def process_singletons_additional(self):
        if not self.singleton_processed_additional:
            # mapping from original entity_id to new entity_id for entities
            entity_id_mapping = {}
            # number of singletons already processed
            singleton_count = 0
            # number of entities already processed
            entity_count = 0
            for seq_event in self.seq_event_list:
                for seq_arg in seq_event.seq_arg_list:
                    # if the argument is an entity
                    if seq_arg.additional_entity_id != -1:
                        entity_id = seq_arg.additional_entity_id
                        # if the entity_id has not been processed before
                        if entity_id not in entity_id_mapping:
                            # assign the new entity_id
                            entity_id_mapping[entity_id] = \
                                entity_count + singleton_count
                            # increase entity_count
                            entity_count += 1
                        seq_arg.additional_entity_id = entity_id_mapping[entity_id]
                    # if the argument is a singleton
                    else:
                        # assign the pseudo entity_id
                        seq_arg.additional_entity_id = \
                            len(entity_id_mapping) + singleton_count
                        # assign the pseudo mention_id (0)
                        seq_arg.additional_mention_id = 0
                        # increase singleton_count
                        singleton_count += 1

            self.singleton_processed_additional = True
            self.additional_entity_id_mapping = entity_id_mapping
            # store the reverse mapping for convenient restoration.
            self.additional_entity_id_mapping_rev = \
                {val: key for key, val in entity_id_mapping.items()}

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
    def build(cls, vocab: Vocab, script: Script, use_lemma=True, use_unk=True,
              pred_vocab_count=None, prep_vocab_list=None,
              filter_repetitive_prep=False):
        seq_event_list = []
        for event in script.events:
            seq_event = SeqEvent.build(
                vocab=vocab,
                event=event,
                use_lemma=use_lemma,
                use_unk=use_unk,
                pred_vocab_count=pred_vocab_count,
                prep_vocab_list=prep_vocab_list,
                filter_repetitive_prep=filter_repetitive_prep
            )
            seq_event_list.append(seq_event)
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

    def get_all_examples(
            self, stop_pred_ids=None, filter_single_candidate=True,
            filter_single_argument=False, query_type='normal',
            include_salience=False, include_coref_pred_pairs=False,
            use_additional_coref=True):
        all_examples = []

        # start the query from the second not-down-sampled event
        query_idx_list = [
            idx for idx, seq_event in enumerate(self.seq_event_list)
            if not seq_event.seq_pred.drop]
        query_idx_list = query_idx_list[1:]

        # or start the query from the second event
        # query_idx_list = [
        #     idx for idx, seq_event in enumerate(self.seq_event_list)
        #     if not seq_event.seq_pred.drop and idx > 0]

        assert query_type in [
            'normal', 'single_arg', 'multi_hop', 'multi_arg', 'multi_slot'], \
            'Unrecognized example_type: {}'.format(query_type)

        for query_idx in query_idx_list:
            doc_event_list = self.seq_event_list[:query_idx]
            query_event = self.seq_event_list[query_idx]

            # drop down-sampled predicates
            if query_event.seq_pred.drop:
                continue

            # filter stop predicates
            if stop_pred_ids and query_event.seq_pred.token_id in stop_pred_ids:
                continue

            all_examples.extend(SeqExample.build(
                doc_event_list=doc_event_list,
                query_event=query_event,
                filter_single_candidate=filter_single_candidate,
                filter_single_argument=filter_single_argument,
                query_type=query_type,
                include_salience=include_salience,
                include_coref_pred_pairs=include_coref_pred_pairs,
                use_additional_coref=use_additional_coref))

        return all_examples
