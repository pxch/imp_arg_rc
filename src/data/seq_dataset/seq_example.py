from collections import Counter
from itertools import combinations
from typing import List

import numpy as np
from torchtext.data import Example


class SeqExample(Example):
    def __init__(self, doc_input: List[int], query_input: List[int],
                 doc_entity_ids: List[int], target_entity_id: int, **kwargs):
        # doc_entity_ids must have the same length as doc_input
        assert len(doc_entity_ids) == len(doc_input)

        self.doc_input = doc_input
        self.query_input = query_input
        self.doc_entity_ids = doc_entity_ids
        self.target_entity_id = target_entity_id

        # additional information
        if 'num_mentions_total' in kwargs:
            assert len(kwargs['num_mentions_total']) == len(doc_input)
        if 'num_mentions_named' in kwargs:
            assert len(kwargs['num_mentions_named']) == len(doc_input)
        if 'num_mentions_nominal' in kwargs:
            assert len(kwargs['num_mentions_nominal']) == len(doc_input)
        if 'num_mentions_pronominal' in kwargs:
            assert len(kwargs['num_mentions_pronominal']) == len(doc_input)
        if 'argument_mask' in kwargs:
            assert len(kwargs['argument_mask']) == len(doc_input)
        if 'neg_query_input' in kwargs:
            assert len(kwargs['neg_query_input']) == len(query_input)

        for key, val in kwargs.items():
            setattr(self, key, val)

    def __str__(self):
        return str(self.__dict__)

    @classmethod
    def from_text(cls, text):
        return cls(**eval(text))

    @staticmethod
    def get_coref_pred_pairs(doc_event_list, doc_entity_ids):
        pred_idx_list = [
            token_idx for token_idx, entity_id in enumerate(doc_entity_ids)
            if entity_id == -1]

        kwargs = {'coref_pred_1': [], 'coref_pred_2': []}

        for i in range(len(doc_event_list)):
            for j in range(i+1, len(doc_event_list)):
                if doc_event_list[i].has_shared_arg(doc_event_list[j]):
                    kwargs['coref_pred_1'].append(pred_idx_list[i])
                    kwargs['coref_pred_2'].append(pred_idx_list[j])

        # will cause error when the two fields of all examples
        # within a batch are empty
        if len(kwargs['coref_pred_1']) == 0:
            kwargs['coref_pred_1'] = [-1]
            kwargs['coref_pred_2'] = [-1]

        return kwargs

    @staticmethod
    def get_entity_salience(doc_event_list, doc_entity_ids):
        entity_salience_mapping = {}
        for entity_id in doc_entity_ids:
            if entity_id not in entity_salience_mapping:
                entity_salience_mapping[entity_id] = [0, 0, 0, 0]
        for seq_event in doc_event_list:
            for seq_arg in seq_event.seq_arg_list:
                entity_id = seq_arg.entity_id
                mention_type = seq_arg.mention_type
                if mention_type != 0:
                    entity_salience_mapping[entity_id][mention_type] += 1
                entity_salience_mapping[entity_id][0] += 1

        kwargs = {}

        kwargs['num_mentions_total'] = \
            [entity_salience_mapping[entity_id][0]
             for entity_id in doc_entity_ids]
        kwargs['num_mentions_named'] = \
            [entity_salience_mapping[entity_id][1]
             for entity_id in doc_entity_ids]
        kwargs['num_mentions_nominal'] = \
            [entity_salience_mapping[entity_id][2]
             for entity_id in doc_entity_ids]
        kwargs['num_mentions_pronominal'] = \
            [entity_salience_mapping[entity_id][3]
             for entity_id in doc_entity_ids]

        return kwargs

    @staticmethod
    def get_query_input(query_event, arg_idx, arg_entity_mapping,
                        query_type='normal'):
        assert query_type in ['normal', 'single_arg', 'multi_hop']

        query_input = np.array(query_event.token_id_list())

        # target component_id == 1 --> TARGET-SUBJ (1)
        # target component_id == 2 --> TARGET-OBJ (2)
        # target component_id == 3 --> TARGET-PREP (3)
        query_input[arg_idx + 1] = \
            query_event.seq_arg_list[arg_idx].component_id

        if query_type == 'single_arg':
            query_input = np.delete(
                query_input,
                [i + 1 for i in arg_entity_mapping.keys() if i != arg_idx])

        if query_type == 'multi_hop':
            for miss_arg_idx in arg_entity_mapping.keys():
                if miss_arg_idx != arg_idx:
                    # missing component_id == 1 --> MISS-SUBJ (4)
                    # missing component_id == 2 --> MISS-OBJ (5)
                    # missing component_id == 3 --> MISS-PREP (6)
                    query_input[miss_arg_idx + 1] = \
                        query_event.seq_arg_list[miss_arg_idx].component_id + 3

        return list(query_input)

    @staticmethod
    def get_query_input_pair(query_event, arg_idx_1, arg_idx_2):
        query_input_1 = np.array(query_event.token_id_list())
        query_input_1[arg_idx_1 + 1] = \
            query_event.seq_arg_list[arg_idx_1].component_id
        query_input_1 = np.delete(query_input_1, arg_idx_2 + 1)

        query_input_2 = np.array(query_event.token_id_list())
        query_input_2[arg_idx_2 + 1] = \
            query_event.seq_arg_list[arg_idx_2].component_id
        query_input_2 = np.delete(query_input_2, arg_idx_1 + 1)

        return list(query_input_1), list(query_input_2)

    @classmethod
    def build(cls, doc_event_list, query_event, filter_single_candidate=True,
              filter_single_argument=False, query_type='normal',
              include_salience=False, include_coref_pred_pairs=False):
        assert query_type in \
               ['normal', 'single_arg', 'multi_hop', 'multi_arg', 'multi_slot']

        examples = []

        doc_input = [
            token_id for seq_event in doc_event_list
            for token_id in seq_event.token_id_list()]

        doc_entity_ids = [
            entity_id for seq_event in doc_event_list
            for entity_id in seq_event.entity_id_list()]

        # do not generate examples with a single candidate, when doc_entity_ids
        # contain only 2 distinct numbers (with a -1 for predicate).
        if filter_single_candidate and len(Counter(doc_entity_ids)) <= 2:
            return examples

        arg_entity_mapping = {
            arg_idx: seq_arg.entity_id
            for arg_idx, seq_arg in enumerate(query_event.seq_arg_list)
            if seq_arg.entity_id in doc_entity_ids}

        if len(arg_entity_mapping) == 0:
            return examples

        if len(arg_entity_mapping) <= 1:
            if (filter_single_argument or
                    query_type in ['multi_arg', 'multi_slot']):
                return examples

        kwargs = {}

        if include_salience:
            kwargs.update(
                cls.get_entity_salience(doc_event_list, doc_entity_ids))

        if include_coref_pred_pairs:
            kwargs.update(
                cls.get_coref_pred_pairs(doc_event_list, doc_entity_ids))

        if query_type == 'multi_hop':
            kwargs['argument_mask'] = [
                1 if entity_id in arg_entity_mapping.values() else 0
                for entity_id in doc_entity_ids]

        if query_type in ['normal', 'single_arg', 'multi_hop']:
            for arg_idx, target_entity_id in arg_entity_mapping.items():
                query_input = cls.get_query_input(
                    query_event, arg_idx, arg_entity_mapping,
                    query_type=query_type)

                examples.append(cls(
                    doc_input=doc_input,
                    query_input=query_input,
                    doc_entity_ids=doc_entity_ids,
                    target_entity_id=target_entity_id,
                    **kwargs
                ))

        else:
            for arg_idx_1, arg_idx_2 in combinations(
                    arg_entity_mapping.keys(), r=2):
                target_entity_id_1 = arg_entity_mapping[arg_idx_1]
                target_entity_id_2 = arg_entity_mapping[arg_idx_2]

                if target_entity_id_1 == target_entity_id_2:
                    continue

                query_input_1, query_input_2 = cls.get_query_input_pair(
                    query_event, arg_idx_1, arg_idx_2)

                if query_type == 'multi_arg':
                    examples.append(cls(
                        doc_input=doc_input,
                        query_input=query_input_1,
                        doc_entity_ids=doc_entity_ids,
                        target_entity_id=target_entity_id_1,
                        neg_target_entity_id=target_entity_id_2,
                        **kwargs
                    ))
                    examples.append(cls(
                        doc_input=doc_input,
                        query_input=query_input_2,
                        doc_entity_ids=doc_entity_ids,
                        target_entity_id=target_entity_id_2,
                        neg_target_entity_id=target_entity_id_1,
                        **kwargs
                    ))
                else:
                    examples.append(cls(
                        doc_input=doc_input,
                        query_input=query_input_1,
                        doc_entity_ids=doc_entity_ids,
                        target_entity_id=target_entity_id_1,
                        neg_query_input=query_input_2,
                        **kwargs
                    ))
                    examples.append(cls(
                        doc_input=doc_input,
                        query_input=query_input_2,
                        doc_entity_ids=doc_entity_ids,
                        target_entity_id=target_entity_id_2,
                        neg_query_input=query_input_1,
                        **kwargs
                    ))

        return examples
