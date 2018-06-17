from collections import Counter
from itertools import combinations
from typing import List

import numpy as np
from torchtext.data import Example


class SeqExample(Example):
    def __init__(self, doc_input: List[int], query_input: List[int],
                 doc_entity_ids: List[int], target_entity_id: int):
        # doc_entity_ids must have the same length as doc_input
        assert len(doc_entity_ids) == len(doc_input)
        # # target_mask must have the same length as document_input
        # assert len(target_mask) == len(document_input)
        # # All 1s in target_mask must be a subset of the 1s in softmax_mask
        # assert target_mask == [
        #     ms * mt for ms, mt in zip(softmax_mask, target_mask)]

        self.doc_input = doc_input
        self.query_input = query_input
        self.doc_entity_ids = doc_entity_ids
        self.target_entity_id = target_entity_id

    def __str__(self):
        return '({}, {}, {}, {})'.format(
            self.doc_input, self.query_input, self.doc_entity_ids,
            self.target_entity_id)

    @classmethod
    def from_text(cls, text):
        return cls(*eval(text))

    @classmethod
    def build(cls, doc_event_list, query_event, filter_single_candidate=True,
              filter_single_argument=False):
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

        # softmax_mask = (doc_entity_ids != -1)

        base_query_input = np.array(query_event.token_id_list())

        arg_entity_mapping = {
            arg_idx: seq_arg.entity_id
            for arg_idx, seq_arg in enumerate(query_event.seq_arg_list)
            if seq_arg.entity_id in doc_entity_ids}

        if len(arg_entity_mapping) == 0:
            return examples

        if filter_single_argument and len(arg_entity_mapping) <= 1:
            return examples

        for arg_idx, seq_arg in enumerate(query_event.seq_arg_list):
            target_entity_id = seq_arg.entity_id
            if target_entity_id != -1 and \
                    target_entity_id in doc_entity_ids:
                query_input = base_query_input.copy()
                # component_id is exactly the token id for MISS-SUBJ/OBJ/PREP
                query_input[arg_idx + 1] = seq_arg.component_id
                # target_mask = (doc_entity_ids == target_entity_id)

                examples.append(cls(
                    doc_input=doc_input,
                    query_input=query_input.tolist(),
                    doc_entity_ids=doc_entity_ids,
                    target_entity_id=target_entity_id
                ))

        return examples

    @classmethod
    def build_single(
            cls, doc_event_list, query_event, filter_single_candidate=True,
            filter_single_argument=False):
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

        # softmax_mask = (doc_entity_ids != -1)

        base_query_input = np.array(query_event.token_id_list())

        arg_entity_mapping = {
            arg_idx: seq_arg.entity_id
            for arg_idx, seq_arg in enumerate(query_event.seq_arg_list)
            if seq_arg.entity_id in doc_entity_ids}

        if len(arg_entity_mapping) == 0:
            return examples

        if filter_single_argument and len(arg_entity_mapping) <= 1:
            return examples

        for arg_idx in arg_entity_mapping.keys():

            query_input = base_query_input.copy()
            query_input[arg_idx + 1] = \
                query_event.seq_arg_list[arg_idx].component_id

            query_input = np.delete(
                query_input,
                [i + 1 for i in arg_entity_mapping.keys() if i != arg_idx])

            examples.append(cls(
                doc_input=doc_input,
                query_input=query_input.tolist(),
                doc_entity_ids=doc_entity_ids,
                target_entity_id=arg_entity_mapping[arg_idx],
            ))

        return examples


class SeqExampleMultiHop(SeqExample):
    def __init__(self, doc_input: List[int], query_input: List[int],
                 doc_entity_ids: List[int], target_entity_id: int,
                 argument_mask: List[int]):
        super().__init__(
            doc_input=doc_input,
            query_input=query_input,
            doc_entity_ids=doc_entity_ids,
            target_entity_id=target_entity_id
        )
        assert len(argument_mask) == len(doc_input)
        self.argument_mask=argument_mask

    def __str__(self):
        return '({}, {}, {}, {}, {})'.format(
            self.doc_input, self.query_input, self.doc_entity_ids,
            self.target_entity_id, self.argument_mask)

    @classmethod
    def from_text(cls, text):
        return cls(*eval(text))

    @classmethod
    def build(cls, doc_event_list, query_event, filter_single_candidate=True,
              filter_single_argument=False):
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

        # softmax_mask = (doc_entity_ids != -1)

        base_query_input = np.array(query_event.token_id_list())

        arg_entity_mapping = {
            arg_idx: seq_arg.entity_id
            for arg_idx, seq_arg in enumerate(query_event.seq_arg_list)
            if seq_arg.entity_id in doc_entity_ids}

        if len(arg_entity_mapping) == 0:
            return examples

        if filter_single_argument and len(arg_entity_mapping) <= 1:
            return examples

        argument_mask = [1 if entity_id in arg_entity_mapping.values() else 0
                         for entity_id in doc_entity_ids]

        for arg_idx in arg_entity_mapping.keys():

            query_input = base_query_input.copy()
            # component_id == 1 --> TARGET-SUBJ (4)
            # component_id == 2 --> TARGET_OBJ (5)
            # component_id == 3 --> TARGET_PREP (6)
            query_input[arg_idx + 1] = \
                query_event.seq_arg_list[arg_idx].component_id + 3

            for miss_arg_idx in arg_entity_mapping.keys():
                if miss_arg_idx != arg_idx:
                    # component_id == 1 --> MISS-SUBJ (1)
                    # component_id == 2 --> MISS-OBJ (2)
                    # component_id == 3 --> MISS-PREP (3)
                    query_input[miss_arg_idx + 1] = \
                        query_event.seq_arg_list[miss_arg_idx].component_id

            examples.append(cls(
                doc_input=doc_input,
                query_input=query_input.tolist(),
                doc_entity_ids=doc_entity_ids,
                target_entity_id=arg_entity_mapping[arg_idx],
                argument_mask=argument_mask
            ))

        return examples


class SeqExampleMultiArg(SeqExample):
    def __init__(self, doc_input: List[int], query_input: List[int],
                 doc_entity_ids: List[int], target_entity_id: int,
                 neg_target_entity_id: int):
        super().__init__(
            doc_input=doc_input,
            query_input=query_input,
            doc_entity_ids=doc_entity_ids,
            target_entity_id=target_entity_id)
        self.neg_target_entity_id = neg_target_entity_id

    def __str__(self):
        return '({}, {}, {}, {}, {})'.format(
            self.doc_input, self.query_input, self.doc_entity_ids,
            self.target_entity_id, self.neg_target_entity_id)

    @classmethod
    def from_text(cls, text):
        return cls(*eval(text))

    @classmethod
    def build(cls, doc_event_list, query_event, filter_single_candidate=True,
              filter_single_argument=False):
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

        # softmax_mask = (doc_entity_ids != -1)

        base_query_input = np.array(query_event.token_id_list())

        arg_entity_mapping = {
            arg_idx: seq_arg.entity_id
            for arg_idx, seq_arg in enumerate(query_event.seq_arg_list)
            if seq_arg.entity_id in doc_entity_ids}

        if len(arg_entity_mapping) < 2:
            return examples

        for arg_idx_1, arg_idx_2 in combinations(
                arg_entity_mapping.keys(), r=2):
            if arg_entity_mapping[arg_idx_1] == arg_entity_mapping[arg_idx_2]:
                continue

            # use arg_idx_1 as positive and arg_idx_2 as negative
            query_input = base_query_input.copy()
            query_input[arg_idx_1 + 1] = \
                query_event.seq_arg_list[arg_idx_1].component_id
            query_input = np.delete(query_input, arg_idx_2 + 1)

            examples.append(cls(
                doc_input=doc_input,
                query_input=query_input.tolist(),
                doc_entity_ids=doc_entity_ids,
                target_entity_id=arg_entity_mapping[arg_idx_1],
                neg_target_entity_id=arg_entity_mapping[arg_idx_2]
            ))

            # use arg_idx_2 as positive and arg_idx_1 as negative
            query_input = base_query_input.copy()
            query_input[arg_idx_2 + 1] = \
                query_event.seq_arg_list[arg_idx_2].component_id
            query_input = np.delete(query_input, arg_idx_1 + 1)

            examples.append(cls(
                doc_input=doc_input,
                query_input=query_input.tolist(),
                doc_entity_ids=doc_entity_ids,
                target_entity_id=arg_entity_mapping[arg_idx_2],
                neg_target_entity_id=arg_entity_mapping[arg_idx_1]
            ))

        return examples


class SeqExampleMultiSlot(SeqExample):
    def __init__(self, doc_input: List[int], query_input: List[int],
                 neg_query_input: List[int], doc_entity_ids: List[int],
                 target_entity_id: int):
        super().__init__(
            doc_input=doc_input,
            query_input=query_input,
            doc_entity_ids=doc_entity_ids,
            target_entity_id=target_entity_id)
        assert len(neg_query_input) == len(query_input)
        self.neg_query_input = neg_query_input

    def __str__(self):
        return '({}, {}, {}, {}, {})'.format(
            self.doc_input, self.query_input, self.neg_query_input,
            self.doc_entity_ids, self.target_entity_id)

    @classmethod
    def from_text(cls, text):
        return cls(*eval(text))

    @classmethod
    def build(cls, doc_event_list, query_event, filter_single_candidate=True,
              filter_single_argument=False):
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

        # softmax_mask = (doc_entity_ids != -1)

        base_query_input = np.array(query_event.token_id_list())

        arg_entity_mapping = {
            arg_idx: seq_arg.entity_id
            for arg_idx, seq_arg in enumerate(query_event.seq_arg_list)
            if seq_arg.entity_id in doc_entity_ids}

        if len(arg_entity_mapping) < 2:
            return examples

        for arg_idx_1, arg_idx_2 in combinations(
                arg_entity_mapping.keys(), r=2):
            target_entity_id_1 = arg_entity_mapping[arg_idx_1]
            target_entity_id_2 = arg_entity_mapping[arg_idx_2]

            if target_entity_id_1 == target_entity_id_2:
                continue

            # use arg_idx_1 as positive slot and arg_idx_2 as negative slot
            query_input_1 = base_query_input.copy()
            query_input_1[arg_idx_1 + 1] = \
                query_event.seq_arg_list[arg_idx_1].component_id
            query_input_1 = np.delete(query_input_1, arg_idx_2 + 1)

            # use arg_idx_2 as positive slot and arg_idx_1 as negative slot
            query_input_2 = base_query_input.copy()
            query_input_2[arg_idx_2 + 1] = \
                query_event.seq_arg_list[arg_idx_2].component_id
            query_input_2 = np.delete(query_input_2, arg_idx_1 + 1)

            examples.append(cls(
                doc_input=doc_input,
                query_input=query_input_1.tolist(),
                neg_query_input=query_input_2.tolist(),
                doc_entity_ids=doc_entity_ids,
                target_entity_id=target_entity_id_1
            ))

            examples.append(cls(
                doc_input=doc_input,
                query_input=query_input_2.tolist(),
                neg_query_input=query_input_1.tolist(),
                doc_entity_ids=doc_entity_ids,
                target_entity_id=target_entity_id_2
            ))

        return examples


class SeqExampleWithSalience(SeqExample):
    def __init__(self, doc_input: List[int], query_input: List[int],
                 doc_entity_ids: List[int], target_entity_id: int,
                 num_mentions_total: List[int], num_mentions_named: List[int],
                 num_mentions_nominal: List[int],
                 num_mentions_pronominal: List[int]):
        super().__init__(
            doc_input=doc_input,
            query_input=query_input,
            doc_entity_ids=doc_entity_ids,
            target_entity_id=target_entity_id)
        assert len(num_mentions_total) == len(doc_input)
        self.num_mentions_total = num_mentions_total
        assert len(num_mentions_named) == len(doc_input)
        self.num_mentions_named = num_mentions_named
        assert len(num_mentions_nominal) == len(doc_input)
        self.num_mentions_nominal = num_mentions_nominal
        assert len(num_mentions_pronominal) == len(doc_input)
        self.num_mentions_pronominal = num_mentions_pronominal

    def __str__(self):
        return '({}, {}, {}, {}, {}, {}, {}, {})'.format(
            self.doc_input, self.query_input, self.doc_entity_ids,
            self.target_entity_id, self.num_mentions_total,
            self.num_mentions_named, self.num_mentions_nominal,
            self.num_mentions_pronominal)

    @classmethod
    def from_text(cls, text):
        return cls(*eval(text))

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

    @classmethod
    def build(cls, doc_event_list, query_event, filter_single_candidate=True,
              filter_single_argument=False):
        examples = []

        doc_input = [
            token_id for seq_event in doc_event_list
            for token_id in seq_event.token_id_list()]

        doc_entity_ids = [
            entity_id for seq_event in doc_event_list
            for entity_id in seq_event.entity_id_list()]

        kwargs = SeqExampleWithSalience.get_entity_salience(
            doc_event_list, doc_entity_ids)

        # do not generate examples with a single candidate, when doc_entity_ids
        # contain only 2 distinct numbers (with a -1 for predicate).
        if filter_single_candidate and len(Counter(doc_entity_ids)) <= 2:
            return examples

        # softmax_mask = (doc_entity_ids != -1)

        base_query_input = np.array(query_event.token_id_list())

        arg_entity_mapping = {
            arg_idx: seq_arg.entity_id
            for arg_idx, seq_arg in enumerate(query_event.seq_arg_list)
            if seq_arg.entity_id in doc_entity_ids}

        if len(arg_entity_mapping) == 0:
            return examples

        if filter_single_argument and len(arg_entity_mapping) <= 1:
            return examples

        for arg_idx, seq_arg in enumerate(query_event.seq_arg_list):
            target_entity_id = seq_arg.entity_id
            if target_entity_id != -1 and \
                    target_entity_id in doc_entity_ids:
                query_input = base_query_input.copy()
                # component_id is exactly the token id for MISS-SUBJ/OBJ/PREP
                query_input[arg_idx + 1] = seq_arg.component_id
                # target_mask = (doc_entity_ids == target_entity_id)

                examples.append(cls(
                    doc_input=doc_input,
                    query_input=query_input.tolist(),
                    doc_entity_ids=doc_entity_ids,
                    target_entity_id=target_entity_id,
                    **kwargs
                ))

        return examples

    @classmethod
    def build_single(
            cls, doc_event_list, query_event, filter_single_candidate=True,
            filter_single_argument=False):
        examples = []

        doc_input = [
            token_id for seq_event in doc_event_list
            for token_id in seq_event.token_id_list()]

        doc_entity_ids = [
            entity_id for seq_event in doc_event_list
            for entity_id in seq_event.entity_id_list()]

        kwargs = SeqExampleWithSalience.get_entity_salience(
            doc_event_list, doc_entity_ids)

        # do not generate examples with a single candidate, when doc_entity_ids
        # contain only 2 distinct numbers (with a -1 for predicate).
        if filter_single_candidate and len(Counter(doc_entity_ids)) <= 2:
            return examples

        # softmax_mask = (doc_entity_ids != -1)

        base_query_input = np.array(query_event.token_id_list())

        arg_entity_mapping = {
            arg_idx: seq_arg.entity_id
            for arg_idx, seq_arg in enumerate(query_event.seq_arg_list)
            if seq_arg.entity_id in doc_entity_ids}

        if len(arg_entity_mapping) == 0:
            return examples

        if filter_single_argument and len(arg_entity_mapping) <= 1:
            return examples

        for arg_idx in arg_entity_mapping.keys():

            query_input = base_query_input.copy()
            query_input[arg_idx + 1] = \
                query_event.seq_arg_list[arg_idx].component_id

            query_input = np.delete(
                query_input,
                [i + 1 for i in arg_entity_mapping.keys() if i != arg_idx])

            examples.append(cls(
                doc_input=doc_input,
                query_input=query_input.tolist(),
                doc_entity_ids=doc_entity_ids,
                target_entity_id=arg_entity_mapping[arg_idx],
                **kwargs
            ))

        return examples
