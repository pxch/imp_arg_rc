from collections import Counter
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
    def build(cls, doc_event_list, query_event, filter_single_candidate=True):
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
