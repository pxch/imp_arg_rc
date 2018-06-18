from typing import List

import torch
from torchtext.data import Example, Field

input_field = Field(
    use_vocab=False,
    tensor_type=torch.LongTensor,
    include_lengths=True,
    pad_token=0
)

doc_entity_ids_field = Field(
    use_vocab=False,
    tensor_type=torch.LongTensor,
    include_lengths=False,
    pad_token=-1
)

candidate_mask_field = Field(
    use_vocab=False,
    tensor_type=torch.ByteTensor,
    include_lengths=False,
    pad_token=0
)

dice_scores_field = Field(
    use_vocab=False,
    tensor_type=torch.FloatTensor,
    include_lengths=False,
    pad_token=0
)

mask_field = Field(
    use_vocab=False,
    tensor_type=torch.ByteTensor,
    include_lengths=False,
    pad_token=0
)

num_mentions_field = Field(
    use_vocab=False,
    tensor_type=torch.LongTensor,
    include_lengths=False,
    pad_token=0
)

gc_seq_fields = [
    ('doc_input', input_field),
    ('query_input', input_field),
    ('doc_entity_ids', doc_entity_ids_field),
    ('candidate_mask', candidate_mask_field),
    ('dice_scores', dice_scores_field),
    ('argument_mask', mask_field)
]

gc_seq_with_salience_fields = [
    ('doc_input', input_field),
    ('query_input', input_field),
    ('doc_entity_ids', doc_entity_ids_field),
    ('candidate_mask', candidate_mask_field),
    ('dice_scores', dice_scores_field),
    ('argument_mask', mask_field),
    ('num_mentions_total', num_mentions_field),
    ('num_mentions_named', num_mentions_field),
    ('num_mentions_nominal', num_mentions_field),
    ('num_mentions_pronominal', num_mentions_field)
]


class GCSeqExample(Example):
    def __init__(self, doc_input: List[int], query_input: List[int],
                 doc_entity_ids: List[int], candidate_mask: List[int],
                 dice_scores: List[float], argument_mask: List[int],
                 max_len=None):
        assert len(doc_input) == len(doc_entity_ids)
        assert len(doc_input) == len(candidate_mask)
        assert len(doc_input) == len(dice_scores)
        assert len(doc_input) == len(argument_mask)

        keep_len = len(doc_input)
        if max_len is not None and len(doc_input) > max_len:
            for keep_len in range(max_len, 0, -1):
                if doc_entity_ids[-keep_len] == -1:
                    break

        self.doc_input = doc_input[-keep_len:]
        self.query_input = query_input
        self.doc_entity_ids = doc_entity_ids[-keep_len:]
        self.candidate_mask = candidate_mask[-keep_len:]
        self.dice_scores = dice_scores[-keep_len:]
        self.argument_mask = argument_mask[-keep_len:]

    def __str__(self):
        return '({}, {}, {}, {}, {}, {})'.format(
            self.doc_input, self.query_input, self.doc_entity_ids,
            self.candidate_mask, self.dice_scores, self.argument_mask)

    @classmethod
    def from_text(cls, text):
        return cls(*eval(text))


class GCSeqExampleWithSalience(GCSeqExample):
    def __init__(self, doc_input: List[int], query_input: List[int],
                 doc_entity_ids: List[int], candidate_mask: List[int],
                 dice_scores: List[float], argument_mask: List[int],
                 num_mentions_total: List[int], num_mentions_named: List[int],
                 num_mentions_nominal: List[int],
                 num_mentions_pronominal: List[int]):
        super().__init__(
            doc_input=doc_input,
            query_input=query_input,
            doc_entity_ids=doc_entity_ids,
            candidate_mask=candidate_mask,
            dice_scores=dice_scores,
            argument_mask=argument_mask
        )
        assert len(num_mentions_total) == len(doc_input)
        self.num_mentions_total = num_mentions_total
        assert len(num_mentions_named) == len(doc_input)
        self.num_mentions_named = num_mentions_named
        assert len(num_mentions_nominal) == len(doc_input)
        self.num_mentions_nominal = num_mentions_nominal
        assert len(num_mentions_pronominal) == len(doc_input)
        self.num_mentions_pronominal = num_mentions_pronominal

    def __str__(self):
        return '({}, {}, {}, {}, {}, {}, {}, {}, {}, {})'.format(
            self.doc_input, self.query_input, self.doc_entity_ids,
            self.candidate_mask, self.dice_scores, self.argument_mask,
            self.num_mentions_total, self.num_mentions_named,
            self.num_mentions_nominal, self.num_mentions_pronominal)

    @classmethod
    def from_text(cls, text):
        return cls(*eval(text))
