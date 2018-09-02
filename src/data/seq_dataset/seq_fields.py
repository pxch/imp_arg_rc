import torch
from torchtext.data import Field

__all__ = [
    'input_field', 'doc_entity_ids_field', 'target_entity_id_field',
    'num_mentions_field', 'mask_field',
    'coref_pred_1_field', 'coref_pred_2_field'
]

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

target_entity_id_field = Field(
    sequential=False,
    use_vocab=False,
    tensor_type=torch.LongTensor,
    include_lengths=False
)

num_mentions_field = Field(
    use_vocab=False,
    tensor_type=torch.LongTensor,
    include_lengths=False,
    pad_token=0
)

mask_field = Field(
    use_vocab=False,
    tensor_type=torch.ByteTensor,
    include_lengths=False,
    pad_token=0
)


def coref_pred_postprocessing(arr, _, __):
    batch_indices = []
    pred_indices = []
    for batch_idx, x in enumerate(arr):
        batch_indices.extend([batch_idx] * len(x))
        pred_indices.extend(x)
    return [batch_indices, pred_indices]


coref_pred_1_field = Field(
    sequential=False,
    use_vocab=False,
    tensor_type=torch.LongTensor,
    include_lengths=False,
    postprocessing=coref_pred_postprocessing
)

coref_pred_2_field = Field(
    sequential=False,
    use_vocab=False,
    tensor_type=torch.LongTensor,
    include_lengths=False,
    postprocessing=lambda arr, _, __: [[val for x in arr for val in x]]
)
