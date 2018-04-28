import bz2
from pathlib import Path

import torch
from joblib import Parallel, delayed
from torchtext.data import BucketIterator, Dataset, Field, Iterator

from utils import log
from .seq_example import SeqExample

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

# mask_field = Field(
#     use_vocab=False,
#     tensor_type=torch.ByteTensor,
#     include_lengths=False,
#     pad_token=0
# )

seq_fields = [
    ('doc_input', input_field),
    ('query_input', input_field),
    ('doc_entity_ids', doc_entity_ids_field),
    ('target_entity_id', target_entity_id_field)
    # ('softmax_mask', mask_field),
    # ('target_mask', mask_field)
]


def read_examples_from_file(file_path):
    log.info('Reading examples from file {}'.format(file_path.name))
    return [SeqExample.from_text(line)
            for line in bz2.open(file_path, 'rt').readlines()]


def read_examples(dataset_path, n_jobs=1):
    log.info('Reading examples from directory {}'.format(dataset_path))

    if n_jobs == 1:
        examples = []
        for example_file in sorted(Path(dataset_path).glob('*.bz2')):
            examples.extend(read_examples_from_file(example_file))
    else:
        examples_list = Parallel(n_jobs=n_jobs)(
            delayed(read_examples_from_file)(example_file)
            for example_file in sorted(Path(dataset_path).glob('*.bz2')))
        examples = [ex for ex_list in examples_list for ex in ex_list]

    log.info('Found {} examples'.format(len(examples)))
    return examples


def build_dataset(examples, max_len=None):
    log.info('Creating Dataset')

    filter_pred = None
    if max_len:
        log.info('Filter examples by length of document input (<= {})'.format(
            max_len))

        def filter_pred(example):
            return len(example.doc_input) <= max_len

    dataset = Dataset(examples, seq_fields, filter_pred=filter_pred)
    log.info('Dataset created with {} examples'.format(len(dataset)))
    return dataset


def build_iterator(
        dataset, use_bucket=True, device=None, batch_size=32, sort_query=True,
        train=True, sort_within_batch=True, **kwargs):
    if sort_query:
        def sort_key(example):
            return len(example.doc_input), len(example.query_input)
    else:
        def sort_key(example):
            return len(example.doc_input)

    log.info(
        'Creating {} on {} with batch_size = {}, sort_key = {}, train = {}, '
        'sort_within_batch = {}{}'.format(
            'BucketIterator' if use_bucket else 'Iterator',
            'cpu' if device == -1 else 'cuda',
            batch_size,
            '(document_len, query_len)' if sort_query else '(document_len)',
            train,
            sort_within_batch,
            ''.join([', {} = {}'.format(k, v) for k, v in kwargs.items()])))

    if use_bucket:
        iterator = BucketIterator(
            dataset=dataset,
            batch_size=batch_size,
            sort_key=sort_key,
            device=device,
            train=train,
            sort_within_batch=sort_within_batch,
            **kwargs
        )
    else:
        iterator = Iterator(
            dataset=dataset,
            batch_size=batch_size,
            sort_key=sort_key,
            device=device,
            train=train,
            sort_within_batch=sort_within_batch,
            **kwargs
        )

    log.info('{} created with {} mini-batches'.format(
        'BucketIterator' if use_bucket else 'Iterator', len(iterator)))

    return iterator


def load_seq_dataset(
        dataset_path, n_jobs=1, max_len=None, use_bucket=True, device=None,
        batch_size=32, sort_query=True, train=True, sort_within_batch=True,
        **kwargs):
    examples = read_examples(dataset_path, n_jobs=n_jobs)

    dataset = build_dataset(examples, max_len=max_len)

    iterator = build_iterator(
        dataset, use_bucket=use_bucket, device=device, batch_size=batch_size,
        sort_query=sort_query, train=train, sort_within_batch=sort_within_batch,
        **kwargs
    )

    return iterator
