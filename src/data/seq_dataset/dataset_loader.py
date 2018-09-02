import bz2
from pathlib import Path

from joblib import Parallel, delayed
from torchtext.data import BucketIterator, Dataset, Iterator

from utils import log
from .seq_example import SeqExample
from .seq_fields import *


def get_fields(query_type='normal', include_salience=False,
               include_coref_pred_pairs=False):
    fields = [
        ('doc_input', input_field),
        ('query_input', input_field),
        ('doc_entity_ids', doc_entity_ids_field),
        ('target_entity_id', target_entity_id_field)
    ]
    if include_salience:
        fields.extend([
            ('num_mentions_total', num_mentions_field),
            ('num_mentions_named', num_mentions_field),
            ('num_mentions_nominal', num_mentions_field),
            ('num_mentions_pronominal', num_mentions_field)
        ])
    if include_coref_pred_pairs:
        fields.extend([
            ('coref_pred_1', coref_pred_1_field),
            ('coref_pred_2', coref_pred_2_field)
        ])
    if query_type == 'multi_hop':
        fields.append(('argument_mask', mask_field))
    if query_type == 'multi_arg':
        fields.append(('neg_target_entity_id', target_entity_id_field))
    if query_type == 'multi_slot':
        fields.append(('neg_query_input', input_field))
    return fields


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


def build_dataset(examples, max_len=None, query_type='normal',
                  include_salience=False, include_coref_pred_pairs=False):
    log.info('Creating Dataset')

    filter_pred = None
    if max_len:
        log.info('Filter examples by length of document input (<= {})'.format(
            max_len))

        def filter_pred(example):
            return len(example.doc_input) <= max_len

    assert query_type in [
        'normal', 'single_arg', 'multi_hop', 'multi_arg', 'multi_slot']
    fields = get_fields(
        query_type=query_type, include_salience=include_salience,
        include_coref_pred_pairs=include_coref_pred_pairs)

    dataset = Dataset(examples, fields, filter_pred=filter_pred)

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
            device,
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
        dataset_path, n_jobs=1, max_len=None, query_type='normal',
        include_salience=False, include_coref_pred_pairs=False, use_bucket=True,
        device=None, batch_size=32, sort_query=True, train=True,
        sort_within_batch=True, **kwargs):
    examples = read_examples(dataset_path, n_jobs=n_jobs)

    dataset = build_dataset(
        examples, max_len=max_len, query_type=query_type,
        include_salience=include_salience,
        include_coref_pred_pairs=include_coref_pred_pairs)

    iterator = build_iterator(
        dataset, use_bucket=use_bucket, device=device, batch_size=batch_size,
        sort_query=sort_query, train=train, sort_within_batch=sort_within_batch,
        **kwargs
    )

    return iterator
