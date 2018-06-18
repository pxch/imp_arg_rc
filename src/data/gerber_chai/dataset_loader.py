import json
import random

from torchtext.data import Dataset, Iterator

from .gc_seq_example import GCSeqExampleWithSalience, GCSeqExample
from .gc_seq_example import gc_seq_fields, gc_seq_with_salience_fields

emnlp_ddl_seed = 1527058799

def save_examples_by_fold(examples_by_fold, file_path):
    json_examples_by_fold = []
    for train_examples, val_examples, test_examples in examples_by_fold:
        json_train = [str(ex) for ex in train_examples]
        json_val = [str(ex) for ex in val_examples]
        json_test = [str(ex) for ex in test_examples]
        json_examples_by_fold.append((json_train, json_val, json_test))
    with open(file_path, 'wt') as fout:
        json.dump(json_examples_by_fold, fout, indent=2)


def load_examples_by_fold(file_path, use_salience=False):
    with open(file_path, 'rt') as fin:
        json_examples_by_fold = json.load(fin)

    if use_salience:
        build_fn = GCSeqExampleWithSalience.from_text
    else:
        build_fn = GCSeqExample.from_text

    examples_by_fold = []
    for json_train, json_val, json_test in json_examples_by_fold:
        train_examples = [build_fn(json_ex) for json_ex in json_train]
        val_examples = [build_fn(json_ex) for json_ex in json_val]
        test_examples = [build_fn(json_ex) for json_ex in json_test]

        examples_by_fold.append((train_examples, val_examples, test_examples))
    return examples_by_fold


def build_cross_validation_iterators(
        examples_by_fold, use_salience=False, batch_sizes=None,
        sort_query=False):
    iterators_by_fold = []

    if use_salience:
        fields = gc_seq_with_salience_fields
    else:
        fields = gc_seq_fields

    if sort_query:
        def sort_key(example):
            return len(example.doc_input), len(example.query_input)
    else:
        def sort_key(example):
            return len(example.doc_input)

    for train_examples, val_examples, test_examples in examples_by_fold:
        train_dataset = Dataset(train_examples, fields)
        val_dataset = Dataset(val_examples, fields)
        test_dataset = Dataset(test_examples, fields)

        iterators = Iterator.splits(
            (train_dataset, val_dataset, test_dataset),
            batch_sizes=batch_sizes,
            sort_key=sort_key,
            sort_within_batch=True,
            repeat=False)

        iterators_by_fold.append(iterators)

    return iterators_by_fold


def set_train_iter_random_seed(iterators_by_fold, seed=emnlp_ddl_seed):
    random.seed(seed)
    for train_iter, _, _ in iterators_by_fold:
        train_iter.random_shuffler.random_state = random.getstate()
