from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import List, Union

import numpy as np
from nltk.corpus.reader.nombank import NombankChainTreePointer
from nltk.corpus.reader.propbank import PropbankChainTreePointer
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import KFold
from torchtext.vocab import Vocab

from common import event_script
from config import cfg
from data.nltk import NombankReader, PropbankReader, PTBReader
from data.seq_dataset import SeqScript
from utils import log
from utils import read_vocab_list
from .candidate import CandidateDict
from .corenlp_reader import CoreNLPReader
from .helper import convert_nombank_label, core_arg_list
from .helper import expand_wsj_fileid, shorten_wsj_fileid
from .helper import nominal_predicate_mapping
from .imp_arg_instance import ImpArgInstance
from .predicate import Predicate
from .rich_tree_pointer import RichTreePointer
from .stats import print_stats


class ImpArgDataset(object):
    def __init__(self, vocab: Vocab, corenlp_root: Path, n_splits=10,
                 split_by_original=False, max_candidate_dist=2,
                 include_non_head_entity=True, suppress_warning=False):
        # vocabulary to build sequential representations
        self.vocab = vocab

        # root path to the CoreNLP parsed wsj corpus
        self.corenlp_root = Path(corenlp_root)

        # number of splits in n-fold cross validation
        self.n_splits = n_splits
        # if True, split the dataset by the original order,
        # otherwise by the sorted order
        self.split_by_original = split_by_original

        # maximum sentence distance between a candidate and the predicate
        self.max_candidate_dist = max_candidate_dist

        # if True, count the entity indices of non-head words in a candidate
        # when determining the entity idx of the candidate, otherwise only
        # use the entity idx of the head word
        self.include_non_head_entity = include_non_head_entity

        # if True, do not print warning message to stderr
        if suppress_warning:
            log.warning = log.debug

        self.all_instances = []
        self.instance_order_list = []
        self.train_test_folds = []

        self.all_predicates = []

        self._treebank_reader = None
        self._nombank_reader = None
        self._propbank_reader = None
        self._predicate_mapping = None
        self._corenlp_reader = None
        self._candidate_dict = None

    def read_dataset(self, file_path):
        log.info('Reading implicit argument dataset from {}'.format(file_path))
        input_xml = open(file_path, 'r')

        all_instances = []
        for line in input_xml.readlines()[1:-1]:
            instance = ImpArgInstance.parse(line.strip())
            all_instances.append(instance)

        log.info('Found {} instances'.format(len(all_instances)))

        self.all_instances = sorted(
            all_instances, key=lambda ins: str(ins.pred_pointer))

        self.instance_order_list = [self.all_instances.index(instance)
                                    for instance in all_instances]

        instance_order_list = np.asarray(self.instance_order_list)

        kf = KFold(n_splits=self.n_splits, shuffle=False)
        if self.split_by_original:
            for train, test in kf.split(self.instance_order_list):
                train_indices = instance_order_list[train]
                test_indices = instance_order_list[test]
                self.train_test_folds.append((train_indices, test_indices))
        else:
            self.train_test_folds = list(kf.split(self.instance_order_list))

    def print_dataset(self, file_path):
        fout = open(file_path, 'w')
        fout.write('<annotations>\n')

        for instance in self.all_instances:
            fout.write(str(instance) + '\n')

        fout.write('</annotations>\n')
        fout.close()

    def print_dataset_by_pred(self, dir_path: Path):
        all_instances_by_pred = defaultdict(list)
        for instance in self.all_instances:
            n_pred = self.predicate_mapping[str(instance.pred_pointer)]
            all_instances_by_pred[n_pred].append(instance)

        for n_pred in all_instances_by_pred:
            fout = open(dir_path / n_pred, 'w')
            for instance in all_instances_by_pred[n_pred]:
                fout.write(str(instance) + '\n')
            fout.close()

    @property
    def treebank_reader(self):
        if self._treebank_reader is None:
            self._treebank_reader = PTBReader()
        return self._treebank_reader

    @property
    def nombank_reader(self):
        if self._nombank_reader is None:
            self._nombank_reader = NombankReader()
            self._nombank_reader.build_index()
        return self._nombank_reader

    @property
    def propbank_reader(self):
        if self._propbank_reader is None:
            self._propbank_reader = PropbankReader()
            self._propbank_reader.build_index()
        return self._propbank_reader

    @property
    def predicate_mapping(self):
        if self._predicate_mapping is None:
            assert self.treebank_reader is not None

            log.info('Building predicate mapping')
            self._predicate_mapping = {}

            lemmatizer = WordNetLemmatizer()

            for instance in self.all_instances:
                pred_pointer = instance.pred_pointer

                self.treebank_reader.read_file(
                    expand_wsj_fileid(pred_pointer.fileid, '.mrg'))

                word = self.treebank_reader.all_sents[
                    pred_pointer.sentnum][pred_pointer.tree_pointer.wordnum]

                n_pred = lemmatizer.lemmatize(word.lower(), pos='n')

                if n_pred not in nominal_predicate_mapping:
                    for subword in n_pred.split('-'):
                        if subword in nominal_predicate_mapping:
                            n_pred = subword
                            break

                assert n_pred in nominal_predicate_mapping, \
                    'unexpected nominal predicate: {}'.format(n_pred)
                assert str(pred_pointer) not in self._predicate_mapping, \
                    'pred_node {} already found'.format(pred_pointer)
                self._predicate_mapping[str(pred_pointer)] = n_pred

        return self._predicate_mapping

    @property
    def corenlp_reader(self):
        if self._corenlp_reader is None:
            self._corenlp_reader = CoreNLPReader.build(
                self.all_instances, self.corenlp_root, vocab=self.vocab)
        return self._corenlp_reader

    def build_extra_event(self, instance, pred, sent):
        subj = None
        obj = None
        pobj_list = []

        fileid = shorten_wsj_fileid(instance.fileid)
        sentnum = instance.sentnum

        for tree_pointer, label in instance.arguments:
            cvt_label = convert_nombank_label(label)
            if cvt_label not in core_arg_list:
                continue
            arg_pointer_list = []
            if isinstance(tree_pointer, NombankChainTreePointer) or \
                    isinstance(tree_pointer, PropbankChainTreePointer):
                for p in tree_pointer.pieces:
                    arg_pointer_list.append(RichTreePointer(
                        fileid, sentnum, p, tree=instance.tree))
            else:
                arg_pointer_list.append(RichTreePointer(
                    fileid, sentnum, tree_pointer, tree=instance.tree))

            argument_list = []
            for arg_pointer in arg_pointer_list:
                arg_pointer.parse_treebank()
                arg_pointer.parse_corenlp(self.corenlp_reader,
                                          include_non_head_entity=True)

                arg_token_idx = arg_pointer.head_idx()
                if arg_token_idx != -1 and arg_token_idx != pred.wordnum:
                    arg_token = sent.get_token(arg_token_idx)
                    argument = event_script.Argument.from_token(arg_token)
                    argument_list.append(argument)

            if cvt_label == 'arg0' and argument_list:
                subj = argument_list[0]
            elif cvt_label == 'arg1' and argument_list:
                obj = argument_list[0]
            else:
                pobj_list.extend(
                    [('', argument) for argument in argument_list])

        event = event_script.Event(pred, subj, obj, pobj_list)

        return event

    def get_extra_events(self, fileid, idx_mapping, doc, verbification_dict,
                         use_nombank=True, use_propbank=True):
        extra_events = []

        if use_nombank:
            for instance in self.nombank_reader.search_by_fileid(fileid):
                sentnum = instance.sentnum
                sent = doc.get_sent(sentnum)

                try:
                    pred_token_idx = \
                        idx_mapping[sentnum].index(instance.wordnum)
                except ValueError:
                    continue

                nom_pred_token = sent.get_token(pred_token_idx)
                if nom_pred_token.lemma not in verbification_dict:
                    continue
                pred_lemma = verbification_dict[nom_pred_token.lemma]
                pred = event_script.Predicate(
                    word=pred_lemma,
                    lemma=pred_lemma,
                    pos='VB',
                    sentnum=sentnum,
                    wordnum=pred_token_idx)

                event = self.build_extra_event(instance, pred, sent)
                extra_events.append(event)

        if use_propbank:
            for instance in self.propbank_reader.search_by_fileid(fileid):
                sentnum = instance.sentnum
                sent = doc.get_sent(sentnum)

                try:
                    pred_token_idx = \
                        idx_mapping[sentnum].index(instance.wordnum)
                except ValueError:
                    continue

                pred_token = sent.get_token(pred_token_idx)

                pred = event_script.Predicate(
                    pred_token.word,
                    pred_token.lemma,
                    'VB',
                    sentnum=sentnum,
                    wordnum=pred_token_idx)

                event = self.build_extra_event(instance, pred, sent)
                extra_events.append(event)

        return extra_events

    def add_extra_events(self, verbification_dict, use_nombank=True,
                         use_propbank=True):
        if use_nombank:
            log.info('Adding extra events from NomBank to CoreNLP scripts')
        if use_propbank:
            log.info('Adding extra events from PropBank to CoreNLP scripts')

        prep_vocab_list = read_vocab_list(
            cfg.vocab_path / cfg.prep_vocab_list_file)

        for fileid in self.corenlp_reader.corenlp_dict.keys():
            log.info('Process file {}'.format(fileid))

            idx_mapping = self.corenlp_reader.get_idx_mapping(fileid)
            doc = self.corenlp_reader.get_doc(fileid)

            extra_events = self.get_extra_events(
                fileid, idx_mapping, doc, verbification_dict,
                use_nombank=use_nombank, use_propbank=use_propbank)

            script = self.corenlp_reader.get_script(fileid)
            for extra_event in extra_events:
                script.add_extra_event(extra_event)

            seq_script = SeqScript.build(
                vocab=self.vocab, script=script, use_lemma=True, use_unk=True,
                prep_vocab_list=prep_vocab_list,
                filter_repetitive_prep=False)

            self.corenlp_reader.corenlp_dict[fileid] = (
                idx_mapping, doc, deepcopy(script), deepcopy(seq_script))

    def build_predicates(self):
        assert len(self.all_instances) > 0
        assert self.treebank_reader is not None
        assert self.nombank_reader is not None
        assert self.predicate_mapping is not None
        assert self.corenlp_reader is not None

        if len(self.all_predicates) > 0:
            log.warning('Overriding existing predicates')
            self.all_predicates = []

        log.info('Building predicates')
        for instance in self.all_instances:
            predicate = Predicate.build(instance)
            predicate.set_pred(
                self.predicate_mapping[str(predicate.pred_pointer)])
            self.all_predicates.append(predicate)

        log.info('Checking explicit arguments with Nombank instances')
        for predicate in self.all_predicates:
            nombank_instance = self.nombank_reader.search_by_pointer(
                predicate.pred_pointer)
            predicate.check_exp_args(
                nombank_instance, add_missing_args=False,
                remove_conflict_imp_args=False, verbose=False)

        log.info('Parsing all implicit and explicit arguments')
        for predicate in self.all_predicates:
            predicate.parse_args(
                self.treebank_reader, self.corenlp_reader,
                include_non_head_entity=self.include_non_head_entity)
        log.info('Done')

    @property
    def candidate_dict(self):
        if self._candidate_dict is None:
            assert len(self.all_predicates) > 0
            assert self.propbank_reader is not None
            assert self.nombank_reader is not None
            assert self.corenlp_reader is not None
            log.info('Building candidate dict from Propbank and Nombank')
            self._candidate_dict = CandidateDict(
                propbank_reader=self.propbank_reader,
                nombank_reader=self.nombank_reader,
                corenlp_reader=self.corenlp_reader,
                max_dist=self.max_candidate_dist)

            for predicate in self.all_predicates:
                self._candidate_dict.add_candidates(
                    predicate.pred_pointer,
                    include_non_head_entity=self.include_non_head_entity)
            log.info('Done')

        return self._candidate_dict

    def add_candidates(self):
        assert len(self.all_predicates) > 0
        assert self.candidate_dict is not None
        log.info('Adding candidates to predicates')
        for predicate in self.all_predicates:
            for candidate in self.candidate_dict.get_candidates(
                    predicate.pred_pointer):
                predicate.candidates.append(candidate)

    def print_stats(self, verbose=0):
        print_stats(self.all_predicates, verbose=verbose)

    def get_all_examples(self, missing_labels_mapping=None,
                         labeled_arg_only=False, max_dist=2, use_salience=False,
                         multi_hop=False, return_dict=False):
        all_examples = {}

        for predicate in self.all_predicates:
            pred_key = str(predicate.pred_pointer)
            if missing_labels_mapping is not None:
                missing_labels = missing_labels_mapping[pred_key]
            else:
                missing_labels = None
            all_examples[pred_key] = predicate.get_all_examples(
                corenlp_reader=self.corenlp_reader,
                vocab=self.vocab,
                missing_labels=missing_labels,
                labeled_arg_only=labeled_arg_only,
                max_dist=max_dist,
                use_salience=use_salience,
                multi_hop=multi_hop,
                return_dict=return_dict)

        if return_dict:
            return all_examples
        else:
            return [ex for examples in all_examples.values() for ex in examples]

    def get_pred_fold_mapping(self):
        pred_fold_mapping = {}

        for fold in range(self.n_splits):
            for predicate_idx in self.train_test_folds[fold][1]:
                predicate = self.all_predicates[predicate_idx]
                pred_fold_mapping[str(predicate.pred_pointer)] = fold

        return pred_fold_mapping

    def get_predicate_mapping(self):
        predicate_mapping = {}

        for predicate in self.all_predicates:
            predicate_mapping[str(predicate.pred_pointer)] = predicate

        return predicate_mapping

    @staticmethod
    def get_train_val_test_splits(split_type='val_on_prev', n_splits=10):
        assert split_type in [
            'val_on_train', 'val_on_test', 'val_on_next', 'val_on_prev']
        train_val_test_fold_splits = []

        for test in range(n_splits):
            if split_type == 'val_on_train':
                train = [f for f in range(n_splits) if f != test]
                val = [f for f in range(n_splits) if f != test]
            elif split_type == 'val_on_test':
                train = [f for f in range(n_splits) if f != test]
                val = [test]
            elif split_type == 'val_on_next':
                if test == n_splits - 1:
                    val = 0
                else:
                    val = test + 1
                train = [f for f in range(n_splits) if f != test and f != val]
            else:
                if test == 0:
                    val = n_splits - 1
                else:
                    val = test - 1
                train = [f for f in range(n_splits) if f != test and f != val]

            train_val_test_fold_splits.append((train, val, test))

        return train_val_test_fold_splits

    def get_predicate_indices_by_fold(
            self, fold_indices: Union[int, List[int]]):
        if isinstance(fold_indices, int):
            return list(self.train_test_folds[fold_indices][1])
        results = []
        for fold in fold_indices:
            results.extend(list(self.train_test_folds[fold][1]))
        return results

    def get_cross_validation_examples(
            self, split_type, missing_labels_mapping=None,
            use_missing_labels=True, labeled_arg_only=False, max_dist=2,
            filter_none_candidate=False, use_salience=False, multi_hop=False):
        log.info('Building cross validation examples')

        train_val_test_fold_splits = ImpArgDataset.get_train_val_test_splits(
            split_type, n_splits=self.n_splits)

        log.info(
            'Use training / validation / testing folds:\n' + '\n'.join(
                map(str, train_val_test_fold_splits)))

        examples_by_fold = []

        for train, val, test in train_val_test_fold_splits:

            train_examples = []
            for predicate_idx in self.get_predicate_indices_by_fold(train):
                predicate = self.all_predicates[predicate_idx]
                missing_labels = None
                if missing_labels_mapping and use_missing_labels:
                    missing_labels = \
                        missing_labels_mapping[str(predicate.pred_pointer)]

                examples = predicate.get_all_examples(
                    corenlp_reader=self.corenlp_reader,
                    vocab=self.vocab,
                    labeled_arg_only=labeled_arg_only,
                    missing_labels=missing_labels,
                    max_dist=max_dist,
                    use_salience=use_salience,
                    multi_hop=multi_hop,
                    return_dict=False
                )
                if filter_none_candidate:
                    examples = \
                        [ex for ex in examples if max(ex.dice_scores) > 0]
                train_examples.extend(examples)

            val_examples = []
            for predicate_idx in self.get_predicate_indices_by_fold(val):
                predicate = self.all_predicates[predicate_idx]
                missing_labels = None
                if missing_labels_mapping and use_missing_labels:
                    missing_labels = \
                        missing_labels_mapping[str(predicate.pred_pointer)]

                examples = predicate.get_all_examples(
                    corenlp_reader=self.corenlp_reader,
                    vocab=self.vocab,
                    labeled_arg_only=labeled_arg_only,
                    missing_labels=missing_labels,
                    max_dist=max_dist,
                    use_salience=use_salience,
                    multi_hop=multi_hop,
                    return_dict=False
                )
                if filter_none_candidate:
                    examples = \
                        [ex for ex in examples if max(ex.dice_scores) > 0]
                val_examples.extend(examples)

            test_examples = []
            for predicate_idx in self.get_predicate_indices_by_fold(test):
                predicate = self.all_predicates[predicate_idx]
                missing_labels = None
                if missing_labels_mapping:
                    missing_labels = \
                        missing_labels_mapping[str(predicate.pred_pointer)]

                test_examples.extend(predicate.get_all_examples(
                    corenlp_reader=self.corenlp_reader,
                    vocab=self.vocab,
                    labeled_arg_only=False,
                    missing_labels=missing_labels,
                    max_dist=max_dist,
                    use_salience=use_salience,
                    multi_hop=multi_hop,
                    return_dict=False
                ))

            examples_by_fold.append(
                (train_examples, val_examples, test_examples))

        return examples_by_fold
