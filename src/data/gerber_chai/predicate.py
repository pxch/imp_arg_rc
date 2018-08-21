from collections import OrderedDict, defaultdict

from nltk.corpus.reader.nombank import NombankChainTreePointer
from nltk.corpus.reader.nombank import NombankSplitTreePointer
from torchtext.vocab import Vocab

from common.event_script import Argument
from data.nltk import PTBReader
from data.seq_dataset.seq_script import SeqArgument
from utils import log
from .candidate import Candidate
from .corenlp_reader import CoreNLPReader
from .gc_seq_example import GCSeqExample, GCSeqExampleWithSalience
from .helper import convert_nombank_label
from .helper import core_arg_list, nominal_predicate_mapping
from .helper import predicate_core_arg_mapping
from .imp_arg_instance import ImpArgInstance
from .rich_tree_pointer import RichTreePointer


class Predicate(object):
    def __init__(self, pred_pointer, imp_args, exp_args):
        self.pred_pointer = pred_pointer
        self.fileid = pred_pointer.fileid
        self.sentnum = pred_pointer.sentnum
        self.n_pred = ''
        self.v_pred = ''
        self.imp_args = imp_args
        self.exp_args = exp_args
        self.candidates = []

    def set_pred(self, n_pred):
        self.n_pred = n_pred
        self.v_pred = nominal_predicate_mapping[n_pred]

    def has_imp_arg(self, label, max_dist=-1):
        if label in self.imp_args:
            if max_dist == -1:
                return True
            else:
                for arg_pointer in self.imp_args[label]:
                    if 0 <= self.sentnum - arg_pointer.sentnum <= max_dist:
                        return True
        return False

    def num_imp_arg(self, max_dist=-1):
        return sum([1 for label in self.imp_args
                    if self.has_imp_arg(label, max_dist)])

    def has_oracle(self, label):
        for candidate in self.candidates:
            if candidate.is_oracle(self.imp_args[label]):
                return True
        return False

    def num_oracle(self):
        return sum([1 for label in self.imp_args if self.has_oracle(label)])

    def parse_args(
            self, treebank_reader: PTBReader, corenlp_reader: CoreNLPReader,
            include_non_head_entity=True):
        for label in self.imp_args:
            for arg in self.imp_args[label]:
                arg.get_treebank(treebank_reader)
                arg.parse_treebank()
                arg.parse_corenlp(
                    corenlp_reader,
                    include_non_head_entity=include_non_head_entity)

        for label, fillers in self.exp_args.items():
            for arg in fillers:
                arg.get_treebank(treebank_reader)
                arg.parse_treebank()
                arg.parse_corenlp(
                    corenlp_reader,
                    include_non_head_entity=include_non_head_entity)
            if label in core_arg_list and len(fillers) > 1:
                assert len(fillers) == 2
                new_fillers = []
                for arg in fillers:
                    # remove pointer pointing to WH-determiner
                    if arg.tree.pos()[arg.tree_pointer.wordnum][1] != 'WDT':
                        new_fillers.append(arg)
                # should only exists one non-WH-determiner pointer
                assert len(new_fillers) == 1
                self.exp_args[label] = new_fillers

    def get_candidate_keys(self, max_dist=2):
        key_list = []
        for sentnum in range(max(0, self.sentnum - max_dist), self.sentnum + 1):
            key_list.append('{}:{}'.format(self.fileid, sentnum))
        return key_list

    def add_candidates(self, instances, max_dist=2):
        for instance in instances:
            if 0 <= self.sentnum - instance.sentnum <= max_dist:
                candidate_list = Candidate.from_instance(instance)
                self.candidates.extend(candidate_list)

    def check_exp_args(self, instance, add_missing_args=False,
                       remove_conflict_imp_args=False, verbose=False):

        unmatched_labels = list(self.exp_args.keys())

        if instance is not None:
            nombank_arg_dict = defaultdict(list)
            for arg_pointer, label in instance.arguments:
                cvt_label = convert_nombank_label(label)
                if cvt_label:
                    nombank_arg_dict[cvt_label].append(arg_pointer)

            for label in nombank_arg_dict:
                nombank_args = nombank_arg_dict[label]

                if label not in self.exp_args:
                    message = \
                        '{} has {} in Nombank but not found in explicit ' \
                        'arguments.'.format(self.pred_pointer, label)
                    if add_missing_args:
                        message += \
                            '\n\tAdding missing explicit {}: {}.'.format(
                                label, nombank_args)
                        self.exp_args[label] = \
                            [RichTreePointer(self.fileid, self.sentnum, arg)
                             for arg in nombank_args]
                        if remove_conflict_imp_args and label in self.imp_args:
                            message += '\n\tRemoving implicit {}.'.format(label)
                            self.imp_args.pop(label, None)
                    else:
                        message += '\n\tIgnored...'
                    if verbose:
                        log.info(message)
                    continue

                exp_args = [p.tree_pointer for p in self.exp_args[label]]
                unmatched_labels.remove(label)

                if exp_args != nombank_args:
                    message = '{} has mismatch in {}: {} --> {}'.format(
                        self.pred_pointer, label, exp_args, nombank_args)
                    if len(nombank_args) == 1:
                        nombank_arg = nombank_args[0]
                        if isinstance(nombank_arg, NombankSplitTreePointer):
                            if all(p in nombank_arg.pieces for p in exp_args):
                                self.exp_args[label] = [RichTreePointer(
                                    self.fileid, self.sentnum, nombank_arg)]
                                if verbose:
                                    log.info(message + '\n\tReplaced...')
                                continue
                        if isinstance(nombank_arg, NombankChainTreePointer):
                            if all(p in nombank_arg.pieces for p in exp_args):
                                if verbose:
                                    log.info(message + '\n\tIgnored...')
                                continue

                    raise AssertionError(message)

        if unmatched_labels:
            message = '{} has {} in explicit arguments but not found in ' \
                      'Nombank.'.format(self.pred_pointer, unmatched_labels)
            raise AssertionError(message)

    def get_query_token_ids(self, corenlp_reader: CoreNLPReader, vocab: Vocab):
        pred_id = vocab.stoi[self.v_pred + '-PRED']

        doc = corenlp_reader.get_doc(self.pred_pointer.fileid)
        sent = doc.get_sent(self.sentnum)

        exp_arg_id_mapping = defaultdict(list)
        for label, fillers in self.exp_args.items():
            if label in core_arg_list:
                assert len(fillers) == 1
                arg_pointer = fillers[0]
                arg_type = predicate_core_arg_mapping[self.v_pred][label]

                token = sent.get_token(arg_pointer.head_idx())
                argument = Argument.from_token(token)
                seq_arg = SeqArgument.build(
                    vocab=vocab, argument=argument, use_lemma=True,
                    arg_type=arg_type, use_unk=True)

                if arg_type == 'SUBJ':
                    exp_arg_id_mapping['SUBJ'] = [seq_arg.token_id]
                elif arg_type == 'OBJ':
                    exp_arg_id_mapping['OBJ'] = [seq_arg.token_id]
                else:
                    exp_arg_id_mapping['PREP'].append(seq_arg.token_id)

        return pred_id, exp_arg_id_mapping

    def get_candidate_idx_mapping(self, seq_event_list):
        candidate_idx_mapping_list = []
        matched_candidate_idx_list = []

        for seq_event in seq_event_list:
            candidate_idx_mapping_list.append([])

            for seq_arg in seq_event.seq_arg_list:
                candidate_idx_mapping = []

                for candidate_idx, candidate in enumerate(self.candidates):
                    matched = False
                    c_pointer = candidate.arg_pointer
                    if seq_event.sentnum == c_pointer.sentnum and \
                            seq_arg.wordnum == c_pointer.head_idx():
                        matched = True
                    if seq_arg.entity_id != -1 and \
                            seq_arg.entity_id == c_pointer.entity_idx:
                        matched = True
                    if matched:
                        candidate_idx_mapping.append(candidate_idx)
                        if candidate_idx not in matched_candidate_idx_list:
                            matched_candidate_idx_list.append(candidate_idx)

                candidate_idx_mapping_list.append(candidate_idx_mapping)

        matched_candidate_idx_list = sorted(matched_candidate_idx_list)
        return candidate_idx_mapping_list, matched_candidate_idx_list

    def get_all_examples(self, corenlp_reader: CoreNLPReader, vocab: Vocab,
                         missing_labels=None, labeled_arg_only=False,
                         max_dist=2, use_salience=False, return_dict=False,
                         multi_hop=False):
        examples = OrderedDict()

        if missing_labels is not None and len(missing_labels) == 0:
            return examples if return_dict else list(examples.values())

        if missing_labels is None and labeled_arg_only:
            missing_labels = list(self.imp_args.keys())
            if len(missing_labels) == 0:
                return examples if return_dict else list(examples.values())

        seq_script = corenlp_reader.get_seq_script(self.fileid)
        if max_dist is not None:
            seq_event_list = [
                seq_event for seq_event in seq_script.seq_event_list
                if self.sentnum - max_dist <= seq_event.sentnum <= self.sentnum]
        else:
            seq_event_list = [
                seq_event for seq_event in seq_script.seq_event_list
                if seq_event.sentnum <= self.sentnum]

        if len(seq_event_list) == 0:
            log.warning('Predicate {} has not context events'.format(
                self.pred_pointer))
            return examples if return_dict else list(examples.values())

        c_idx_mapping_list, matched_candidate_idx_list = \
            self.get_candidate_idx_mapping(seq_event_list)

        seq_script.process_singletons()

        doc_input = [
            token_id for seq_event in seq_event_list
            for token_id in seq_event.token_id_list()]

        doc_entity_ids = [
            entity_id for seq_event in seq_event_list
            for entity_id in seq_event.entity_id_list()]

        candidate_mask = [
            1 if len(c_idx_mapping) > 0 else 0
            for c_idx_mapping in c_idx_mapping_list]

        correct_candidate_idx_list = []
        for label, fillers in self.imp_args.items():
            candidate_dice_scores_dict = {
                candidate_idx: self.candidates[candidate_idx].dice_score(
                    fillers, use_corenlp_tokens=True)
                for candidate_idx in matched_candidate_idx_list}
            max_dice_score = max(candidate_dice_scores_dict.values()) \
                if candidate_dice_scores_dict else 0.0
            correct_candidate_idx_list.extend([
                candidate_idx for candidate_idx, dice_score
                in candidate_dice_scores_dict.items()
                if dice_score == max_dice_score])

        correct_candidate_idx_list = \
            sorted(list(set(correct_candidate_idx_list)))

        argument_mask = [
            1 if any(c_idx in correct_candidate_idx_list
                     for c_idx in c_idx_mapping) else 0
            for c_idx_mapping in c_idx_mapping_list]

        kwargs = {}

        if use_salience:
            entity_salience_mapping = {}
            for entity_id in doc_entity_ids:
                if entity_id not in entity_salience_mapping:
                    entity_salience_mapping[entity_id] = [0, 0, 0, 0]

            for seq_event in seq_event_list:
                for seq_arg in seq_event.seq_arg_list:
                    entity_id = seq_arg.entity_id
                    mention_type = seq_arg.mention_type
                    if mention_type != 0:
                        entity_salience_mapping[entity_id][mention_type] += 1
                    entity_salience_mapping[entity_id][0] += 1

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

        pred_id, exp_arg_id_mapping = self.get_query_token_ids(
            corenlp_reader=corenlp_reader, vocab=vocab)

        imp_arg_type_list = [
            predicate_core_arg_mapping[self.v_pred][label]
            for label in predicate_core_arg_mapping[self.v_pred]
            if label not in self.exp_args and (
                    not missing_labels or label in missing_labels)
        ]

        for label in predicate_core_arg_mapping[self.v_pred]:
            if label in self.exp_args:
                continue
            if missing_labels and label not in missing_labels:
                continue
            if label in self.imp_args:
                fillers = self.imp_args[label]
                candidate_dice_scores = [
                    candidate.dice_score(fillers, use_corenlp_tokens=True)
                    for candidate in self.candidates]
                dice_scores = []
                for c_idx_mapping in c_idx_mapping_list:
                    if c_idx_mapping:
                        dice_scores.append(max(
                            candidate_dice_scores[c_idx]
                            for c_idx in c_idx_mapping))
                    else:
                        dice_scores.append(0.0)
            else:
                dice_scores = [0.0 for _ in c_idx_mapping_list]

            query_input = [pred_id]
            arg_type = predicate_core_arg_mapping[self.v_pred][label]
            if arg_type == 'SUBJ':
                assert exp_arg_id_mapping['SUBJ'] == []
                if multi_hop:
                    query_input.append(4)
                else:
                    query_input.append(1)
            else:
                query_input.extend(exp_arg_id_mapping['SUBJ'])
                if multi_hop and 'SUBJ' in imp_arg_type_list:
                    query_input.append(1)

            if arg_type == 'OBJ':
                assert exp_arg_id_mapping['OBJ'] == []
                if multi_hop:
                    query_input.append(5)
                else:
                    query_input.append(2)
            else:
                query_input.extend(exp_arg_id_mapping['OBJ'])
                if multi_hop and 'OBJ' in imp_arg_type_list:
                    query_input.append(2)

            if arg_type.startswith('PREP'):
                if multi_hop:
                    query_input.append(6)
                else:
                    query_input.append(3)
            query_input.extend(exp_arg_id_mapping['PREP'])

            if use_salience:
                examples[label] = GCSeqExampleWithSalience(
                    doc_input=doc_input,
                    query_input=query_input,
                    doc_entity_ids=doc_entity_ids,
                    candidate_mask=candidate_mask,
                    dice_scores=dice_scores,
                    argument_mask=argument_mask,
                    *kwargs
                )
            else:
                examples[label] = GCSeqExample(
                    doc_input=doc_input,
                    query_input=query_input,
                    doc_entity_ids=doc_entity_ids,
                    candidate_mask=candidate_mask,
                    dice_scores=dice_scores,
                    argument_mask=argument_mask
                )

        seq_script.restore_singletons()
        return examples if return_dict else list(examples.values())

    @classmethod
    def build(cls, instance: ImpArgInstance):
        pred_pointer = instance.pred_pointer

        tmp_imp_args = defaultdict(list)
        exp_args = defaultdict(list)

        for argument in instance.arguments:

            label = argument[0].lower()
            arg_pointer = argument[1]
            attribute = argument[2]

            # remove arguments located in sentences following the predicate
            if arg_pointer.fileid != pred_pointer.fileid or \
                    arg_pointer.sentnum > pred_pointer.sentnum:
                continue

            # add explicit arguments to exp_args
            if attribute == 'Explicit':
                exp_args[label].append(arg_pointer)
                # remove the label from tmp_imp_args, as we do not process
                # an implicit argument if some explicit arguments with
                # the same label exist
                tmp_imp_args.pop(label, None)

            # add non-explicit arguments to tmp_imp_args
            else:
                # do not add the argument when some explicit arguments with
                # the same label exist
                if label not in exp_args:
                    tmp_imp_args[label].append((arg_pointer, attribute))

        # process implicit arguments
        imp_args = {}
        for label, fillers in tmp_imp_args.items():

            # remove incorporated arguments from tmp_imp_args
            # incorporated argument: argument with the same node as
            # the predicate itself
            if pred_pointer in [pointer for pointer, _ in fillers]:
                continue

            # add non-split arguments to imp_args
            imp_args[label] = [pointer for pointer, attribute in fillers
                               if attribute == '']
            split_pointers = [pointer for pointer, attribute in fillers
                              if attribute == 'Split']

            sentnum_set = set([pointer.sentnum for pointer in split_pointers])

            # group split arguments by their sentnum,
            # and sort pieces by nombank_pointer.wordnum within each group
            grouped_split_pointers = []
            for sentnum in sentnum_set:
                grouped_split_pointers.append(sorted(
                    [pointer for pointer in split_pointers
                     if pointer.sentnum == sentnum],
                    key=lambda p: p.tree_pointer.wordnum))

            # add each split pointer to imp_args
            for split_pointers in grouped_split_pointers:
                imp_args[label].append(RichTreePointer.merge(split_pointers))

        return cls(pred_pointer, imp_args, exp_args)

    def pretty_print(self, verbose=False, include_candidates=False,
                     include_dice_score=False, corenlp_reader=None):
        result = '{}\t{}\n'.format(self.pred_pointer, self.n_pred)

        for label, fillers in self.imp_args.items():
            result += '\tImplicit {}:\n'.format(label)
            for filler in fillers:
                if verbose:
                    result += '\t\t{}\n'.format(
                        filler.pretty_print(corenlp_reader))
                else:
                    result += '\t\t{}\n'.format(filler)

        for label, fillers in self.exp_args.items():
            result += '\tExplicit {}:\n'.format(label)
            for filler in fillers:
                if verbose:
                    result += '\t\t{}\n'.format(
                        filler.pretty_print(corenlp_reader))
                else:
                    result += '\t\t{}\n'.format(filler)

        if include_candidates:
            result += '\tCandidates:\n'
            for candidate in self.candidates:
                if verbose:
                    result += '\t\t{}'.format(
                        candidate.arg_pointer.pretty_print(corenlp_reader))
                else:
                    result += '\t\t{}'.format(candidate.arg_pointer)

                if include_dice_score:
                    dice_list = {}
                    for label, fillers in self.imp_args.items():
                        dice_list[label] = candidate.dice_score(fillers)
                        result += '\t{}\n'.format(
                            ', '.join(['{}: {:.2f}'.format(label, dice)
                                       for label, dice in dice_list.items()]))
                else:
                    result += '\n'

        return result
