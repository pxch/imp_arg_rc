import pickle as pkl
from pathlib import Path

from torchtext.vocab import Vocab

from common.event_script import Script
from config import cfg
from data.document_reader import read_corenlp_doc
from data.seq_dataset import SeqScript
from utils import log, read_vocab_list
from .helper import expand_wsj_fileid


class CoreNLPReader(object):
    def __init__(self, corenlp_dict):
        self.corenlp_dict = corenlp_dict

    def get_all(self, fileid):
        return self.corenlp_dict[fileid]

    def get_idx_mapping(self, fileid):
        return self.corenlp_dict[fileid][0]

    def get_doc(self, fileid):
        return self.corenlp_dict[fileid][1]

    def get_script(self, fileid):
        return self.corenlp_dict[fileid][2]

    def get_seq_script(self, fileid):
        return self.corenlp_dict[fileid][3]

    @classmethod
    def build(cls, instances, corenlp_root: Path, vocab: Vocab, verbose=False):
        prep_vocab_list = read_vocab_list(
            cfg.vocab_path / cfg.prep_vocab_list_file)

        log.info('Building CoreNLP Reader from {}'.format(corenlp_root))
        corenlp_dict = {}

        for instance in instances:
            pred_pointer = instance.pred_pointer
            if pred_pointer.fileid not in corenlp_dict:
                fileid = expand_wsj_fileid(pred_pointer.fileid)

                idx_mapping = []
                with open(corenlp_root / 'idx' / fileid, 'r') as fin:
                    for line in fin:
                        idx_mapping.append([int(i) for i in line.split()])

                doc = read_corenlp_doc(
                    corenlp_root / 'parsed' / (fileid + '.xml.bz2'),
                    verbose=verbose)

                script = Script.from_doc(doc)

                seq_script = SeqScript.build(
                    vocab=vocab, script=script, use_lemma=True, use_unk=True,
                    prep_vocab_list=prep_vocab_list,
                    filter_repetitive_prep=False)

                corenlp_dict[pred_pointer.fileid] = \
                    (idx_mapping, doc, script, seq_script)

        log.info('Done')
        return cls(corenlp_dict)

    @classmethod
    def load(cls, corenlp_dict_path):
        log.info('Loading CoreNLP Reader from {}'.format(corenlp_dict_path))
        corenlp_dict = pkl.load(open(corenlp_dict_path, 'r'))
        log.info('Done')

        return cls(corenlp_dict)

    def save(self, corenlp_dict_path):
        log.info('Saving CoreNLP dict to {}'.format(corenlp_dict_path))
        pkl.dump(self.corenlp_dict, open(corenlp_dict_path, 'w'))
