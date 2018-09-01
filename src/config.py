import os
from pathlib import Path


class DefaultConfig(object):
    # absolute path to the root of this repository (parent directory of src)
    repo_root = Path(__file__).resolve().parents[1]

    # root directory for all corpora (from environment variable CORPUS_ROOT)
    corpus_root = Path(os.environ['CORPUS_ROOT']).resolve()

    # path to Penn Treebank WSJ corpus (relative to corpus_root)
    wsj_path = 'penn-treebank-rel3/parsed/mrg/wsj'
    # file pattern to read PTB data from WSJ corpus
    wsj_file_pattern = '\d\d/wsj_.*\.mrg'

    @property
    def wsj_root(self):
        return self.corpus_root / self.wsj_path

    # path to Propbank corpus (relative to corpus_root)
    propbank_path = 'propbank-LDC2004T14/data'
    # file name of propositions in Propbank corpus
    propbank_file = 'prop.txt'
    # file name of verb list in Propbank corpus
    propbank_verbs_file = 'verbs.txt'

    @property
    def propbank_root(self):
        return self.corpus_root / self.propbank_path

    # path to Nombank corpus (relative to corpus_root)
    nombank_path = 'nombank.1.0'
    # file name of propositions in Nombank corpus
    nombank_file = 'nombank.1.0_sorted_old'
    # file name of noun list in Nombank corpus
    nombank_nouns_file = 'nombank.1.0.words'

    @property
    def nombank_root(self):
        return self.corpus_root / self.nombank_path

    # file pattern to read frame data from Propbank/Nombank corpus
    frame_file_pattern = 'frames/.*\.xml'

    # path to Ontonotes corpus (relative to corpus_root)
    ontonotes_path = 'ontonotes-release-5.0/data/files/data/'

    @property
    def ontonotes_root(self):
        return self.corpus_root / self.ontonotes_path

    # path to the data directory
    @property
    def data_path(self):
        return self.repo_root / 'data'

    @property
    def vocab_path(self):
        return self.data_path / 'vocab'

    pred_vocab_list_file = 'predicate_min_100'
    arg_vocab_list_file = 'argument_min_500'
    ner_vocab_list_file = 'name_entity_min_500'
    prep_vocab_list_file = 'preposition'

    pred_vocab_count_file = 'predicate_min_100_count'

    @property
    def gc_path(self):
        return self.data_path / 'gerber_chai'

    gc_dataset_url = \
        'http://lair.cse.msu.edu/projects/implicit_argument_annotations.zip'
    gc_dataset_name = 'implicit_argument_annotations.xml'

    @property
    def word2vec_path(self):
        return self.data_path / 'word2vec'

    word2vec_name = 'min_500_dim300vecs'

    @property
    def on_scripts_path(self):
        return self.data_path / 'ontonotes'

    on_short_scripts_file = 'on_short_scripts.txt'
    on_long_scripts_file = 'on_long_scripts.txt'


cfg = DefaultConfig()
