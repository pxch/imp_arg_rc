import argparse
import bz2
from pathlib import Path

from common.event_script import ScriptCorpus
from config import cfg
from data.seq_dataset import SeqScript
from utils import load_vocab, log, read_vocab_count, read_vocab_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', help='directory of scripts for input')
    parser.add_argument('output_file', help='filename to output all samples')
    parser.add_argument('word2vec_dir', help='directory for word2vec files')
    parser.add_argument('word2vec_name', help='name of word2vec files')
    parser.add_argument('--prep_vocab', help='path to preposition vocab file')
    parser.add_argument('--subsampling', action='store_true',
                        help='if turned on, most frequent predicates would be '
                             'randomly subsampled according to their frequency')
    parser.add_argument('--filter_repetitive_prep', action='store_true',
                        help='if turned on, remove pobjs with repetitive '
                             'prepositions in a event')
    parser.add_argument('--query_type', default='normal',
                        help='type of query to generate, can be either '
                             'normal (default), single_arg, multi_hop, '
                             'multi_arg, or multi_slot')
    parser.add_argument('--include_salience', action='store_true',
                        help='if turned on, include salience features')
    parser.add_argument('--include_coref_pred_pairs', action='store_true',
                        help='if turned on, include pairs of predicate indices'
                             'with shared arguments')

    args = parser.parse_args()

    assert args.query_type in [
        'normal', 'single_arg', 'multi_hop', 'multi_arg', 'multi_slot']

    word2vec_dir = Path(args.word2vec_dir)
    fname = word2vec_dir / (args.word2vec_name + '.bin')
    fvocab = word2vec_dir / (args.word2vec_name + '.vocab')
    assert fname.exists() and fvocab.exists()

    vocab = load_vocab(fname=str(fname), fvocab=str(fvocab), binary=True,
                       use_target_specials=True, use_miss_specials=True)

    input_dir = Path(args.input_dir)
    assert input_dir.exists()

    output_file = Path(args.output_file).with_suffix('.bz2')
    assert output_file.parent.is_dir()

    if args.prep_vocab:
        prep_vocab_list = read_vocab_list(Path(args.prep_vocab))
    else:
        prep_vocab_list = read_vocab_list(
            cfg.vocab_path / cfg.prep_vocab_list_file)

    pred_vocab_count = None
    if args.subsampling:
        pred_vocab_count = read_vocab_count(
            cfg.vocab_path / cfg.pred_vocab_count_file)

    all_examples = []
    for corpus_path in sorted(input_dir.iterdir()):
        log.info('Reading script corpus from {}'.format(corpus_path))
        corpus = ScriptCorpus.from_text(bz2.open(corpus_path, 'rt').read())
        for script in corpus.scripts:
            seq_script = SeqScript.build(
                vocab=vocab, script=script, use_lemma=True, use_unk=True,
                pred_vocab_count=pred_vocab_count,
                prep_vocab_list=prep_vocab_list,
                filter_repetitive_prep=args.filter_repetitive_prep)
            seq_script.process_singletons()
            all_examples.extend(
                seq_script.get_all_examples(
                    filter_single_candidate=True,
                    filter_single_argument=False,
                    query_type=args.query_type,
                    include_salience=args.include_salience,
                    include_coref_pred_pairs=args.include_coref_pred_pairs
                ))

    log.info('Writing {} examples to {}'.format(len(all_examples), output_file))
    with bz2.open(output_file, 'wt') as fout:
        fout.write('\n'.join(map(str, all_examples)))
        fout.write('\n')
