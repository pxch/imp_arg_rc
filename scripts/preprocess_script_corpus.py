import argparse
import bz2
from pathlib import Path

from common.event_script import ScriptCorpus
from data.seq_dataset import SeqScript
from utils import load_vocab, log, read_vocab_count, read_vocab_list


def preprocess_script_corpus(
        word2vec_fname, word2vec_fvocab, input_path: Path, output_path: Path,
        pred_vocab_count=None, prep_vocab_list=None,
        filter_repetitive_prep=False):
    vocab = load_vocab(
        fname=str(word2vec_fname), fvocab=str(word2vec_fvocab), binary=True)

    all_examples = []
    for corpus_path in sorted(input_path.iterdir()):
        log.info('Reading script corpus from {}'.format(corpus_path))
        corpus = ScriptCorpus.from_text(bz2.open(corpus_path, 'rt').read())
        for script in corpus.scripts:
            seq_script = SeqScript.build(
                vocab=vocab, script=script, use_lemma=True, use_unk=True,
                pred_vocab_count=pred_vocab_count,
                prep_vocab_list=prep_vocab_list,
                filter_repetitive_prep=filter_repetitive_prep)
            seq_script.process_singletons()
            all_examples.extend(
                seq_script.get_all_examples(filter_single_candidate=True))

    log.info('Writing {} examples to {}'.format(len(all_examples), output_path))
    with bz2.open(output_path, 'wt') as fout:
        fout.write('\n'.join(map(str, all_examples)))
        fout.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', help='directory of scripts for input')
    parser.add_argument('output_file', help='filename to output all samples')
    parser.add_argument('word2vec_dir', help='directory for word2vec files')
    parser.add_argument('word2vec_name', help='name of word2vec files')
    parser.add_argument('--pred_vocab_count',
                        help='path to predicate vocab count file')
    parser.add_argument('--prep_vocab_list',
                        help='path to preposition vocab list file')
    parser.add_argument('--filter_repetitive_prep', action='store_true',
                        help='if turned on, remove pobjs with repetitive'
                             'prepositions in a event')

    args = parser.parse_args()

    word2vec_dir = Path(args.word2vec_dir)
    fname = word2vec_dir / (args.word2vec_name + '.bin')
    fvocab = word2vec_dir / (args.word2vec_name + '.vocab')
    assert fname.exists() and fvocab.exists()

    input_dir = Path(args.input_dir)
    assert input_dir.exists()

    output_file = Path(args.output_file).with_suffix('.bz2')
    assert output_file.parent.is_dir()

    pred_vocab_count = None
    if args.pred_vocab_count:
        pred_vocab_count = read_vocab_count(args.pred_vocab_count)

    prep_vocab_list = None
    if args.prep_vocab_list:
        prep_vocab_list = read_vocab_list(args.prep_vocab_list)

    preprocess_script_corpus(
        word2vec_fname=fname,
        word2vec_fvocab=fvocab,
        input_path=input_dir,
        output_path=output_file,
        pred_vocab_count=pred_vocab_count,
        prep_vocab_list=prep_vocab_list,
        filter_repetitive_prep=args.filter_repetitive_prep
    )