import argparse
import bz2
from pathlib import Path

from common.event_script import ScriptCorpus
from data.seq_dataset import SeqScript
from utils import log
from utils.vocabulary import load_vocab


def preprocess_script_corpus(
        word2vec_fname, word2vec_fvocab, input_path: Path, output_path: Path):
    vocab = load_vocab(
        fname=str(word2vec_fname), fvocab=str(word2vec_fvocab), binary=True)

    all_examples = []
    for corpus_path in sorted(input_path.iterdir()):
        log.info('Reading script corpus from {}'.format(corpus_path))
        corpus = ScriptCorpus.from_text(bz2.open(corpus_path, 'rt').read())
        for script in corpus.scripts:
            seq_script = SeqScript.build(vocab, script)
            seq_script.process_singletons()
            all_examples.extend(seq_script.get_all_examples())

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

    args = parser.parse_args()

    word2vec_dir = Path(args.word2vec_dir)
    fname = word2vec_dir / (args.word2vec_name + '.bin')
    fvocab = word2vec_dir / (args.word2vec_name + '.vocab')
    assert fname.exists() and fvocab.exists()

    input_dir = Path(args.input_dir)
    assert input_dir.exists()

    output_file = Path(args.output_file).with_suffix('.bz2')
    assert output_file.parent.is_dir()

    preprocess_script_corpus(
        word2vec_fname=fname,
        word2vec_fvocab=fvocab,
        input_path=input_dir,
        output_path=output_file)
