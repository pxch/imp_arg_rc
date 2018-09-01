import argparse
from pathlib import Path

import torch

from common.event_script import ScriptCorpus
from config import cfg
from data.seq_dataset import SeqScript
from data.seq_dataset.dataset_loader import build_dataset, build_iterator
from model.pointer_net import PointerNet
from model.pointer_net.trainer import evaluate
from utils import consts, log, load_vocab, read_vocab_list


def prepare_dataset(
        script_corpus, vocab, prep_vocab_list, stop_pred_ids,
        filter_single_argument=False, query_type='normal',
        include_salience=False, use_bucket=True, device=None, batch_size=8,
        sort_query=False):
    examples = []
    for script in script_corpus.scripts:
        seq_script = SeqScript.build(
            vocab=vocab, script=script,
            prep_vocab_list=prep_vocab_list,
            filter_repetitive_prep=True)
        seq_script.process_singletons()
        examples.extend(seq_script.get_all_examples(
            stop_pred_ids=stop_pred_ids,
            filter_single_candidate=True,
            filter_single_argument=filter_single_argument,
            query_type=query_type,
            include_salience=include_salience))

    dataset = build_dataset(
        examples, query_type=query_type, include_salience=include_salience)

    iterator = build_iterator(
        dataset, use_bucket=use_bucket, device=device, batch_size=batch_size,
        sort_query=sort_query, train=False, sort_within_batch=True)

    return iterator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('load_model_state',
                        help='path to state dict file to load pretrained model')

    # arguments for i/o paths
    parser.add_argument('--on_short_path', help='Path to ON-Short scripts')
    parser.add_argument('--on_long_path', help='Path to ON-Long scripts')
    parser.add_argument('--word2vec_dir', help='Directory for word2vec files')
    parser.add_argument('--word2vec_name', help='Name of word2vec files')
    parser.add_argument('--prep_vocab', help='Path to preposition vocab file')
    # general purpose arguments
    parser.add_argument('--disable_cuda', action='store_true',
                        help='Disable CUDA')

    # arguments for loading datasets
    parser.add_argument('--filter_single_argument', action='store_true',
                        help='only evaluate on a subset of the dataset '
                             'with multiple implicit arguments')

    parser.add_argument('--query_type', default='normal',
                        help='type of queries in dataset, can be one of '
                             'normal (default), multi_hop, multi_arg '
                             '(deprecated), or multi_slot (deprecated)')
    parser.add_argument('--batch_size', help='Number of examples in minibatch',
                        type=int, default=8)
    parser.add_argument('--use_bucket', help='Use BucketIterator',
                        action='store_true')
    parser.add_argument('--sort_query', action='store_true',
                        help='Sort examples by both document length and '
                             'query length')

    # arguments for document/query encoders
    parser.add_argument('--hidden_size', help='Size of hidden state in GRU',
                        type=int, default=100)
    parser.add_argument('--num_layers', help='Number of layers in GRU',
                        type=int, default=1)
    parser.add_argument('--query_num_layers', type=int,
                        help='Number of layers in query encoder GRU, '
                             'if different from num_layers')
    parser.add_argument('--dropout_p', help='Dropout rate for encoder',
                        type=float, default=0.2)
    # arguments for attention layer
    parser.add_argument('--attn_method', default='general',
                        help='Attention function to use, can be one of '
                             'dot, general (default), concat')
    parser.add_argument('--rescale_attn_energy', action='store_true',
                        help='rescale dot / general attention energy')
    # arguments for self attention
    parser.add_argument('--use_self_attn', action='store_true',
                        help='use self attention on document encoder')
    parser.add_argument('--self_attn_method',
                        help='Attention function for self attention,'
                             'same as attn_method if not specified')
    # arguments for multi_hop inference
    parser.add_argument('--multi_hop', action='store_true',
                        help='use multi hop inference in query')
    parser.add_argument('--extra_query_linear', action='store_true',
                        help='add an extra linear mapping to query '
                             'hidden state in multi_hop models')
    parser.add_argument('--extra_doc_encoder', action='store_true',
                        help='use another layer of doc encoder for the second '
                             'layer of attention in multi_hop models')
    parser.add_argument('--query_aware', action='store_true',
                        help='concat query hidden state to the input of the '
                             'second layer of doc encoder')
    # arguments for salience features
    parser.add_argument('--use_salience', action='store_true',
                        help='if turned on, use salience features in model')
    parser.add_argument('--salience_vocab_size', type=int,
                        help='size of salience vocab (max_num_mentions + 1), '
                             'leave blank to use numerical features')
    parser.add_argument('--salience_embedding_size', type=int,
                        help='embedding size of salience features, leave blank '
                             'to use numerical features')

    parser.add_argument('--report_entity_freq', action='store_true',
                        help='report accuracy by entity frequency')

    args = parser.parse_args()

    args.device = None
    if not args.disable_cuda and torch.cuda.is_available():
        # default to cuda:0, as only 1 CUDA device would be visible
        args.device = torch.device('cuda:0')
    else:
        args.device = torch.device('cpu')

    if args.word2vec_dir:
        word2vec_dir = Path(args.word2vec_dir)
    else:
        word2vec_dir = cfg.word2vec_path
    if args.word2vec_name:
        word2vec_name = args.word2vec_name
    else:
        word2vec_name = cfg.word2vec_name

    fname = word2vec_dir / (word2vec_name + '.bin')
    fvocab = word2vec_dir / (word2vec_name + '.vocab')
    assert fname.exists() and fvocab.exists()

    vocab = load_vocab(fname=str(fname), fvocab=str(fvocab),
                       use_target_specials=True, use_miss_specials=True)

    stop_pred_ids = [
        vocab.stoi[stop_pred + '-PRED'] for stop_pred in consts.stop_preds]

    if args.prep_vocab:
        prep_vocab_list = read_vocab_list(Path(args.prep_vocab))
    else:
        prep_vocab_list = read_vocab_list(
            cfg.vocab_path / cfg.prep_vocab_list_file)

    on_short_path = args.on_short_path
    if not on_short_path:
        on_short_path = cfg.on_scripts_path / cfg.on_short_scripts_file

    on_short_corpus = ScriptCorpus.from_text(open(on_short_path, 'r').read())

    on_short_iter = prepare_dataset(
        on_short_corpus, vocab, prep_vocab_list, stop_pred_ids,
        filter_single_argument=args.filter_single_argument,
        query_type=args.query_type, include_salience=args.use_salience,
        use_bucket=args.use_bucket, device=args.device,
        batch_size=args.batch_size, sort_query=args.sort_query)

    on_long_path = args.on_long_path
    if not on_long_path:
        on_long_path = cfg.on_scripts_path / cfg.on_long_scripts_file

    on_long_corpus = ScriptCorpus.from_text(open(on_long_path, 'r').read())

    on_long_iter = prepare_dataset(
        on_long_corpus, vocab, prep_vocab_list, stop_pred_ids,
        filter_single_argument=args.filter_single_argument,
        query_type=args.query_type, include_salience=args.use_salience,
        use_bucket=args.use_bucket, device=args.device,
        batch_size=args.batch_size, sort_query=args.sort_query)

    vocab_size, input_size = vocab.vectors.shape

    pointer_net = PointerNet(
        vocab_size=vocab_size,
        input_size=input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        query_num_layers=args.query_num_layers,
        bidirectional=True,
        dropout_p=args.dropout_p,
        attn_method=args.attn_method,
        rescale_attn_energy=args.rescale_attn_energy,
        use_self_attn=args.use_self_attn,
        self_attn_method=args.self_attn_method,
        multi_hop=args.multi_hop,
        extra_query_linear=args.extra_query_linear,
        extra_doc_encoder=args.extra_doc_encoder,
        query_aware=args.query_aware,
        use_salience=args.use_salience,
        salience_vocab_size=args.salience_vocab_size,
        salience_embedding_size=args.salience_embedding_size
    )
    log.info('Moving all model parameters to {}.'.format(args.device))
    pointer_net.to(device=args.device)

    log.info('Loading pretrained model parameters from {}'.format(
        args.load_model_state))
    pointer_net.load_state_dict(torch.load(
        args.load_model_state, map_location=str(args.device)))

    log.info('Evaluate OnShort')
    evaluate(pointer_net, on_short_iter,
             report_entity_freq=args.report_entity_freq)
    log.info('Evaluate OnLong')
    evaluate(pointer_net, on_long_iter,
             report_entity_freq=args.report_entity_freq)


if __name__ == '__main__':
    main()
