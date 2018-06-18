import argparse

import torch
from data.gerber_chai.dataset_loader import load_examples_by_fold
from data.gerber_chai.dataset_loader import build_cross_validation_iterators
from data.gerber_chai.dataset_loader import set_train_iter_random_seed
from data.gerber_chai.trainer import cross_val
from utils import load_vocab, log, add_file_handler
import datetime
from pathlib import Path
from model.pointer_net import PointerNet


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('examples_path',
                        help='path to cross validation examples')

    parser.add_argument('word2vec_dir', help='Directory for word2vec files')
    parser.add_argument('word2vec_name', help='Name of word2vec files')
    parser.add_argument('model_state_dict_path',
                        help='path to state dict file to load pretrained model')

    parser.add_argument('output_path', help='Path to save trained models')

    parser.add_argument('--disable_cuda', action='store_true',
                        help='Disable CUDA')
    parser.add_argument('--log_to_file', action='store_true',
                        help='Log to file rather than stream output')

    parser.add_argument('--batch_sizes', help='Number of examples in minibatch')
    parser.add_argument('--sort_query', action='store_true',
                        help='Sort examples by both document length and '
                             'query length')

    parser.add_argument('--multi_hop', action='store_true',
                        help='multi hop model or normal model')
    parser.add_argument('--hidden_size', help='Size of hidden state in GRU',
                        type=int, default=100)
    parser.add_argument('--num_layers', help='Number of layers in GRU',
                        type=int, default=1)
    parser.add_argument('--query_num_layers', type=int,
                        help='Number of layers in query encoder GRU, '
                             'if different from num_layers')
    parser.add_argument('--dropout', help='Dropout rate for encoder',
                        type=float, default=0.2)
    parser.add_argument('--attention', default='general',
                        help='Attention function to use, can be one of '
                             'dot, general (default), concat')
    parser.add_argument('--use_salience', action='store_true',
                        help='if turned on, use salience features in model')
    parser.add_argument('--salience_vocab_size', type=int,
                        help='size of salience vocab (max_num_mentions + 1), '
                             'leave blank to use numerical features')
    parser.add_argument('--salience_embedding_size', type=int,
                        help='embedding size of salience features, leave blank '
                             'to use numerical features')
    parser.add_argument('--use_self_attention', action='store_true',
                        help='use self attention on document encoder')
    parser.add_argument('--rescale_attn_energy', action='store_true',
                        help='rescale dot / general attention energy')
    parser.add_argument('--extra_query_linear', action='store_true',
                        help='add an extra linear mapping to query '
                             'hidden state in multi_hop models')
    parser.add_argument('--extra_doc_encoder', action='store_true',
                        help='use another layer of doc encoder for the second '
                             'layer of attention in multi_hop models')
    parser.add_argument('--query_aware', action='store_true',
                        help='concat query hidden state to the input of the '
                             'second layer of doc encoder')

    parser.add_argument('--optimizer', help='Optimization method',
                        default='adam')
    parser.add_argument('--lr', help='Learning rate for optimizer', type=float)
    parser.add_argument('--regularization', help='Regularization rate',
                        type=float, default=0.0)
    parser.add_argument('--num_epochs', help='Number of epochs to train',
                        type=int, default=10)
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='maximum gradient norm in gradient clipping')
    parser.add_argument('--val_on_dice_score', action='store_true',
                        help='validate on dice score, not loss')
    parser.add_argument('--backward_with_attn_loss', action='store_true',
                        help='compute gradient with the extra attention loss')
    parser.add_argument('--max_margin', action='store_true',
                        help='minimize negative probability in training')
    parser.add_argument('--use_sum', action='store_true',
                        help='use the sum of all correct attention scores '
                             'in computing loss function')
    parser.add_argument('--predict_entity', action='store_true',
                        help='predict based on entity score, not mention score')
    parser.add_argument('--fix_embedding', action='store_true',
                        help='freeze the weight of word embedding in tuning')
    parser.add_argument('--fix_doc_encoder', action='store_true',
                        help='freeze the weight of doc encoder in tuning')
    parser.add_argument('--fix_query_encoder', action='store_true',
                        help='freeze the weight of query encoder in tuning')
    parser.add_argument('--verbose', type=int,
                        help='verbosity level in logging, can be 0, 1, or 2')

    args = parser.parse_args()

    args.cuda = not args.disable_cuda and torch.cuda.is_available()

    # make sure output path exists
    output_path = Path(args.output_path)
    if not output_path.exists():
        output_path.mkdir(parents=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    if args.log_to_file:
        # only log to file at {output_path}/log
        add_file_handler(
            log,
            file_path=output_path / '{}.log'.format(timestamp),
            exclusive=True)

    log.info(
        'Loading cross validation dataset from {}, use_salience = {}'.format(
            args.examples_path, args.use_salience))
    examples_by_fold = load_examples_by_fold(
        args.examples_path, use_salience=args.use_salience)

    batch_sizes = [int(s) for s in args.batch_sizes.split(',')]

    log.info(
        'Building train / val / test iterators for each fold, '
        'use_salience = {}, batch_sizes = {}, sort_query = {}'.format(
            args.use_salience, batch_sizes, args.sort_query))
    iterators_by_fold = build_cross_validation_iterators(
        examples_by_fold,
        use_salience=args.use_salience,
        batch_sizes=batch_sizes,
        sort_query=args.sort_query
    )

    word2vec_dir = Path(args.word2vec_dir)
    fname = word2vec_dir / (args.word2vec_name + '.bin')
    fvocab = word2vec_dir / (args.word2vec_name + '.vocab')
    assert fname.exists() and fvocab.exists()

    if args.multi_hop:
        vocab = load_vocab(fname=str(fname), fvocab=str(fvocab),
                           use_target_specials=True)
    else:
        vocab = load_vocab(fname=str(fname), fvocab=str(fvocab))

    vocab_size, input_size = vocab.vectors.shape


    log.info(
        'Initializing pointer network with vocab_size = {}, input_size = {}, '
        'hidden_size = {}, num_layers = {}, query_num_layers = {}, '
        'dropout = {}, use_self_attention = {}, attention_method = {}, '
        'rescale_attn_energy = {}, use_salience = {}, '
        'salience_vocab_size = {},  salience_embedding_size = {}, '
        'multi_hop = {},  extra_query_linear = {}, extra_doc_encoder = {}, '
        'query_aware = {}'.format(
            vocab_size, input_size, args.hidden_size, args.num_layers,
            args.query_num_layers, args.dropout, args.use_self_attention,
            args.attention, args.rescale_attn_energy, args.use_salience,
            args.salience_vocab_size,args.salience_embedding_size,
            args.multi_hop, args.extra_query_linear, args.extra_doc_encoder,
            args.query_aware))
    pointer_net = PointerNet(
        vocab_size=vocab_size,
        input_size=input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        query_num_layers=args.query_num_layers,
        bidirectional=True,
        dropout_p=args.dropout,
        use_self_attention=args.use_self_attention,
        attn_method=args.attention,
        rescale_attn_energy=args.rescale_attn_energy,
        use_salience=args.use_salience,
        salience_vocab_size=args.salience_vocab_size,
        salience_embedding_size=args.salience_embedding_size,
        multi_hop=args.multi_hop,
        extra_query_linear=args.extra_query_linear,
        extra_doc_encoder=args.extra_doc_encoder,
        query_aware=args.query_aware
    )

    if args.cuda:
        log.info('Moving all model parameters to GPU.')
        pointer_net.cuda()

    log.info('Model specifications:\n{}'.format(pointer_net))

    log.info('Loading pretrained model parameters from {}'.format(
        args.model_state_dict_path))
    pointer_net.load_state_dict(torch.load(
        args.model_state_dict_path,
        map_location=lambda storage, loc: storage.cuda(0)))

    log.info('Setting random seed of training iterators')
    set_train_iter_random_seed(iterators_by_fold)

    param_grid = [{'optimizer': args.optimizer, 'lr': args.lr}]

    log.info(
        'Cross validation training with param_grid = {}, num_epochs = {}, '
        'regularization = {}, max_grad_norm = {}, val_on_dice_score = {}, '
        'multi_hop = {}, predict_entity = {}, backward_with_attn_loss = {}, '
        'max_margin = {}, use_sum = {}, fix_embedding = {},'
        'fix_doc_encoder = {}. fix_query_encoder = {}'.format(
            param_grid, args.num_epochs, args.regularization,
            args.max_grad_norm, args.val_on_dice_score, args.multi_hop,
            args.predict_entity, args.backward_with_attn_loss, args.max_margin,
            args.use_sum, args.fix_embedding, args.fix_doc_encoder,
            args.fix_query_encoder))

    model_state_dict_list = cross_val(
        pointer_net,
        args.model_state_dict_path,
        iterators_by_fold,
        param_grid=param_grid,
        verbose=args.verbose,
        num_gt=970,
        num_epochs=args.num_epochs,
        regularization=args.regularization,
        max_grad_norm=args.max_grad_norm,
        val_on_dice_score=args.val_on_dice_score,
        multi_hop=args.multi_hop,
        predict_entity=args.predict_entity,
        backward_with_attn_loss=args.backward_with_attn_loss,
        max_margin=args.max_margin,
        use_sum=args.use_sum,
        fix_embedding=args.fix_embedding,
        fix_doc_encoder=args.fix_doc_encoder,
        fix_query_encoder=args.fix_query_encoder,
        keep_models=True)

    model_output_path = output_path / '{}-model'.format(timestamp)

    log.info('Save model state dict list to {}'.format(model_output_path))

    torch.save(model_state_dict_list, model_output_path)
