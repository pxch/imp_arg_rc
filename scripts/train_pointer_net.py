import argparse
from pathlib import Path

import torch
from torch import optim

from data.seq_dataset import load_seq_dataset
from model.pointer_net import PointerNet, train_epoch, validate
from utils import load_vocab, log, add_file_handler

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('training_path', help='Path to training dataset')
    parser.add_argument('validation_path', help='Path to validation dataset')
    parser.add_argument('word2vec_dir', help='Directory for word2vec files')
    parser.add_argument('word2vec_name', help='Name of word2vec files')
    parser.add_argument('output_path', help='Path to save trained models')
    parser.add_argument('--disable_cuda', action='store_true',
                        help='Disable CUDA')
    parser.add_argument('--log_to_file', action='store_true',
                        help='Log to file rather than stream output')
    parser.add_argument('--example_type', default='normal',
                        help='type of examples in dataset, can be one of '
                             'normal (default), multi_arg, multi_slot, '
                             'or multi_hop')
    parser.add_argument('--max_len', help='maximum length of document allowed',
                        type=int, default=100)
    parser.add_argument('--batch_size', help='Number of examples in minibatch',
                        type=int, default=32)
    parser.add_argument('--use_bucket', help='Use BucketIterator',
                        action='store_true')
    parser.add_argument('--sort_query', action='store_true',
                        help='Sort examples by both document length and '
                             'query length')
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
    parser.add_argument('--optimizer', help='Optimization method',
                        default='adam')
    parser.add_argument('--lr', help='Learning rate for optimizer', type=float)
    parser.add_argument('--regularization', help='Regularization rate',
                        type=float, default=0.0)
    parser.add_argument('--log_every',
                        help='Log training loss after number of batches',
                        type=int, default=1000)
    parser.add_argument('--val_every',
                        help='Validate after number of batches',
                        type=int, default=50000)
    parser.add_argument('--num_epochs', help='Number of epochs to train',
                        type=int, default=10)
    parser.add_argument('--num_jobs', type=int, default=1,
                        help='Number of parallel jobs in loading data')
    parser.add_argument('--load_model_state',
                        help='path to state dict file to load pretrained model')
    parser.add_argument('--load_optimizer_state',
                        help='path to state dict file to load optimizer state')
    parser.add_argument('--objective_type',
                        help='type of objective function in training, can be '
                             'either normal or multi_arg, if not specified, '
                             'default to be the same as example_type')
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
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='maximum gradient norm in gradient clipping')
    parser.add_argument('--rescale_attn_energy', action='store_true',
                        help='rescale dot / general attention energy')
    parser.add_argument('--backward_with_attn_loss', action='store_true',
                        help='compute gradient with the extra attention loss')
    parser.add_argument('--extra_query_linear', action='store_true',
                        help='add an extra linear mapping to query '
                             'hidden state in multi_hop models')
    parser.add_argument('--extra_doc_encoder', action='store_true',
                        help='use another layer of doc encoder for the second '
                             'layer of attention in multi_hop models')
    parser.add_argument('--query_aware', action='store_true',
                        help='concat query hidden state to the input of the '
                             'second layer of doc encoder')
    parser.add_argument('--use_sum', action='store_true',
                        help='use the sum of all correct attention scores '
                             'in computing loss function')
    parser.add_argument('--use_sigmoid', action='store_true',
                        help='use sigmoid of attention energies in computing '
                             'loss function')

    args = parser.parse_args()

    args.cuda = not args.disable_cuda and torch.cuda.is_available()

    device = None if args.cuda else -1

    # make sure output path exists
    output_path = Path(args.output_path)
    if not output_path.exists():
        output_path.mkdir(parents=True)

    if args.log_to_file:
        # only log to file at {output_path}/log
        add_file_handler(log, file_path=output_path / 'log', exclusive=True)

    # make sure the directory to save model checkpoints exists
    checkpoint_path = output_path / 'checkpoints'
    if not checkpoint_path.exists():
        checkpoint_path.mkdir(parents=True)

    training_iter = load_seq_dataset(
        args.training_path,
        example_type=args.example_type,
        n_jobs=args.num_jobs,
        max_len=args.max_len,
        use_bucket=args.use_bucket,
        device=device,
        batch_size=args.batch_size,
        sort_query=args.sort_query,
        train=True,
        sort_within_batch=True,
        repeat=False
    )

    validation_iter = load_seq_dataset(
        args.validation_path,
        example_type=args.example_type,
        n_jobs=args.num_jobs,
        max_len=args.max_len,
        use_bucket=args.use_bucket,
        device=device,
        batch_size=args.batch_size,
        sort_query=args.sort_query,
        train=False,
        sort_within_batch=True
    )

    word2vec_dir = Path(args.word2vec_dir)
    fname = word2vec_dir / (args.word2vec_name + '.bin')
    fvocab = word2vec_dir / (args.word2vec_name + '.vocab')
    assert fname.exists() and fvocab.exists()

    if args.example_type == 'multi_hop':
        vocab = load_vocab(fname=str(fname), fvocab=str(fvocab),
                           use_target_specials=True)
    else:
        vocab = load_vocab(fname=str(fname), fvocab=str(fvocab))

    vocab_size, input_size = vocab.vectors.shape

    multi_hop = (args.example_type == 'multi_hop')

    log.info(
        'Initializing pointer network with vocab_size = {}, input_size = {}, '
        'hidden_size = {}, num_layers = {}, query_num_layers = {}, '
        'dropout = {}, use_self_attention = {}, attention_method = {}, '
        'rescale_attn_energy = {}, use_salience = {}, '
        'salience_vocab_size = {}, salience_embedding_size = {}, '
        'multi_hop = {}, extra_query_linear = {}, extra_doc_encoder = {}, '
        'query_aware = {}'.format(
            vocab_size, input_size, args.hidden_size, args.num_layers,
            args.query_num_layers, args.dropout, args.use_self_attention,
            args.attention, args.rescale_attn_energy, args.use_salience,
            args.salience_vocab_size, args.salience_embedding_size, multi_hop,
            args.extra_query_linear, args.extra_doc_encoder, args.query_aware))
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
        multi_hop=multi_hop,
        extra_query_linear=args.extra_query_linear,
        extra_doc_encoder=args.extra_doc_encoder,
        query_aware=args.query_aware
    )
    log.info('Initializing embedding layer with pretrained word vectors.')
    pointer_net.init_embedding(vocab.vectors)

    if args.cuda:
        log.info('Moving all model parameters to GPU.')
        pointer_net.cuda()

    log.info('Model specifications:\n{}'.format(pointer_net))

    if args.load_model_state:
        log.info('Loading pretrained model parameters from {}'.format(
            args.load_model_state))
        pointer_net.load_state_dict(torch.load(
            args.load_model_state,
            map_location=lambda storage, loc: storage.cuda(0)))

    optim_dict = {
        'adadelta': optim.Adadelta,
        'adagrad': optim.Adagrad,
        'adam': optim.Adam,
        'sparseadam': optim.SparseAdam,
        'adamax': optim.Adamax,
        'rmsprop': optim.RMSprop,
    }
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(pointer_net.parameters(), lr=args.lr)
    elif args.optimizer in optim_dict:
        if args.lr:
            optimizer = optim_dict[args.optimizer](
                pointer_net.parameters(), lr=args.lr)
        else:
            optimizer = optim_dict[args.optimizer](
                pointer_net.parameters())
    else:
        raise NotImplementedError()

    log.info('Initializing {} optimizer with lr = {}'.format(
        args.optimizer, optimizer.defaults['lr']))

    if args.load_optimizer_state:
        log.info('Loading optimizer state from {}'.format(
            args.load_optimizer_state))
        optimizer.load_state_dict(torch.load(
            args.load_optimizer_state,
            map_location=lambda storage, loc: storage.cuda(0)))

    if not args.objective_type:
        objective_type = args.example_type
    else:
        objective_type = args.objective_type
    assert objective_type in [
        'normal', 'multi_arg', 'multi_slot', 'multi_hop', 'max_margin']

    log.info(
        'Training with objective_type = {}, backward_with_attn_loss = {}, '
        'regularization = {}, max_grad_norm = {}, log_every = {}, '
        'val_every = {}, use_sum = {}, use_sigmoid = {}'.format(
            objective_type, args.backward_with_attn_loss, args.regularization,
            args.max_grad_norm, args.log_every, args.val_every, args.use_sum,
            args.use_sigmoid))

    validate(pointer_net, validation_iter, objective_type=objective_type,
             msg='Before training')

    for epoch in range(args.num_epochs):
        log.info('Start training epoch {:3d}/{:3d}'.format(
            epoch + 1, args.num_epochs))
        train_epoch(
            pointer_net=pointer_net,
            training_iter=training_iter,
            validation_iter=validation_iter,
            optimizer=optimizer,
            objective_type=objective_type,
            backward_with_attn_loss=args.backward_with_attn_loss,
            regularization=args.regularization,
            log_every=args.log_every,
            val_every=args.val_every,
            max_grad_norm=args.max_grad_norm,
            use_sum=args.use_sum,
            use_sigmoid=args.use_sigmoid
        )

        validate(
            pointer_net, validation_iter, objective_type=objective_type,
            msg='After {:3d}/{:3d} epochs'.format(epoch + 1, args.num_epochs))

        model_output_path = \
            checkpoint_path / 'epoch-{}-model'.format(epoch + 1)
        log.info('Save model states to {}'.format(model_output_path))
        torch.save(pointer_net.state_dict(), model_output_path)

        optim_output_path = \
            checkpoint_path / 'epoch-{}-optim'.format(epoch + 1)
        log.info('Save optimizer states to {}'.format(optim_output_path))
        torch.save(optimizer.state_dict(), optim_output_path)
