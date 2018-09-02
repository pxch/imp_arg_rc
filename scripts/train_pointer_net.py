import argparse
from pathlib import Path

import torch
from torch import optim

from data.seq_dataset import load_seq_dataset
from model.pointer_net import PointerNet, train_epoch, validate
from utils import load_vocab, log, add_file_handler

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # arguments for i/o paths
    parser.add_argument('training_path', help='Path to training dataset')
    parser.add_argument('validation_path', help='Path to validation dataset')
    parser.add_argument('word2vec_dir', help='Directory for word2vec files')
    parser.add_argument('word2vec_name', help='Name of word2vec files')
    parser.add_argument('output_path', help='Path to save trained models')
    # general purpose arguments
    parser.add_argument('--disable_cuda', action='store_true',
                        help='Disable CUDA')
    parser.add_argument('--log_to_file', action='store_true',
                        help='Log to file rather than stream output')

    # arguments for loading datasets
    parser.add_argument('--query_type', default='normal',
                        help='type of queries in dataset, can be one of '
                             'normal (default), multi_hop, multi_arg '
                             '(deprecated), or multi_slot (deprecated)')
    parser.add_argument('--max_len', help='maximum length of document allowed',
                        type=int, default=100)
    parser.add_argument('--batch_size', help='Number of examples in minibatch',
                        type=int, default=32)
    parser.add_argument('--use_bucket', help='Use BucketIterator',
                        action='store_true')
    parser.add_argument('--sort_query', action='store_true',
                        help='Sort examples by both document length and '
                             'query length')
    parser.add_argument('--num_jobs', type=int, default=1,
                        help='Number of parallel jobs in loading data')

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

    # arguments for optimizer
    parser.add_argument('--optimizer', help='Optimization method',
                        default='adam')
    parser.add_argument('--lr', help='Learning rate for optimizer', type=float)
    parser.add_argument('--use_lr_annealing', action='store_true',
                        help='if turned on, reduce learning rate after a '
                             'certain number of epochs with no improvement '
                             'on validation set')
    parser.add_argument('--lr_annealing_epochs', type=int, default=2,
                        help='number of epochs with no improvement after which '
                             'learning rate will be reduced')
    parser.add_argument('--lr_annealing_factor', type=float, default=0.5,
                        help='factor by which learning rate will be reduced')
    # arguments for loading pretrained models
    parser.add_argument('--load_model_state',
                        help='path to state dict file to load pretrained model')
    parser.add_argument('--load_optimizer_state',
                        help='path to state dict file to load optimizer state')

    # arguments for computing loss functions
    parser.add_argument('--neg_loss_type', default='none',
                        help='type of negative loss term, can be'
                             'none (default), multi_arg, multi_slot,'
                             'or max_margin')
    parser.add_argument('--regularization', help='Regularization rate',
                        type=float, default=0.0)
    parser.add_argument('--use_sum', action='store_true',
                        help='use the sum of all correct attention scores '
                             'in computing loss function')
    parser.add_argument('--use_sigmoid', action='store_true',
                        help='use sigmoid of attention energies in computing '
                             'loss function')
    parser.add_argument('--self_attn_target_for_pred', default='none',
                        help='type of self_attn_target for predicate tokens, '
                             'can be one of none (default), self, or self_attn')
    # arguments for training models
    parser.add_argument('--num_epochs', help='Number of epochs to train',
                        type=int, default=10)
    parser.add_argument('--backward_self_attn_loss', action='store_true',
                        help='compute gradient with the additional'
                             'self attention loss')
    parser.add_argument('--backward_first_hop_attn_loss', action='store_true',
                        help='compute gradient with the additional'
                             'first hop attention loss (in multi_hop setting)')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='maximum gradient norm in gradient clipping')
    parser.add_argument('--log_every',
                        help='Log training loss after number of batches',
                        type=int, default=1000)
    parser.add_argument('--val_every',
                        help='Validate after number of batches',
                        type=int, default=50000)

    args = parser.parse_args()

    args.device = None
    if not args.disable_cuda and torch.cuda.is_available():
        # default to cuda:0, as only 1 CUDA device would be visible
        args.device = torch.device('cuda:0')
    else:
        args.device = torch.device('cpu')

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

    assert args.query_type in ['normal', 'multi_hop', 'multi_arg', 'multi_slot']

    # load training dataset
    training_iter = load_seq_dataset(
        args.training_path,
        n_jobs=args.num_jobs,
        max_len=args.max_len,
        query_type=args.query_type,
        include_salience=args.use_salience,
        include_coref_pred_pairs=args.use_self_attn,
        use_bucket=args.use_bucket,
        device=args.device,
        batch_size=args.batch_size,
        sort_query=args.sort_query,
        train=True,
        sort_within_batch=True,
        repeat=False
    )

    # load validation dataset
    validation_iter = load_seq_dataset(
        args.validation_path,
        n_jobs=args.num_jobs,
        max_len=args.max_len,
        query_type=args.query_type,
        include_salience=args.use_salience,
        include_coref_pred_pairs=args.use_self_attn,
        use_bucket=args.use_bucket,
        device=args.device,
        batch_size=args.batch_size,
        sort_query=args.sort_query,
        train=False,
        sort_within_batch=True
    )

    # load word2vec embeddings
    word2vec_dir = Path(args.word2vec_dir)
    fname = word2vec_dir / (args.word2vec_name + '.bin')
    fvocab = word2vec_dir / (args.word2vec_name + '.vocab')
    assert fname.exists() and fvocab.exists()

    vocab = load_vocab(fname=str(fname), fvocab=str(fvocab),
                       use_target_specials=True, use_miss_specials=True)

    vocab_size, input_size = vocab.vectors.shape

    # initialize pointer net model
    log.info(
        'Initializing pointer network with vocab_size = {}, input_size = {}, '
        'hidden_size = {}, num_layers = {}, query_num_layers = {}, '
        'dropout_p = {}, attn_method = {}, rescale_attn_energy = {}, '
        'use_self_attn = {}, self_attn_method = {}, multi_hop = {}, '
        'extra_query_linear = {}, extra_doc_encoder = {}, query_aware = {}, '
        'use_salience = {}, salience_vocab_size = {}, '
        'salience_embedding_size = {}'
        ''.format(
            vocab_size, input_size, args.hidden_size, args.num_layers,
            args.query_num_layers, args.dropout_p, args.attn_method,
            args.rescale_attn_energy, args.use_self_attn, args.self_attn_method,
            args.multi_hop, args.extra_query_linear, args.extra_doc_encoder,
            args.query_aware, args.use_salience, args.salience_vocab_size,
            args.salience_embedding_size))

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
    log.info('Initializing embedding layer with pretrained word vectors.')
    pointer_net.init_embedding(vocab.vectors)

    log.info('Moving all model parameters to {}.'.format(args.device))
    pointer_net.to(device=args.device)

    log.info('Model specifications:\n{}'.format(pointer_net))

    # load pretrained model state
    if args.load_model_state:
        log.info('Loading pretrained model parameters from {}'.format(
            args.load_model_state))
        pointer_net.load_state_dict(torch.load(
            args.load_model_state, map_location=str(args.device)))

    # initialize optimizer
    optim_dict = {
        'adadelta': optim.Adadelta,
        'adagrad': optim.Adagrad,
        'adam': optim.Adam,
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

    # load pretrained optimizer state
    if args.load_optimizer_state:
        log.info('Loading optimizer state from {}'.format(
            args.load_optimizer_state))
        optimizer.load_state_dict(torch.load(
            args.load_optimizer_state, map_location=str(args.device)))

    scheduler = None
    if args.use_lr_annealing:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=args.lr_annealing_factor,
            patience=args.lr_annealing_epochs, verbose=True)

    # start training
    assert args.neg_loss_type in [
        'none', 'multi_arg', 'multi_slot', 'max_margin']

    log.info(
        'Training with neg_loss_type = {}, regularization = {}, use_sum = {}, '
        'use_sigmoid = {}, self_attn_target_for_pred = {}, '
        'backward_self_attn_loss = {}, backward_first_hop_attn_loss = {}, '
        'max_grad_norm = {}, log_every = {}, val_every = {}'.format(
            args.neg_loss_type, args.regularization, args.use_sum,
            args.use_sigmoid, args.self_attn_target_for_pred,
            args.backward_self_attn_loss, args.backward_first_hop_attn_loss,
            args.max_grad_norm, args.log_every, args.val_every))

    _ = validate(
        pointer_net, validation_iter,
        self_attn_target_for_pred=args.self_attn_target_for_pred,
        msg='Before training')

    for epoch in range(args.num_epochs):
        log.info('Start training epoch {:3d}/{:3d}'.format(
            epoch + 1, args.num_epochs))
        train_epoch(
            pointer_net=pointer_net,
            training_iter=training_iter,
            validation_iter=validation_iter,
            optimizer=optimizer,
            neg_loss_type=args.neg_loss_type,
            regularization=args.regularization,
            use_sum=args.use_sum,
            use_sigmoid=args.use_sigmoid,
            self_attn_target_for_pred=args.self_attn_target_for_pred,
            backward_self_attn_loss=args.backward_self_attn_loss,
            backward_first_hop_attn_loss=args.backward_first_hop_attn_loss,
            max_grad_norm=args.max_grad_norm,
            log_every=args.log_every,
            val_every=args.val_every
        )

        val_loss, _, _ = validate(
            pointer_net, validation_iter,
            self_attn_target_for_pred=args.self_attn_target_for_pred,
            msg='After {:3d}/{:3d} epochs'.format(epoch + 1, args.num_epochs))

        if args.use_lr_annealing:
            scheduler.step(val_loss)

        model_output_path = \
            checkpoint_path / 'epoch-{}-model'.format(epoch + 1)
        log.info('Save model states to {}'.format(model_output_path))
        torch.save(pointer_net.state_dict(), model_output_path)

        optim_output_path = \
            checkpoint_path / 'epoch-{}-optim'.format(epoch + 1)
        log.info('Save optimizer states to {}'.format(optim_output_path))
        torch.save(optimizer.state_dict(), optim_output_path)
